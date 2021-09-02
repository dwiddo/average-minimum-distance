"""
The base _Reader class is intended to be inherited from
to read from different sources.
For example, given a source for structures X which yields
Python objects called Y from some reader, one can inherit
from this class to make a reader for PeriodicSets.

The reader class should contain the parser/converter
going from Y type objects to PeriodicSets. The child class
has a constructor which should set up a generator named
self._generator which yields PeriodicSets (or None for
invaild structures.)
"""

import itertools
import numpy as np
from ase.io.cif import NoStructureData
from ase.spacegroup.spacegroup import parse_sitesym  # string symop -> rot, trans

from ..core.PeriodicSet import PeriodicSet

import warnings
def _warning(message, category, filename, lineno, file=None, line=None):
    return f'{filename}:{lineno}: {category.__name__}: {message}\n'
warnings.formatwarning = _warning


def cellpar_to_cell(a, b, c, alpha, beta, gamma):
    """Simplified version of function from ase.geometry.
    
    Unit cell params a,b,c,α,β,γ --> cell as 3x3 ndarray.
    """

    # Handle orthorhombic cells separately to avoid rounding errors
    eps = 2 * np.spacing(90.0, dtype=np.float64)  # around 1.4e-14
    
    cos_alpha = 0. if abs(abs(alpha) - 90.) < eps else np.cos(alpha * np.pi / 180.)
    cos_beta  = 0. if abs(abs(beta)  - 90.) < eps else np.cos(beta * np.pi / 180.)
    cos_gamma = 0. if abs(abs(gamma) - 90.) < eps else np.cos(gamma * np.pi / 180.)
    
    if abs(gamma - 90) < eps:
        sin_gamma = 1.
    elif abs(gamma + 90) < eps:
        sin_gamma = -1.
    else:
        sin_gamma = np.sin(gamma * np.pi / 180.)

    cy = (cos_alpha - cos_beta * cos_gamma) / sin_gamma
    cz_sqr = 1. - cos_beta ** 2 - cy ** 2
    assert cz_sqr >= 0
    
    return np.array([
            [a,           0,           0], 
            [b*cos_gamma, b*sin_gamma, 0],
            [c*cos_beta,  c*cy,        c*np.sqrt(cz_sqr)]])


class _Reader:
    """Base Reader class. Contains parsers for converting ase CifBlock 
    and ccdc Entry objects to PeriodicSets.
    
    Intended use:
    
    First make a new method for _Reader converting object to PeriodicSet 
    (e.g. named _X_to_PSet). Then make this class outline:
    
    class XReader(_Reader):
        def __init__(self, ..., **kwargs):
            
        super().__init__(**kwargs)
        
        # setup and checks
        
        # make 'iterable' which yields objects to be converted (e.g. CIFBlock, Entry)
        
        # set self._generator like this
        self._generator = self._read(iterable, self._X_to_PSet)
    
    """

    disorder_options = {'skip', 'ordered_sites', 'all_sites'}
    
    def __init__(self, 
                 remove_hydrogens=False,      
                 disorder='skip',
                 heaviest_component=False,
                 dtype=np.float64, 
                 types=False,
                ):
        
        # settings
        self.remove_hydrogens = remove_hydrogens
        if disorder not in _Reader.disorder_options:
            raise ValueError(f'disorder parameter {disorder} must be one of {_Reader.disorder_options}')
        self.disorder = disorder
        self.dtype = dtype
        self.heaviest_component = heaviest_component

        if types:
            warnings.warn("types=True as an argument to a reader is deprecated. Atomic types are now always read regardless.", category=DeprecationWarning)

    # basically the builtin map, but skips items if the function returned None
    def _map(self, func, iterable):
        """Iterates over iterable, passing items through parser and
        yielding the result if it is not None.
        """
        for item in iterable:
            periodic_set = func(item)
            if periodic_set is not None:
                yield periodic_set

    def _expand(self, asym_frac_motif, sitesym):
        """
        Asymmetric unit's fractional coords + sitesyms (as strings)
        -->
        frac_motif, asym_unit, multiplicities, inverses
        """
        
        rotations, translations = parse_sitesym(sitesym)
        
        all_sites = []
        for site in asym_frac_motif:
            
            # equivalent sites
            sites = np.array([np.mod(np.dot(rot, site) + trans, 1) 
                              for rot, trans in zip(rotations, translations)])
            
            # find overlapping sites (asym unit site invariant under symop) and keep unique sites
            site_diffs = np.abs(sites[:, None] - sites)
            reduced_sites = sites[~np.triu(np.all(np.logical_or(site_diffs <= 1e-3, np.abs(site_diffs - 1) <= 1e-3), axis=-1), 1).any(0)]
            
            all_sites.append(reduced_sites)
            
        # Wyckoff multiplicites = unique sites generated for each point in asym unit
        multiplicities = np.array([sites.shape[0] for sites in all_sites])
        inverses = list(itertools.chain.from_iterable([itertools.repeat(i, m) for i, m in enumerate(multiplicities)]))
        asym_unit = np.array([0] + list(itertools.accumulate(multiplicities[:-1])))
        
        frac_motif = np.concatenate(all_sites)
        
        return frac_motif, asym_unit, multiplicities, inverses
   

    ### Parsers. Intended to be passed to _read ###
    
    def _CIFBlock_to_PeriodicSet(self, block):
        """ase.io.cif.CIFBlock --> PeriodicSet. Returns None for a "bad" set."""
        
        # could replace with get_cellpar + cell_to_cellpar. Is there any point?
        cell = block.get_cell().array
        
        # get asymmetric unit's fractional motif
        asym_frac_motif = [block.get(name) for name in ['_atom_site_fract_x',
                                                        '_atom_site_fract_y',
                                                        '_atom_site_fract_z']]
        if None in asym_frac_motif:
            asym_motif = [block.get(name) for name in ['_atom_site_cartn_x',
                                                       '_atom_site_cartn_y',
                                                       '_atom_site_cartn_z']]
            if None in asym_motif:
                warnings.warn(f'Skipping {block.name} as coordinates were not found')
                return None
                
            asym_frac_motif = np.array(asym_motif) @ np.linalg.inv(cell)
            
        asym_frac_motif = np.array(asym_frac_motif).T     # fract coords in cif
        
        # remove will contain indices of all sites to remove
        remove = []
        
        try:
            asym_symbols = block.get_symbols()
        except NoStructureData:
            asym_symbols = ['Unknown' for _ in range(len(asym_frac_motif))]

        # remove Hydrogens if requested
        if self.remove_hydrogens:
            remove.extend([i for i, sym in enumerate(asym_symbols) if sym == 'H'])
        
        # disorder. if 'all_sites', include all sites. if 'ordered_sites', remove sites with disorder.
        # if 'skip', don't read this structure
        disordered = []             # list of indices where disorder is
        asym_is_disordered = []     # True/False list same length as asym unit
        occupancies = block.get('_atom_site_occupancy')
        # if no disorder information, assume there is no disorder
        if occupancies is not None:
            # get indices of sites with fractional occupancy
            for i, occ in enumerate(occupancies):
                if i not in remove and isinstance(occ, (float, int)) and occ < 1:
                    disordered.append(i)
                    asym_is_disordered.append(True)
                else:
                    asym_is_disordered.append(False)

            if self.disorder == 'skip' and len(disordered) > 0:
                warnings.warn(f'Skipping {block.name} as structure is disordered')
                return None
            elif self.disorder == 'ordered_sites':
                remove.extend(disordered)
        
        # get rid of unwanted sites in asym unit
        asym_frac_motif = np.mod(np.delete(asym_frac_motif, remove, axis=0), 1)
        asym_symbols = [s for i, s in enumerate(asym_symbols) if i not in remove]
        asym_is_disordered = [v for i, v in enumerate(asym_is_disordered) if i not in remove]

        site_diffs = np.abs(asym_frac_motif[:, None] - asym_frac_motif)
        overlapping = np.triu(np.all((site_diffs <= 1e-3) | (np.abs(site_diffs - 1) <= 1e-3), axis=-1), 1)
        
        # if disorder is 'all_sites' and overlapping sites are disordered, don't remove either
        if self.disorder == 'all_sites':
            for i, j in np.argwhere(overlapping):
                if asym_is_disordered[i] and asym_is_disordered[j]:
                    overlapping[i, j] = False
        
        # if sites in asym unit overlap, warn and keep only one
        if overlapping.any():
            warnings.warn(f'{block.name} may have overlapping sites')
            keep_sites = ~overlapping.any(0)
            asym_frac_motif = asym_frac_motif[keep_sites]
            asym_symbols = [t for t, keep in zip(asym_symbols, keep_sites) if keep]
        
        if asym_frac_motif.shape[0] == 0:
            warnings.warn(f'Skipping {block.name} with no accepted sites')
            return None
        
        symop_tags = ['_space_group_symop_operation_xyz',
                      '_space_group_symop.operation_xyz',
                      '_symmetry_equiv_pos_as_xyz']

        sitesym = ('x,y,z',)
        for tag in symop_tags:
            if tag in block:
                sitesym = block[tag]
                break

        if isinstance(sitesym, str):
            sitesym = [sitesym]
        
        frac_motif, asym_unit, multiplicities, inverses = self._expand(asym_frac_motif, sitesym)
        motif = frac_motif @ cell
        motif, cell = motif.astype(self.dtype), cell.astype(self.dtype)
        types = [asym_symbols[i] for i in inverses]
    
        kwargs = {
            'name': block.name, 
            'asymmetric_unit': asym_unit,
            'wyckoff_multiplicities': multiplicities,
            'types': types
        }

        periodic_set = PeriodicSet(motif, cell, **kwargs)
        
        return periodic_set
    
    
    def _Entry_to_PeriodicSet(self, entry):
        """ccdc.entry.Entry --> PeriodicSet. Returns None for a "bad" set."""
        
        # basic checks first. Some exclude the structure from consideration
        try:
            crystal = entry.crystal
        except RuntimeError as e:
            warnings.warn(f'Skipping {entry.identifier}: {e}')
            return None
            
        if not entry.has_3d_structure:
            warnings.warn(f'Skipping {entry.identifier}: has no 3D structure')
            return None
        
        # heaviest component (removes all but the heaviest component of the asym unit)
        if self.heaviest_component:
            if len(crystal.molecule.components) > 1:
                component_weights = []
                for component in crystal.molecule.components:
                    weight = 0
                    for a in component.atoms:
                        if isinstance(a.occupancy, (float, int)):
                            weight += a.occupancy * a.atomic_weight
                        else:
                            weight += a.atomic_weight
                    component_weights.append(weight)
                crystal.molecule = crystal.molecule.components[np.argmax(np.array(component_weights))]
        
        # remove hydrogens
        if self.remove_hydrogens:
            mol = crystal.molecule
            mol.remove_atoms(a for a in mol.atoms if a.atomic_symbol in 'HD')
            crystal.molecule = mol
        
        # check all atoms have coords
        # if not crystal.molecule.all_atoms_have_sites:
        #     warnings.warn(f'Skipping {entry.identifier} as some atoms do not have sites')
        #     return None
        
        # above check doesn't catch everything?
        # mol = crystal.molecule
        # remove = []
        # for atom in crystal.molecule.atoms:
        #     if atom.fractional_coordinates is None:
        #         remove.append(atom)
        # mol.remove_atoms(remove)
        # crystal.molecule = mol
        
                # warnings.warn(f'Skipping {entry.identifier} as some atoms do not have sites')
                # return None
        
        # disorder. if 'all_sites', get indices of disordered atoms
        # otherwise, look for disorder and remove if 'ordered_sites' or skip if 'skip'
        if self.disorder != 'all_sites':
            
            mol = crystal.molecule
            
            disordered = []
            for atom in mol.atoms:
                occ = atom.occupancy
                if isinstance(occ, (float, int)) and occ < 1:
                    disordered.append(atom)
            
            if len(disordered) > 0:
                if self.disorder == 'skip':
                    warnings.warn(f'Skipping {entry.identifier} as structure is disordered')
                    return None
                elif self.disorder == 'ordered_sites':
                    mol.remove_atoms(disordered)
            
            crystal.molecule = mol
            
        else:
            
            asym_is_disordered = []
            for atom in crystal.asymmetric_unit_molecule.atoms:
                val = True if (isinstance(atom.occupancy, (float, int)) and atom.occupancy < 1) else False
                asym_is_disordered.append(val)
            
        
        mol = crystal.molecule
        remove = []
        for atom in mol.atoms:
            if atom.fractional_coordinates is None:
                remove.append(atom)
        mol.remove_atoms(remove)
        crystal.molecule = mol
        
        
        # asymmetric unit fractional coordinates
        asym_frac_motif = np.array([[
                            a.fractional_coordinates.x, 
                            a.fractional_coordinates.y, 
                            a.fractional_coordinates.z]
                            for a in crystal.asymmetric_unit_molecule.atoms])
        
        asym_frac_motif = np.mod(asym_frac_motif, 1)
        asym_symbols = [a.atomic_symbol for a in crystal.asymmetric_unit_molecule.atoms]
        
        # check for overlapping sites
        site_diffs = np.abs(asym_frac_motif[:, None] - asym_frac_motif)
        overlapping = np.triu(np.all((site_diffs <= 1e-3) | (np.abs(site_diffs - 1) <= 1e-3), axis=-1), 1)
        
        # if disorder is 'all_sites' and overlapping sites are disordered, don't remove either
        if self.disorder == 'all_sites':
            for i, j in np.argwhere(overlapping):
                if asym_is_disordered[i] and asym_is_disordered[j]:
                    overlapping[i, j] = False
        
        # if sites in asym unit overlap, warn and keep only one
        if overlapping.any():
            warnings.warn(f'{entry.identifier} may have overlapping sites')
            keep_sites = ~overlapping.any(0)
            asym_frac_motif = asym_frac_motif[keep_sites]
            asym_symbols = [t for t, keep in zip(asym_symbols, keep_sites) if keep]
        
        # if the above removed everything, skip this structure
        if asym_frac_motif.shape[0] == 0:
            warnings.warn(f'Skipping {entry.identifier} with no accepted sites')
            return None
        
        sitesym = crystal.symmetry_operators
        if not sitesym: sitesym = ('x,y,z',)
        
        frac_motif, asym_unit, multiplicities, inverses = self._expand(asym_frac_motif, sitesym)
        
        cell = cellpar_to_cell(*crystal.cell_lengths, *crystal.cell_angles)
        motif = frac_motif @ cell
        motif, cell = motif.astype(self.dtype), cell.astype(self.dtype)
        
        kwargs = {
            'name': crystal.identifier, 
            'asymmetric_unit': asym_unit,
            'wyckoff_multiplicities': multiplicities,
            'types': [asym_symbols[i] for i in inverses]
        }

        periodic_set = PeriodicSet(motif, cell, **kwargs)
        
        return periodic_set


    ### public interface ###
    
    def __iter__(self):
        yield from self._generator
