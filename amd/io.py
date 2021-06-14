import numpy as np
from .core.PeriodicSet import PeriodicSet

import warnings
def _warning(message, category, filename, lineno, file=None, line=None):
    return f'{filename}:{lineno}: {category.__name__}: {message}\n'
warnings.formatwarning = _warning


def lexsort(x):
    """lexicographically sort rows of matrix x."""
    return x[np.lexsort(np.rot90(x))]

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
            [a, 0, 0], 
            [b*cos_gamma, b*sin_gamma, 0],
            [c*cos_beta, c*cy, c*np.sqrt(cz_sqr)]
    ])


class Reader:
    """Base Reader class. Contains parser functions for
    converting ase CifBlock and ccdc Entry objects to PeriodicSets.
    """
    
    def __init__(self, remove_hydrogens=False,
                       allow_disorder=False,
                       dtype=np.float64,
                       heaviest_component=False     # ccdc only
                       ):
        
        # settings
        self.remove_hydrogens = remove_hydrogens
        self.allow_disorder = allow_disorder
        self.dtype = dtype
        self.heaviest_component = heaviest_component
        
    def _read(self, iterable, parser):
        """Iterates over iterable, passing items through parser and
        yielding the result if it is not None.
        """
        for item in iterable:
            periodic_set = parser(item)
            if periodic_set is not None:
                yield periodic_set
                
    ### Parsers. Intended to be passed to _read ###
    
    def _CIFBlock_to_PeriodicSet(self, block):
        """ase.io.cif.CIFBlock --> PeriodicSet. Returns None for a "bad" set."""
        
        if not self.allow_disorder:
            occupancies = block.get('_atom_site_occupancy')
            if occupancies is not None:
                occupancies = {occ for occ in occupancies if isinstance(occ, (float, int))}
                if any(o != 1 for o in occupancies):
                    warnings.warn(f'Skipping disordered structure {block.name}')
                    return None
        
        # often raises a UserWarning about space groups that doesn't appear to matter.
        failed = True
        from ase.spacegroup.spacegroup import SpacegroupValueError
        try:
            with warnings.catch_warnings(): 
                warnings.simplefilter('ignore')
                
                atoms = block.get_atoms()
                
                if self.remove_hydrogens:
                    nums = atoms.get_atomic_numbers()
                    for i in sorted(np.argwhere(nums == 1), reverse=True):
                        del atoms[i]
                        
                motif, cell = atoms.get_positions(wrap=True), atoms.get_cell().array
                failed = False
                
        except SpacegroupValueError as e:
            # if get_atoms fails (ase wanted more spacegroup info) and the only symop
            # is the identity, try getting data manually
            
            # to work, need excatly one symop (identity), cellpars and fract coords

            symops = block.get('_space_group_symop_operation_xyz')
            if symops is None:
                symops = block.get('_symmetry_equiv_pos_as_xyz')    # superceded by the above

            if symops is not None and len(symops) == 1:
                
                op = ','.join([s.replace('+', '') for s in symops[0].split(',')])
                if op == 'x,y,z':
                    
                    cellpar = block.get_cellpar()
                    if cellpar is not None:
                        
                        cell = cellpar_to_cell(*cellpar)
                        
                        coords = [block.get('_atom_site_fract_x'), block.get('_atom_site_fract_y'), block.get('_atom_site_fract_z')]
                        if None not in coords:
                        
                            frac_motif = np.array(coords).T
                            
                            if self.remove_hydrogens:
                                types = block.get('_atom_site_type_symbol')
                                non_H_indexes = [i for i, t in enumerate(types) if t not in 'HD']
                                frac_motif = frac_motif[non_H_indexes]
                            
                            motif = frac_motif @ cell
                            failed = False

        if failed:
            warnings.warn(f'Skipping {block.name}: insufficient spacegroup data for ase to read structure.')
            return None
        
        motif, cell = self._normalise_motif_cell(motif, cell)
        
        return PeriodicSet(name=block.name, motif=motif, cell=cell)
    
    
    def _Entry_to_PeriodicSet(self, entry):
        """ccdc.entry.Entry --> PeriodicSet. Returns None for a "bad" set."""
        
        try:
            crystal = entry.crystal
        except RuntimeError as e:
            warnings.warn(f'Skipping {entry.identifier}: {e}')
            return None
            
        if not entry.has_3d_structure:
            warnings.warn(f'Skipping {entry.identifier}: has no 3D structure')
            return None
        
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
        
        if self.remove_hydrogens:
            mol = crystal.molecule
            mol.remove_atoms(a for a in mol.atoms if a.atomic_symbol in 'HD')
            crystal.molecule = mol
            
        if not crystal.molecule.all_atoms_have_sites:
            warnings.warn(f'Skipping {entry.identifier} as some atoms do not have sites')
            return None
            
        # disorder check
        occupancies = {a.occupancy for a in crystal.molecule.atoms if isinstance(a.occupancy, (float, int))}
        if not self.allow_disorder:
            if crystal.has_disorder or any(occ != 1 for occ in occupancies):    # or entry.has_disorder:
                warnings.warn(f'Skipping disordered structure {entry.identifier}')
                return None
        
        atoms = crystal.packing(inclusion='OnlyAtomsIncluded').atoms
            
        frac_motif = np.array([[
                        a.fractional_coordinates.x, 
                        a.fractional_coordinates.y, 
                        a.fractional_coordinates.z]
                        for a in atoms])
        
        
        # packing duplicates points on the edge (within 0.001) of the unit cell, this removes dupes
        frac_motif = np.around(frac_motif, decimals=10)
        frac_motif = np.array([p for p in frac_motif if (p >= 0).all() and (p < 1).all()])
        
        cell = cellpar_to_cell(*crystal.cell_lengths, *crystal.cell_angles)
        motif = frac_motif @ cell
        
        motif, cell = self._normalise_motif_cell(motif, cell)
        
        return PeriodicSet(name=crystal.identifier, motif=motif, cell=cell)

    # gives more consistent outputs beteween readers 
    def _normalise_motif_cell(self, motif, cell):
        return lexsort(np.around(motif, decimals=10)).astype(self.dtype), np.around(cell, decimals=10).astype(self.dtype)


    ### public interface ###
    
    def __iter__(self):
        yield from self._generator

# Subclasses of Reader read from different sources (CSD, .cif).
# The __init__ function should initialise self._generator by calling
# self._read(iterable, parser) with an iterable which 

class CSDReader(Reader):
    
    def __init__(self, refcodes, families=False, **kwargs):
        super().__init__(**kwargs)
        
        if families:
            from ccdc.search import TextNumericSearch
            
            all_refcodes = []
            for refcode in refcodes:
                query = TextNumericSearch()
                query.add_identifier(refcode)
                all_refcodes.extend([hit.identifier for hit in query.search()])
        
            seen = set()
            seen_add = seen.add
            refcodes = [refcode for refcode in all_refcodes if not (refcode in seen or seen_add(refcode))]

        iterable = self._init_ccdc_reader(refcodes)
        self._generator = self._read(iterable, self._Entry_to_PeriodicSet)
    
    def _init_ccdc_reader(self, refcodes):
        """Generates ccdc Entries from CSD refcodes"""
        from ccdc.io import EntryReader
        reader = EntryReader('CSD')
        for refcode in refcodes:
            try:
                entry = reader.entry(refcode)
                yield entry
            except RuntimeError:
                warnings.warn(f'Identifier {refcode} not found in database.')

class CifReader(Reader):
    
    READERS = {'ase', 'ccdc'}    
        
    def __init__(self, filename, reader='ase', **kwargs):
        
        super().__init__(**kwargs)
        
        if reader not in CifReader.READERS:
            raise ValueError(f'Invalid reader {reader}. Reader must be one of {CifReader.READERS}')
        
        if self.heaviest_component and reader != 'ccdc':
            raise NotImplementedError(f'Parameter heaviest_component not implimented for {reader}, only ccdc.')
        
        # sets up self._generator which yields PeriodicSet objects
        if reader == 'ase':
            from ase.io.cif import parse_cif
            self._generator = self._read(parse_cif(filename), self._CIFBlock_to_PeriodicSet)
            
        elif reader == 'ccdc':
            from ccdc.io import EntryReader
            self._generator = self._read(EntryReader(filename), self._Entry_to_PeriodicSet)

if __name__ == '__main__':
    pass