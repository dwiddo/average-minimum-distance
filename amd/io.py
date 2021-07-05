from .core.PeriodicSet import PeriodicSet
import numpy as np
import warnings
def _warning(message, category, filename, lineno, file=None, line=None):
    return f'{filename}:{lineno}: {category.__name__}: {message}\n'
warnings.formatwarning = _warning

# error checks?
try:
    from ase.io.cif import parse_cif                     # .cif -> Iterable[CIFBlock]
    from ase.spacegroup.spacegroup import parse_sitesym  # string symop -> rot, trans
    _ASE_ENABLED = True
except ImportError:
    _ASE_ENABLED = False

try:
    from ccdc.io import EntryReader             # .cif or 'CSD' -> Iterable[Entry]
    from ccdc.search import TextNumericSearch   # refcode -> family of refcodes
    _CCDC_ENABLED = True
except ImportError:
    _CCDC_ENABLED = False

try:
    import h5py
    _H5PY_ENABLED = True
except ImportError:
    _H5PY_ENABLED = False


# def lexsort(x):
#     """lexicographically sort rows of matrix x."""
#     return x[np.lexsort(np.rot90(x))]

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
    """Base Reader class. Contains parser functions for
    converting ase CifBlock and ccdc Entry objects to PeriodicSets.
    """
    
    disorder_options = {'skip', 'ordered_sites', 'all_sites'}
    
    def __init__(self, 
                 remove_hydrogens=False,      
                 disorder='skip',
                 dtype=np.float64, 
                 heaviest_component=False,
                 types=False,
                ):
        
        # settings
        self.remove_hydrogens = remove_hydrogens
        if disorder not in _Reader.disorder_options:
            raise ValueError(f'disorder parameter {disorder} must be one of {_Reader.disorder_options}')
        self.disorder = disorder
        self.dtype = dtype
        self.heaviest_component = heaviest_component
        self.types = types
    
    # basically wraps an iterable passing its outputs through a function
    # and skips if the function returned None
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
                warnings.warn(f'Skipping {block.name}, coordinates not found in structure')
                return None
                
            asym_frac_motif = np.array(asym_motif) @ np.linalg.inv(cell)
            
        asym_frac_motif = np.array(asym_frac_motif).T     # fract coords in cif
        
        # get indexes of sites to remove in asymmetric unit
        remove = []
        
        # hydrogens
        asym_symbols = block.get_symbols()
        if self.remove_hydrogens:
            remove.extend([i for i, sym in enumerate(asym_symbols) if sym == 'H'])
        
        # disorder 
        disordered = []
        if self.disorder != 'all_sites':
            occupancies = block.get('_atom_site_occupancy')
            if occupancies is not None:
                # indices with fractional occupancy
                for i, occ in enumerate(occupancies):
                    if i not in remove and isinstance(occ, (float, int)) and occ < 1:
                        disordered.append(i)
                
                if self.disorder == 'skip' and len(disordered) > 0:
                    warnings.warn(f'Skipping {block.name} as structure is disordered')
                    return None
                else:
                    remove.extend(disordered)
        
        asym_frac_motif = np.delete(asym_frac_motif, remove, axis=0)
        asym_symbols = [s for i, s in enumerate(asym_symbols) if i not in remove]
        
        if asym_frac_motif.shape[0] == 0:
            warnings.warn('Skipping {block.name} with no accepted sites')
            return None
        
        # remove/replace?
        sitesym = block._get_sitesym()
        if not sitesym:
            sitesym = ('x,y,z',)

        rotations, translations = parse_sitesym(sitesym)
        
        frac_motif = []
        symbols = []
        for i, site in enumerate(asym_frac_motif):
            for rot, trans in zip(rotations, translations):
                site_ = np.mod(np.dot(rot, site) + trans, 1)
                if not frac_motif:
                    frac_motif.append(site_)
                    symbols.append(asym_symbols[i])
                    continue
                t = site_ - frac_motif
                mask = np.all( (abs(t) < 1e-3) | (abs(abs(t) - 1.0) < 1e-3), axis=1)
                if not np.any(mask) or (self.disorder == 'all_sites' and i in disordered):
                    frac_motif.append(site_)
                    symbols.append(asym_symbols[i])
                else:
                    warnings.warn(f'{block.name} has overlapping sites')
                    print(rot, trans, site, site_, frac_motif)
        
        frac_motif = np.array(frac_motif)
        
        frac_motif = np.mod(frac_motif, 1)  # wrap
        
        motif = frac_motif @ cell
        
        motif, cell = self._normalise_motif_cell(motif, cell)
        
        if self.types:
            periodic_set = PeriodicSet(motif, cell, name=block.name, types=symbols)
        else:
            periodic_set = PeriodicSet(motif, cell, name=block.name)
        
        return periodic_set
    
    
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
        
        
        # hydrogens
        if self.remove_hydrogens:
            mol = crystal.molecule
            mol.remove_atoms(a for a in mol.atoms if a.atomic_symbol in 'HD')
            crystal.molecule = mol
            
        if not crystal.molecule.all_atoms_have_sites:
            warnings.warn(f'In {entry.identifier} some atoms do not have sites')
            
        # disorder 
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
        
        """
        # not sure why, but ccdc may alter the symops from those in the cif.
        # tried to pack the cell directly with the symops but it didn't work.
        
        asym_frac_motif = np.array([[
                            a.fractional_coordinates.x, 
                            a.fractional_coordinates.y, 
                            a.fractional_coordinates.z]
                            for a in crystal.molecule.atoms])
        
        sitesym = crystal.symmetry_operators
        if not sitesym:
            sitesym = ('x,y,z',)
        
        rotations = []
        translations = []
        for op in sitesym:
            rot = crystal.symmetry_rotation(op)
            trans = crystal.symmetry_translation(op)
            rotations.append(np.array([rot[0:3], rot[3:6], rot[6:9]]))
            translations.append(trans)
        
        rotations = np.array(rotations)
        translations = np.array(translations)

        frac_motif = []
        for site in asym_frac_motif:
            for rot, trans in zip(rotations, translations):
                site_ = np.mod(np.dot(rot, site) + trans, 1)
                if not frac_motif:
                    frac_motif.append(site)
                    continue
                t = site_ - frac_motif
                mask = np.all((abs(t) < 1e-3) | (abs(abs(t) - 1.0) < 1e-3), axis=1)
                if not np.any(mask):
                    frac_motif.append(site_)
                else:
                    warnings.warn(f'{crystal.identifier} has overlapping sites')
        
        # frac_motif = np.array(frac_motif)
        """
        try:
            atoms = crystal.packing(inclusion='OnlyAtomsIncluded').atoms
        except RuntimeError as e:
            atoms = crystal.molecule.atoms
                    
        frac_motif = np.array([[
                        a.fractional_coordinates.x, 
                        a.fractional_coordinates.y, 
                        a.fractional_coordinates.z]
                        for a in atoms])
        
        if frac_motif.shape[0] == 0:
            warnings.warn('Skipping {block.name} with no accepted sites')
            return None
        
        # packing duplicates points on the edge (within 0.001) of the unit cell, this removes dupes
        frac_motif = np.around(frac_motif, decimals=10)
        
        types = [a.atomic_symbol for a in atoms]
        wrapped_inds = {i for i, p in enumerate(frac_motif) if (p >= 0).all() and (p < 1).all()}
        frac_motif = np.array([frac_motif[i] for i in range(len(frac_motif)) if i in wrapped_inds])
        types = [types[i] for i in range(len(types)) if i in wrapped_inds]
        
        cell = cellpar_to_cell(*crystal.cell_lengths, *crystal.cell_angles)
        motif = frac_motif @ cell
        
        motif, cell = self._normalise_motif_cell(motif, cell)
        
        if self.types:
            periodic_set = PeriodicSet(motif, cell, name=crystal.identifier, types=types)
        else:
            periodic_set = PeriodicSet(motif, cell, name=crystal.identifier)
        
        return periodic_set

    def _normalise_motif_cell(self, motif, cell):
        return motif.astype(self.dtype), cell.astype(self.dtype)


    ### public interface ###
    
    def __iter__(self):
        yield from self._generator

# Subclasses of _Reader read from different sources (CSD, .cif).
# The __init__ function should initialise self._generator by calling
# self._read(iterable, parser) with an iterable which 

class CSDReader(_Reader):
    """Given CSD refcodes, reads PeriodicSets with ccdc."""
    
    def __init__(self, refcodes, families=False, **kwargs):
        super().__init__(**kwargs)
        
        if families:
            
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
        reader = EntryReader('CSD')
        for refcode in refcodes:
            try:
                entry = reader.entry(refcode)
                yield entry
            except RuntimeError:
                warnings.warn(f'Identifier {refcode} not found in database.')

class CifReader(_Reader):
    """Read all structures in a .cif with ase or ccdc, yielding PeriodicSets."""
    
    READERS = {'ase', 'ccdc'}    
        
    def __init__(self, filename, reader='ase', **kwargs):
        
        super().__init__(**kwargs)
        
        if reader not in CifReader.READERS:
            raise ValueError(f'Invalid reader {reader}. Reader must be one of {CifReader.READERS}')

        
        if self.heaviest_component and reader != 'ccdc':
            raise NotImplementedError(f'Parameter heaviest_component not implimented for {reader}, only ccdc.')
        
        # sets up self._generator which yields PeriodicSet objects
        if reader == 'ase':
            
            if not _ASE_ENABLED:
                raise ImportError(f'Failed to import ase')
            
            self._generator = self._read(parse_cif(filename), self._CIFBlock_to_PeriodicSet)
            
        elif reader == 'ccdc':
            
            if not _CCDC_ENABLED:
                raise ImportError(f'Failed to import ccdc.')
            
            self._generator = self._read(EntryReader(filename), self._Entry_to_PeriodicSet)



class SetWriter:
    """Write PeriodicSets to a .hdf5 file. Requires h5py.
    Reading the .hdf5 is much faster than parsing a big .cif.
    Can optionally pass other data, intended to store AMDs/PDDs.
    """

    str_dtype = h5py.special_dtype(vlen=str)
    
    def __init__(self, filename):
        
        if not _H5PY_ENABLED:
            raise ImportError('Failed to import h5py.')
        
        self.file = h5py.File(filename, 'w', track_order=True)
        
    def write(self, periodic_set):
        
        # need a name to store or you can't access items by key
        if periodic_set.name is None:
            raise ImportError('Periodic set must have a name to save to .hdf5.')
        
        # this group is the PeriodicSet
        group = self.file.create_group(periodic_set.name)
        
        # datasets in the group for motif and cell
        group.create_dataset('motif', data=periodic_set.motif)
        group.create_dataset('cell', data=periodic_set.cell)
        
        if periodic_set.tags:
            
            # a subgroup contains tags that are lists or ndarrays
            tags_group = group.create_group('tags')
            for tag in periodic_set.tags:
                
                data = periodic_set.tags[tag]
                
                # scalars (nums and strs) stored as attrs
                if np.isscalar(data):
                    tags_group.attrs[tag] = data
                
                elif isinstance(data, np.ndarray):
                    tags_group.create_dataset(tag, data=data)
                    
                elif isinstance(data, list):
                    
                    # lists of strings stored as special type for some reason
                    # if any string is in the tags
                    if any(isinstance(d, str) for d in data):
                        data = [str(d) for d in data]
                        tags_group.create_dataset(tag, data=data, 
                                                  dtype=SetWriter.str_dtype)
                        
                    # everything else castable to ndarray
                    else:
                        data = np.asarray(data)
                        tags_group.create_dataset(tag, data=np.array(data))
                else:
                    raise ValueError(f'Cannot store object {data} of type {type(data)} in hdf5')
    
    def iwrite(self, periodic_sets):
        for periodic_set in periodic_sets:
            self.write(periodic_set)
        
    def close(self):
        self.file.close()
        
    def __enter__(self):
        return self

    # handle exceptions?
    def __exit__(self, exc_type, exc_value, tb):
        self.file.close()

class SetReader:
    """Read PeriodicSets from a .hdf5 file written with SetWriter. Requires h5py.
    Reading the .hdf5 is much faster than parsing a big .cif.
    """
    
    def __init__(self, filename):
        
        if not _H5PY_ENABLED:
            raise ImportError('Failed to import h5py.')
        
        self.file = h5py.File(filename, 'r', track_order=True)

    def _get_set(self, name):
        # take a name in the set and return the PeriodicSet
        
        group = self.file[name]
        periodic_set = PeriodicSet(group['motif'][:], group['cell'][:], name=name)
        
        if 'tags' in group:
            for tag in group['tags']:
                # tag: data e.g. 'types': ['C', 'H', 'O',...]
                data = group['tags'][tag][:]
                
                if any(isinstance(d, (bytes, bytearray)) for d in data):
                    periodic_set.tags[tag] = [d.decode() for d in data]
                else:
                    periodic_set.tags[tag] = data
        
        for attr in group.attrs:
            periodic_set.tags[tag] = group.attrs[attr]

        return periodic_set
    
    def __getitem__(self, name):
        # index by name. Not found exc?
        return self._get_set(name)

    def __iter__(self):
        # interface to loop over the SetReader
        # does not close the SetReader when done
        for name in self.file['/'].keys():
            yield self._get_set(name)

    def close(self):
        self.file.close()
        
    def __enter__(self):
        return self

    # handle exceptions?
    def __exit__(self, exc_type, exc_value, tb):
        self.file.close()


if __name__ == '__main__':
    pass