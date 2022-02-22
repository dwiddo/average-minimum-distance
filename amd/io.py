"""Contains I/O tools, including a .CIF reader and CSD reader 
(``csd-python-api`` only) to extract periodic set representations 
of crystals which can be passed to :func:`.calculate.AMD` and :func:`.calculate.PDD`.

The :class:`CifReader` and :class:`CSDReader` return :class:`.periodicset.PeriodicSet`
objects, which can be given to :func:`.calculate.AMD` and :func:`.calculate.PDD` to 
calculate the AMD/PDD of a crystal. 

These intermediate :class:`.periodicset.PeriodicSet` representations can be written 
to a .hdf5 file with :class:`SetWriter` (along with their metadata), which can be 
read back with :class:`SetReader`. This is much faster than rereading a .CIF or
recomputing invariants.
"""

from typing import Callable, Iterable, Sequence, Tuple, Optional
from itertools import chain
import os
import numpy as np
from ase.io.cif import NoStructureData, CIFBlock, parse_cif
from ase.spacegroup.spacegroup import parse_sitesym     # string symop -> rot, trans
import h5py

from .periodicset import PeriodicSet
from .utils import _extend_signature, cellpar_to_cell

import warnings
def _warning(message, category, filename, lineno, file=None, line=None):
    return f'{filename}:{lineno}: {category.__name__}: {message}\n'
warnings.formatwarning = _warning

try:
    from ccdc.io import EntryReader             # .cif or 'CSD' -> Iterable[Entry]
    from ccdc.search import TextNumericSearch   # refcode -> family of refcodes
    _CCDC_ENABLED = True
except Exception:
    _CCDC_ENABLED = False
    

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
    reserved_tags = {'motif', 'cell', 'name', 'asymmetric_unit', 'wyckoff_multiplicities', 'types'}
    atom_site_fract_tags = ['_atom_site_fract_x', '_atom_site_fract_y', '_atom_site_fract_z']
    atom_site_cartn_tags = ['_atom_site_cartn_x', '_atom_site_cartn_y', '_atom_site_cartn_z']
    symop_tags = ['_space_group_symop_operation_xyz', '_space_group_symop.operation_xyz', '_symmetry_equiv_pos_as_xyz']
    
    equiv_site_tol = 1e-3
    
    def __init__(self, 
                 remove_hydrogens=False,      
                 disorder='skip',
                 heaviest_component=False,
                 extract_data=None,
                 include_if=None,
                 show_warnings=True,
                ):
        
        # settings
        if disorder not in _Reader.disorder_options:
            raise ValueError(f'disorder parameter {disorder} must be one of {_Reader.disorder_options}')
        
        if extract_data:
            if not isinstance(extract_data, dict):
                raise ValueError('extract_data must be a dict with callable values')
            for key in extract_data:
                if not callable(extract_data[key]):
                    raise ValueError('extract_data must be a dict with callable values')
                if key in _Reader.reserved_tags:
                    raise ValueError(f'extract_data includes reserved key {key}; change {key} to a different value to resolve')
        
        if include_if:
            for f in include_if:
                if not callable(f):
                    raise ValueError('include_if must be a list of callables')
                
        self.remove_hydrogens = remove_hydrogens
        self.disorder = disorder
        self.heaviest_component = heaviest_component
        self.extract_data = extract_data
        self.include_if = include_if
        self.show_warnings = show_warnings
    
    ### public interface ###
    
    def __iter__(self):
        yield from self._generator

    def read_one(self):
        return next(iter(self._generator))
    
    # basically the builtin map, but skips items if the function returned None.
    # The object returned by this function (Iterable of PeriodicSets) is set to
    # self._generator; then iterating over the Reader iterates over self._generator.
    @staticmethod
    def _map(func: Callable, iterable: Iterable) -> Iterable[PeriodicSet]:
        """Iterates over iterable, passing items through parser and
        yielding the result if it is not None.
        """
        
        for item in iterable:
            periodic_set = func(item)
            if periodic_set is not None:
                yield periodic_set

    @staticmethod
    def _expand(asym_frac_motif: np.ndarray, sitesym: Sequence[str], show_warnings=True) -> Tuple[np.ndarray, ...]:
        """
        Asymmetric unit's fractional coords + sitesyms (as strings)
        -->
        frac_motif, asym_unit, multiplicities, inverses
        """

        rotations, translations = parse_sitesym(sitesym)
        all_sites = []
        asym_unit = [0]
        multiplicities = []
        inverses = []
        
        for inv, site in enumerate(asym_frac_motif):
            multiplicity = 0
            
            for rot, trans in zip(rotations, translations):
                site_ = np.mod(np.dot(rot, site) + trans, 1)
                
                if not all_sites:
                    all_sites.append(site_)
                    inverses.append(inv)
                    multiplicity += 1
                    continue
                
                diffs = np.abs(site_ - all_sites)
                mask = np.all(np.logical_or(diffs <= _Reader.equiv_site_tol, np.abs(diffs - 1) <= _Reader.equiv_site_tol), axis=-1)
                
                if np.any(mask):
                    where_equal = np.argwhere(mask).flatten()
                    for ind in where_equal:
                        if inverses[ind] == inv:
                            pass
                        else:
                            if show_warnings:
                                warnings.warn(f'positions {inverses[ind]} and {inv} are equivalent')
                else:
                    all_sites.append(site_)
                    inverses.append(inv)
                    multiplicity += 1

            if multiplicity > 0:
                multiplicities.append(multiplicity)
                asym_unit.append(len(all_sites))
        
        frac_motif = np.array(all_sites)
        asym_unit = np.array(asym_unit[:-1])
        multiplicities = np.array(multiplicities)
        return frac_motif, asym_unit, multiplicities, inverses
   
    ### Parsers. Intended to be passed to _map ###
    
    def _CIFBlock_to_PeriodicSet(self, block: CIFBlock) -> PeriodicSet:
        """ase.io.cif.CIFBlock --> PeriodicSet. Returns None for a "bad" set."""
        
        if self.include_if:
            if not all(check(block) for check in self.include_if):
                return None
            
        # cell & asymmetric unit
        cell = block.get_cell().array
        asym_frac_motif = [block.get(name) for name in _Reader.atom_site_fract_tags]
        if None in asym_frac_motif:
            asym_motif = [block.get(name) for name in _Reader.atom_site_cartn_tags]
            if None in asym_motif:
                if self.show_warnings:
                    warnings.warn(f'Skipping {block.name} as coordinates were not found')
                return None
            asym_frac_motif = np.array(asym_motif) @ np.linalg.inv(cell)   
        asym_frac_motif = np.array(asym_frac_motif).T
        
        try:
            asym_symbols = block.get_symbols()
        except NoStructureData as e:
            asym_symbols = ['Unknown' for _ in range(len(asym_frac_motif))]

        remove = [] # indices of sites to remove
        if self.remove_hydrogens:
            remove.extend((i for i, sym in enumerate(asym_symbols) if sym in 'HD'))
        
        # disorder
        asym_is_disordered = []
        occupancies = block.get('_atom_site_occupancy')
        labels = block.get('_atom_site_label')
        if occupancies is not None:
            disordered = []     # indices where there is disorder
            for i, (occ, label) in enumerate(zip(occupancies, labels)):
                if (isinstance(occ, (float, int)) and occ < 1) or label.endswith('?'):
                    if i not in remove:
                        disordered.append(i)
                        asym_is_disordered.append(True)
                else:
                    asym_is_disordered.append(False)

            if self.disorder == 'skip' and len(disordered) > 0:
                if self.show_warnings:
                    warnings.warn(f'Skipping {block.name} as structure is disordered')
                return None
            elif self.disorder == 'ordered_sites':
                remove.extend(disordered)
        
        # remove sites
        asym_frac_motif = np.mod(np.delete(asym_frac_motif, remove, axis=0), 1)
        asym_symbols = [s for i, s in enumerate(asym_symbols) if i not in remove]
        asym_is_disordered = [v for i, v in enumerate(asym_is_disordered) if i not in remove]
        
        # if there are overlapping sites in asym unit, warn and keep only one
        site_diffs = np.abs(asym_frac_motif[:, None] - asym_frac_motif)
        overlapping = np.triu(np.all((site_diffs <= _Reader.equiv_site_tol) | (np.abs(site_diffs - 1) <= _Reader.equiv_site_tol), axis=-1), 1)
        
        if self.disorder == 'all_sites':    # don't remove overlapping sites if one is disordered
            for i, j in np.argwhere(overlapping):
                if asym_is_disordered[i] or asym_is_disordered[j]:
                    overlapping[i, j] = False
        
        if overlapping.any():
            if self.show_warnings:
                warnings.warn(f'{block.name} may have overlapping sites; duplicates will be removed')
            keep_sites = ~overlapping.any(0)
            asym_frac_motif = asym_frac_motif[keep_sites]
            asym_symbols = [sym for sym, keep in zip(asym_symbols, keep_sites) if keep]
        
        if asym_frac_motif.shape[0] == 0:
            if self.show_warnings:
                warnings.warn(f'Skipping {block.name} as there are no sites with coordinates')
            return None

        sitesym = ('x,y,z',)
        for tag in _Reader.symop_tags:
            if tag in block:
                sitesym = block[tag]
                break
            
        if isinstance(sitesym, str): sitesym = [sitesym]
        
        frac_motif, asym_unit, multiplicities, inverses = self._expand(asym_frac_motif, sitesym, 
                                                                       show_warnings=self.show_warnings)
        motif = frac_motif @ cell

        kwargs = {
            'name':                     block.name, 
            'asymmetric_unit':          asym_unit,
            'wyckoff_multiplicities':   multiplicities,
            'types':                    [asym_symbols[i] for i in inverses],
        }
        
        if self.extract_data is not None:
            for key in self.extract_data:
                kwargs[key] = self.extract_data[key](block)
                
        periodic_set = PeriodicSet(motif, cell, **kwargs)
        
        return periodic_set
    
    def _Entry_to_PeriodicSet(self, entry) -> PeriodicSet:
        """ccdc.entry.Entry --> PeriodicSet. Returns None for a "bad" set."""
        
        if self.include_if:
            if not all(check(entry) for check in self.include_if):
                return None
        
        # basic checks first. Some exclude the structure from consideration
        if not entry.has_3d_structure:
            if self.show_warnings:
                warnings.warn(f'Skipping {entry.identifier} as entry has no 3D structure')
            return None
        
        # at least one entry raised RuntimeError when reading its crystal
        try:
            crystal = entry.crystal
        except RuntimeError as e:
            if self.show_warnings:
                warnings.warn(f'Skipping {entry.identifier}: {e}')
            return None
        
        # first disorder check, if skipping. If occ == 1 for all atoms but the entry
        # or crystal is listed as having disorder, skip (can't know where disorder is).
        # If occ != 1 for any atoms, we wait to see if we remove them before skipping.
        molecule = crystal.disordered_molecule
        if self.disorder == 'ordered_sites':
            molecule.remove_atoms(a for a in molecule.atoms if a.label.endswith('?'))
        
        may_have_disorder = False
        if self.disorder == 'skip':
            for a in molecule.atoms:
                occ = a.occupancy
                if a.label.endswith('?') or (isinstance(occ, (int, float)) and occ < 1):
                    may_have_disorder = True
                    break
                
            if not may_have_disorder:
                if crystal.has_disorder or entry.has_disorder:
                    if self.show_warnings:
                        warnings.warn(f'Skipping {crystal.identifier} as structure is disordered')
                    return None
            
        if self.remove_hydrogens:
            molecule.remove_atoms(a for a in molecule.atoms if a.atomic_symbol in 'HD')
            
        # heaviest component (removes all but the heaviest component of the asym unit)
        # intended for removing solvents. probably doesn't play well with disorder
        if self.heaviest_component:
            if len(molecule.components) > 1:
                component_weights = []
                for component in molecule.components:
                    weight = 0
                    for a in component.atoms:
                        if isinstance(a.atomic_weight, (float, int)):
                            if isinstance(a.occupancy, (float, int)):
                                weight += a.occupancy * a.atomic_weight
                            else:
                                weight += a.atomic_weight
                    component_weights.append(weight)
                molecule = molecule.components[np.argmax(np.array(component_weights))]
            
        crystal.molecule = molecule
        
        # by here all atoms to be removed have been (except via ordered_sites).
        # If disorder == 'skip' and there were atom(s) with occ < 1 found 
        # eariler, we check if all such atoms were removed. If not, skip.
        if self.disorder == 'skip' and may_have_disorder:
            for a in crystal.disordered_molecule.atoms:
                occ = a.occupancy
                if a.label.endswith('?') or (isinstance(occ, (int, float)) and occ < 1):
                    if self.show_warnings:
                        warnings.warn(f'Skipping {crystal.identifier} as structure is disordered')
                    return None
        
        # if disorder is all_sites, we need to know where disorder is to ignore overlaps
        asym_is_disordered = []     # True/False list same length as asym unit
        if self.disorder == 'all_sites':
            for atom in crystal.asymmetric_unit_molecule.atoms:
                occ = atom.occupancy
                if a.label.endswith('?') or (isinstance(occ, (int, float)) and occ < 1):
                    asym_is_disordered.append(True)
                else:
                    asym_is_disordered.append(False)
    
        # check all atoms have coords. option/default remove unknown sites?
        if not molecule.all_atoms_have_sites or any(a.fractional_coordinates is None for a in molecule.atoms):
            if self.show_warnings:
                warnings.warn(f'Skipping {crystal.identifier} as some atoms do not have sites')
            return None
        
        # cell & asymmetric unit
        cell = cellpar_to_cell(*crystal.cell_lengths, *crystal.cell_angles)
        asym_frac_motif = np.array([tuple(a.fractional_coordinates) 
                                    for a in crystal.asymmetric_unit_molecule.atoms])
        asym_frac_motif = np.mod(asym_frac_motif, 1)
        asym_symbols = [a.atomic_symbol for a in crystal.asymmetric_unit_molecule.atoms]
        
        # if there are overlapping sites in asym unit, warn and remove one
        site_diffs = np.abs(asym_frac_motif[:, None] - asym_frac_motif)
        overlapping = np.triu(np.all((site_diffs <= _Reader.equiv_site_tol) | (np.abs(site_diffs - 1) <= _Reader.equiv_site_tol), axis=-1), 1)

        if self.disorder == 'all_sites':    # don't remove overlapping sites if either is disordered
            for i, j in np.argwhere(overlapping):
                if asym_is_disordered[i] or asym_is_disordered[j]:
                    overlapping[i, j] = False
        
        if overlapping.any():
            if self.show_warnings:
                warnings.warn(f'{crystal.identifier} may have overlapping sites; duplicates will be removed')
            keep_sites = ~overlapping.any(0)
            asym_frac_motif = asym_frac_motif[keep_sites]
            asym_symbols = [sym for sym, keep in zip(asym_symbols, keep_sites) if keep]

        if asym_frac_motif.shape[0] == 0:
            if self.show_warnings:
                warnings.warn(f'Skipping {crystal.identifier} as there are no sites with coordinates')
            return None
        
        sitesym = crystal.symmetry_operators
        if not sitesym: sitesym = ('x,y,z',)
        
        frac_motif, asym_unit, multiplicities, inverses = self._expand(asym_frac_motif, sitesym, 
                                                                       show_warnings=self.show_warnings)
        motif = frac_motif @ cell
        
        kwargs = {
            'name':                     crystal.identifier, 
            'asymmetric_unit':          asym_unit,
            'wyckoff_multiplicities':   multiplicities,
            'types':                    [asym_symbols[i] for i in inverses],
        }
        print(asym_symbols)
        if self.extract_data is not None:
            entry.crystal.molecule = crystal.disordered_molecule
            for key in self.extract_data:
                kwargs[key] = self.extract_data[key](entry)

        periodic_set = PeriodicSet(motif, cell, **kwargs)
        
        return periodic_set

class CifReader(_Reader):
    """Read all structures in a .CIF with ``ase`` or ``ccdc``, yielding  
    :class:`.periodicset.PeriodicSet` objects which can be passed
    to :func:`.calculate.AMD` or :func:`.calculate.PDD`.
    
    Examples:
    
        Put all crystals in mycif.cif in a list::
        
            structures = list(amd.CifReader('mycif.cif'))   
            
        Reads just one, convinient when the .CIF has one crystal::
        
            periodic_set = amd.CifReader('mycif.cif').read_one()
        
        If the folder 'cifs' has many .CIFs each with one crystal::
        
            import os
            folder = 'cifs'
            structures = []
            for filename in os.listdir(folder):
                p_set = amd.CifReader(os.path.join(folder, filename)).read_one()
                structures.append(p_set)
            
        Make list of AMD (with k=100) for crystals in mycif.cif::
        
            amds = []
            for periodic_set in amd.CifReader('mycif.cif'):
                amds.append(amd.AMD(periodic_set, 100))
    """
    
    @_extend_signature(_Reader.__init__)
    def __init__(self, path, reader='ase', folder=False, **kwargs):
        
        super().__init__(**kwargs)
        
        if reader not in ('ase', 'ccdc'):
            raise ValueError(f'Invalid reader {reader}; must be ase or ccdc.')

        if reader == 'ase' and self.heaviest_component:
            raise NotImplementedError(f'Parameter heaviest_component not implimented for {reader}, only ccdc.')

        if reader == 'ase':
            if folder:
                generators = (parse_cif(os.path.join(path, file)) for file in os.listdir(path) if file.endswith('.cif'))
                raw_generator = chain(*generators)
            else:
                raw_generator = parse_cif(path)
            
            self._generator = self._map(self._CIFBlock_to_PeriodicSet, raw_generator)
            
        elif reader == 'ccdc':
            if not _CCDC_ENABLED:
                raise ImportError(f"Failed to import csd-python-api; please check it is installed and licensed.")
            
            if folder:
                generators = []
                for file in os.listdir(path):
                    suff = os.path.splitext(file)[1][1:]
                    if suff.lower() in EntryReader.known_suffixes:
                        generators.append(EntryReader(os.path.join(path, file)))
                raw_generator = chain(*generators)
            else:
                raw_generator = EntryReader(path)
                
            self._generator = self._map(self._Entry_to_PeriodicSet, raw_generator)

class CSDReader(_Reader):
    """Read Entries from the CSD, yielding :class:`.periodicset.PeriodicSet` objects.
    
    The CSDReader returns :class:`.periodicset.PeriodicSet` objects which can be passed
    to :func:`.calculate.AMD` or :func:`.calculate.PDD`.
    
    Examples:
    
        Get crystals with refcodes in a list::

            refcodes = ['DEBXIT01', 'DEBXIT05', 'HXACAN01']
            structures = list(amd.CSDReader(refcodes))
            
        Read refcode families (any whose refcode starts with strings in the list)::
        
            refcodes = ['ACSALA', 'HXACAN']
            structures = list(amd.CSDReader(refcodes, families=True))
        
        Create a generic reader, read crystals 'on demand' with :meth:`CSDReader.entry()`::
        
            reader = amd.CSDReader()
            debxit01 = reader.entry('DEBXIT01')
            
            # looping over this generic reader will yield all CSD entries
            for periodic_set in reader:
                ...
        
        Make list of AMD (with k=100) for crystals in these families::

            refcodes = ['ACSALA', 'HXACAN']
            amds = []
            for periodic_set in amd.CSDReader(refcodes, families=True):
                amds.append(amd.AMD(periodic_set, 100))
    """
    
    @_extend_signature(_Reader.__init__)
    def __init__(self, refcodes=None, families=False, **kwargs):

        if not _CCDC_ENABLED:
            raise ImportError(f"Failed to import csd-python-api; please check it is installed and licensed.")

        super().__init__(**kwargs)

        if isinstance(refcodes, str) and refcodes.lower() == 'csd':
            refcodes = None

        if refcodes is None:
            families = False
        else:
            refcodes = [refcodes] if isinstance(refcodes, str) else list(refcodes)
        
        if families:
            all_refcodes = []
            for refcode in refcodes:
                query = TextNumericSearch()
                query.add_identifier(refcode)
                all_refcodes.extend((hit.identifier for hit in query.search()))

            # filter to unique refcodes
            seen = set()
            seen_add = seen.add
            refcodes = [refcode for refcode in all_refcodes if not (refcode in seen or seen_add(refcode))]

        self._entry_reader = EntryReader('CSD')
        self._generator = self._map(self._Entry_to_PeriodicSet, self._init_ccdc_reader(refcodes))
    
    def _init_ccdc_reader(self, refcodes):
        """Generates ccdc Entries from CSD refcodes"""
        
        if refcodes is None:
            for entry in self._entry_reader:
                yield entry
        else: 
            for refcode in refcodes:
                try:
                    entry = self._entry_reader.entry(refcode)
                    yield entry
                except RuntimeError:
                    warnings.warn(f'Identifier {refcode} not found in database')

    def entry(self, refcode: str) -> PeriodicSet:
        """Read a PeriodicSet given any CSD refcode."""
        
        entry = self._entry_reader.entry(refcode)
        periodic_set = self._Entry_to_PeriodicSet(entry)
        return periodic_set

class SetWriter:
    """Write several :class:`.periodicset.PeriodicSet` objects to a .hdf5 file. 
    Reading the .hdf5 is much faster than parsing a .CIF file.
    """

    _str_dtype = h5py.vlen_dtype(str)
    
    def __init__(self, filename: str):
        
        self.file = h5py.File(filename, 'w', track_order=True)
        
    def write(self, periodic_set: PeriodicSet, name: Optional[str] = None):
        """Write a PeriodicSet object to file."""
        
        if not isinstance(periodic_set, PeriodicSet):
            raise ValueError(f'Object type {periodic_set.__class__.__name__} cannot be written with SetWriter')
        
        # need a name to store or you can't access items by key
        if name is None:
            if periodic_set.name is None:
                raise ValueError('Periodic set must have a name to be written. Either set the name '
                                 'attribute of the PeriodicSet or pass a name to SetWriter.write()')
            name = periodic_set.name
        
        # this group is the PeriodicSet
        group = self.file.create_group(name)
        
        # datasets in the group for motif and cell
        group.create_dataset('motif', data=periodic_set.motif)
        group.create_dataset('cell',  data=periodic_set.cell)
        
        if periodic_set.tags:
            # a subgroup contains tags that are lists or ndarrays
            tags_group = group.create_group('tags')
            
            for tag in periodic_set.tags:
                data = periodic_set.tags[tag]

                if data is None:                    # nonce to handle None
                    tags_group.attrs[tag] = '__None' 
                elif np.isscalar(data):             # scalars (nums and strs) stored as attrs
                    tags_group.attrs[tag] = data
                elif isinstance(data, np.ndarray):
                    tags_group.create_dataset(tag, data=data)
                elif isinstance(data, list):    
                    # lists of strings stored as special type for some reason
                    if any(isinstance(d, str) for d in data):
                        data = [str(d) for d in data]
                        tags_group.create_dataset(tag, data=data, dtype=SetWriter._str_dtype)
                    else:    # other lists must be castable to ndarray
                        data = np.asarray(data)
                        tags_group.create_dataset(tag, data=np.array(data))           
                else:
                    raise ValueError(f'Cannot store tag of type {type(data)} with SetWriter')
    
    def iwrite(self, periodic_sets: Iterable[PeriodicSet]):
        """Write :class:`.periodicset.PeriodicSet` objects from an iterable to file."""
        for periodic_set in periodic_sets:
            self.write(periodic_set)
        
    def close(self):
        """Close the :class:`SetWriter`."""
        self.file.close()
        
    def __enter__(self):
        return self

    # handle exceptions?
    def __exit__(self, exc_type, exc_value, tb):
        self.file.close()

class SetReader:
    """Read :class:`.periodicset.PeriodicSet` objects from a .hdf5 file written
    with :class:`SetWriter`. Acts like a read-only dict that can be iterated 
    over (preserves write order)."""
    
    def __init__(self, filename: str):
        
        self.file = h5py.File(filename, 'r', track_order=True)

    def _get_set(self, name: str) -> PeriodicSet:
        # take a name in the set and return the PeriodicSet
        
        group = self.file[name]
        periodic_set = PeriodicSet(group['motif'][:], group['cell'][:], name=name)
        
        if 'tags' in group:
            for tag in group['tags']:
                data = group['tags'][tag][:]
                     
                if any(isinstance(d, (bytes, bytearray)) for d in data):
                    periodic_set.tags[tag] = [d.decode() for d in data]
                else:
                    periodic_set.tags[tag] = data
        
            for attr in group['tags'].attrs:
                data = group['tags'].attrs[attr]
                periodic_set.tags[attr] = None if data == '__None' else data

        return periodic_set
    
    def close(self):
        """Close the :class:`SetReader`."""
        self.file.close()
    
    def family(self, refcode: str) -> Iterable[PeriodicSet]:
        """Yield any :class:`.periodicset.PeriodicSet`s whose name starts with input refcode."""
        
        for name in self.keys():
            if name.startswith(refcode):
                yield self._get_set(name)
    
    def __getitem__(self, name):
        # index by name. Not found exc?
        return self._get_set(name)
    
    def __len__(self):
        return len(self.keys())
    
    def __iter__(self):
        # interface to loop over the SetReader; does not close the SetReader when done
        for name in self.keys():
            yield self._get_set(name)

    def __contains__(self, item):
        if item in self.keys():
            return True
        else:
            return False
        
    def keys(self):
        """Yield names of items in the :class:`SetReader`."""
        return self.file['/'].keys()
    
    def __enter__(self):
        return self

    # handle exceptions?
    def __exit__(self, exc_type, exc_value, tb):
        self.file.close()
    
    def extract_tags(self) -> dict:
        """Return ``dict`` with scalar data in the tags of items in :class:`SetReader`.
        
        Dict is in format easily passable to ``pandas.DataFrame``, as in::
        
            reader = amd.SetReader('periodic_sets.hdf5')
            data = reader.extract_tags()
            df = pd.DataFrame(data, index=reader.keys(), columns=data.keys())
        
        Format of returned dict is for example:: 
        
            {
                'density': [1.231, 2.532, ...],
                'family':  ['CBMZPN', 'SEMFAU', ...], 
                ...
            }
            
        where the inner lists have 
        the same order as the items in the SetReader.
        """
        
        names = list(self.keys())
        data = []
        columns = []
        
        for name in names:
            
            row = {}
            group = self.file[name]
            
            for tag in group['tags']:
                value = group['tags'][tag][:]
                if np.isscalar(value):
                    if tag not in columns:
                        columns.append(tag)
                    row[tag] = value
            
            for attr in group['tags'].attrs:
                value = group['tags'].attrs[attr]
                if np.isscalar(value):
                    if attr not in columns:
                        columns.append(attr)
                    row[attr] = value
            
            data.append(row)
            
        data_ = {}
        for col_name in columns:
            column_data = []
            for row in data:
                if col_name in row:
                    column_data.append(row[col_name])
                else:
                    column_data.append(None)
            data_[col_name] = column_data

        return data_

### ccdc.crystal.Crystal --> amd.periodicset.PeriodicSet
### ignores disorder, missing sites/coords, checks & no options
def crystal_to_periodicset(crystal):
    
    cell = cellpar_to_cell(*crystal.cell_lengths, *crystal.cell_angles)
        
    # asymmetric unit fractional coordinates
    asym_frac_motif = np.array([[
                        a.fractional_coordinates.x, 
                        a.fractional_coordinates.y, 
                        a.fractional_coordinates.z]
                        for a in crystal.asymmetric_unit_molecule.atoms])
    
    asym_frac_motif = np.mod(asym_frac_motif, 1)
    
    # if the above removed everything, skip this structure
    if asym_frac_motif.shape[0] == 0:
        raise ValueError(f'{crystal.identifier} has no coordinates')
    
    sitesym = crystal.symmetry_operators
    if not sitesym: sitesym = ('x,y,z',)
    
    frac_motif, asym_unit, multiplicities, _ = _Reader._expand(asym_frac_motif, sitesym)
    motif = frac_motif @ cell
    
    kwargs = {
        'name': crystal.identifier, 
        'asymmetric_unit': asym_unit,
        'wyckoff_multiplicities': multiplicities,
    }

    periodic_set = PeriodicSet(motif, cell, **kwargs)

    return periodic_set

### ase.io.cif.CIFBlock --> amd.periodicset.PeriodicSet
### ignores disorder, missing sites/coords, checks & no options

def cifblock_to_periodicset(block):
    
    cell = block.get_cell().array
    
    asym_frac_motif = [block.get(name) for name in _Reader.atom_site_fract_tags]
    if None in asym_frac_motif:
        asym_motif = [block.get(name) for name in _Reader.atom_site_cartn_tags]
        if None in asym_motif:
            warnings.warn(f'Skipping {block.name} as coordinates were not found')
            return None
            
        asym_frac_motif = np.array(asym_motif) @ np.linalg.inv(cell)
        
    asym_frac_motif = np.mod(np.array(asym_frac_motif).T, 1)
    
    if asym_frac_motif.shape[0] == 0:
        raise ValueError(f'{block.name} has no coordinates')
    
    sitesym = ('x,y,z',)
    for tag in _Reader.symop_tags:
        if tag in block:
            sitesym = block[tag]
            break
        
    if isinstance(sitesym, str):
        sitesym = [sitesym]
    
    frac_motif, asym_unit, multiplicities, inverses = _Reader._expand(asym_frac_motif, sitesym)
    motif = frac_motif @ cell

    kwargs = {
        'name': block.name, 
        'asymmetric_unit': asym_unit,
        'wyckoff_multiplicities': multiplicities
    }
    
    periodic_set = PeriodicSet(motif, cell, **kwargs)
    
    return periodic_set