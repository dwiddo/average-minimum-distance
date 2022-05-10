"""Contains I/O tools, including a .CIF reader and CSD reader
(``csd-python-api`` only) to extract periodic set representations
of crystals which can be passed to :func:`.calculate.AMD` and :func:`.calculate.PDD`.

These intermediate :class:`.periodicset.PeriodicSet` representations can be written
to a .hdf5 file with :class:`SetWriter`, which can be read back with :class:`SetReader`.
This is much faster than rereading a .CIF and recomputing invariants.
"""

import os
import functools
import warnings
from typing import Callable, Iterable, Sequence, Tuple

import numpy as np
import ase.io.cif
import ase.spacegroup.spacegroup

from .periodicset import PeriodicSet
from .utils import cellpar_to_cell

try:
    import ccdc.io
    import ccdc.search
    _CSD_PYTHON_API_ENABLED = True
except (ImportError, RuntimeError) as _:
    _CSD_PYTHON_API_ENABLED = False

def _custom_warning(message, category, filename, lineno, *args, **kwargs):
    return f'{category.__name__}: {message}\n'

warnings.formatwarning = _custom_warning

class ParseError(ValueError):
    pass

_EQUIV_SITE_TOL = 1e-3
_DISORDER_OPTIONS = {'skip', 'ordered_sites', 'all_sites',}
_ATOM_SITE_FRACT_TAGS = [
    '_atom_site_fract_x',
    '_atom_site_fract_y',
    '_atom_site_fract_z',]
_ATOM_SITE_CARTN_TAGS = [
    '_atom_site_cartn_x',
    '_atom_site_cartn_y',
    '_atom_site_cartn_z',]
_SYMOP_TAGS = [
    '_space_group_symop_operation_xyz',
    '_space_group_symop.operation_xyz',
    '_symmetry_equiv_pos_as_xyz',]


class CifReader:
    """Read all structures in a .CIF with ``ase`` or ``ccdc``
    (``csd-python-api`` only), yielding  :class:`.periodicset.PeriodicSet`
    objects which can be passed to :func:`.calculate.AMD` or
    :func:`.calculate.PDD`.

    Examples:

        ::

            # Put all crystals in a .CIF in a list
            structures = list(amd.CifReader('mycif.cif'))

            # Reads just one if the .CIF has just one crystal
            periodic_set = amd.CifReader('mycif.cif').read_one()

            # If a folder has several .CIFs each with one crystal, use
            structures = list(amd.CifReader('path/to/folder', folder=True))

            # Make list of AMDs (with k=100) of crystals in a .CIF
            amds = [amd.AMD(periodic_set, 100) for periodic_set in amd.CifReader('mycif.cif')]
    """

    def __init__(self,
                 path,
                 reader='ase',
                 folder=False,
                 remove_hydrogens=False,
                 disorder='skip',
                 heaviest_component=False,
                 show_warnings=True,
                 extract_data=None,
                 include_if=None
    ):
  
        if disorder not in _DISORDER_OPTIONS:
            raise ValueError(f'disorder parameter {disorder} must be one of {_DISORDER_OPTIONS}')

        if reader not in ('ase', 'ccdc'):
            raise ValueError(f'Invalid reader {reader}; must be ase or ccdc.')

        if reader == 'ase' and heaviest_component:
            raise NotImplementedError('Parameter heaviest_component not implimented for ase, only ccdc.')

        extract_data, include_if = _validate_kwargs(extract_data, include_if)

        self.show_warnings = show_warnings
        self.extract_data = extract_data
        self.include_if = include_if
        self.current_filename = None

        if reader == 'ase':
            extensions = {'cif'}
            file_parser = ase.io.cif.parse_cif
            converter = functools.partial(cifblock_to_periodicset, 
                                          remove_hydrogens=remove_hydrogens,
                                          disorder=disorder)

        elif reader == 'ccdc':
            if not _CSD_PYTHON_API_ENABLED:
                raise ImportError("Failed to import csd-python-api; check it is installed and licensed.")
            extensions = ccdc.io.EntryReader.known_suffixes
            file_parser = ccdc.io.EntryReader
            converter = functools.partial(entry_to_periodicset,
                                          remove_hydrogens=remove_hydrogens,
                                          disorder=disorder,
                                          heaviest_component=heaviest_component)

        if folder:
            generator = self._folder_generator(path, file_parser, extensions)
        else:
            generator = file_parser(path)

        self._generator = self._map(converter, generator)

    def __iter__(self):
        yield from self._generator

    def read_one(self):
        """Read the next (usually first and only) item."""
        return next(iter(self._generator))

    def _folder_generator(self, path, file_parser, extensions):
        for file in os.listdir(path):
            suff = os.path.splitext(file)[1][1:]
            if suff.lower() in extensions:
                self.current_filename = file
                yield from file_parser(os.path.join(path, file))

    def _map(self, func: Callable, iterable: Iterable) -> Iterable[PeriodicSet]:
        """Iterates over iterable, passing items through parser and yielding the result.
        Applies warning and include_if filters, catches bad structures and warns.
        """

        for item in iterable:

            with warnings.catch_warnings(record=True) as warning_msgs:

                if not self.show_warnings:
                    warnings.simplefilter('ignore')

                if any(not check(item) for check in self.include_if):
                    continue

                try:
                    periodic_set = func(item)
                except ParseError as err:
                    warnings.warn(err, category=UserWarning)
                    continue

            for warning in warning_msgs:
                msg = f'{periodic_set.name}: {warning.message}'
                warnings.warn(msg, category=warning.category)

            if self.current_filename:
                periodic_set.tags['filename'] = self.current_filename

            for key, func in self.extract_data.items():
                periodic_set.tags[key] = func(item)

            yield periodic_set


class CSDReader:
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

        Create a generic reader, read crystals by name with :meth:`CSDReader.entry()`::

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

    def __init__(self,
                 refcodes=None,
                 families=False,
                 remove_hydrogens=False,
                 disorder='skip',
                 heaviest_component=False,
                 show_warnings=True,
                 extract_data=None,
                 include_if=None,
    ):

        if not _CSD_PYTHON_API_ENABLED:
            raise ImportError('Failed to import csd-python-api; check it is installed and licensed.')

        if disorder not in _DISORDER_OPTIONS:
            raise ValueError(f'disorder parameter {disorder} must be one of {_DISORDER_OPTIONS}')

        extract_data, include_if = _validate_kwargs(extract_data, include_if)

        self.show_warnings = show_warnings
        self.extract_data = extract_data
        self.include_if = include_if
        self.current_filename = None

        if isinstance(refcodes, str) and refcodes.lower() == 'csd':
            refcodes = None

        if refcodes is None:
            families = False
        else:
            refcodes = [refcodes] if isinstance(refcodes, str) else list(refcodes)

        # families parameter reads all crystals with ids starting with passed refcodes
        if families:
            all_refcodes = []
            for refcode in refcodes:
                query = ccdc.search.TextNumericSearch()
                query.add_identifier(refcode)
                all_refcodes.extend((hit.identifier for hit in query.search()))

            # filter to unique refcodes
            seen = set()
            seen_add = seen.add
            refcodes = [
                refcode for refcode in all_refcodes
                if not (refcode in seen or seen_add(refcode))]

        self._entry_reader = ccdc.io.EntryReader('CSD')

        converter = functools.partial(entry_to_periodicset, 
                                      remove_hydrogens=remove_hydrogens,
                                      disorder=disorder,
                                      heaviest_component=heaviest_component)
        
        generator = self._ccdc_generator(refcodes)
        
        self._generator = self._map(converter, generator)

    def __iter__(self):
        yield from self._generator

    def read_one(self):
        """Read the next (usually first and only) item."""
        return next(iter(self._generator))

    def entry(self, refcode: str, **kwargs) -> PeriodicSet:
        """Read a PeriodicSet given any CSD refcode."""

        entry = self._entry_reader.entry(refcode)
        periodic_set = entry_to_periodicset(entry, **kwargs)
        return periodic_set

    def _ccdc_generator(self, refcodes):
        """Generates ccdc Entries from CSD refcodes."""

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

    def _map(self, func: Callable, iterable: Iterable) -> Iterable[PeriodicSet]:
        """Iterates over iterable, passing items through parser and yielding the result.
        Applies warning and include_if filters, catches bad structures and warns.
        """

        for item in iterable:

            with warnings.catch_warnings(record=True) as warning_msgs:

                if not self.show_warnings:
                    warnings.simplefilter('ignore')

                if any(not check(item) for check in self.include_if):
                    continue

                try:
                    periodic_set = func(item)
                except ParseError as err:
                    warnings.warn(err, category=UserWarning)
                    continue

            for warning in warning_msgs:
                msg = f'{periodic_set.name}: {warning.message}'
                warnings.warn(msg, category=warning.category)

            for key, func in self.extract_data.items():
                periodic_set.tags[key] = func(item)

            yield periodic_set


def entry_to_periodicset(entry,
                         remove_hydrogens=False,
                         disorder='skip',
                         heaviest_component=False
) -> PeriodicSet:
    """ccdc.entry.Entry --> PeriodicSet."""

    crystal = entry.crystal

    if not entry.has_3d_structure:
        raise ParseError(f'Has no 3D structure')

    molecule = crystal.disordered_molecule

    if disorder == 'skip':
        if crystal.has_disorder or entry.has_disorder or \
            any(atom_has_disorder(a.label, a.occupancy) for a in molecule.atoms):
            raise ParseError(f'Has disorder')

    elif disorder == 'ordered_sites':
        molecule.remove_atoms(a for a in molecule.atoms
                                if atom_has_disorder(a.label, a.occupancy))

    if remove_hydrogens:
        molecule.remove_atoms(a for a in molecule.atoms if a.atomic_symbol in 'HD')

    if heaviest_component and len(molecule.components) > 1:
        molecule = _heaviest_component(molecule)

    if not molecule.all_atoms_have_sites or \
        any(a.fractional_coordinates is None for a in molecule.atoms):
        raise ParseError(f'Has atoms without sites')

    crystal.molecule = molecule
    asym_atoms = crystal.asymmetric_unit_molecule.atoms
    asym_unit = np.array([tuple(a.fractional_coordinates) for a in asym_atoms])
    asym_unit = np.mod(asym_unit, 1)
    asym_symbols = [a.atomic_symbol for a in asym_atoms]
    cell = cellpar_to_cell(*crystal.cell_lengths, *crystal.cell_angles)

    sitesym = crystal.symmetry_operators
    if not sitesym:
        sitesym = ['x,y,z', ]

    if disorder != 'all_sites':
        keep_sites = _unique_sites(asym_unit)
        if np.any(keep_sites == False):
            warnings.warn(f'May have overlapping sites; duplicates will be removed')
        asym_unit = asym_unit[keep_sites]
        asym_symbols = [sym for sym, keep in zip(asym_symbols, keep_sites) if keep]

    if asym_unit.shape[0] == 0:
        raise ParseError(f'Has no valid sites')
    
    frac_motif, asym_inds, multiplicities, inverses = expand_asym_unit(asym_unit, sitesym)
    full_types = [asym_symbols[i] for i in inverses]
    motif = frac_motif @ cell

    tags = {
        'name': entry.identifier,
        'asymmetric_unit': asym_inds,
        'wyckoff_multiplicities': multiplicities,
        'types': full_types,
    }

    return PeriodicSet(motif, cell, **tags)


def cifblock_to_periodicset(block, 
                            remove_hydrogens=False,
                            disorder='skip'
) -> PeriodicSet:
    """ase.io.cif.CIFBlock --> PeriodicSet."""

    cell = block.get_cell().array

    # asymmetric unit fractional coords
    asym_unit = [block.get(name) for name in _ATOM_SITE_FRACT_TAGS]
    if None in asym_unit:
        asym_motif = [block.get(name) for name in _ATOM_SITE_CARTN_TAGS]
        if None in asym_motif:
            raise ParseError(f'Has no sites')
        asym_unit = np.array(asym_motif) @ np.linalg.inv(cell)
    asym_unit = np.mod(np.array(asym_unit).T, 1)

    try:
        asym_symbols = block.get_symbols()
    except ase.io.cif.NoStructureData as _:
        asym_symbols = ['Unknown' for _ in range(len(asym_unit))]

    sitesym = ['x,y,z', ]
    for tag in _SYMOP_TAGS:
        if tag in block:
            sitesym = block[tag]
            break
    if isinstance(sitesym, str):
        sitesym = [sitesym]

    remove_sites = []

    occupancies = block.get('_atom_site_occupancy')
    labels = block.get('_atom_site_label')
    if occupancies is not None:
        if disorder == 'skip':
            if any(atom_has_disorder(lab, occ) for lab, occ in zip(labels, occupancies)):
                raise ParseError(f'Has disorder')
        elif disorder == 'ordered_sites':
            remove_sites.extend(
                (i for i, (lab, occ) in enumerate(zip(labels, occupancies))
                    if atom_has_disorder(lab, occ)))

    if remove_hydrogens:
        remove_sites.extend((i for i, sym in enumerate(asym_symbols) if sym in 'HD'))

    asym_unit = np.delete(asym_unit, remove_sites, axis=0)
    asym_symbols = [s for i, s in enumerate(asym_symbols) if i not in remove_sites]

    if disorder != 'all_sites':
        keep_sites = _unique_sites(asym_unit)
        if np.any(keep_sites == False):
            warnings.warn(f'May have overlapping sites; duplicates will be removed')
        asym_unit = asym_unit[keep_sites]
        asym_symbols = [sym for sym, keep in zip(asym_symbols, keep_sites) if keep]
    
    if asym_unit.shape[0] == 0:
        raise ParseError(f'Has no valid sites')

    frac_motif, asym_inds, multiplicities, inverses = expand_asym_unit(asym_unit, sitesym)
    full_types = [asym_symbols[i] for i in inverses]
    motif = frac_motif @ cell

    tags = {
        'name': block.name,
        'asymmetric_unit': asym_inds,
        'wyckoff_multiplicities': multiplicities,
        'types': full_types,
    }

    return PeriodicSet(motif, cell, **tags)


def expand_asym_unit(
    asym_unit: np.ndarray, 
    sitesym: Sequence[str]
) -> Tuple[np.ndarray, ...]:
    """
    Asymmetric unit's fractional coords + sitesyms (as strings)
    -->
    frac motif, asym unit inds, multiplicities, inverses
    """

    rotations, translations = ase.spacegroup.spacegroup.parse_sitesym(sitesym)
    all_sites = []
    asym_inds = [0]
    multiplicities = []
    inverses = []

    for inv, site in enumerate(asym_unit):
        multiplicity = 0

        for rot, trans in zip(rotations, translations):
            site_ = np.mod(np.dot(rot, site) + trans, 1)

            if not all_sites:
                all_sites.append(site_)
                inverses.append(inv)
                multiplicity += 1
                continue

            # check if site_ overlaps with existing sites
            diffs1 = np.abs(site_ - all_sites)
            diffs2 = np.abs(diffs1 - 1)
            mask = np.all((diffs1 <= _EQUIV_SITE_TOL) | (diffs2 <= _EQUIV_SITE_TOL), axis=-1)

            if np.any(mask):
                where_equal = np.argwhere(mask).flatten()
                for ind in where_equal:
                    if inverses[ind] == inv:
                        pass
                    else:
                        warnings.warn(f'Equivalent sites at positions {inverses[ind]}, {inv}')
            else:
                all_sites.append(site_)
                inverses.append(inv)
                multiplicity += 1

        if multiplicity > 0:
            multiplicities.append(multiplicity)
            asym_inds.append(len(all_sites))

    frac_motif = np.array(all_sites)
    asym_inds = np.array(asym_inds[:-1])
    multiplicities = np.array(multiplicities)
    return frac_motif, asym_inds, multiplicities, inverses


def atom_has_disorder(label, occupancy):
    return label.endswith('?') or (np.isscalar(occupancy) and occupancy < 1)


def _unique_sites(asym_unit):
    site_diffs1 = np.abs(asym_unit[:, None] - asym_unit)
    site_diffs2 = np.abs(site_diffs1 - 1)
    overlapping = np.triu(np.all(
        (site_diffs1 <= _EQUIV_SITE_TOL) | (site_diffs2 <= _EQUIV_SITE_TOL),
        axis=-1), 1)
    return ~overlapping.any(axis=0)


def _heaviest_component(molecule):
    """Heaviest component (removes all but the heaviest component of the asym unit).
    Intended for removing solvents. Probably doesn't play well with disorder"""
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
    largest_component_ind = np.argmax(np.array(component_weights))
    molecule = molecule.components[largest_component_ind]
    return molecule


def _validate_kwargs(extract_data, include_if):
    
    reserved_tags = {'motif', 'cell', 'name',
                     'asymmetric_unit', 'wyckoff_multiplicities',
                     'types', 'filename'}
    
    if extract_data is None:
        extract_data = {}
    else:
        if not isinstance(extract_data, dict):
            raise ValueError('extract_data must be a dict of callables')
        for key in extract_data:
            if not callable(extract_data[key]):
                raise ValueError('extract_data must be a dict of callables')
            if key in reserved_tags:
                raise ValueError(f'extract_data includes reserved key {key}')
        extract_data = extract_data

    if include_if is None:
        include_if = ()
    elif not all(callable(func) for func in include_if):
        raise ValueError('include_if must be a list of callables')
    else:
        include_if = include_if

    return extract_data, include_if
