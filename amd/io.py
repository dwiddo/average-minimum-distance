"""Tools for reading crystals from files, or from the CSD with ``csd-python-api``. 
The readers return :class:`.periodicset.PeriodicSet` objects representing the 
crystal which can be passed to :func:`.calculate.AMD` and :func:`.calculate.PDD` 
to get their invariants.
"""

import os
import functools
import warnings
from typing import Callable, Iterable, Sequence, Tuple

import numpy as np
import numba
import ase.io.cif
import ase.data
from ase.spacegroup.spacegroup import parse_sitesym

from . import utils
from .periodicset import PeriodicSet

try:
    import ccdc.io
    import ccdc.search
    _CSD_PYTHON_API_ENABLED = True
except (ImportError, RuntimeError) as _:
    _CSD_PYTHON_API_ENABLED = False

def _custom_warning(message, category, filename, lineno, *args, **kwargs):
    return f'{category.__name__}: {message}\n'

warnings.formatwarning = _custom_warning

_READERS = {'ase', 'ccdc', 'pycodcif'}
_DISORDER_OPTIONS = {'skip', 'ordered_sites', 'all_sites'}
_EQUIV_SITE_TOL = 1e-3
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


class _ParseError(ValueError):
    """Raised when an item cannot be parsed into a periodic set."""
    pass


class _Reader:
    """Base Reader class. Contains parsers for converting ase CifBlock
    and ccdc Entry objects to PeriodicSets.
    Intended to be inherited and then a generator set to self._generator.
    First make a new method for _Reader converting object to PeriodicSet
    (e.g. named _X_to_PSet). Then make this class outline:
    class XReader(_Reader):
        def __init__(self, ..., **kwargs):
        super().__init__(**kwargs)
        # setup and checks
        # make 'iterable' which yields objects to be converted (e.g. CIFBlock, Entry)
        # set self._generator like this
        self._generator = self._map(iterable, self._X_to_PSet)
    """

    __slots__ = [
        'remove_hydrogens',
        'disorder',
        'heaviest_component',
        'show_warnings',
        'current_filename',
        '_generator',
    ]

    def __init__(
            self,
            remove_hydrogens=False,
            disorder='skip',
            heaviest_component=False,
            show_warnings=True,
    ):

        if disorder not in _DISORDER_OPTIONS:
            raise ValueError(f'disorder parameter {disorder} must be one of {_DISORDER_OPTIONS}')

        self.remove_hydrogens = remove_hydrogens
        self.disorder = disorder
        self.heaviest_component = heaviest_component
        self.show_warnings = show_warnings
        self.current_filename = None
        self._generator = []

    def __iter__(self):
        yield from self._generator

    def read_one(self):
        """Read the next (usually first and/or only) item."""
        return next(iter(self._generator))

    def _map(self, func: Callable, iterable: Iterable) -> Iterable[PeriodicSet]:
        """Iterates over iterable, passing items through parser and 
        yielding the result. Catches bad structures and warns.
        """

        if not self.show_warnings:
            warnings.simplefilter('ignore')

        for item in iterable:

            with warnings.catch_warnings(record=True) as warning_msgs:

                parse_failed = False
                try:
                    periodic_set = func(item)
                except _ParseError as err:
                    parse_failed = str(err)

            if parse_failed:
                warnings.warn(parse_failed)
                continue

            for warning in warning_msgs:
                msg = f'{periodic_set.name}: {warning.message}'
                warnings.warn(msg, category=warning.category)

            if self.current_filename:
                periodic_set.tags['filename'] = self.current_filename

            yield periodic_set


class CifReader(_Reader):
    """Read all structures in a .cif file or all files in a folder 
    with ase or csd-python-api (if installed), yielding 
    :class:`.periodicset.PeriodicSet` s.

    Parameters
    ----------
    path : str
        Path to a .cif file or directory. (Other files are accepted when using
        ``reader='ccdc'``, if csd-python-api is installed.)
    reader : str, optional
        The backend package used for parsing. Default is :code:`ase`,
        to use csd-python-api change to :code:`ccdc`. The ccdc reader should 
        be able to read any format accepted by :class:`ccdc.io.EntryReader`,
        though only cifs have been tested.
    remove_hydrogens : bool, optional
        Remove Hydrogens from the crystal.
    disorder : str, optional
        Controls how disordered structures are handled. Default is ``skip`` which skips any crystal 
        with disorder, since disorder conflicts with the periodic set model. To read disordered 
        structures anyway, choose either :code:`ordered_sites` to remove sites with disorder or 
        :code:`all_sites` include all sites regardless.
    heaviest_component : bool, optional
        csd-python-api only. Removes all but the heaviest molecule in the asymmeric unit,
        intended for removing solvents. 
    show_warnings : bool, optional
        Controls whether warnings that arise during reading are printed.

    Yields
    ------
    :class:`.periodicset.PeriodicSet`
        Represents the crystal as a periodic set, consisting of a finite set of points (motif)
        and lattice (unit cell). Contains other useful data, e.g. the crystal's name and
        information about the asymmetric unit for calculation.

    Examples
    --------
    
        ::

            # Put all crystals in a .CIF in a list
            structures = list(amd.CifReader('mycif.cif'))

            # Can also accept path to a directory, reading all files inside
            structures = list(amd.CifReader('path/to/folder'))

            # Reads just one if the .CIF has just one crystal
            periodic_set = amd.CifReader('mycif.cif').read_one()

            # List of AMDs (k=100) of crystals in a .CIF
            amds = [amd.AMD(periodic_set, 100) for periodic_set in amd.CifReader('mycif.cif')]
    """

    def __init__(
            self,
            path,
            reader='ase',
            remove_hydrogens=False,
            disorder='skip',
            heaviest_component=False,
            show_warnings=True,
    ):

        super().__init__(
            remove_hydrogens=remove_hydrogens,
            disorder=disorder,
            heaviest_component=heaviest_component,
            show_warnings=show_warnings,
        )

        if reader not in _READERS:
            raise ValueError(f'Invalid reader {reader}; must be one of f{_READERS}.')

        if reader in ('ase', 'pycodcif'):
            if heaviest_component:
                raise NotImplementedError('Parameter heaviest_component only implimented for reader="ccdc".')

            extensions = {'cif'}
            file_parser = functools.partial(ase.io.cif.parse_cif, reader=reader)
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

        if os.path.isfile(path):
            generator = file_parser(path)
        elif os.path.isdir(path):
            generator = self._generate_from_dir(path, file_parser, extensions)
        else:
            raise FileNotFoundError(f'No such file or directory: {path}')

        self._generator = self._map(converter, generator)

    def _generate_from_dir(self, path, file_parser, extensions):
        for file in os.listdir(path):
            suff = os.path.splitext(file)[1][1:]
            if suff.lower() in extensions:
                self.current_filename = file
                yield from file_parser(os.path.join(path, file))


class CSDReader(_Reader):
    """Read structures from the CSD with csd-python-api, yielding 
    :class:`.periodicset.PeriodicSet` s.

    Parameters
    ----------
    refcodes : List[str], optional
        List of CSD refcodes to read. If None or 'CSD', iterates over the whole CSD.
    families : bool, optional
        Read all entries whose refcode starts with the given strings, or 'families' 
        (e.g. giving 'DEBXIT' reads all entries starting with DEBXIT).
    remove_hydrogens : bool, optional
        Remove hydrogens from the crystal.
    disorder : str, optional
        Controls how disordered structures are handled. Default is ``skip`` which skips any crystal 
        with disorder, since disorder conflicts with the periodic set model. To read disordered 
        structures anyway, choose either :code:`ordered_sites` to remove sites with disorder or 
        :code:`all_sites` include all sites regardless.
    heaviest_component : bool, optional
        csd-python-api only. Removes all but the heaviest molecule in the asymmeric unit,
        intended for removing solvents. 
    show_warnings : bool, optional
        Controls whether warnings that arise during reading are printed.

    Yields
    ------
    :class:`.periodicset.PeriodicSet`
        Represents the crystal as a periodic set, consisting of a finite set of points (motif)
        and lattice (unit cell). Contains other useful data, e.g. the crystal's name and
        information about the asymmetric unit for calculation.

    Examples
    --------
    
        ::

            # Put these entries in a list
            refcodes = ['DEBXIT01', 'DEBXIT05', 'HXACAN01']
            structures = list(amd.CSDReader(refcodes))

            # Read refcode families (any whose refcode starts with strings in the list)
            refcode_families = ['ACSALA', 'HXACAN']
            structures = list(amd.CSDReader(refcode_families, families=True))

            # Get AMDs (k=100) for crystals in these families
            refcodes = ['ACSALA', 'HXACAN']
            amds = []
            for periodic_set in amd.CSDReader(refcodes, families=True):
                amds.append(amd.AMD(periodic_set, 100))

            # Giving the reader nothing reads from the whole CSD.
            reader = amd.CSDReader()
            
            # looping over this generic reader will yield all CSD entries
            for periodic_set in reader:
                ...
            
            # or, read structures by refcode on demand
            debxit01 = reader.entry('DEBXIT01')
    """

    __slots__ = ['_entry_reader']

    def __init__(
            self,
            refcodes=None,
            families=False,
            remove_hydrogens=False,
            disorder='skip',
            heaviest_component=False,
            show_warnings=True,
    ):

        super().__init__(
            remove_hydrogens=remove_hydrogens,
            disorder=disorder,
            heaviest_component=heaviest_component,
            show_warnings=show_warnings,
        )

        if not _CSD_PYTHON_API_ENABLED:
            raise ImportError('Failed to import csd-python-api; check it is installed and licensed.')

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
                hits = [hit.identifier for hit in query.search()]
                all_refcodes.extend(hits)

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
                except RuntimeError:    # if self.show_warnings?
                    warnings.warn(f'Identifier {refcode} not found in database')

    def entry(self, refcode: str, **kwargs) -> PeriodicSet:
        """Read a crystal given a CSD refcode, returning a :class:`.periodicset.PeriodicSet`."""
        entry = self._entry_reader.entry(refcode)
        periodic_set = entry_to_periodicset(entry, **kwargs)
        return periodic_set


def entry_to_periodicset(
        entry,
        remove_hydrogens=False,
        disorder='skip',
        heaviest_component=False
) -> PeriodicSet:
    """:class:`ccdc.entry.Entry` --> :class:`amd.periodicset.PeriodicSet`.
    Entry is the type returned by :class:`ccdc.io.EntryReader`.

    Parameters
    ----------
    entry : :class:`ccdc.entry.Entry`
        A ccdc Entry object representing a database entry.
    remove_hydrogens : bool, optional
        Remove Hydrogens from the crystal.
    disorder : str, optional
        Controls how disordered structures are handled. Default is ``skip`` which skips any crystal 
        with disorder, since disorder conflicts with the periodic set model. To read disordered 
        structures anyway, choose either :code:`ordered_sites` to remove sites with disorder or 
        :code:`all_sites` include all sites regardless.
    heaviest_component : bool, optional
        Removes all but the heaviest molecule in the asymmeric unit,
        intended for removing solvents. 

    Returns
    -------
    :class:`.periodicset.PeriodicSet`
        Represents the crystal as a periodic set, consisting of a finite set of points (motif)
        and lattice (unit cell). Contains other useful data, e.g. the crystal's name and
        information about the asymmetric unit for calculation.

    Raises
    ------
    _ParseError
        Raised if the structure cannot (or should not) be parsed.
        Specifically raised when:
            - entry.has_3d_structure is False
            - disorder == 'skip' and any of:
                any disorder flag is True,
                any atom has fractional occupancy,
                any atom's label ends with '?'
            - entry.crystal.molecule.all_atoms_have_sites is False
            - a.fractional_coordinates is None for any a in entry.crystal.disordered_molecule
            - motif is empty after removing H or disordered sites
    """

    crystal = entry.crystal

    if not entry.has_3d_structure:
        raise _ParseError(f'{crystal.identifier}: Has no 3D structure')

    molecule = crystal.disordered_molecule

    if disorder == 'skip':
        if crystal.has_disorder or entry.has_disorder or \
            any(_atom_has_disorder(a.label, a.occupancy) for a in molecule.atoms):
            raise _ParseError(f'{crystal.identifier}: Has disorder')

    elif disorder == 'ordered_sites':
        molecule.remove_atoms(a for a in molecule.atoms
                              if _atom_has_disorder(a.label, a.occupancy))

    if remove_hydrogens:
        molecule.remove_atoms(a for a in molecule.atoms if a.atomic_symbol in 'HD')

    if heaviest_component and len(molecule.components) > 1:
        molecule = _heaviest_component(molecule)

    if not molecule.all_atoms_have_sites or \
        any(a.fractional_coordinates is None for a in molecule.atoms):
        raise _ParseError(f'{crystal.identifier}: Has atoms without sites')

    crystal.molecule = molecule
    asym_atoms = crystal.asymmetric_unit_molecule.atoms
    asym_unit = np.array([tuple(a.fractional_coordinates) for a in asym_atoms])
    asym_unit = np.mod(asym_unit, 1)
    asym_types = [a.atomic_number for a in asym_atoms]
    cell = utils.cellpar_to_cell(*crystal.cell_lengths, *crystal.cell_angles)

    sitesym = crystal.symmetry_operators
    if not sitesym:
        sitesym = ['x,y,z', ]

    if disorder != 'all_sites':
        keep_sites = _unique_sites(asym_unit)
        if not np.all(keep_sites):
            warnings.warn(f'{crystal.identifier}: May have overlapping sites; duplicates will be removed')
        asym_unit = asym_unit[keep_sites]
        asym_types = [sym for sym, keep in zip(asym_types, keep_sites) if keep]

    if asym_unit.shape[0] == 0:
        raise _ParseError(f'{crystal.identifier}: Has no valid sites')

    frac_motif, asym_inds, multiplicities, inverses = _expand_asym_unit(asym_unit, sitesym)
    full_types = np.array([asym_types[i] for i in inverses])
    motif = frac_motif @ cell

    tags = {
        'name': crystal.identifier,
        'asymmetric_unit': asym_inds,
        'wyckoff_multiplicities': multiplicities,
        'types': full_types
    }

    return PeriodicSet(motif, cell, **tags)


def cifblock_to_periodicset(
        block,
        remove_hydrogens=False,
        disorder='skip'
) -> PeriodicSet:
    """:class:`ase.io.cif.CIFBlock` --> :class:`amd.periodicset.PeriodicSet`. 
    CIFBlock is the type returned by :class:`ase.io.cif.parse_cif`.

    Parameters
    ----------
    block : :class:`ase.io.cif.CIFBlock`
        An ase CIFBlock object representing a crystal.
    remove_hydrogens : bool, optional
        Remove Hydrogens from the crystal.
    disorder : str, optional
        Controls how disordered structures are handled. Default is ``skip`` which skips any crystal 
        with disorder, since disorder conflicts with the periodic set model. To read disordered 
        structures anyway, choose either :code:`ordered_sites` to remove sites with disorder or 
        :code:`all_sites` include all sites regardless.

    Returns
    -------
    :class:`.periodicset.PeriodicSet`
        Represents the crystal as a periodic set, consisting of a finite set of points (motif)
        and lattice (unit cell). Contains other useful data, e.g. the crystal's name and
        information about the asymmetric unit for calculation.

    Raises
    ------
    _ParseError
        Raised if the structure cannot (or should not) be parsed.
        Specifically raised when:
            - no sites found or motif is empty after removing H or disordered sites
            - a site has missing coordinates
            - disorder == 'skip' and any of:
                any disorder flag is True,
                any atom has fractional occupancy,
                any atom's label ends with '?'
    """

    cell = block.get_cell().array

    # asymmetric unit fractional coords
    asym_unit = [block.get(name) for name in _ATOM_SITE_FRACT_TAGS]
    if None in asym_unit:
        asym_motif = [block.get(name) for name in _ATOM_SITE_CARTN_TAGS]
        if None in asym_motif:
            raise _ParseError(f'{block.name}: Has no sites')
        asym_unit = np.array(asym_motif) @ np.linalg.inv(cell)

    if any(None in coords for coords in asym_unit):
        raise _ParseError(f'{block.name}: Has atoms without sites')

    asym_unit = np.mod(np.array(asym_unit).T, 1)

    try:
        asym_types = [ase.data.atomic_numbers[s] for s in block.get_symbols()]
    except ase.io.cif.NoStructureData as _:
        asym_types = [0 for _ in range(len(asym_unit))]

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
            if any(_atom_has_disorder(lab, occ) for lab, occ in zip(labels, occupancies)):
                raise _ParseError(f'{block.name}: Has disorder')
        elif disorder == 'ordered_sites':
            remove_sites.extend(
                (i for i, (lab, occ) in enumerate(zip(labels, occupancies))
                 if _atom_has_disorder(lab, occ)))

    if remove_hydrogens:
        remove_sites.extend((i for i, num in enumerate(asym_types) if num == 1))

    asym_unit = np.delete(asym_unit, remove_sites, axis=0)
    asym_types = [s for i, s in enumerate(asym_types) if i not in remove_sites]

    if disorder != 'all_sites':
        keep_sites = _unique_sites(asym_unit)
        if not np.all(keep_sites):
            warnings.warn(f'{block.name}: May have overlapping sites; duplicates will be removed')
        asym_unit = asym_unit[keep_sites]
        asym_types = [sym for sym, keep in zip(asym_types, keep_sites) if keep]

    if asym_unit.shape[0] == 0:
        raise _ParseError(f'{block.name}: Has no valid sites')

    frac_motif, asym_inds, multiplicities, inverses = _expand_asym_unit(asym_unit, sitesym)
    full_types = np.array([asym_types[i] for i in inverses])
    motif = frac_motif @ cell

    tags = {
        'name': block.name,
        'asymmetric_unit': asym_inds,
        'wyckoff_multiplicities': multiplicities,
        'types': full_types
    }

    return PeriodicSet(motif, cell, **tags)


def _expand_asym_unit(
        asym_unit: np.ndarray,
        sitesym: Sequence[str]
) -> Tuple[np.ndarray, ...]:
    """
    Asymmetric unit's fractional coords + site symmetries (as strings)
    -->
    fractional motif, asymmetric unit indices, multiplicities and inverses.
    """

    rotations, translations = parse_sitesym(sitesym)
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


def _atom_has_disorder(label, occupancy):
    """Return True if label ends with ? or occupancy < 1."""
    return label.endswith('?') or (np.isscalar(occupancy) and occupancy < 1)


@numba.njit()
def _unique_sites(asym_unit):
    site_diffs1 = np.abs(np.expand_dims(asym_unit, 1) - asym_unit)
    site_diffs2 = np.abs(site_diffs1 - 1)
    sites_neq_mask = np.logical_and((site_diffs1 > _EQUIV_SITE_TOL), (site_diffs2 > _EQUIV_SITE_TOL))
    overlapping = np.triu(sites_neq_mask.sum(axis=-1) == 0, 1)
    return overlapping.sum(axis=0) == 0


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
