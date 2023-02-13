"""Tools for reading crystals from files, or from the CSD with
``csd-python-api``. The readers return
:class:`amd.PeriodicSet <.periodicset.PeriodicSet>` objects representing
the crystal which can be passed to :func:`amd.AMD() <.calculate.AMD>`
and :func:`amd.PDD() <.calculate.PDD>` to get their invariants.
"""

import os
import re
import functools
import warnings
import errno
from typing import Tuple

import numpy as np
import numba

from .utils import cellpar_to_cell
from .periodicset import PeriodicSet


def _custom_warning(message, category, filename, lineno, *args, **kwargs):
    return f'{category.__name__}: {message}\n'
warnings.formatwarning = _custom_warning

_EQUIV_SITE_TOL = 1e-3

CIF_TAGS = {
    'cellpar': [
        '_cell_length_a',
        '_cell_length_b',
        '_cell_length_c',
        '_cell_angle_alpha',
        '_cell_angle_beta',
        '_cell_angle_gamma',],

    'atom_site_fract': [
        '_atom_site_fract_x',
        '_atom_site_fract_y',
        '_atom_site_fract_z',],

    'atom_site_cartn': [
        '_atom_site_cartn_x',
        '_atom_site_cartn_y',
        '_atom_site_cartn_z',],

    'atom_symbol': [
        '_atom_site_type_symbol',
        '_atom_site_label',],

    'symop': [
        '_symmetry_equiv_pos_as_xyz',
        '_space_group_symop_operation_xyz',
        '_space_group_symop.operation_xyz',
        '_symmetry_equiv_pos_as_xyz_',
        '_space_group_symop_operation_xyz_',],

    'spacegroup_name': [
        '_space_group_name_Hall',
        '_symmetry_space_group_name_hall',
        '_space_group_name_H-M_alt',
        '_symmetry_space_group_name_H-M',
        '_symmetry_space_group_name_H_M',
        '_symmetry_space_group_name_h-m',],

    'spacegroup_number': [
        '_space_group_IT_number',
        '_symmetry_Int_Tables_number',
        '_space_group_IT_number_',
        '_symmetry_Int_Tables_number_',],
}


class _Reader:

    _DISORDER_OPTIONS = {'skip', 'ordered_sites', 'all_sites'}
    _CCDC_IMPORT_ERR_MSG = 'Failed to import csd-python-api, please check ' \
                           'it is installed and licensed.'

    def __init__(self, iterable, converter, show_warnings=True):
        self._iterator = iter(iterable)
        self._converter = converter
        self.show_warnings = show_warnings

    def __iter__(self):
        if self._iterator is None or self._converter is None:
            raise RuntimeError(f'{self.__class__.__name__} not initialized.')
        return self

    def __next__(self):
        """Iterates over self._iterator, passing items through
        self._converter. Catches :class:`ParseError <.io.ParseError>`
        and warnings raised in self._converter, optionally printing
        them.
        """

        if not self.show_warnings:
            warnings.simplefilter('ignore')

        while True:

            item = next(self._iterator)

            with warnings.catch_warnings(record=True) as warning_msgs:
                msg = None
                try:
                    periodic_set = self._converter(item)
                except ParseError as err:
                    msg = str(err)

            if msg:
                warnings.warn(msg)
                continue

            for warning in warning_msgs:
                msg = f'(name={periodic_set.name}) {warning.message}'
                warnings.warn(msg, category=warning.category)

            return periodic_set

    def read(self):
        """Reads the crystal(s), returns one
        :class:`amd.PeriodicSet <.periodicset.PeriodicSet>` if there is
        only one, otherwise returns a list.
        """

        l = list(self)
        if len(l) == 1:
            return l[0]
        return l


class CifReader(_Reader):
    """Read all structures in a .cif file or all files in a folder
    with ase or csd-python-api (if installed), yielding
    :class:`amd.PeriodicSet <.periodicset.PeriodicSet>` s.

    Parameters
    ----------
    path : str
        Path to a .CIF file or directory. (Other files are accepted when
        using ``reader='ccdc'``, if csd-python-api is installed.)
    reader : str, optional
        The backend package used to parse the CIF. The default is 
        :code:`ase`, :code:`pymatgen` and :code:`gemmi` are also
        accepted, as well as :code:`ccdc` if csd-python-api is 
        installed. The ccdc reader should be able to read any format
        accepted by :class:`ccdc.io.EntryReader`, though only CIFs have
        been tested.
    remove_hydrogens : bool, optional
        Remove Hydrogens from the crystals.
    disorder : str, optional
        Controls how disordered structures are handled. Default is
        ``skip`` which skips any crystal with disorder, since disorder
        conflicts with the periodic set model. To read disordered
        structures anyway, choose either :code:`ordered_sites` to remove
        atoms with disorder or :code:`all_sites` include all atoms
        regardless of disorder.
    heaviest_component : bool, optional
        csd-python-api only. Removes all but the heaviest molecule in
        the asymmeric unit, intended for removing solvents.
    molecular_centres : bool, default False
        csd-python-api only. Extract the centres of molecules in the
        unit cell and store in the attribute molecular_centres.
    show_warnings : bool, optional
        Controls whether warnings that arise during reading are printed.

    Yields
    ------
    :class:`amd.PeriodicSet <.periodicset.PeriodicSet>`
        Represents the crystal as a periodic set, consisting of a finite
        set of points (motif) and lattice (unit cell). Contains other
        data, e.g. the crystal's name and information about the
        asymmetric unit.

    Examples
    --------

        ::

            # Put all crystals in a .CIF in a list
            structures = list(amd.CifReader('mycif.cif'))

            # Can also accept path to a directory, reading all files inside
            structures = list(amd.CifReader('path/to/folder'))

            # Reads just one if the .CIF has just one crystal
            periodic_set = amd.CifReader('mycif.cif').read()

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
            molecular_centres=False,
            show_warnings=True,
    ):

        if disorder not in CifReader._DISORDER_OPTIONS:
            msg = 'disorder parameter must be one of ' \
                  f'{CifReader._DISORDER_OPTIONS} (passed "{disorder}")'
            raise ValueError(msg)

        if reader != 'ccdc':
            if heaviest_component:
                msg = 'Parameter heaviest_component only implemented for ' \
                      'reader="ccdc".'
                raise NotImplementedError(msg)

            if molecular_centres:
                msg = 'Parameter molecular_centres only implemented for ' \
                      'reader="ccdc".'
                raise NotImplementedError(msg)

        if reader in ('ase', 'pycodcif'):
            from ase.io.cif import parse_cif
            extensions = {'cif'}
            file_parser = functools.partial(parse_cif, reader=reader)
            converter = functools.partial(
                periodicset_from_ase_cifblock,
                remove_hydrogens=remove_hydrogens,
                disorder=disorder
            )

        elif reader == 'pymatgen':
            extensions = {'cif'}
            file_parser = CifReader._pymatgen_cifblock_generator
            converter = functools.partial(
                periodicset_from_pymatgen_cifblock,
                remove_hydrogens=remove_hydrogens,
                disorder=disorder
            )

        elif reader == 'gemmi':
            import gemmi
            extensions = {'cif'}
            file_parser = gemmi.cif.read_file
            converter = functools.partial(
                periodicset_from_gemmi_block,
                remove_hydrogens=remove_hydrogens,
                disorder=disorder
            )

        elif reader == 'ccdc':
            try:
                import ccdc.io
            except (ImportError, RuntimeError) as e:
                raise ImportError(_Reader._CCDC_IMPORT_ERR_MSG) from e

            extensions = ccdc.io.EntryReader.known_suffixes
            file_parser = ccdc.io.EntryReader
            converter = functools.partial(
                periodicset_from_ccdc_entry,
                remove_hydrogens=remove_hydrogens,
                disorder=disorder,
                molecular_centres=molecular_centres,
                heaviest_component=heaviest_component
            )

        else:
            raise ValueError(f'Unknown reader "{reader}".')

        if os.path.isfile(path):
            iterable = file_parser(path)
        elif os.path.isdir(path):
            iterable = CifReader._dir_generator(path, file_parser, extensions)
        else:
            raise FileNotFoundError(
                errno.ENOENT, os.strerror(errno.ENOENT), path)

        super().__init__(iterable, converter, show_warnings=show_warnings)

    @staticmethod
    def _dir_generator(path, callable, extensions):
        for file in os.listdir(path):
            suff = os.path.splitext(file)[1][1:]
            if suff.lower() in extensions:
                yield from callable(os.path.join(path, file))

    @staticmethod
    def _pymatgen_cifblock_generator(path):
        """Path to .cif --> generator of pymatgen CifBlock objects."""
        from pymatgen.io.cif import CifFile
        yield from CifFile.from_file(path).data.values()


class CSDReader(_Reader):
    """Read structures from the CSD with csd-python-api, yielding
    :class:`amd.PeriodicSet <.periodicset.PeriodicSet>` s.

    Parameters
    ----------
    refcodes : str or List[str], optional
        Single or list of CSD refcodes to read. If None or 'CSD',
        iterates over the whole CSD.
    families : bool, optional
        Read all entries whose refcode starts with the given strings, or
        'families' (e.g. giving 'DEBXIT' reads all entries starting with
        DEBXIT).
    remove_hydrogens : bool, optional
        Remove hydrogens from the crystals.
    disorder : str, optional
        Controls how disordered structures are handled. Default is
        ``skip`` which skips any crystal with disorder, since disorder
        conflicts with the periodic set model. To read disordered
        structures anyway, choose either :code:`ordered_sites` to remove
        atoms with disorder or :code:`all_sites` include all atoms
        regardless of disorder.
    heaviest_component : bool, optional
        Removes all but the heaviest molecule in the asymmeric unit,
        intended for removing solvents.
    molecular_centres : bool, default False
        Extract the centres of molecules in the unit cell and store in
        attribute molecular_centres.
    show_warnings : bool, optional
        Controls whether warnings that arise during reading are printed.

    Yields
    ------
    :class:`amd.PeriodicSet <.periodicset.PeriodicSet>`
        Represents the crystal as a periodic set, consisting of a finite
        set of points (motif) and lattice (unit cell). Contains other
        useful data, e.g. the crystal's name and information about the
        asymmetric unit for calculation.

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
            for periodic_set in amd.CSDReader():
                ...
    """

    def __init__(
            self,
            refcodes=None,
            families=False,
            remove_hydrogens=False,
            disorder='skip',
            heaviest_component=False,
            molecular_centres=False,
            show_warnings=True,
    ):

        try:
            import ccdc.io
        except (ImportError, RuntimeError) as _:
            raise ImportError(_Reader._CCDC_IMPORT_ERR_MSG)

        if disorder not in _Reader._DISORDER_OPTIONS:
            msg = 'disorder parameter must be one of ' \
                  f'{_Reader._DISORDER_OPTIONS} (passed {disorder})'
            raise ValueError(msg)

        if isinstance(refcodes, str) and refcodes.lower() == 'csd':
            refcodes = None

        if refcodes is None:
            families = False
        else:
            if isinstance(refcodes, str):
                refcodes = [refcodes]
            else:
                refcodes = list(refcodes)

        if families:
            refcodes = self._refcodes_from_families_ccdc(refcodes)

        entry_reader = ccdc.io.EntryReader('CSD')
        converter = functools.partial(
            periodicset_from_ccdc_entry,
            remove_hydrogens=remove_hydrogens,
            disorder=disorder,
            molecular_centres=molecular_centres,
            heaviest_component=heaviest_component
        )
        iterable = self._ccdc_generator(refcodes, entry_reader)
        super().__init__(iterable, converter, show_warnings=show_warnings)

    @staticmethod
    def _ccdc_generator(refcodes, entry_reader):
        """Generates ccdc Entries from CSD refcodes.
        """

        if refcodes is None:
            for entry in entry_reader:
                yield entry
        else:
            for refcode in refcodes:
                try:
                    entry = entry_reader.entry(refcode)
                    yield entry
                except RuntimeError:
                    warnings.warn(f'{refcode} not found in database')

    @staticmethod
    def _refcodes_from_families_ccdc(refcode_families):
        """List of strings --> all CSD refcodes starting with any of the
        strings. Intended to be passed a list of families and return all
        refcodes in them.
        """

        try:
            import ccdc.search
        except (ImportError, RuntimeError) as _:
            raise ImportError(_Reader._CCDC_IMPORT_ERR_MSG)

        all_refcodes = []
        for refcode in refcode_families:
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

        return refcodes


class ParseError(ValueError):
    """Raised when an item cannot be parsed into a periodic set.
    """
    pass


def periodicset_from_ase_cifblock(
        block,
        remove_hydrogens=False,
        disorder='skip'
) -> PeriodicSet:
    """:class:`ase.io.cif.CIFBlock` --> 
    :class:`amd.PeriodicSet <.periodicset.PeriodicSet>`.
    :class:`ase.io.cif.CIFBlock` is the type returned by
    :class:`ase.io.cif.parse_cif`.

    Parameters
    ----------
    block : :class:`ase.io.cif.CIFBlock`
        An ase :class:`ase.io.cif.CIFBlock` object representing a
        crystal.
    remove_hydrogens : bool, optional
        Remove Hydrogens from the crystal.
    disorder : str, optional
        Controls how disordered structures are handled. Default is
        ``skip`` which skips any crystal with disorder, since disorder
        conflicts with the periodic set model. To read disordered
        structures anyway, choose either :code:`ordered_sites` to remove
        atoms with disorder or :code:`all_sites` include all atoms
        regardless of disorder.

    Returns
    -------
    :class:`amd.PeriodicSet <.periodicset.PeriodicSet>`
        Represents the crystal as a periodic set, consisting of a finite
        set of points (motif) and lattice (unit cell). Contains other
        useful data, e.g. the crystal's name and information about the
        asymmetric unit for calculation.

    Raises
    ------
    ParseError
        Raised if the structure fails to be parsed for any of the
        following: 1. Required data is missing (e.g. cell parameters),
        2. The motif is empty after removing H or disordered sites,
        3. :code:``disorder == 'skip'`` and disorder is found on any
        atom.
    """

    import ase

    # Unit cell
    cellpar = [block.get(tag) for tag in CIF_TAGS['cellpar']]
    if None in cellpar:
        raise ParseError(f'{block.name} has missing cell data')
    cell = cellpar_to_cell(*cellpar)

    # Asymmetric unit coordinates. ase removes uncertainty brackets
    cartesian = False # flag needed for later
    asym_unit = [block.get(name) for name in CIF_TAGS['atom_site_fract']]
    if None in asym_unit: # missing scaled coords, try Cartesian
        asym_unit = [block.get(name) for name in CIF_TAGS['atom_site_cartn']]
        if None in asym_unit:
            raise ParseError(f'{block.name} has missing coordinates')
        cartesian = True
    asym_unit = list(zip(*asym_unit)) # transpose [xs,ys,zs] -> [p1,p2,...]

    # Atomic types
    asym_symbols = block._get_any(CIF_TAGS['atom_symbol'])
    if asym_symbols is None:
        warnings.warn('missing atomic types will be labelled 0')
        asym_types = [0] * len(asym_unit)
    else:
        asym_types = []
        for label in asym_symbols:
            if label in ('.', '?'):
                warnings.warn('missing atomic types will be labelled 0')
                num = 0
            else:
                sym = re.search(r'([A-Z][a-z]?)', label).group(0)
                if sym == 'D':
                    sym = 'H'
                num = ase.data.atomic_numbers[sym]
            asym_types.append(num)

    # Find where sites have disorder if necassary
    has_disorder = []
    if disorder != 'all_sites':
        occupancies = block.get('_atom_site_occupancy')
        if occupancies is None:
            occupancies = np.ones((len(asym_unit), ))
        labels = block.get('_atom_site_label')
        if labels is None:
            labels = [''] * len(asym_unit)
        for lab, occ in zip(labels, occupancies):
            has_disorder.append(_has_disorder(lab, occ))

    # Remove sites with ?, . or other invalid string for coordinates
    invalid = []
    for i, xyz in enumerate(asym_unit):
        if not all(isinstance(coord, (int, float)) for coord in xyz):
            invalid.append(i)
    if invalid:
        warnings.warn('atoms without sites or missing data will be removed')
        asym_unit = [c for i, c in enumerate(asym_unit) if i not in invalid]
        asym_types = [t for i, t in enumerate(asym_types) if i not in invalid]
        if disorder != 'all_sites':
            has_disorder = [d for i, d in enumerate(has_disorder)
                            if i not in invalid]

    remove_sites = []

    if remove_hydrogens:
        remove_sites.extend(i for i, num in enumerate(asym_types) if num == 1)

    # Remove atoms with fractional occupancy or raise ParseError
    if disorder != 'all_sites':
        for i, dis in enumerate(has_disorder):
            if i in remove_sites:
                continue
            if dis:
                if disorder == 'skip':
                    msg = f"{block.name} has disorder, pass " \
                           "disorder='ordered_sites'or 'all_sites' to " \
                           "remove/ignore disorder"
                    raise ParseError(msg)
                elif disorder == 'ordered_sites':
                    remove_sites.append(i)

    # Asymmetric unit
    asym_unit = [c for i, c in enumerate(asym_unit) if i not in remove_sites]
    asym_types = [t for i, t in enumerate(asym_types) if i not in remove_sites]
    if len(asym_unit) == 0:
        raise ParseError(f'{block.name} has no valid sites')
    asym_unit = np.array(asym_unit)

    # If Cartesian coords were given, convert to scaled
    if cartesian:
        asym_unit = asym_unit @ np.linalg.inv(cell)
    asym_unit = np.mod(asym_unit, 1)
    # asym_unit = _snap_small_prec_coords(asym_unit, 1e-4) # recommended by pymatgen

    # Remove overlapping sites unless disorder == 'all_sites'
    if disorder != 'all_sites':
        keep_sites = _unique_sites(asym_unit, _EQUIV_SITE_TOL)
        if not np.all(keep_sites):
            msg = 'may have overlapping sites, duplicates will be removed'
            warnings.warn(msg)
            asym_unit = asym_unit[keep_sites]
            asym_types = [t for t, keep in zip(asym_types, keep_sites) if keep]

    # Symmetry operations
    sitesym = block._get_any(CIF_TAGS['symop'])
    if sitesym is None: # no symops, use spacegroup
        try:
            spg = block.get_spacegroup(True)
            rot, trans = spg.rotations, spg.translations
        except: # no spacegroup, assume P1
            rot, trans = _parse_sitesym(['x,y,z'])
    else:
        if isinstance(sitesym, str):
            sitesym = [sitesym]
        rot, trans = _parse_sitesym(sitesym)

    # Apply symmetries to asymmetric unit
    frac_motif, asym_inds, wyc_muls, inverses = _expand_asym_unit(asym_unit, rot, trans)
    types = np.array([asym_types[i] for i in inverses])
    motif = frac_motif @ cell

    return PeriodicSet(
        motif=motif,
        cell=cell,
        name=block.name,
        asymmetric_unit=asym_inds,
        wyckoff_multiplicities=wyc_muls,
        types=types
    )


def periodicset_from_ase_atoms(
        atoms,
        remove_hydrogens=False
) -> PeriodicSet:
    """:class:`ase.atoms.Atoms` -->
    :class:`amd.PeriodicSet <.periodicset.PeriodicSet>`. Does not have
    the option to remove disorder.

    Parameters
    ----------
    atoms : :class:`ase.atoms.Atoms`
        An ase :class:`ase.atoms.Atoms` object representing a crystal.
    remove_hydrogens : bool, optional
        Remove Hydrogens from the crystal.

    Returns
    -------
    :class:`amd.PeriodicSet <.periodicset.PeriodicSet>`
        Represents the crystal as a periodic set, consisting of a finite
        set of points (motif) and lattice (unit cell). Contains other
        useful data, e.g. the crystal's name and information about the
        asymmetric unit for calculation.

    Raises
    ------
    ParseError
        Raised if there are no valid sites in atoms.
    """

    from ase.spacegroup import get_basis

    cell = atoms.get_cell().array

    remove_inds = []
    if remove_hydrogens:
        for i in np.where(atoms.get_atomic_numbers() == 1)[0]:
            remove_inds.append(i)
    for i in sorted(remove_inds, reverse=True):
        atoms.pop(i)

    if len(atoms) == 0:
        raise ParseError(f'ase Atoms object has no valid sites')

    # Symmetry operations from spacegroup 
    spg = None
    if 'spacegroup' in atoms.info:
        spg = atoms.info['spacegroup']
        rot, trans = spg.rotations, spg.translations
    # else assume no symmetries?

    # Asymmetric unit. ase default tol is 1e-5
    asym_unit = get_basis(atoms, spacegroup=spg, tol=_EQUIV_SITE_TOL)
    frac_motif, asym_inds, wyc_muls, _ = _expand_asym_unit(asym_unit, rot, trans)
    motif = frac_motif @ cell

    return PeriodicSet(
        motif=motif,
        cell=cell,
        asymmetric_unit=asym_inds,
        wyckoff_multiplicities=wyc_muls,
        types=atoms.get_atomic_numbers()
    )


def periodicset_from_pymatgen_cifblock(
        block,
        remove_hydrogens=False,
        disorder='skip'
) -> PeriodicSet:
    """:class:`pymatgen.io.cif.CifBlock` -->
    :class:`amd.PeriodicSet <.periodicset.PeriodicSet>`.

    Parameters
    ----------
    block : :class:`pymatgen.io.cif.CifBlock`
        A pymatgen CifBlock object representing a crystal.
    remove_hydrogens : bool, optional
        Remove Hydrogens from the crystal.
    disorder : str, optional
        Controls how disordered structures are handled. Default is
        ``skip`` which skips any crystal with disorder, since disorder
        conflicts with the periodic set model. To read disordered
        structures anyway, choose either :code:`ordered_sites` to remove
        atoms with disorder or :code:`all_sites` include all atoms
        regardless of disorder.

    Returns
    -------
    :class:`amd.PeriodicSet <.periodicset.PeriodicSet>`
        Represents the crystal as a periodic set, consisting of a finite
        set of points (motif) and lattice (unit cell). Contains other
        useful data, e.g. the crystal's name and information about the
        asymmetric unit for calculation.

    Raises
    ------
    ParseError
        Raised if the structure can/should not be parsed for the
        following reasons: 1. No sites found or motif is empty after
        removing Hydrogens & disorder, 2. A site has missing
        coordinates, 3. :code:``disorder == 'skip'`` and disorder is
        found on any atom.
    """

    from pymatgen.io.cif import str2float
    import pymatgen.core.periodic_table as periodic_table

    odict = block.data

    # Unit cell
    cellpar = [odict.get(tag) for tag in CIF_TAGS['cellpar']]
    if None in cellpar:
        raise ParseError(f'{block.header} has missing cell data')
    cellpar = [str2float(v) for v in cellpar]
    cell = cellpar_to_cell(*cellpar)

    # Asymmetric unit coordinates
    cartesian = False
    asym_unit = [odict.get(tag) for tag in CIF_TAGS['atom_site_fract']]

    if None in asym_unit: # missing scaled coordinates, try Cartesian
        asym_unit = [odict.get(tag) for tag in CIF_TAGS['atom_site_cartn']]
        if None in asym_unit:
            raise ParseError(f'{block.header} has no coordinates')
        cartesian = True

    asym_unit = list(zip(*asym_unit)) # transpose [xs,ys,zs] -> [p1,p2,...]
    asym_unit = [[str2float(coord) for coord in xyz] for xyz in asym_unit]

    # Atomic types
    for tag in CIF_TAGS['atom_symbol']:
        asym_symbols = odict.get(tag)
        if asym_symbols is not None:
            asym_types = []
            for label in asym_symbols:
                if label in ('.', '?'):
                    warnings.warn('missing atomic types will be labelled 0')
                    num = 0
                else:
                    sym = re.search(r'([A-Z][a-z]?)', label).group(0)
                    if sym == 'D':
                        sym = 'H'
                    num = periodic_table.Element[sym].number
                asym_types.append(num)
            break
    else:
        warnings.warn('missing atomic types will be labelled 0')
        asym_types = [0] * len(asym_unit)

    # Find where sites have disorder if necassary
    has_disorder = []
    if disorder != 'all_sites':
        occupancies = odict.get('_atom_site_occupancy')
        if occupancies is None:
            occupancies = np.ones((len(asym_unit), ))
        labels = odict.get('_atom_site_label')
        if labels is None:
            labels = [''] * len(asym_unit)
        for lab, occ in zip(labels, occupancies):
            has_disorder.append(_has_disorder(lab, occ))

    # Remove sites with ?, . or other invalid string for coordinates
    invalid = []
    for i, xyz in enumerate(asym_unit):
        if not all(isinstance(coord, (int, float)) for coord in xyz):
            invalid.append(i)

    if invalid:
        warnings.warn('atoms without sites or missing data will be removed')
        asym_unit = [c for i, c in enumerate(asym_unit) if i not in invalid]
        asym_types = [c for i, c in enumerate(asym_types) if i not in invalid]
        if disorder != 'all_sites':
            has_disorder = [d for i, d in enumerate(has_disorder) if i not in invalid]

    remove_sites = []

    if remove_hydrogens:
        remove_sites.extend((i for i, num in enumerate(asym_types) if num == 1))

    # Remove atoms with fractional occupancy or raise ParseError
    if disorder != 'all_sites':
        for i, dis in enumerate(has_disorder):
            if i in remove_sites:
                continue
            if dis:
                if disorder == 'skip':
                    msg = f"{block.header} has disorder, pass " \
                            "disorder='ordered_sites' or 'all_sites' to " \
                            "remove/ignore disorder"
                    raise ParseError(msg)
                elif disorder == 'ordered_sites':
                    remove_sites.append(i)

    # Asymmetric unit
    asym_unit = [c for i, c in enumerate(asym_unit) if i not in remove_sites]
    asym_types = [t for i, t in enumerate(asym_types) if i not in remove_sites]
    if len(asym_unit) == 0:
        raise ParseError(f'{block.header} has no valid sites')
    asym_unit = np.array(asym_unit)

    # If Cartesian coords were given, convert to scaled
    if cartesian:
        asym_unit = asym_unit @ np.linalg.inv(cell)
    asym_unit = np.mod(asym_unit, 1)
    # asym_unit = _snap_small_prec_coords(asym_unit) # recommended by pymatgen

    # Remove overlapping sites unless disorder == 'all_sites'
    if disorder != 'all_sites':      
        keep_sites = _unique_sites(asym_unit, _EQUIV_SITE_TOL)
        if not np.all(keep_sites):
            msg = 'may have overlapping sites; duplicates will be removed'
            warnings.warn(msg)
        asym_unit = asym_unit[keep_sites]
        asym_types = [sym for sym, keep in zip(asym_types, keep_sites) if keep]

    # Apply symmetries to asymmetric unit
    rot, trans = _parse_sitesym_pymatgen(odict)
    frac_motif, asym_inds, wyc_muls, inverses = _expand_asym_unit(asym_unit, rot, trans)
    types = np.array([asym_types[i] for i in inverses])
    motif = frac_motif @ cell

    return PeriodicSet(
        motif=motif,
        cell=cell,
        name=block.header,
        asymmetric_unit=asym_inds,
        wyckoff_multiplicities=wyc_muls,
        types=types
    )


def periodicset_from_pymatgen_structure(
        structure,
        remove_hydrogens=False,
        disorder='skip'
) -> PeriodicSet:
    """:class:`pymatgen.core.structure.Structure` -->
    :class:`amd.PeriodicSet <.periodicset.PeriodicSet>`.
    Does not set the name of the periodic set, as there seems to be no
    name attribute in the pymatgen Structure object.

    Parameters
    ----------
    structure : :class:`pymatgen.core.structure.Structure`
        A pymatgen Structure object representing a crystal.
    remove_hydrogens : bool, optional
        Remove Hydrogens from the crystal.
    disorder : str, optional
        Controls how disordered structures are handled. Default is
        ``skip`` which skips any crystal with disorder, since disorder
        conflicts with the periodic set model. To read disordered
        structures anyway, choose either :code:`ordered_sites` to remove
        atoms with disorder or :code:`all_sites` include all atoms
        regardless of disorder.

    Returns
    -------
    :class:`amd.PeriodicSet <.periodicset.PeriodicSet>`
        Represents the crystal as a periodic set, consisting of a finite
        set of points (motif) and lattice (unit cell). Contains other
        useful data, e.g. the crystal's name and information about the
        asymmetric unit for calculation.
    
    Raises
    ------
    ParseError
        Raised if the :code:`disorder == 'skip'` and 
        :code:`not structure.is_ordered`
    """

    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

    if remove_hydrogens:
        structure.remove_species(['H', 'D'])

    # Disorder
    if disorder == 'skip':
        if not structure.is_ordered:
            msg = f"pymatgen Structure has disorder, pass " \
                   "disorder='ordered_sites' or 'all_sites' to " \
                   "remove/ignore disorder"
            raise ParseError(msg)
    elif disorder == 'ordered_sites':
        remove_inds = []
        for i, comp in enumerate(structure.species_and_occu):
            if comp.num_atoms < 1:
                remove_inds.append(i)
        structure.remove_sites(remove_inds)

    motif = structure.cart_coords
    cell = structure.lattice.matrix
    sym_structure = SpacegroupAnalyzer(structure).get_symmetrized_structure()
    asym_unit = np.array([l[0] for l in sym_structure.equivalent_indices])
    wyc_muls = np.array([len(l) for l in sym_structure.equivalent_indices])
    types = np.array(sym_structure.atomic_numbers)

    return PeriodicSet(
        motif=motif,
        cell=cell,
        asymmetric_unit=asym_unit,
        wyckoff_multiplicities=wyc_muls,
        types=types
    )


def periodicset_from_ccdc_entry(
        entry,
        remove_hydrogens=False,
        disorder='skip',
        heaviest_component=False,
        molecular_centres=False
) -> PeriodicSet:
    """:class:`ccdc.entry.Entry` -->
    :class:`amd.PeriodicSet <.periodicset.PeriodicSet>`.
    Entry is the type returned by :class:`ccdc.io.EntryReader`.

    Parameters
    ----------
    entry : :class:`ccdc.entry.Entry`
        A ccdc Entry object representing a database entry.
    remove_hydrogens : bool, optional
        Remove Hydrogens from the crystal.
    disorder : str, optional
        Controls how disordered structures are handled. Default is
        ``skip`` which skips any crystal with disorder, since disorder
        conflicts with the periodic set model. To read disordered
        structures anyway, choose either :code:`ordered_sites` to remove
        atoms with disorder or :code:`all_sites` include all atoms
        regardless of disorder.
    heaviest_component : bool, optional
        Removes all but the heaviest molecule in the asymmeric unit,
        intended for removing solvents.
    molecular_centres : bool, default False
        Extract the centres of molecules in the unit cell and store in
        the attribute molecular_centres of the returned PeriodicSet.

    Returns
    -------
    :class:`amd.PeriodicSet <.periodicset.PeriodicSet>`
        Represents the crystal as a periodic set, consisting of a finite
        set of points (motif) and lattice (unit cell). Contains other
        useful data, e.g. the crystal's name and information about the
        asymmetric unit for calculation.

    Raises
    ------
    ParseError
        Raised if the structure fails parsing for any of the following:
        1. entry.has_3d_structure is False, 2.
        :code:``disorder == 'skip'`` and disorder is found on any atom,
        3. entry.crystal.molecule.all_atoms_have_sites is False,
        4. a.fractional_coordinates is None for any a in
        entry.crystal.disordered_molecule, 5. The motif is empty after
        removing Hydrogens and disordered sites.
    """

    # Entry specific flag
    if not entry.has_3d_structure:
        raise ParseError(f'{entry.identifier} has no 3D structure')

    # Disorder
    if disorder == 'skip' and entry.has_disorder:
        msg = f"{entry.identifier} has disorder, pass " \
                "disorder='ordered_sites' or 'all_sites' to remove/ignore" \
                "disorder"
        raise ParseError(msg)

    return periodicset_from_ccdc_crystal(
        entry.crystal,
        remove_hydrogens=remove_hydrogens,
        disorder=disorder,
        heaviest_component=heaviest_component,
        molecular_centres=molecular_centres
    )


def periodicset_from_ccdc_crystal(
        crystal,
        remove_hydrogens=False,
        disorder='skip',
        heaviest_component=False,
        molecular_centres=False
) -> PeriodicSet:
    """:class:`ccdc.crystal.Crystal` -->
    :class:`amd.PeriodicSet <.periodicset.PeriodicSet>`.
    Crystal is the type returned by :class:`ccdc.io.CrystalReader`.

    Parameters
    ----------
    crystal : :class:`ccdc.crystal.Crystal`
        A ccdc Crystal object representing a crystal structure.
    remove_hydrogens : bool, optional
        Remove Hydrogens from the crystal.
    disorder : str, optional
        Controls how disordered structures are handled. Default is
        ``skip`` which skips any crystal with disorder, since disorder
        conflicts with the periodic set model. To read disordered
        structures anyway, choose either :code:`ordered_sites` to remove
        atoms with disorder or :code:`all_sites` include all atoms
        regardless of disorder.
    heaviest_component : bool, optional
        Removes all but the heaviest molecule in the asymmeric unit,
        intended for removing solvents.
    molecular_centres : bool, default False
        Extract the centres of molecules in the unit cell and store in
        the attribute molecular_centres of the returned PeriodicSet.

    Returns
    -------
    :class:`amd.PeriodicSet <.periodicset.PeriodicSet>`
        Represents the crystal as a periodic set, consisting of a finite
        set of points (motif) and lattice (unit cell). Contains other
        useful data, e.g. the crystal's name and information about the
        asymmetric unit for calculation.

    Raises
    ------
    ParseError
        Raised if the structure fails parsing for any of the following:
        1. :code:``disorder == 'skip'`` and disorder is found on any
        atom, 2. crystal.molecule.all_atoms_have_sites is False,
        3. a.fractional_coordinates is None for any a in
        crystal.disordered_molecule, 4. The motif is empty after
        removing H, disordered sites or solvents.
    """

    molecule = crystal.disordered_molecule

    # Disorder
    if disorder == 'skip':
        if crystal.has_disorder or \
         any(_has_disorder(a.label, a.occupancy) for a in molecule.atoms):
            msg = f"{crystal.identifier} has disorder, pass " \
                   "disorder='ordered_sites' or 'all_sites' to remove/ignore" \
                   "disorder"
            raise ParseError(msg)

    elif disorder == 'ordered_sites':
        molecule.remove_atoms(
            a for a in molecule.atoms if _has_disorder(a.label, a.occupancy)
        )

    if remove_hydrogens:
        molecule.remove_atoms(
            a for a in molecule.atoms if a.atomic_symbol in 'HD'
        )

    if heaviest_component and len(molecule.components) > 1:
        molecule = _heaviest_component_ccdc(molecule)

    # Remove atoms with missing coordinates and warn
    if any(a.fractional_coordinates is None for a in molecule.atoms):
        warnings.warn('atoms without sites or missing data will be removed')
        molecule.remove_atoms(
            a for a in molecule.atoms if a.fractional_coordinates is None
        )

    crystal.molecule = molecule
    cell = cellpar_to_cell(*crystal.cell_lengths, *crystal.cell_angles)

    if molecular_centres:
        frac_centres = _frac_molecular_centres_ccdc(crystal)
        mol_centres = frac_centres @ cell
        return PeriodicSet(mol_centres, cell, name=crystal.identifier)

    asym_atoms = crystal.asymmetric_unit_molecule.atoms
    # check for None?
    asym_unit = np.array([tuple(a.fractional_coordinates) for a in asym_atoms])

    if asym_unit.shape[0] == 0:
        raise ParseError(f'{crystal.identifier} has no valid sites')

    asym_unit = np.mod(asym_unit, 1)
    # asym_unit = _snap_small_prec_coords(asym_unit) # recommended by pymatgen
    asym_types = [a.atomic_number for a in asym_atoms]

    # Disorder
    if disorder != 'all_sites':
        keep_sites = _unique_sites(asym_unit, _EQUIV_SITE_TOL)
        if not np.all(keep_sites):
            msg = 'may have overlapping sites; duplicates will be removed'
            warnings.warn(msg)
        asym_unit = asym_unit[keep_sites]
        asym_types = [sym for sym, keep in zip(asym_types, keep_sites) if keep]

    # Symmetry operations
    sitesym = crystal.symmetry_operators
    # try spacegroup numbers?
    if not sitesym:
        sitesym = ['x,y,z']
    rot = np.array([np.array(crystal.symmetry_rotation(op)).reshape((3, 3)) 
                    for op in sitesym])
    trans = np.array([crystal.symmetry_translation(op) for op in sitesym])

    # Apply symmetries to asymmetric unit
    frac_motif, asym_inds, wyc_muls, inverses = _expand_asym_unit(asym_unit, rot, trans)
    motif = frac_motif @ cell
    types = np.array([asym_types[i] for i in inverses])

    return PeriodicSet(
        motif=motif,
        cell=cell,
        name=crystal.identifier,
        asymmetric_unit=asym_inds,
        wyckoff_multiplicities=wyc_muls,
        types=types
    )


# Not quite finished
def periodicset_from_gemmi_block(
        block,
        remove_hydrogens=False,
        disorder='skip'
) -> PeriodicSet:
    """:class:`gemmi.cif.Block` -->
    :class:`amd.PeriodicSet <.periodicset.PeriodicSet>`.
    Block is the type returned by :class:`gemmi.cif.read_file`.

    Parameters
    ----------
    block : :class:`ase.io.cif.CIFBlock`
        An ase CIFBlock object representing a crystal.
    remove_hydrogens : bool, optional
        Remove Hydrogens from the crystal.
    disorder : str, optional
        Controls how disordered structures are handled. Default is
        ``skip`` which skips any crystal with disorder, since disorder
        conflicts with the periodic set model. To read disordered
        structures anyway, choose either :code:`ordered_sites` to remove
        atoms with disorder or :code:`all_sites` include all atoms
        regardless of disorder.

    Returns
    -------
    :class:`amd.PeriodicSet <.periodicset.PeriodicSet>`
        Represents the crystal as a periodic set, consisting of a finite
        set of points (motif) and lattice (unit cell). Contains other
        useful data, e.g. the crystal's name and information about the
        asymmetric unit for calculation.

    Raises
    ------
    ParseError
        Raised if the structure fails to be parsed for any of the
        following: 1. Required data is missing (e.g. cell parameters),
        2. :code:``disorder == 'skip'`` and disorder is found on any
        atom, 3. The motif is empty after removing H or disordered
        sites.
    """

    import gemmi

    # Unit cell
    cellpar = [gemmi.cif.as_number(block.find_value(tag))
               for tag in CIF_TAGS['cellpar']]
    if None in cellpar:
        raise ParseError(f'{block.name} has missing cell data')
    cell = cellpar_to_cell(*cellpar)

    xyz_loop = block.find(CIF_TAGS['atom_site_fract']).loop
    if xyz_loop is None:
        # check for Cartesian coordinates
        raise ParseError(f'{block.name} has missing coordinate data')

    # Asymmetric unit coordinates
    loop_dict = _loop_to_dict_gemmi(xyz_loop)
    xyz_str = [loop_dict[t] for t in CIF_TAGS['atom_site_fract']]
    asym_unit = [[gemmi.cif.as_number(c) for c in coords] for coords in xyz_str]
    asym_unit = np.mod(np.array(asym_unit).T, 1)
    # asym_unit = _snap_small_prec_coords(asym_unit) # recommended by pymatgen

    # Asymmetric unit types
    if '_atom_site_type_symbol' in loop_dict:
        asym_syms = loop_dict['_atom_site_type_symbol']
        asym_types = [gemmi.Element(s).atomic_number for s in asym_syms]
    else:
        warnings.warn('missing atomic types will be labelled 0')
        asym_types = [0 for _ in range(len(asym_unit))]

    remove_sites = []

    # Disorder
    if '_atom_site_label' in loop_dict:
        labels = loop_dict['_atom_site_label']
    else:
        labels = [''] * xyz_loop.length()

    if '_atom_site_occupancy' in loop_dict:
        occupancies = [gemmi.cif.as_number(occ)
                       for occ in loop_dict['_atom_site_occupancy']]
    else:
        occupancies = [None for _ in range(xyz_loop.length())]

    if disorder == 'skip':
        if any(_has_disorder(l, o) for l, o in zip(labels, occupancies)):
            msg = f"{block.name} has disorder, pass " \
                   "disorder='ordered_sites' or 'all_sites' to " \
                   "remove/ignore disorder"
            raise ParseError(msg)
    elif disorder == 'ordered_sites':
        for i, (lab, occ) in enumerate(zip(labels, occupancies)):
            if _has_disorder(lab, occ):
                remove_sites.append(i)

    if remove_hydrogens:
        remove_sites.extend(
            i for i, num in enumerate(asym_types) if num == 1
        )

    # Asymmetric unit
    asym_unit = np.delete(asym_unit, remove_sites, axis=0)
    asym_types = [s for i, s in enumerate(asym_types) if i not in remove_sites]
    if asym_unit.shape[0] == 0:
        raise ParseError(f'{block.name} has no valid sites')

    # Remove overlapping sites unless disorder == 'all_sites'
    if disorder != 'all_sites':
        keep_sites = _unique_sites(asym_unit, _EQUIV_SITE_TOL)
        if not np.all(keep_sites):
            msg = 'may have overlapping sites; duplicates will be removed'
            warnings.warn(msg)
        asym_unit = asym_unit[keep_sites]
        asym_types = [sym for sym, keep in zip(asym_types, keep_sites) if keep]

    # Symmetry operations
    sitesym = []
    for tag in CIF_TAGS['symop']:
        symop_loop = block.find([tag]).loop
        if symop_loop is not None:
            symop_loop_dict = _loop_to_dict_gemmi(symop_loop)
            sitesym = symop_loop_dict[tag]
            break
    # Try spacegroup names/numbers?
    if not sitesym:
        sitesym = ['x,y,z',]

    # Apply symmetries to asymmetric unit
    rot, trans = _parse_sitesym(sitesym)
    frac_motif, asym_inds, wyc_muls, inverses = _expand_asym_unit(asym_unit, rot, trans)
    types = np.array([asym_types[i] for i in inverses])
    motif = frac_motif @ cell

    return PeriodicSet(
        motif=motif,
        cell=cell,
        name=block.name,
        asymmetric_unit=asym_inds,
        wyckoff_multiplicities=wyc_muls,
        types=types
    )


def _parse_sitesym(symmetries):
    """Parses a sequence of symmetries in xyz form and returns rotation
    and translation arrays. Similar to function found in
    ase.spacegroup.spacegroup.
    """

    nsyms = len(symmetries)
    rotations = np.zeros((nsyms, 3, 3))
    translations = np.zeros((nsyms, 3))

    for i, sym in enumerate(symmetries):
        for ind, element in enumerate(sym.split(',')):

            is_positive = True
            is_fraction = False
            sng_trans = None
            fst_trans = []
            snd_trans = []

            for char in element.lower():
                if char == '+':
                    is_positive = True
                elif char == '-':
                    is_positive = False
                elif char == '/':
                    is_fraction = True
                elif char in 'xyz':
                    rot_sgn = 1 if is_positive else -1
                    rotations[i][ind][ord(char) - ord('x')] = rot_sgn
                elif char.isdigit() or char == '.':
                    if sng_trans is None:
                        sng_trans = 1.0 if is_positive else -1.0
                    if is_fraction:
                        snd_trans.append(char)
                    else:
                        fst_trans.append(char)

            if not fst_trans:
                e_trans = 0.0
            else:
                e_trans = sng_trans * float(''.join(fst_trans))

            if is_fraction:
                e_trans /= float(''.join(snd_trans))

            translations[i][ind] = e_trans

    return rotations, translations


def _expand_asym_unit(
        asym_unit: np.ndarray,
        rotations: np.ndarray,
        translations: np.ndarray
) -> Tuple[np.ndarray, ...]:
    """
    Asymmetric unit frac coords, list of rotations & translations -->
    full fractional motif, asymmetric unit indices, multiplicities and
    inverses (which motif points come from where in the asym unit).
    """

    frac_motif = []     # Full motif
    asym_inds = [0]     # Indices of asymmetric unit 
    multiplicities = [] # Wyckoff multiplicities
    inverses = []       # Motif -> Asymmetric unit
    m, dims = asym_unit.shape

    # Apply all symmetries first
    expanded_sites = np.zeros((m, len(rotations), dims))
    for i in range(m):
        expanded_sites[i] = np.dot(rotations, asym_unit[i]) + translations
    expanded_sites = np.mod(expanded_sites, 1)

    for inv, sites in enumerate(expanded_sites):

        multiplicity = 0

        for site_ in sites:

            if not frac_motif:
                frac_motif.append(site_)
                inverses.append(inv)
                multiplicity += 1
                continue

            # check if site_ overlaps with existing sites
            diffs1 = np.abs(site_ - frac_motif)
            diffs2 = np.abs(diffs1 - 1)
            mask = np.all((diffs1 <= _EQUIV_SITE_TOL) | 
                          (diffs2 <= _EQUIV_SITE_TOL), axis=-1)

            if np.any(mask): # site is not new
                where_equal = np.argwhere(mask).flatten()
                for ind in where_equal:
                    if inverses[ind] == inv: # site is invariant
                        pass
                    else: # equivalent to a different site
                        msg = f'has equivalent sites at positions ' \
                              f'{inverses[ind]}, {inv}'
                        warnings.warn(msg)
            else: # new site
                frac_motif.append(site_)
                inverses.append(inv)
                multiplicity += 1

        if multiplicity > 0:
            multiplicities.append(multiplicity)
            asym_inds.append(len(frac_motif))

    frac_motif = np.array(frac_motif)
    asym_inds = np.array(asym_inds[:-1])
    multiplicities = np.array(multiplicities)

    return frac_motif, asym_inds, multiplicities, inverses


@numba.njit()
def _unique_sites(asym_unit, tol):
    """Uniquify (within tol) a list of fractional coordinates,
    considering all points modulo 1. Returns an array of bools such that
    asym_unit[_unique_sites(asym_unit, tol)] is the uniquified list.
    """
    
    site_diffs1 = np.abs(np.expand_dims(asym_unit, 1) - asym_unit)
    site_diffs2 = np.abs(site_diffs1 - 1)
    sites_neq_mask = (site_diffs1 > tol) & (site_diffs2 > tol)
    overlapping = np.triu(sites_neq_mask.sum(axis=-1) == 0, 1)
    return overlapping.sum(axis=0) == 0


def _has_disorder(label, occupancy):
    """Return True if label ends with ? or occupancy is a number < 1.
    """
    return (np.isscalar(occupancy) and occupancy < 1) or label.endswith('?')


def _parse_sitesym_pymatgen(data):
    """Parse symmetry operations given data = block.data where block is
    a pymatgen CifBlock object. If the symops are not present the space
    group symbol is parsed and symops are generated.
    """

    from pymatgen.symmetry.groups import SpaceGroup
    from pymatgen.core.operations import SymmOp
    import pymatgen.io.cif

    symops = []

    # Try to parse xyz symmetry operations
    for symmetry_label in CIF_TAGS['symop']:

        xyz = data.get(symmetry_label)
        if not xyz:
            continue
        if isinstance(xyz, str):
            xyz = [xyz]
        try:
            symops = [SymmOp.from_xyz_string(s) for s in xyz]
            break
        except ValueError:
            continue

    # Spacegroup symbol
    if not symops:

        for symmetry_label in CIF_TAGS['spacegroup_name']:

            sg = data.get(symmetry_label)
            if not sg:
                continue
            sg = re.sub(r'[\s_]', '', sg)

            try:
                spg = pymatgen.io.cif.space_groups.get(sg)
                if not spg:
                    continue
                symops = SpaceGroup(spg).symmetry_ops
                break
            except ValueError:
                pass

            try:
                for d in pymatgen.io.cif._get_cod_data():
                    if sg == re.sub(r'\s+', '', d['hermann_mauguin']):
                        xyz = d['symops']
                        symops = [SymmOp.from_xyz_string(s) for s in xyz]
                        break
            except Exception as e:
                continue

            if symops:
                break

    # International number
    if not symops:
        for symmetry_label in CIF_TAGS['spacegroup_number']:
            num = data.get(symmetry_label)
            if not num:
                continue

            try:
                i = int(pymatgen.io.cif.str2float(num))
                symops = SpaceGroup.from_int_number(i).symmetry_ops
                break
            except ValueError:
                continue

    if not symops:
        symops = [SymmOp.from_xyz_string(s) for s in ['x', 'y', 'z']]

    rotations = [op.rotation_matrix for op in symops]
    translations = [op.translation_vector for op in symops]

    return rotations, translations


def _frac_molecular_centres_ccdc(crystal):
    """Returns the geometric centres of molecules in the unit cell.
    Expects a ccdc Crystal object and returns fractional coordiantes.
    """

    frac_centres = []
    for comp in crystal.packing(inclusion='CentroidIncluded').components:
        coords = [a.fractional_coordinates for a in comp.atoms]
        frac_centres.append((sum(ax) / len(coords) for ax in zip(*coords)))
    frac_centres = np.mod(np.array(frac_centres), 1)
    return frac_centres[_unique_sites(frac_centres, _EQUIV_SITE_TOL)]


def _heaviest_component_ccdc(molecule):
    """Removes all but the heaviest component of the asymmetric unit.
    Intended for removing solvents. Expects and returns a ccdc Molecule
    object.
    """

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


def _loop_to_dict_gemmi(gemmi_loop):
    """gemmi Loop object --> dict, tags: values
    """

    tablified_loop = [[] for _ in range(len(gemmi_loop.tags))]
    n_cols = gemmi_loop.width()
    for i, item in enumerate(gemmi_loop.values):
        tablified_loop[i % n_cols].append(item)
    return {tag: l for tag, l in zip(gemmi_loop.tags, tablified_loop)}


def _snap_small_prec_coords(frac_coords, tol):
    """Find where frac_coords is within 1e-4 of 1/3 or 2/3, change to
    1/3 and 2/3. Recommended by pymatgen's CIF parser.
    """
    frac_coords[np.abs(1 - 3 * frac_coords) < tol] = 1 / 3.
    frac_coords[np.abs(1 - 3 * frac_coords / 2) < tol] = 2 / 3.
    return frac_coords
