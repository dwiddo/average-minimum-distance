"""Tools for reading crystals from files, or from the CSD with 
``csd-python-api``. The readers return :class:`.periodicset.PeriodicSet`
objects representing the crystal which can be passed to 
:func:`.calculate.AMD` and :func:`.calculate.PDD` to get their invariants.
"""

import os
import re
import functools
import warnings
from typing import Callable, Iterable, Tuple

import numpy as np
import numba

import ase.io.cif
import ase.data
import ase.spacegroup
import ase.spacegroup.spacegroup

from .utils import cellpar_to_cell
from .periodicset import PeriodicSet
from .data import CIF_TAGS


def _custom_warning(message, category, filename, lineno, *args, **kwargs):
    return f'{category.__name__}: {message}\n'

warnings.formatwarning = _custom_warning

_EQUIV_SITE_TOL = 1e-3
_DISORDER_OPTIONS = {'skip', 'ordered_sites', 'all_sites'}


class _Reader:

    def __init__(
            self,
            remove_hydrogens=False,
            disorder='skip',
            heaviest_component=False,
            molecular_centres=False,
            show_warnings=True
    ):

        if disorder not in _DISORDER_OPTIONS:
            msg = 'disorder parameter must be one of ' \
                  f'{_DISORDER_OPTIONS} (passed {disorder})'
            raise ValueError(msg)

        self.remove_hydrogens = remove_hydrogens
        self.disorder = disorder
        self.heaviest_component = heaviest_component
        self.molecular_centres = molecular_centres
        self.show_warnings = show_warnings
        self._generator = None

    def __iter__(self):
        yield from self._generator

    def read_one(self):
        """Read the first item."""
        try:
            return next(iter(self._generator))
        except StopIteration:
            return None


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
        Controls how disordered structures are handled. Default is ``skip``
        which skips any crystal with disorder, since disorder conflicts with
        the periodic set model. To read disordered structures anyway, choose
        either :code:`ordered_sites` to remove atoms with disorder or
        :code:`all_sites` include all atoms regardless of disorder.
    heaviest_component : bool, optional
        csd-python-api only. Removes all but the heaviest molecule in the
        asymmeric unit, intended for removing solvents.
    molecular_centres : bool, default False
        csd-python-api only. Extract the centres of molecules in the unit cell
        and store in the attribute molecular_centres.
    show_warnings : bool, optional
        Controls whether warnings that arise during reading are printed.

    Yields
    ------
    :class:`.periodicset.PeriodicSet`
        Represents the crystal as a periodic set, consisting of a finite set
        of points (motif) and lattice (unit cell). Contains other data, e.g.
        the crystal's name and information about the asymmetric unit.

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
            molecular_centres=False,
            show_warnings=True,
    ):

        super().__init__(
            remove_hydrogens=remove_hydrogens,
            disorder=disorder,
            heaviest_component=heaviest_component,
            molecular_centres=molecular_centres,
            show_warnings=show_warnings
        )

        if reader != 'ccdc':

            if heaviest_component:
                msg = 'Parameter heaviest_component ' \
                      'only implemented for reader="ccdc".'
                raise NotImplementedError(msg)

            if molecular_centres:
                msg = 'Parameter molecular_centres ' \
                      'only implemented for reader="ccdc".'
                raise NotImplementedError(msg)

        if reader in ('ase', 'pycodcif'):

            extensions = {'cif'}
            file_parser = functools.partial(ase.io.cif.parse_cif, 
                                            reader=reader)
            converter = functools.partial(
                periodicset_from_ase_cifblock,
                remove_hydrogens=self.remove_hydrogens,
                disorder=self.disorder
            )

        elif reader == 'pymatgen':

            extensions = {'cif'}
            file_parser = self._pymatgen_cifblock_generator
            converter = functools.partial(
                periodicset_from_pymatgen_cifblock,
                remove_hydrogens=self.remove_hydrogens,
                disorder=self.disorder
            )

        elif reader == 'gemmi':

            try:
                import gemmi.cif
            except ImportError as _:
                raise ImportError('Failed to import gemmi.')

            extensions = {'cif'}
            file_parser = gemmi.cif.read_file
            converter = functools.partial(
                periodicset_from_gemmi_block,
                remove_hydrogens=self.remove_hydrogens,
                disorder=self.disorder
            )

        elif reader == 'ccdc':

            try:
                import ccdc.io
            except (ImportError, RuntimeError) as _:
                msg = 'Failed to import csd-python-api, please'\
                      'check it is installed and licensed.'
                raise ImportError(msg)

            extensions = ccdc.io.EntryReader.known_suffixes
            file_parser = ccdc.io.EntryReader
            converter = functools.partial(
                periodicset_from_ccdc_entry,
                remove_hydrogens=self.remove_hydrogens,
                disorder=self.disorder,
                molecular_centres=self.molecular_centres,
                heaviest_component=self.heaviest_component
            )

        else:
            raise ValueError(f'Unknown reader {reader}.')

        if os.path.isfile(path):
            generator = file_parser(path)
        elif os.path.isdir(path):
            generator = self._generate_from_dir(path, file_parser, extensions)
        else:
            raise FileNotFoundError(f'No such file or directory: {path}')

        self._generator = _map(converter, generator, self.show_warnings)

    def _generate_from_dir(self, path, file_parser, extensions):
        for file in os.listdir(path):
            suff = os.path.splitext(file)[1][1:]
            if suff.lower() in extensions:
                yield from file_parser(os.path.join(path, file))

    def _pymatgen_cifblock_generator(path):
        """Path to .cif --> generator of pymatgen CifBlocks."""
        from pymatgen.io.cif import CifFile
        yield from CifFile.from_file(path).data.values()


class CSDReader(_Reader):
    """Read structures from the CSD with csd-python-api, yielding 
    :class:`.periodicset.PeriodicSet` s.

    Parameters
    ----------
    refcodes : str or List[str], optional
        Single or list of CSD refcodes to read. If None or 'CSD', iterates
        over the whole CSD.
    families : bool, optional
        Read all entries whose refcode starts with the given strings, or
        'families' (e.g. giving 'DEBXIT' reads all entries starting with
        DEBXIT).
    remove_hydrogens : bool, optional
        Remove hydrogens from the crystal.
    disorder : str, optional
        Controls how disordered structures are handled. Default is ``skip``
        which skips any crystal with disorder, since disorder conflicts with
        the periodic set model. To read disordered structures anyway, choose
        either :code:`ordered_sites` to remove atoms with disorder or
        :code:`all_sites` include all atoms regardless of disorder.
    heaviest_component : bool, optional
        Removes all but the heaviest molecule in the asymmeric unit, intended
        for removing solvents.
    molecular_centres : bool, default False
        Extract the centres of molecules in the unit cell and store in
        attribute molecular_centres.
    show_warnings : bool, optional
        Controls whether warnings that arise during reading are printed.

    Yields
    ------
    :class:`.periodicset.PeriodicSet`
        Represents the crystal as a periodic set, consisting of a finite set
        of points (motif) and lattice (unit cell). Contains other useful data,
        e.g. the crystal's name and information about the asymmetric unit for
        calculation.

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

        super().__init__(
            remove_hydrogens=remove_hydrogens,
            disorder=disorder,
            heaviest_component=heaviest_component,
            molecular_centres=molecular_centres,
            show_warnings=show_warnings,
        )

        try:
            import ccdc.io
        except (ImportError, RuntimeError) as _:
            msg = 'Failed to import csd-python-api, please'\
                  'check it is installed and licensed.'
            raise ImportError(msg)

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
            refcodes = _refcodes_from_families(refcodes)

        self._entry_reader = ccdc.io.EntryReader('CSD')
        converter = functools.partial(
            periodicset_from_ccdc_entry,
            remove_hydrogens=self.remove_hydrogens,
            disorder=self.disorder,
            molecular_centres=self.molecular_centres,
            heaviest_component=self.heaviest_component
        )
        generator = self._ccdc_generator(refcodes)
        self._generator = _map(converter, generator, self.show_warnings)

    def entry(self, refcode: str, **kwargs) -> PeriodicSet:
        """Read a crystal given a CSD refcode, returning a
        :class:`.periodicset.PeriodicSet`. If given kwargs, overrides the
        kwargs given to the Reader."""

        try:
            entry = self._entry_reader.entry(refcode)
        except RuntimeError:
            warnings.warn(f'{refcode} not found in database')

        kwargs_ = {
            'remove_hydrogens': self.remove_hydrogens,
            'disorder': self.disorder,
            'heaviest_component': self.heaviest_component,
            'molecular_centres': self.molecular_centres,
        }
        kwargs_.update(kwargs)
        converter = functools.partial(periodicset_from_ccdc_entry, **kwargs_)

        if 'show_warnings' in kwargs:
            show_warnings = kwargs['show_warnings']
        else:
            show_warnings = self.show_warnings

        try:
            periodic_set = next(iter(_map(converter, [entry], show_warnings)))
        except StopIteration:
            periodic_set = None

        return periodic_set

    def family(self, refcode_family: str, **kwargs):

        kwargs_ = {
            'remove_hydrogens': self.remove_hydrogens,
            'disorder': self.disorder,
            'heaviest_component': self.heaviest_component,
            'molecular_centres': self.molecular_centres
        }

        kwargs_.update(kwargs)
        converter = functools.partial(periodicset_from_ccdc_entry, **kwargs_)
        refcodes = _refcodes_from_families([refcode_family])
        generator = self._ccdc_generator(refcodes)

        if 'show_warnings' in kwargs:
            show_warnings = kwargs['show_warnings']
        else:
            show_warnings = self.show_warnings

        yield from _map(converter, generator, show_warnings)

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
                    warnings.warn(f'{refcode} not found in database')


class ParseError(ValueError):
    """Raised when an item cannot be parsed into a periodic set."""
    pass


def periodicset_from_ase_cifblock(
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
        Controls how disordered structures are handled. Default is ``skip``
        which raises a ParseError if the input has any disorder, since
        disorder conflicts with the periodic set model. To read disordered
        structures anyway, choose either :code:`ordered_sites` to remove sites
        with disorder or :code:`all_sites` include all sites regardless.

    Returns
    -------
    :class:`.periodicset.PeriodicSet`
        Represents the crystal as a periodic set, consisting of a finite set
        of points (motif) and lattice (unit cell). Contains other useful data,
        e.g. the crystal's name and information about the asymmetric unit for
        calculation.

    Raises
    ------
    ParseError :
        Raised if the structure fails to be parsed for any of the following:
        1. Required data is missing (e.g. cell parameters),
        2. disorder == 'skip' and any of:
            (a) any atom has fractional occupancy,
            (b) any atom's label ends with '?',
        3. The motif is empty after removing H or disordered sites.
    """

    # get cell
    try:
        cellpar = [block[tag] for tag in CIF_TAGS['cellpar']]
    except KeyError:
        raise ParseError(f'{block.name} has missing cell data')
    cell = cellpar_to_cell(*cellpar)

    # asymmetric unit fractional coords
    asym_unit = [block.get(name) for name in CIF_TAGS['atom_site_fract']]
    if None in asym_unit: # missing fractional coords, try cartesian
        asym_motif = [block.get(name) for name in CIF_TAGS['atom_site_cartn']]
        if None in asym_motif:
            raise ParseError(f'{block.name} has missing coordinate data')
        asym_unit = np.array(asym_motif) @ np.linalg.inv(cell)

    elif any(None in coords for coords in asym_unit):
        raise ParseError(f'{block.name} has missing coordinate data')

    # if there's a ?, . or other string, remove site
    if any(isinstance(c, str) for coords in asym_unit for c in coords):
        warnings.warn(f'atoms without sites or unparsable data will be removed')
        asym_unit_ = [[], [], []]
        for xyz in zip(*asym_unit):
            if not any(isinstance(coord, str) for coord in xyz):
                for i in range(3):
                    asym_unit_[i].append(xyz[i])
        asym_unit = asym_unit_

    asym_unit = np.mod(np.array(asym_unit).T, 1)
    # asym_unit = _snap_small_prec_coords(asym_unit)

    # atomic types
    for tag in CIF_TAGS['atom_symbol']:
        if tag in block:
            asym_types = []
            for label in block[tag]:
                if label in ('.', '?'):
                    warnings.warn(f'missing atomic type data')
                    asym_types.append(0)
                # remove additional labels
                sym = re.search(r'([A-Z][a-z]?)', label).group(0)
                if sym == 'D': sym = 'H'
                asym_types.append(ase.data.atomic_numbers[sym])
            break
    else:
        warnings.warn(f'missing atomic types')
        asym_types = [0 for _ in range(len(asym_unit))]

    # symmetry operations
    sitesym = []
    for tag in CIF_TAGS['symop']: # try symop tags first
        if tag in block:
            sitesym = block[tag]
            break
    if isinstance(sitesym, str):
        sitesym = [sitesym]
    if not sitesym: # if no symops, get ops from spacegroup
        try:
            spg = block.get_spacegroup(True)
            rot, trans = spg.rotations, spg.translations
        except:
            rot, trans = ase.spacegroup.spacegroup.parse_sitesym(['x,y,z'])
    else:
        rot, trans = ase.spacegroup.spacegroup.parse_sitesym(sitesym)

    # (if needed) remove Hydrogens, fract occupancy atoms
    remove_sites = [] # indices of asymmetric unit sites to remove
    occupancies = block.get('_atom_site_occupancy')
    labels = block.get('_atom_site_label')
    if occupancies is not None:
        if disorder == 'skip':
            if any(_atom_has_disorder(lab, occ) for lab, occ in zip(labels, occupancies)):
                raise ParseError(f'{block.name} has disorder')
        elif disorder == 'ordered_sites':
            remove_sites.extend(
                (i for i, (lab, occ) in enumerate(zip(labels, occupancies))
                 if _atom_has_disorder(lab, occ)))

    if remove_hydrogens:
        remove_sites.extend((i for i, num in enumerate(asym_types) if num == 1))

    asym_unit = np.delete(asym_unit, remove_sites, axis=0)
    asym_types = [s for i, s in enumerate(asym_types) if i not in remove_sites]

    # if disorder == 'all_sites', don't remove overlapping sites
    if disorder != 'all_sites':
        keep_sites = _unique_sites(asym_unit)
        if not np.all(keep_sites):
            warnings.warn(f'may have overlapping sites; duplicates will be removed')
        asym_unit = asym_unit[keep_sites]
        asym_types = [sym for sym, keep in zip(asym_types, keep_sites) if keep]

    if asym_unit.shape[0] == 0:
        raise ParseError(f'{block.name} has no valid sites')

    frac_motif, asym_inds, wyc_muls, inverses = _expand_asym_unit(asym_unit, rot, trans)
    full_types = np.array([asym_types[i] for i in inverses])
    motif = frac_motif @ cell

    kwargs = {
        'name': block.name,
        'asymmetric_unit': asym_inds,
        'wyckoff_multiplicities': wyc_muls,
        'types': full_types
    }

    return PeriodicSet(motif, cell, **kwargs)


# TODO: entrie function needs to be finished.
def periodicset_from_ase_atoms(
        atoms,
        remove_hydrogens=False,
        disorder='skip'
) -> PeriodicSet:
    """:class:`ase.atoms.Atoms` --> :class:`amd.periodicset.PeriodicSet`.

    Parameters
    ----------
    atoms : :class:`ase.atoms.Atoms`
        An ase Atoms object representing a crystal.
    remove_hydrogens : bool, optional
        Remove Hydrogens from the crystal.
    disorder : str, optional
        Controls how disordered structures are handled. Default is ``skip``
        which raises a ParseError if the input has any disorder, since
        disorder conflicts with the periodic set model. To read disordered
        structures anyway, choose either :code:`ordered_sites` to remove sites
        with disorder or :code:`all_sites` include all sites regardless.

    Returns
    -------
    :class:`.periodicset.PeriodicSet`
        Represents the crystal as a periodic set, consisting of a finite set
        of points (motif) and lattice (unit cell). Contains other useful data,
        e.g. the crystal's name and information about the asymmetric unit for
        calculation.

    Raises
    ------
    ParseError      ## rewrite!!
        Raised if the structure can/should not be parsed for the following reasons:
        1. no sites found or motif is empty after removing H or disordered sites,
        2. a site has missing coordinates,
        3. disorder == 'skip' and any of:
            (a) any atom has fractional occupancy,
            (b) any atom's label ends with '?'.
    """

    if remove_hydrogens:
        for i in np.where(atoms.get_atomic_numbers() == 1)[0][::-1]:
            atoms.pop(i)

    if disorder == 'skip':
        occ_info = atoms.info['occupancy']
        if occ_info is not None:
            for d in occ_info.values():
                for occ in d.values():
                    if float(occ) < 1:
                        raise ParseError(f'Ase atoms object has disorder')

    elif disorder == 'ordered_sites':
        occ_info = atoms.info['occupancy']
        if occ_info is not None:
            remove_inds = []
            for i, d in occ_info.items():
                for occ in d.values():
                    if float(occ) < 1:
                        remove_inds.append(int(i))

    # .pop removes just one atom. 

    # if disorder == 'skip':
    #     if not structure.is_ordered:
    #         raise ParseError(f'{name} has disorder')
    # elif disorder == 'ordered_sites':
    #     remove_inds = []
    #     for i, comp in enumerate(structure.species_and_occu):
    #         if comp.num_atoms < 1:
    #             remove_inds.append(i)
    #     structure.remove_sites(remove_inds)

    cell = atoms.get_cell().array
    sg = atoms.info['spacegroup']
    asym_unit = ase.spacegroup.get_basis(atoms)
    frac_motif, asym_inds, multiplicities, inverses = _expand_asym_unit(asym_unit, sg.rotations, sg.translations)
    motif = frac_motif @ cell
    
    kwargs = {
        'name': None,
        'asymmetric_unit': asym_inds,
        'wyckoff_multiplicities': multiplicities,
        'types': atoms.get_atomic_numbers()
    }

    return PeriodicSet(motif, cell, **kwargs)


def periodicset_from_pymatgen_cifblock(
        block,
        remove_hydrogens=False,
        disorder='skip',
) -> PeriodicSet:
    """:class:`pymatgen.io.cif.CifBlock` --> 
    :class:`amd.periodicset.PeriodicSet`.

    Parameters
    ----------
    block : :class:`pymatgen.core.cif.CifBlock`
        A pymatgen CifBlock object representing a crystal.
    remove_hydrogens : bool, optional
        Remove Hydrogens from the crystal.
    disorder : str, optional
        Controls how disordered structures are handled. Default is ``skip``
        which raises a ParseError if the input has any disorder, since
        disorder conflicts with the periodic set model. To read disordered
        structures anyway, choose either :code:`ordered_sites` to remove sites
        with disorder or :code:`all_sites` include all sites regardless.

    Returns
    -------
    :class:`.periodicset.PeriodicSet`
        Represents the crystal as a periodic set, consisting of a finite set
        of points (motif) and lattice (unit cell). Contains other useful data,
        e.g. the crystal's name and information about the asymmetric unit for
        calculation.

    Raises
    ------
    ParseError
        Raised if the structure can/should not be parsed for the following reasons:
        1. no sites found or motif is empty after removing H or disordered sites,
        2. a site has missing coordinates,
        3. disorder == 'skip' and any of:
            (a) any atom has fractional occupancy,
            (b) any atom's label ends with '?'.
    """

    from pymatgen.io.cif import str2float
    from pymatgen.core.periodic_table import _pt_data

    odict = block.data

    try:
        cellpar = [str2float(odict[tag]) for tag in CIF_TAGS['cellpar']]
        cell = cellpar_to_cell(*cellpar)
    except KeyError:
        raise ParseError(f'{block.header} has missing cell information')

    # check for missing data ?
    asym_unit = [[str2float(s) for s in odict.get(name)] 
                 for name in CIF_TAGS['atom_site_fract']]
    if None in asym_unit:
        asym_motif = [[str2float(s) for s in odict.get(name)] 
                      for name in CIF_TAGS['atom_site_cartn']]
        if None in asym_motif:
            raise ParseError(f'{block.header} has no sites')
        asym_unit = np.array(asym_motif) @ np.linalg.inv(cell)

    asym_unit = np.mod(np.array(asym_unit).T, 1)
    # asym_unit = _snap_small_prec_coords(asym_unit)

    # check _atom_site_label as well?
    symbols = odict.get('_atom_site_type_symbol')
    if symbols is None:
        warnings.warn(f'missing atomic types')
        asym_types = [0 for _ in range(len(asym_unit))]
    else:
        asym_types = [_pt_data[s]['Atomic no'] for s in symbols]

    remove_sites = []

    occupancies = odict.get('_atom_site_occupancy')
    labels = odict.get('_atom_site_label')
    if occupancies is not None:
        if disorder == 'skip':
            if any(_atom_has_disorder(lab, occ) for lab, occ in zip(labels, occupancies)):
                raise ParseError(f'{block.header} has disorder')
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
            warnings.warn(f'may have overlapping sites; duplicates will be removed')
        asym_unit = asym_unit[keep_sites]
        asym_types = [sym for sym, keep in zip(asym_types, keep_sites) if keep]

    if asym_unit.shape[0] == 0:
        raise ParseError(f'{block.header} has no valid sites')

    rot, trans = _parse_sitesym_pymatgen(odict)
    frac_motif, asym_inds, wyc_muls, inverses = _expand_asym_unit(asym_unit, rot, trans)
    full_types = np.array([asym_types[i] for i in inverses])
    motif = frac_motif @ cell

    kwargs = {
        'name': block.header,
        'asymmetric_unit': asym_inds,
        'wyckoff_multiplicities': wyc_muls,
        'types': full_types
    }

    return PeriodicSet(motif, cell, **kwargs)


# TODO: add Raises to docstr
def periodicset_from_pymatgen_structure(
        structure,
        remove_hydrogens=False,
        disorder='skip'
) -> PeriodicSet:
    """:class:`pymatgen.core.structure.Structure` --> :class:`amd.periodicset.PeriodicSet`.
    Does not set the name of the periodic set, as there seems to be no such attribute in 
    the pymatgen Structure object.

    Parameters
    ----------
    structure : :class:`pymatgen.core.structure.Structure`
        A pymatgen Structure object representing a crystal.
    remove_hydrogens : bool, optional
        Remove Hydrogens from the crystal.
    disorder : str, optional
        Controls how disordered structures are handled. Default is ``skip``
        which raises a ParseError if the input has any disorder, since
        disorder conflicts with the periodic set model. To read disordered
        structures anyway, choose either :code:`ordered_sites` to remove sites
        with disorder or :code:`all_sites` include all sites regardless.

    Returns
    -------
    :class:`.periodicset.PeriodicSet`
        Represents the crystal as a periodic set, consisting of a finite set
        of points (motif) and lattice (unit cell). Contains other useful data,
        e.g. the crystal's name and information about the asymmetric unit for
        calculation.
    """

    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

    if remove_hydrogens:
        structure.remove_species(['H', 'D'])

    if disorder == 'skip':
        if not structure.is_ordered:
            raise ParseError(f'Pymatgen structure object has disorder')
    elif disorder == 'ordered_sites':
        remove_inds = []
        for i, comp in enumerate(structure.species_and_occu):
            if comp.num_atoms < 1:
                remove_inds.append(i)
        structure.remove_sites(remove_inds)

    motif = structure.cart_coords
    cell = structure.lattice.matrix
    sym_structure = SpacegroupAnalyzer(structure).get_symmetrized_structure()
    equiv_inds = sym_structure.equivalent_indices

    kwargs = {
        'asymmetric_unit': np.array([l[0] for l in equiv_inds]),
        'wyckoff_multiplicities': np.array([len(l) for l in equiv_inds]),
        'types': np.array(sym_structure.atomic_numbers)
    }

    return PeriodicSet(motif, cell, **kwargs)


def periodicset_from_ccdc_entry(
        entry,
        remove_hydrogens=False,
        disorder='skip',
        heaviest_component=False,
        molecular_centres=False
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
        Controls how disordered structures are handled. Default is ``skip``
        which raises a ParseError if the input has any disorder, since
        disorder conflicts with the periodic set model. To read disordered
        structures anyway, choose either :code:`ordered_sites` to remove sites
        with disorder or :code:`all_sites` include all sites regardless.
    heaviest_component : bool, optional
        Removes all but the heaviest molecule in the asymmeric unit, intended
        for removing solvents.
    molecular_centres : bool, default False
        Extract the centres of molecules in the unit cell and store in 
        the attribute molecular_centres of the returned PeriodicSet.

    Returns
    -------
    :class:`.periodicset.PeriodicSet`
        Represents the crystal as a periodic set, consisting of a finite set
        of points (motif) and lattice (unit cell). Contains other useful data,
        e.g. the crystal's name and information about the asymmetric unit for
        calculation.

    Raises
    ------
    ParseError :
        Raised if the structure fails parsing for any of the following:
        1. entry.has_3d_structure is False,
        2. disorder == 'skip' and any of:
            (a) any disorder flag is True,
            (b) any atom has fractional occupancy,
            (c) any atom's label ends with '?',
        3. entry.crystal.molecule.all_atoms_have_sites is False,
        4. a.fractional_coordinates is None for any a in entry.crystal.disordered_molecule,
        5. The motif is empty after removing H, disordered sites or solvents.
    """

    # Entry specific flags
    if not entry.has_3d_structure:
        raise ParseError(f'{entry.identifier} has no 3D structure')

    if disorder == 'skip' and entry.has_disorder:
        raise ParseError(f'{entry.identifier} has disorder')

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
    """:class:`ccdc.crystal.Crystal` --> :class:`amd.periodicset.PeriodicSet`.
    Crystal is the type returned by :class:`ccdc.io.CrystalReader`.

    Parameters
    ----------
    crystal : :class:`ccdc.crystal.Crystal`
        A ccdc Crystal object representing a crystal structure.
    remove_hydrogens : bool, optional
        Remove Hydrogens from the crystal.
    disorder : str, optional
        Controls how disordered structures are handled. Default is ``skip``
        which raises a ParseError if the input has any disorder, since
        disorder conflicts with the periodic set model. To read disordered
        structures anyway, choose either :code:`ordered_sites` to remove sites
        with disorder or :code:`all_sites` include all sites regardless.
    heaviest_component : bool, optional
        Removes all but the heaviest molecule in the asymmeric unit,
        intended for removing solvents.
    molecular_centres : bool, default False
        Extract the centres of molecules in the unit cell and store in 
        the attribute molecular_centres of the returned PeriodicSet.

    Returns
    -------
    :class:`.periodicset.PeriodicSet`
        Represents the crystal as a periodic set, consisting of a finite set
        of points (motif) and lattice (unit cell). Contains other useful data,
        e.g. the crystal's name and information about the asymmetric unit for
        calculation.

    Raises
    ------
    ParseError :
        Raised if the structure fails parsing for any of the following:
        1. disorder == 'skip' and any of:
            (a) any disorder flag is True,
            (b) any atom has fractional occupancy,
            (c) any atom's label ends with '?',
        2. crystal.molecule.all_atoms_have_sites is False,
        3. a.fractional_coordinates is None for any a in crystal.disordered_molecule,
        4. The motif is empty after removing H, disordered sites or solvents.
    """

    molecule = crystal.disordered_molecule

    # if skipping disorder, check disorder flags and then all atoms
    if disorder == 'skip':
        if crystal.has_disorder or \
         any(_atom_has_disorder(a.label, a.occupancy) for a in molecule.atoms):
            raise ParseError(f'{crystal.identifier} has disorder')

    elif disorder == 'ordered_sites':
        molecule.remove_atoms(a for a in molecule.atoms
                              if _atom_has_disorder(a.label, a.occupancy))

    if remove_hydrogens:
        molecule.remove_atoms(a for a in molecule.atoms if a.atomic_symbol in 'HD')

    if heaviest_component and len(molecule.components) > 1:
        molecule = _heaviest_component(molecule)

    # remove atoms with missing coord data and warn
    if any(a.fractional_coordinates is None for a in molecule.atoms):
        warnings.warn(f'trying to remove atoms without sites')
        molecule.remove_atoms(a for a in molecule.atoms 
                              if a.fractional_coordinates is None)

    if not molecule.all_atoms_have_sites:
        raise ParseError(f'{crystal.identifier} has atoms without sites')

    crystal.molecule = molecule
    cell = cellpar_to_cell(*crystal.cell_lengths, *crystal.cell_angles)

    if molecular_centres:
        frac_centres = _frac_molecular_centres_ccdc(crystal)
        mol_centres = frac_centres @ cell
        return PeriodicSet(mol_centres, cell, name=crystal.identifier)
    
    asym_atoms = crystal.asymmetric_unit_molecule.atoms
    # check for None?
    asym_unit = np.array([tuple(a.fractional_coordinates) for a in asym_atoms])
    asym_unit = np.mod(asym_unit, 1)
    # asym_unit = _snap_small_prec_coords(asym_unit)
    asym_types = [a.atomic_number for a in asym_atoms]

    if disorder != 'all_sites':
        keep_sites = _unique_sites(asym_unit)
        if not np.all(keep_sites):
            warnings.warn(f'may have overlapping sites; duplicates will be removed')
        asym_unit = asym_unit[keep_sites]
        asym_types = [sym for sym, keep in zip(asym_types, keep_sites) if keep]

    if asym_unit.shape[0] == 0:
        raise ParseError(f'{crystal.identifier} has no valid sites')

    sitesym = crystal.symmetry_operators
    # try spacegroup numbers?
    if not sitesym:
        sitesym = ['x,y,z']

    rot = np.array([np.array(crystal.symmetry_rotation(op)).reshape((3,3)) 
                    for op in sitesym])
    trans = np.array([crystal.symmetry_translation(op) for op in sitesym])
    frac_motif, asym_inds, wyc_muls, inverses = _expand_asym_unit(asym_unit, rot, trans)
    motif = frac_motif @ cell

    kwargs = {
        'name': crystal.identifier,
        'asymmetric_unit': asym_inds,
        'wyckoff_multiplicities': wyc_muls,
        'types': np.array([asym_types[i] for i in inverses])
    }

    periodic_set = PeriodicSet(motif, cell, **kwargs)

    # if molecular_centres:
    #     frac_centres = _frac_molecular_centres_ccdc(crystal)
    #     periodic_set.molecular_centres = frac_centres @ cell

    return periodic_set


def periodicset_from_gemmi_block(
        block,
        remove_hydrogens=False,
        disorder='skip'
) -> PeriodicSet:
    """:class:`gemmi.cif.Block` --> :class:`amd.periodicset.PeriodicSet`. 
    Block is the type returned by :class:`gemmi.cif.read_file`.

    Parameters
    ----------
    block : :class:`ase.io.cif.CIFBlock`
        An ase CIFBlock object representing a crystal.
    remove_hydrogens : bool, optional
        Remove Hydrogens from the crystal.
    disorder : str, optional
        Controls how disordered structures are handled. Default is ``skip``
        which raises a ParseError if the input has any disorder, since
        disorder conflicts with the periodic set model. To read disordered
        structures anyway, choose either :code:`ordered_sites` to remove sites
        with disorder or :code:`all_sites` include all sites regardless.

    Returns
    -------
    :class:`.periodicset.PeriodicSet`
        Represents the crystal as a periodic set, consisting of a finite set
        of points (motif) and lattice (unit cell). Contains other useful data,
        e.g. the crystal's name and information about the asymmetric unit for
        calculation.

    Raises
    ------
    ParseError :
        Raised if the structure fails to be parsed for any of the following:
        1. Required data is missing (e.g. cell parameters),
        2. disorder == 'skip' and any of:
            (a) any atom has fractional occupancy,
            (b) any atom's label ends with '?',
        3. The motif is empty after removing H or disordered sites.
    """

    cellpar = [ase.io.cif.convert_value(block.find_value(tag)) 
               for tag in CIF_TAGS['cellpar']]
    if None in cellpar:
        raise ParseError(f'{block.name} has missing cell information')

    cell = cellpar_to_cell(*cellpar)

    xyz_loop = block.find(CIF_TAGS['atom_site_fract']).loop
    if xyz_loop is None:
        # check for crtsn!
        raise ParseError(f'{block.name} has missing coordinate data')

    # loop will contain all xyz cols, maybe labels, maybe types, maybe occs

    loop_dict = _gemmi_loop_to_dict(xyz_loop)
    xyz_str = [loop_dict[t] for t in CIF_TAGS['atom_site_fract']]
    asym_unit = [[ase.io.cif.convert_value(c) for c in coords] for coords in xyz_str]

    asym_unit = np.mod(np.array(asym_unit).T, 1)
    # asym_unit = _snap_small_prec_coords(asym_unit)

    if '_atom_site_type_symbol' in loop_dict:
        asym_syms = loop_dict['_atom_site_type_symbol']
        asym_types = [ase.data.atomic_numbers[s] for s in asym_syms]
    else:
        warnings.warn(f'missing atomic types')
        asym_types = [0 for _ in range(len(asym_unit))]

    remove_sites = []
    
    # if labels exist, check them for disorder
    if '_atom_site_label' in loop_dict:
        labels = loop_dict['_atom_site_label']
    else:
        labels = ['' for _ in range(xyz_loop.length())]

    # if occupancies exist, check them for disorder
    if '_atom_site_occupancy' in loop_dict:
        occupancies = [ase.io.cif.convert_value(occ) 
                  for occ in loop_dict['_atom_site_occupancy']]
    else:
        occupancies = [None for _ in range(xyz_loop.length())]

    if disorder == 'skip':
        if any(_atom_has_disorder(lab, occ) for lab, occ in zip(labels, occupancies)):
            raise ParseError(f'{block.name} has disorder')
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
            warnings.warn(f'may have overlapping sites; duplicates will be removed')
        asym_unit = asym_unit[keep_sites]
        asym_types = [sym for sym, keep in zip(asym_types, keep_sites) if keep]

    if asym_unit.shape[0] == 0:
        raise ParseError(f'{block.name} has no valid sites')

    # get symops
    sitesym = []
    for tag in CIF_TAGS['symop']:
        symop_loop = block.find([tag]).loop
        if symop_loop is not None:
            symop_loop_dict = _gemmi_loop_to_dict(symop_loop)
            sitesym = symop_loop_dict[tag]
            break
    # try spacegroup names/nums?
    if not sitesym:
        sitesym = ['x,y,z',]

    rot, trans = ase.spacegroup.spacegroup.parse_sitesym(sitesym)
    frac_motif, asym_inds, wyc_muls, inverses = _expand_asym_unit(asym_unit, rot, trans)
    full_types = np.array([asym_types[i] for i in inverses])
    motif = frac_motif @ cell

    kwargs = {
        'name': block.name,
        'asymmetric_unit': asym_inds,
        'wyckoff_multiplicities': wyc_muls,
        'types': full_types
    }

    return PeriodicSet(motif, cell, **kwargs)


def _frac_molecular_centres_ccdc(crystal):
    """Returns any geometric centres of molecules in the unit cell.
    Expects a ccdc Crystal object and returns fractional coordiantes."""

    frac_centres = []
    for comp in crystal.packing(inclusion='CentroidIncluded').components:
        coords = [a.fractional_coordinates for a in comp.atoms]
        x, y, z = zip(*coords)
        m = len(coords)
        frac_centres.append((sum(x) / m, sum(y) / m, sum(z) / m))
    frac_centres = np.mod(np.array(frac_centres), 1)
    return frac_centres[_unique_sites(frac_centres)]


def _expand_asym_unit(
        asym_unit: np.ndarray,
        rotations: np.ndarray,
        translations: np.ndarray
) -> Tuple[np.ndarray, ...]:
    """
    Asymmetric unit frac coords, list of rotations, list of translations
    -->
    full fractional motif, asymmetric unit indices, multiplicities, inverses.
    """

    all_sites = []
    asym_inds = [0]
    multiplicities = []
    inverses = []
    m, dims = asym_unit.shape

    expanded_sites = np.zeros((m, len(rotations), dims))
    for i in range(m):
        expanded_sites[i] = np.dot(rotations, asym_unit[i]) + translations
    expanded_sites = np.mod(expanded_sites, 1)

    for inv, sites in enumerate(expanded_sites):

        multiplicity = 0

        for site_ in sites:

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
                    if inverses[ind] == inv: # invarnant 
                        pass
                    else: # equivalent to a different site
                        warnings.warn(f'has equivalent sites at positions {inverses[ind]}, {inv}')
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


@numba.njit()
def _unique_sites(asym_unit):
    """Uniquify (within io._EQUIV_SITE_TOL) a list of fractional coordinates,
    considering all points modulo 1. Returns an array of bools such that
    asym_unit[_unique_sites(asym_unit)] is a uniquified list."""
    site_diffs1 = np.abs(np.expand_dims(asym_unit, 1) - asym_unit)
    site_diffs2 = np.abs(site_diffs1 - 1)
    sites_neq_mask = np.logical_and((site_diffs1 > _EQUIV_SITE_TOL), 
                                    (site_diffs2 > _EQUIV_SITE_TOL))
    overlapping = np.triu(sites_neq_mask.sum(axis=-1) == 0, 1)
    return overlapping.sum(axis=0) == 0


def _parse_sitesym_pymatgen(data):
    """In order to generate symmetry equivalent positions, the symmetry
    operations are parsed. If the symops are not present, the space
    group symbol is parsed, and symops are generated."""

    from pymatgen.symmetry.groups import SpaceGroup
    from pymatgen.core.operations import SymmOp
    import pymatgen.io.cif

    symops = []

    # try to parse xyz symops
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

    # try to parse symbol
    if not symops:
        for symmetry_label in CIF_TAGS['spacegroup_name']:

            sg = data.get(symmetry_label)
            if not sg:
                continue

            sg = pymatgen.io.cif.sub_spgrp(sg)
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
            except Exception:
                continue

            if symops:
                break

    # try to parse international number
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


def _atom_has_disorder(label, occupancy):
    """Return True if label ends with ? or occupancy is a number < 1."""
    return label.endswith('?') or (np.isscalar(occupancy) and occupancy < 1)


def _snap_small_prec_coords(frac_coords):
    """Find where frac_coords is within 1e-4 of 1/3 or 2/3, change to 1/3 and 2/3.
    Recommended by pymatgen's CIF parser."""
    frac_coords[np.abs(1 - 3 * frac_coords) < 1e-4] = 1 / 3.
    frac_coords[np.abs(1 - 3 * frac_coords / 2) < 1e-4] = 2 / 3.
    return frac_coords


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


def _refcodes_from_families(refcode_families):
    """List of strings --> all CSD refcodes starting with one of the strings.
    Intended to be passed a list of families and return all refcodes in them."""

    try:
        import ccdc.search
    except (ImportError, RuntimeError) as _:
        msg = 'Failed to import csd-python-api, please'\
              'check it is installed and licensed.'
        raise ImportError(msg)

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


def _gemmi_loop_to_dict(loop):
    """gemmi Loop object --> dict, tags: values"""
    tablified_loop = [[] for _ in range(len(loop.tags))]
    n_cols = loop.width()
    for i, item in enumerate(loop.values):
        tablified_loop[i % n_cols].append(item)
    return {tag: l for tag, l in zip(loop.tags, tablified_loop)}


def _map(func: Callable, iterable: Iterable, show_warnings: bool):
    """Iterates over iterable, passing items through func and yielding the result. 
    Catches ParseError and warnings, optionally printing them.
    """

    if not show_warnings:
        warnings.simplefilter('ignore')

    for item in iterable:

        with warnings.catch_warnings(record=True) as warning_msgs:

            parse_failed = False
            try:
                periodic_set = func(item)
            except ParseError as err:
                parse_failed = str(err)

        if parse_failed:
            warnings.warn(parse_failed)
            continue

        for warning in warning_msgs:
            msg = f'{periodic_set.name} {warning.message}'
            warnings.warn(msg, category=warning.category)

        yield periodic_set
