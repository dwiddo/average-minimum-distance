"""Tools for reading crystals from files, or from the CSD with
``csd-python-api``. The readers return
:class:`amd.PeriodicSet <.periodicset.PeriodicSet>` objects representing
the crystal which can be passed to :func:`amd.AMD() <.calculate.AMD>`
and :func:`amd.PDD() <.calculate.PDD>`.
"""

import warnings
import collections
import os
import re
import functools
import errno
import math
import itertools
from pathlib import Path
from typing import Iterable, Iterator, Optional, Union, Callable, Tuple, List

import numpy as np
import numpy.typing as npt
import numba
import tqdm
from scipy.spatial.distance import pdist

from .utils import cellpar_to_cell
from .periodicset import DisorderGroup, DisorderAssembly, PeriodicSet
from .calculate import _collapse_into_groups
from .globals_ import SUBS_DISORDER_TOL, ATOMIC_NUMBERS
from ._types import FloatArray, UIntArray


def _custom_warning(message, category, filename, lineno, *args, **kwargs):
    return f"{category.__name__}: {message}\n"


warnings.formatwarning = _custom_warning


_CIF_TAGS: dict = {
    "cellpar": [
        "_cell_length_a",
        "_cell_length_b",
        "_cell_length_c",
        "_cell_angle_alpha",
        "_cell_angle_beta",
        "_cell_angle_gamma",
    ],
    "atom_site_fract": [
        "_atom_site_fract_x",
        "_atom_site_fract_y",
        "_atom_site_fract_z",
    ],
    "atom_site_cartn": [
        "_atom_site_Cartn_x",
        "_atom_site_Cartn_y",
        "_atom_site_Cartn_z",
    ],
    "symop": [
        "_space_group_symop_operation_xyz",
        "_space_group_symop.operation_xyz",
        "_symmetry_equiv_pos_as_xyz",
    ],
    "spacegroup_name": ["_space_group_name_H-M_alt", "_symmetry_space_group_name_H-M"],
    "spacegroup_number": ["_space_group_IT_number", "_symmetry_Int_Tables_number"],
}

__all__ = [
    "CifReader",
    "CSDReader",
    "ParseError",
    "periodicset_from_gemmi_block",
    "periodicset_from_ccdc_entry",
    "periodicset_from_ccdc_crystal",
    # "periodicset_from_ase_cifblock",
    # "periodicset_from_pymatgen_cifblock",
    # "periodicset_from_ase_atoms",
    "periodicset_from_pymatgen_structure",
]


class ParseError(ValueError):
    """Raised when an item cannot be parsed into a periodic set."""
    pass


class _Reader(collections.abc.Iterator):
    """Base reader class."""

    def __init__(
        self,
        iterable: Iterable,
        converter: Callable[..., PeriodicSet],
        show_warnings: bool,
        verbose: bool,
    ):
        self._iterator = iter(iterable)
        self._converter = converter
        self.show_warnings = show_warnings
        if verbose:
            self._progress_bar = tqdm.tqdm(desc="Reading", delay=1)
        else:
            self._progress_bar = None

    def __next__(self):
        """Iterate over self._iterator, passing items through
        self._converter and yielding. If
        :class:`ParseError <.io.ParseError>` is raised in a call to
        self._converter, the item is skipped. Warnings raised in
        self._converter are printed if self.show_warnings is True.
        """

        if not self.show_warnings:
            warnings.simplefilter("ignore")

        while True:
            try:
                item = next(self._iterator)
            except StopIteration:
                if self._progress_bar is not None:
                    self._progress_bar.close()
                raise StopIteration

            with warnings.catch_warnings(record=True) as warning_msgs:
                try:
                    periodic_set = self._converter(item)
                except ParseError as err:
                    warnings.warn(str(err))
                    continue
                finally:
                    if self._progress_bar is not None:
                        self._progress_bar.update(1)

            for warning in warning_msgs:
                msg = f"(name={periodic_set.name}) {warning.message}"
                warnings.warn(msg, category=warning.category)

            return periodic_set

    def read(self) -> Union[PeriodicSet, List[PeriodicSet]]:
        """Read the crystal(s), return one
        :class:`amd.PeriodicSet <.periodicset.PeriodicSet>` if there is
        only one, otherwise return a list.
        """
        items = list(self)
        if len(items) == 1:
            return items[0]
        return items


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
        :code:`gemmi`, :code:`ccdc` is accepted if csd-python-api is
        installed. The ccdc backend should be able to read any format
        accepted by :class:`ccdc.io.EntryReader`.
    remove_hydrogens : bool
        Remove Hydrogens from the crystals.
    skip_disorder : bool
        Do not read disordered structures.
    missing_coords : str
        Action to take upon finding missing coordinates. "warn" (default) will
        emit a warning when a structure has missing coordinates, and the sites
        will be removed. "skip" will not read structures with missing coords.
    eq_site_tol : float, default 0.001
        Tolerance below which two sites are considered to be at the same
        position. 
    reduce_cell : bool, default False
        Reduce the unit cell (Lenstra-Lenstra-Lovasz lattice basis reduction).
    show_warnings : bool
        Print warnings that arise during reading.
    verbose : bool, default False
        If True, prints a progress bar showing the number of items
        processed.
    heaviest_component : bool
        csd-python-api only. Removes all but the heaviest molecule in
        the asymmeric unit, intended for removing solvents.
    molecular_centres : bool, default False
        csd-python-api only. Extract the centres of molecules in the
        unit cell and store in the attribute molecular_centres.

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
            amds = [amd.AMD(item, 100) for item in amd.CifReader('mycif.cif')]
    """

    def __init__(
        self,
        path: Union[str, os.PathLike],
        reader: str = "gemmi",
        remove_hydrogens: bool = False,
        skip_disorder: bool = False,
        missing_coords: str = "warn",
        eq_site_tol: float = 1e-3,
        reduce_cell: bool = False,
        show_warnings: bool = True,
        verbose: bool = False,
        heaviest_component: bool = False,
        molecular_centres: bool = False,
    ):

        if reader != "ccdc":
            if heaviest_component:
                raise NotImplementedError(
                    "'heaviest_component' parameter of "
                    f"{self.__class__.__name__} only implemented with "
                    "csd-python-api, if installed pass reader='ccdc'"
                )
            if molecular_centres:
                raise NotImplementedError(
                    "'molecular_centres' parameter of "
                    f"{self.__class__.__name__} only implemented with "
                    "csd-python-api, if installed pass reader='ccdc'"
                )

        # cannot handle some characters (�) in cifs
        if reader == "gemmi":
            
            import gemmi

            extensions = {"cif"}
            file_parser = gemmi.cif.read_file
            converter = functools.partial(
                periodicset_from_gemmi_block,
                remove_hydrogens=remove_hydrogens,
                skip_disorder=skip_disorder,
                missing_coords=missing_coords,
                eq_site_tol=eq_site_tol,
                reduce_cell=reduce_cell
            )

        # elif reader in ("ase", "pycodcif"):
        #     from ase.io.cif import parse_cif

        #     extensions = {"cif"}
        #     file_parser = functools.partial(parse_cif, reader=reader)
        #     converter = functools.partial(
        #         periodicset_from_ase_cifblock,
        #         remove_hydrogens=remove_hydrogens,
        #         skip_disorder=skip_disorder,
        #     )

        # elif reader == "pymatgen":

        #     def _pymatgen_cif_parser(path):
        #         from pymatgen.io.cif import CifFile
        #         return CifFile.from_file(path).data.values()

        #     extensions = {"cif"}
        #     file_parser = _pymatgen_cif_parser
        #     converter = functools.partial(
        #         periodicset_from_pymatgen_cifblock,
        #         remove_hydrogens=remove_hydrogens,
        #         skip_disorder=skip_disorder,
        #     )

        elif reader == "ccdc":
            try:
                import ccdc.io
            except (ImportError, RuntimeError) as e:
                raise ImportError("Failed to import csd-python-api") from e

            extensions = set(ccdc.io.EntryReader.known_suffixes.keys())
            file_parser = ccdc.io.EntryReader
            converter = functools.partial(
                periodicset_from_ccdc_entry,
                remove_hydrogens=remove_hydrogens,
                skip_disorder=skip_disorder,
                missing_coords=missing_coords,
                eq_site_tol=eq_site_tol,
                heaviest_component=heaviest_component,
                molecular_centres=molecular_centres,
                reduce_cell=reduce_cell
            )

        else:
            raise ValueError(
                f"'reader' parameter of {self.__class__.__name__} must be one "
                f"of 'gemmi', 'ccdc' (passed '{reader}')"
            )

        path = Path(path)
        if path.is_file():
            iterable = file_parser(str(path))
        elif path.is_dir():
            iterable = CifReader._dir_generator(path, file_parser, extensions)
        else:
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), str(path))

        super().__init__(iterable, converter, show_warnings, verbose)

    @staticmethod
    def _dir_generator(
        directory: Path, file_parser: Callable, extensions: set[str]
    ) -> Iterator:
        """Generate items from all files with extensions in
        ``extensions`` from a directory ``path``."""

        for path in directory.rglob('*'):
            if path.is_file() and path.suffix[1:].lower() in extensions:
                try:
                    yield from file_parser(str(path))
                except Exception as e:
                    warnings.warn(
                        f'Error parsing "{str(path)}", skipping file. '
                        f"Exception: {repr(e)}"
                    )


class CSDReader(_Reader):
    """Read structures from the CSD with csd-python-api, yielding
    :class:`amd.PeriodicSet <.periodicset.PeriodicSet>` s.

    Parameters
    ----------
    refcodes : str or List[str], optional
        Single or list of CSD refcodes to read. If None or 'CSD',
        iterates over the whole CSD.
    refcode_families : bool, optional
        Interpret ``refcodes`` as one or more refcode families, reading
        all entries whose refcode starts with the given strings.
    remove_hydrogens : bool, optional
        Remove hydrogens from the crystals.
    skip_disorder : bool
        Do not read disordered structures.
    missing_coords : str
        Action to take upon finding missing coordinates. "warn" (default) will
        emit a warning when a structure has missing coordinates, and the sites
        will be removed. "skip" will not read structures with missing coords.
    eq_site_tol : float, default 0.001
        Tolerance below which two sites are considered to be at the same
        position. 
    reduce_cell : bool, default False
        Reduce the unit cell (Lenstra-Lenstra-Lovasz lattice basis reduction).
    show_warnings : bool
        Print warnings that arise during reading.
    verbose : bool, default False
        If True, prints a progress bar showing the number of items
        processed.
    heaviest_component : bool, optional
        Removes all but the heaviest molecule in the asymmeric unit,
        intended for removing solvents.
    molecular_centres : bool, default False
        Extract the centres of molecules in the unit cell and store in
        attribute molecular_centres.

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
            families = ['ACSALA', 'HXACAN']
            structures = list(amd.CSDReader(families, refcode_families=True))

            # Get AMDs (k=100) for crystals in these families
            refcodes = ['ACSALA', 'HXACAN']
            amds = []
            for periodic_set in amd.CSDReader(refcodes, refcode_families=True):
                amds.append(amd.AMD(periodic_set, 100))

            # Giving the reader nothing reads from the whole CSD.
            for periodic_set in amd.CSDReader():
                ...
    """

    def __init__(
        self,
        refcodes: Optional[Union[str, List[str]]] = None,
        refcode_families: bool = False,
        remove_hydrogens: bool = False,
        skip_disorder: bool = False,
        missing_coords: str = "warn",
        eq_site_tol: float = 1e-3,
        reduce_cell: bool = False,
        show_warnings: bool = True,
        verbose: bool = False,
        heaviest_component: bool = False,
        molecular_centres: bool = False,
    ):

        try:
            import ccdc.search
            import ccdc.io
        except (ImportError, RuntimeError) as e:
            raise ImportError("Failed to import csd-python-api") from e

        if isinstance(refcodes, str) and refcodes.lower() == "csd":
            refcodes = None
        if refcodes is None:
            refcode_families = False
        elif isinstance(refcodes, str):
            refcodes = [refcodes]
        elif isinstance(refcodes, list):
            if not all(isinstance(refcode, str) for refcode in refcodes):
                raise ValueError(
                    f"List passed to {self.__class__.__name__} contains " "non-strings."
                )
        else:
            raise ValueError(
                f"{self.__class__.__name__} expects None, a string or list of "
                f"strings, got {refcodes.__class__.__name__}"
            )

        if refcode_families:
            all_refcodes = []
            for refcode in refcodes:
                query = ccdc.search.TextNumericSearch()
                query.add_identifier(refcode)
                hits = [hit.identifier for hit in query.search()]
                all_refcodes.extend(hits)
            # filter to unique refcodes while keeping order
            refcodes = []
            seen = set()
            for refcode in all_refcodes:
                if refcode not in seen:
                    refcodes.append(refcode)
                    seen.add(refcode)

        converter = functools.partial(
            periodicset_from_ccdc_entry,
            remove_hydrogens=remove_hydrogens,
            skip_disorder=skip_disorder,
            missing_coords=missing_coords,
            eq_site_tol=eq_site_tol,
            reduce_cell=reduce_cell,
            heaviest_component=heaviest_component,
            molecular_centres=molecular_centres,
        )

        entry_reader = ccdc.io.EntryReader("CSD")
        if refcodes is None:
            iterable = entry_reader
        else:
            iterable = map(entry_reader.entry, refcodes)

        super().__init__(iterable, converter, show_warnings, verbose)


# class MPReader(_Reader):
#     """Read structures from the Materials Project API, yielding
#     :class:`amd.PeriodicSet <.periodicset.PeriodicSet>` s.

#     Parameters
#     ----------
#     mp_api_key : str
#         ...
#     ids : str or list of str, optional
#         ...
#     remove_hydrogens : bool, optional
#         Remove hydrogens from the crystals.
#     show_warnings : bool, optional
#         Controls whether warnings that arise during reading are printed.
#     verbose : bool, default False
#         If True, prints a progress bar showing the number of items
#         processed.

#     Yields
#     ------
#     :class:`amd.PeriodicSet <.periodicset.PeriodicSet>`
#         Represents the crystal as a periodic set, consisting of a finite
#         set of points (motif) and lattice (unit cell). Contains other
#         useful data, e.g. the crystal's name and information about the
#         asymmetric unit for calculation.
#     """

#     def __init__(
#         self,
#         mp_api_key: str,
#         ids: Optional[Union[str, List[str]]] = None,
#         remove_hydrogens: bool = False,
#         show_warnings: bool = True,
#         verbose: bool = False,
#     ):
#         from mp_api.client import MPRester

#         if isinstance(ids, str):
#             ids = [ids]
#         elif isinstance(ids, list):
#             if not all(isinstance(i, str) for i in ids):
#                 raise ValueError(
#                     f"{self.__class__.__name__} expects None, a string or "
#                     "list of strings."
#                 )
#         else:
#             raise ValueError(
#                 f"{self.__class__.__name__} expects None, a string or list of "
#                 f"strings, got {ids.__class__.__name__}"
#             )

#         converter = functools.partial(
#             self._periodicset_from_mp_api_doc, remove_hydrogens=remove_hydrogens
#         )

#         with MPRester(mp_api_key) as mpr:
#             docs = mpr.materials.summary.search(
#                 fields=["material_id", "structure"], material_ids=ids
#             )

#         super().__init__(docs, converter, show_warnings, verbose)

#     @staticmethod
#     def _periodicset_from_mp_api_doc(doc, remove_hydrogens: bool = False):
#         periodic_set = periodicset_from_pymatgen_structure(
#             doc.structure, remove_hydrogens=remove_hydrogens
#         )
#         periodic_set.name = doc.material_id.title().lower()
#         return periodic_set


def periodicset_from_gemmi_block(
    block,
    remove_hydrogens: bool = False,
    skip_disorder: bool = False,
    missing_coords: str = "warn",
    eq_site_tol: float = 1e-3,
    reduce_cell: bool = False
) -> PeriodicSet:
    """Convert a :class:`gemmi.cif.Block` object to a
    :class:`amd.PeriodicSet <.periodicset.PeriodicSet>`.
    :class:`gemmi.cif.Block` is the type returned by
    :func:`gemmi.cif.read_file`.

    Parameters
    ----------
    block : :class:`gemmi.cif.Block`
        An ase CIFBlock object representing a crystal.
    remove_hydrogens : bool, optional
        Remove Hydrogens from the crystal.
    skip_disorder : bool
        Do not read disordered structures.
    missing_coords : str
        Action to take upon finding missing coordinates. "warn" (default) will
        emit a warning when a structure has missing coordinates, and the sites
        will be removed. "skip" will not read structures with missing coords.
    eq_site_tol : float, default 0.001
        Tolerance below which two sites are considered to be at the same
        position. 
    reduce_cell : bool, default False
        Reduce the unit cell (Lenstra-Lenstra-Lovasz lattice basis reduction).

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
        2. :code:``skip_disorder is True`` and disorder is found, 3. The
        motif is empty after removing H or disordered sites.
    """

    import gemmi

    # Unit cell
    cellpar = [block.find_value(t) for t in _CIF_TAGS["cellpar"]]
    if not all(isinstance(par, str) for par in cellpar):
        raise ParseError(f"{block.name} has no unit cell")
    cellpar = np.array([str2float(par) for par in cellpar])
    if np.isnan(np.sum(cellpar)):
        raise ParseError(f"{block.name} has no unit cell")
    cell = cellpar_to_cell(cellpar)

    # Asymmetric unit coordinates
    xyz_loop = block.find(_CIF_TAGS["atom_site_fract"]).loop
    if xyz_loop is None:
        raise ParseError(f"{block.name} has no coordinates")

    tablified_loop = [[] for _ in range(len(xyz_loop.tags))]
    for i, item in enumerate(xyz_loop.values):
        tablified_loop[i % xyz_loop.width()].append(item)
    loop_dict = {tag: l for tag, l in zip(xyz_loop.tags, tablified_loop)}
    xyz_str = [loop_dict[t] for t in _CIF_TAGS["atom_site_fract"]]
    asym_unit = np.transpose(np.array([[str2float(c) for c in xyz] for xyz in xyz_str]))
    asym_unit = np.mod(asym_unit, 1)

    # recommended by pymatgen
    # asym_unit = _snap_small_prec_coords(asym_unit, 1e-4)

    # Atom labels
    if "_atom_site_label" in loop_dict:
        asym_labels = [
            gemmi.cif.as_string(lab) for lab in loop_dict["_atom_site_label"]
        ]
    else:
        asym_labels = [""] * xyz_loop.length()

    # Atomic types
    if "_atom_site_type_symbol" in loop_dict:
        symbols = []
        for s in loop_dict["_atom_site_type_symbol"]:
            sym = gemmi.cif.as_string(s)
            match = re.search(r"([A-Za-z][A-Za-z]?)", sym)
            if match is not None:
                sym = match.group()
            else:
                sym = ""
            sym = list(sym)
            if len(sym) > 0:
                sym[0] = sym[0].upper()
            if len(sym) > 1:
                sym[1] = sym[1].lower()
            symbols.append("".join(sym))
    else:  # Get atomic types from label
        symbols = _atomic_symbols_from_labels(asym_labels)

    asym_types = []
    for s in symbols:
        if s in ATOMIC_NUMBERS:
            asym_types.append(ATOMIC_NUMBERS[s])
        else:
            asym_types.append(0)

    # Fractional occupancies
    if "_atom_site_occupancy" in loop_dict:
        asym_occs = []
        for occ in loop_dict["_atom_site_occupancy"]:
            try:
                occ = str2float(occ)
            except ValueError:
                occ = 1
            if math.isnan(occ):
                occ = 1
            if skip_disorder and occ < 1:
                raise ParseError(f"{block.name} has disorder")
            asym_occs.append(occ)
    else:
        asym_occs = [1] * xyz_loop.length()

    # if all(a == 1 for a in asym_occs):
    #     raise ParseError('no disorder')

    if "_atom_site_disorder_group" in loop_dict:
        groups = loop_dict["_atom_site_disorder_group"]
    else:
        groups = [None] * len(asym_occs)

    if "_atom_site_disorder_assembly" in loop_dict:
        assemblies = loop_dict["_atom_site_disorder_assembly"]
    else:
        assemblies = [None] * len(asym_occs)

    assemblies = [None if x in (".", "?") else x for x in assemblies]
    groups = [None if x in (".", "?") else x for x in groups]

    # Missing coordinates
    remove_sites = []
    where_missing_atoms = np.isnan(asym_unit.min(axis=-1))
    if np.any(where_missing_atoms):
        if missing_coords == "skip":
            raise ParseError(f"{block.name} has missing coordinates")
        elif missing_coords == "warn":
            warnings.warn(f"{block.name} has missing coordinates")
        remove_sites.extend(np.nonzero(where_missing_atoms)[0])

    # Remove dummy sites
    # remove_sites.extend(i for i, occ in enumerate(asym_occs) if occ > 1)
    if "_atom_site_calc_flag" in loop_dict:
        calc_flags = [gemmi.cif.as_string(f) for f in loop_dict["_atom_site_calc_flag"]]
        remove_sites.extend(i for i, f in enumerate(calc_flags) if f == "dum")

    # # Remove sites with disorder if needed or skip
    # if skip_disorder:
    #     if any(_has_disorder(l, o) for l, o in zip(asym_labels, asym_occs)):
    #         raise ParseError(f"{block.name} has disorder")

    if remove_hydrogens:
        remove_sites.extend(i for i, num in enumerate(asym_types) if num == 1)

    asym_unit = np.delete(asym_unit, remove_sites, axis=0)
    if asym_unit.shape[0] == 0:
        raise ParseError(f"{block.name} has no coordinates")

    # Symmetry operations, try xyz strings first
    for tag in _CIF_TAGS["symop"]:
        sitesym = [v.str(0) for v in block.find([tag])]
        if sitesym:
            rot, trans = _parse_sitesyms(sitesym)
            break
    else:
        # Try spacegroup name; can be a pair or in a loop
        spg = None
        for tag in _CIF_TAGS["spacegroup_name"]:
            for value in block.find([tag]):
                try:
                    # Some names cannot be parsed by gemmi.SpaceGroup
                    spg = gemmi.SpaceGroup(value.str(0))
                    break
                except ValueError:
                    continue
            if spg is not None:
                break

        if spg is None:
            # Try international number
            for tag in _CIF_TAGS["spacegroup_number"]:
                spg_num = block.find_value(tag)
                if spg_num is not None:
                    spg_num = gemmi.cif.as_int(spg_num)
                    break
            else:
                warnings.warn(f"{block.name} has no symmetry data, defaulting to P1")
                spg_num = 1
            spg = gemmi.SpaceGroup(spg_num)

        rot = np.array([np.array(o.rot) / o.DEN for o in spg.operations()])
        trans = np.array([np.array(o.tran) / o.DEN for o in spg.operations()])

    frac_motif, invs = _expand_asym_unit(asym_unit, rot, trans, eq_site_tol)
    _, wyc_muls = np.unique(invs, return_counts=True)
    wyc_muls = np.array(wyc_muls, dtype=np.uint64)
    asym_inds = np.zeros_like(wyc_muls, dtype=np.uint64)
    asym_inds[1:] = np.cumsum(wyc_muls, dtype=np.uint64)[:-1]

    motif = np.matmul(frac_motif, cell)
    
    # for i, (type, occ, lab, asm, grp) in zip(asym_types, asym_occs, asym_labels, assemblies, groups):
    #     if i not in remove_sites:

    asym_types = [s for i, s in enumerate(asym_types) if i not in remove_sites]
    asym_occs = [s for i, s in enumerate(asym_occs) if i not in remove_sites]
    asym_labels = [s for i, s in enumerate(asym_labels) if i not in remove_sites]
    assemblies = [s for i, s in enumerate(assemblies) if i not in remove_sites]
    groups = [s for i, s in enumerate(groups) if i not in remove_sites]

    disorder = _disorder_assemblies(
        asym_unit, wyc_muls, cell, assemblies, groups, asym_occs, eq_site_tol
    )
    asym_types = np.array(asym_types, dtype=np.uint64)

    if reduce_cell:
        cell = lll_reduce(cell)
        motif = np.mod(motif @ np.linalg.inv(cell), 1) @ cell

    return PeriodicSet(
        motif=motif,
        cell=cell,
        name=block.name,
        asym_unit=asym_inds,
        multiplicities=wyc_muls,
        types=asym_types,
        labels=asym_labels,
        disorder=disorder,
    )


def periodicset_from_ccdc_entry(
    entry,
    remove_hydrogens: bool = False,
    skip_disorder: bool = False,
    missing_coords: str = "warn",
    eq_site_tol: float = 1e-3,
    reduce_cell: bool = False,
    heaviest_component: bool = False,
    molecular_centres: bool = False
) -> PeriodicSet:
    """Convert a :class:`ccdc.entry.Entry` object to a
    :class:`amd.PeriodicSet <.periodicset.PeriodicSet>`.
    Entry is the type returned by :class:`ccdc.io.EntryReader`.

    Parameters
    ----------
    entry : :class:`ccdc.entry.Entry`
        A ccdc Entry object representing a database entry.
    remove_hydrogens : bool, optional
        Remove Hydrogens from the crystal.
    skip_disorder : bool
        Do not read disordered structures.
    missing_coords : str
        Action to take upon finding missing coordinates. "warn" (default) will
        emit a warning when a structure has missing coordinates, and the sites
        will be removed. "skip" will not read structures with missing coords.
    eq_site_tol : float, default 0.001
        Tolerance below which two sites are considered to be at the same
        position. 
    reduce_cell : bool, default False
        Reduce the unit cell (Lenstra-Lenstra-Lovasz lattice basis reduction).
    heaviest_component : bool, optional
        Removes all but the heaviest molecule in the asymmeric unit,
        intended for removing solvents.
    molecular_centres : bool, default False
        Use molecular centres of mass as the motif instead of centres of
        atoms.

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
        raise ParseError(f"{entry.identifier} has no 3D structure")

    # Disorder
    if skip_disorder and entry.has_disorder:
        raise ParseError(f"{entry.identifier} has disorder")

    return periodicset_from_ccdc_crystal(
        entry.crystal,
        remove_hydrogens=remove_hydrogens,
        skip_disorder=skip_disorder,
        missing_coords=missing_coords,
        eq_site_tol=eq_site_tol,
        heaviest_component=heaviest_component,
        molecular_centres=molecular_centres,
        reduce_cell=reduce_cell
    )


def periodicset_from_ccdc_crystal(
    crystal,
    remove_hydrogens: bool = False,
    skip_disorder: bool = False,
    missing_coords: str = "warn",
    eq_site_tol: float = 1e-3,
    reduce_cell: bool = False,
    heaviest_component: bool = False,
    molecular_centres: bool = False
) -> PeriodicSet:
    """Convert a :class:`ccdc.crystal.Crystal` object to a
    :class:`amd.PeriodicSet <.periodicset.PeriodicSet>`.
    Crystal is the type returned by :class:`ccdc.io.CrystalReader`.

    Parameters
    ----------
    crystal : :class:`ccdc.crystal.Crystal`
        A ccdc Crystal object representing a crystal structure.
    remove_hydrogens : bool, optional
        Remove Hydrogens from the crystal.
    skip_disorder : bool
        Do not read disordered structures.
    missing_coords : str
        Action to take upon finding missing coordinates. "warn" (default) will
        emit a warning when a structure has missing coordinates, and the sites
        will be removed. "skip" will not read structures with missing coords.
    eq_site_tol : float, default 0.001
        Tolerance below which two sites are considered to be at the same
        position. 
    reduce_cell : bool, default False
        Reduce the unit cell (Lenstra-Lenstra-Lovasz lattice basis reduction).
    heaviest_component : bool, optional
        Removes all but the heaviest molecule in the asymmeric unit,
        intended for removing solvents.
    molecular_centres : bool, default False
        Use molecular centres of mass as the motif instead of centres of
        atoms.

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

    molecule = crystal.asymmetric_unit_molecule

    # Disorder
    if skip_disorder:
        if crystal.has_disorder:
            raise ParseError(f"{crystal.identifier} has disorder")

    if remove_hydrogens:
        molecule.remove_atoms(a for a in molecule.atoms if a.atomic_symbol in "HD")

    # Missing coordinates
    if any(a.fractional_coordinates is None for a in molecule.atoms):
        if missing_coords == "skip":
            raise ParseError(f"{crystal.identifier} has missing coordinates")
        elif missing_coords == "warn":
            warnings.warn(f"{crystal.identifier} has missing coordinates")

        molecule.remove_atoms(
            a for a in molecule.atoms if a.fractional_coordinates is None
        )

    if heaviest_component and len(molecule.components) > 1:
        molecule = _heaviest_component_ccdc(molecule)

    # Unit cell
    cellpar = crystal.cell_lengths + crystal.cell_angles
    if None in cellpar:
        raise ParseError(f"{crystal.identifier} has no unit cell")
    cell = cellpar_to_cell(np.array(cellpar))

    if molecular_centres:
        frac_centres = _frac_molecular_centres_ccdc(crystal, eq_site_tol)
        mol_centres = np.matmul(frac_centres, cell)
        return PeriodicSet(mol_centres, cell, name=crystal.identifier)

    asym_atoms = molecule.atoms
    asym_unit = np.array([tuple(a.fractional_coordinates) for a in asym_atoms])

    if asym_unit.shape[0] == 0:
        raise ParseError(f"{crystal.identifier} has no coordinates")

    asym_unit = np.mod(asym_unit, 1)

    # Symmetry operations
    sitesym = crystal.symmetry_operators
    if not sitesym:
        warnings.warn(f"{crystal.identifier} has no symmetry data, defaulting to P1")
        sitesym = ["x,y,z"]

    # Apply symmetries to asymmetric unit
    rot, trans = _parse_sitesyms(sitesym)
    frac_motif, invs = _expand_asym_unit(asym_unit, rot, trans, eq_site_tol)
    _, wyc_muls = np.unique(invs, return_counts=True)
    wyc_muls = np.array(wyc_muls, dtype=np.uint64)
    asym_inds = np.zeros_like(wyc_muls, dtype=np.uint64)
    asym_inds[1:] = np.cumsum(wyc_muls, dtype=np.uint64)[:-1]
    motif = np.matmul(frac_motif, cell)

    asym_types = np.array([a.atomic_number for a in asym_atoms], dtype=np.uint64)
    asym_labels = [a.label for a in asym_atoms]
    asym_occs = np.array([float(a.occupancy) for a in asym_atoms])
    assemblies = [None] * len(asym_unit)
    groups = [None] * len(asym_unit)

    if crystal.has_disorder and not crystal.disorder.is_suppressed:
        for asm in crystal.disorder.assemblies:
            for grp in asm.groups:
                for atom in grp.atoms:
                    if atom.index < len(asym_unit): # fix: asym unit not giving all atoms
                        assemblies[atom.index] = str(asm.id)
                        groups[atom.index] = str(grp.id)

    disorder = _disorder_assemblies(
        asym_unit, wyc_muls, cell, assemblies, groups, asym_occs, eq_site_tol
    )

    if reduce_cell:
        cell = lll_reduce(cell)
        motif = np.mod(motif @ np.linalg.inv(cell), 1) @ cell

    return PeriodicSet(
        motif=motif,
        cell=cell,
        name=crystal.identifier,
        asym_unit=asym_inds,
        multiplicities=wyc_muls,
        types=asym_types,
        labels=asym_labels,
        disorder=disorder,
    )


def _parse_sitesyms(symmetries: List[str]) -> Tuple[FloatArray, FloatArray]:
    """Parse a sequence of symmetries in xyz form and return rotation
    and translation arrays.
    """
    n = len(symmetries)
    rotations = np.empty((n, 3, 3), dtype=np.float64)
    translations = np.empty((n, 3), dtype=np.float64)
    for i, sym in enumerate(symmetries):
        rot, trans = _parse_sitesym(sym)
        rotations[i] = rot
        translations[i] = trans
    return rotations, translations


def memoize(f):
    """Cache for _parse_sitesym()."""
    cache = {}

    def wrapper(arg):
        if arg not in cache:
            cache[arg] = f(arg)
        return cache[arg]

    return wrapper


@memoize
def _parse_sitesym(sym: str) -> Tuple[FloatArray, FloatArray]:
    """Parse a single symmetry as an xyz string and return a 3x3
    rotation matrix and a 3x1 translation vector.
    """

    rot = np.zeros((3, 3), dtype=np.float64)
    trans = np.zeros((3,), dtype=np.float64)

    for ind, element in enumerate(sym.split(",")):
        is_positive = True
        is_fraction = False
        sng_trans = None
        fst_trans = []
        snd_trans = []

        for char in element.lower():
            if char == "+":
                is_positive = True
            elif char == "-":
                is_positive = False
            elif char == "/":
                is_fraction = True
            elif char in "xyz":
                rot_sgn = 1.0 if is_positive else -1.0
                rot[ind][ord(char) - ord("x")] = rot_sgn
            elif char.isdigit() or char == ".":
                if sng_trans is None:
                    sng_trans = 1.0 if is_positive else -1.0
                if is_fraction:
                    snd_trans.append(char)
                else:
                    fst_trans.append(char)

        if not fst_trans:
            e_trans = 0.0
        else:
            e_trans = sng_trans * float("".join(fst_trans))

        if is_fraction:
            e_trans /= float("".join(snd_trans))

        trans[ind] = e_trans

    return rot, trans


def _expand_asym_unit(
    asym_unit: FloatArray,
    rotations: FloatArray,
    translations: FloatArray,
    tol: float,
    uniquify_sites: bool = False,
) -> Tuple[FloatArray, UIntArray]:
    """Expand the asymmetric unit by applying symmetries given by
    ``rotations`` and ``translations`` and removing invariant points.
    """

    asym_unit = asym_unit.astype(np.float64, copy=False)
    rotations = rotations.astype(np.float64, copy=False)
    translations = translations.astype(np.float64, copy=False)
    expanded_sites = _expand_sites(asym_unit, rotations, translations)
    frac_motif, invs = _reduce_expanded_sites(expanded_sites, tol)

    if uniquify_sites:
        if not all(_unique_sites(frac_motif, tol)):
            frac_motif, invs = _reduce_expanded_equiv_sites(expanded_sites, tol)

    return frac_motif, invs


@numba.njit(cache=True)
def _expand_sites(
    asym_unit: FloatArray, rotations: FloatArray, translations: FloatArray
) -> FloatArray:
    """Expand the asymmetric unit by applying ``rotations`` and
    ``translations``, without yet removing points duplicated because
    they are invariant under a symmetry. Returns a 3D array shape
    (#points, #syms, dims).
    """

    m, dims = asym_unit.shape
    n_syms = len(rotations)
    expanded_sites = np.empty((m, n_syms, dims), dtype=np.float64)
    for i in range(m):
        for j in range(n_syms):
            for dim in range(3):
                v = 0
                for dim_ in range(3):
                    v += rotations[j, dim, dim_] * asym_unit[i, dim_]
                expanded_sites[i, j, dim] = v + translations[j, dim]
    return np.mod(expanded_sites, 1)


@numba.njit(cache=True)
def _reduce_expanded_sites(
    expanded_sites: FloatArray, tol: float
) -> Tuple[FloatArray, UIntArray]:
    """Reduce the asymmetric unit after being expended by symmetries by
    removing invariant points. Assumes that no two sites in the
    asymmetric unit are equivalent.
    """

    all_unqiue_inds = []
    n_sites, _, dims = expanded_sites.shape
    multiplicities = np.empty(shape=(n_sites,), dtype=np.uint64)

    for i in range(n_sites):
        unique_inds = _unique_sites(expanded_sites[i], tol)
        all_unqiue_inds.append(unique_inds)
        multiplicities[i] = np.sum(unique_inds)

    m = np.sum(multiplicities)
    frac_motif = np.empty(shape=(m, dims), dtype=np.float64)
    inverses = np.empty(shape=(m,), dtype=np.uint64)

    s = 0
    for i in range(n_sites):
        t = s + multiplicities[i]
        frac_motif[s:t] = expanded_sites[i, all_unqiue_inds[i]]
        inverses[s:t] = i
        s = t

    return frac_motif, inverses


def _reduce_expanded_equiv_sites(
    expanded_sites: FloatArray, tol: float
) -> Tuple[FloatArray, UIntArray]:
    """Reduce the asymmetric unit after being expended by symmetries by
    removing invariant points. This version also removes symmetrically
    equivalent sites in the asymmetric unit.
    """

    sites = expanded_sites[0]
    unique_inds = _unique_sites(sites, tol)
    frac_motif = sites[unique_inds]
    inverses = [0] * len(frac_motif)

    for i in range(1, len(expanded_sites)):
        sites = expanded_sites[i]
        unique_inds = _unique_sites(sites, tol)

        points = []
        for site in sites[unique_inds]:
            diffs1 = np.abs(site - frac_motif)
            diffs2 = np.abs(diffs1 - 1)
            mask = np.all((diffs1 <= tol) | (diffs2 <= tol), axis=-1)

            if not np.any(mask):
                points.append(site)
            else:
                warnings.warn(
                    "has equivalent sites at positions "
                    f"{inverses[np.argmax(mask)]}, {i}"
                )

        if points:
            inverses.extend(i for _ in range(len(points)))
            frac_motif = np.concatenate((frac_motif, np.array(points)))

    return frac_motif, np.array(inverses, dtype=np.uint64)


@numba.njit(cache=True)
def _unique_sites(asym_unit: FloatArray, tol: float) -> npt.NDArray[np.bool_]:
    """Uniquify (within tol) a list of fractional coordinates,
    considering all points modulo 1. Return an array of bools such that
    asym_unit[_unique_sites(asym_unit, tol)] is the uniquified list.
    """

    m, dims = asym_unit.shape
    where_unique = np.full(shape=(m,), fill_value=True, dtype=np.bool_)
    for i in range(1, m):
        for j in range(i):
            for d in range(dims):
                diff1 = np.mod(np.abs(asym_unit[i, d] - asym_unit[j, d]), 1)
                diff2 = np.abs(diff1 - 1)
                if min(diff1, diff2) > tol:
                    break
            else:
                where_unique[i] = False
                break

    return where_unique


def lll_reduce(cell, delta: float = 0.75) -> npt.NDArray[np.float64]:
    """Perform a Lenstra-Lenstra-Lovasz lattice basis reduction to obtain a
    c-reduced basis. This method returns a basis which is as "good" as
    possible, with "good" defined by orthogonality of the lattice vectors.
    This basis is used for all the periodic boundary condition calculations.

    Parameters
    ----------
        delta (float): Reduction parameter. Default of 0.75 is usually fine.

    Returns
    -------
        Reduced lattice matrix, mapping to get to that lattice.
    """

    a = cell.T
    b = np.zeros((3, 3))  # Vectors after the Gram-Schmidt process
    u = np.zeros((3, 3))  # Gram-Schmidt coefficients
    m = np.zeros(3)  # These are the norm squared of each vec

    b[:, 0] = a[:, 0]
    m[0] = np.dot(b[:, 0], b[:, 0])
    for i in range(1, 3):
        u[i, :i] = np.dot(a[:, i].T, b[:, :i]) / m[:i]
        b[:, i] = a[:, i] - np.dot(b[:, :i], u[i, :i].T)
        m[i] = np.dot(b[:, i], b[:, i])

    k = 2
    mapping = np.identity(3, dtype=np.float64)
    while k <= 3:
        # Size reduction
        for i in range(k - 1, 0, -1):
            q = round(u[k - 1, i - 1])
            if q != 0:
                # Reduce the k-th basis vector
                a[:, k - 1] -= q * a[:, i - 1]
                mapping[:, k - 1] -= q * mapping[:, i - 1]
                uu = list(u[i - 1, 0 : (i - 1)])
                uu.append(1)
                # Update the GS coefficients
                u[k - 1, 0:i] -= q * np.array(uu)

        # Check the Lovasz condition
        if np.dot(b[:, k - 1], b[:, k - 1]) >= (delta - abs(u[k - 1, k - 2]) ** 2) * np.dot(
            b[:, (k - 2)], b[:, (k - 2)]
        ):
            # Increment k if the Lovasz condition holds
            k += 1
        else:
            # If the Lovasz condition fails, swap the k-th and (k-1)-th basis vector
            v = a[:, k - 1].copy()
            a[:, k - 1] = a[:, k - 2].copy()
            a[:, k - 2] = v

            v_m = mapping[:, k - 1].copy()
            mapping[:, k - 1] = mapping[:, k - 2].copy()
            mapping[:, k - 2] = v_m

            # Update the Gram-Schmidt coefficients
            for s in range(k - 1, k + 1):
                u[s - 1, : (s - 1)] = np.dot(a[:, s - 1].T, b[:, : (s - 1)]) / m[: (s - 1)]
                b[:, s - 1] = a[:, s - 1] - np.dot(b[:, : (s - 1)], u[s - 1, : (s - 1)].T)
                m[s - 1] = np.dot(b[:, s - 1], b[:, s - 1])

            if k > 2:
                k -= 1
            else:
                # We have to do p/q, so do lstsq(q.T, p.T).T instead
                q = np.diag(m[(k - 2) : k])
                p = np.dot(a[:, k:3].T, b[:, (k - 2) : k])
                result = np.linalg.lstsq(q.T, p.T, rcond=None)[0].T
                # result = np.linalg.inv(q.T) @ p.T  # if invertibility guaranteed
                u[k:3, (k - 2) : k] = result

    return a.T


def _disorder_assemblies(
        asym_unit,
        multiplicities,
        cell,
        assemblies,
        groups,
        asym_occs,
        eq_site_tol
):

    disorder = {}  # asm_name: {grp_name: [inds...], ...}

    # Follow given assemblies and groups
    for i, (asmbly, grp) in enumerate(zip(assemblies, groups)):

        if asym_occs[i] == 1:
            continue

        if asmbly in disorder:
            if grp in disorder[asmbly]:
                disorder[asmbly][grp].append(i)
            else:
                disorder[asmbly][grp] = [i]
        else:
            disorder[asmbly] = {grp: [i]}

    # FIX: what to do with atoms with a given group but no assembly?

    # fractional occupancies with no given groups or assemblies
    if None in disorder and None in disorder[None]:

        # Find substitutional disorder. Overlapping disordered atoms
        # are assumed to be substitutionally disordered.
        inds = disorder[None][None]
        asm_count = 0
        m = len(inds)
        leftover_pdist = _pdist_pbc(asym_unit[inds], cell)
        grps = _collapse_into_groups(leftover_pdist <= SUBS_DISORDER_TOL)

        leftover = []  # atoms not overlapping
        for grp in grps:

            if len(grp) == 1:
                leftover.append(inds[grp[0]])
            else:
                max_d = max(  # max interpoint dist in group
                    leftover_pdist[m * i + j - ((i + 2) * (i + 1)) // 2]
                    for i, j in itertools.combinations(grp, 2)
                )
                displaced_prefix = "d_" if max_d > 1e-10 else ""
                asm_name = f"{displaced_prefix}sub_asm_{asm_count}"
                disorder[asm_name] = {i: [inds[j]] for i, j in enumerate(grp)}
                asm_count += 1

        # Not substiutional
        if leftover:

            # seperate atoms with different occuapancies
            leftover_occs_arr = np.array(asym_occs)[leftover]
            occ_grps_pdist = (
                pdist(leftover_occs_arr[None, :], metric="chebyshev") < 1e-6
            )
            occ_grps = _collapse_into_groups(occ_grps_pdist)
            leftover_occs = np.array([leftover_occs_arr[grp[0]] for grp in occ_grps])

            # for now, put all 1/2 occupancy atoms in one assembly
            where_half_occ = np.argwhere(leftover_occs == 0.5)
            if where_half_occ.size > 0:
                grp_i = where_half_occ[0, 0]
                asm = {0: [leftover[j] for j in occ_grps[grp_i]]}
                disorder[f"half_asm_{asm_count}"] = asm
                asm_count += 1
                del occ_grps[grp_i]
                leftover_occs = np.delete(leftover_occs, grp_i)

            # Assemble atoms with occ > 1
            where_gt_1 = np.argwhere(leftover_occs > 1)
            if where_gt_1.size > 0:
                where_gt1_flat = where_gt_1.flatten()
                for grp_i in where_gt1_flat:
                    asm = {0: [leftover[j] for j in occ_grps[grp_i]]}
                    disorder[f"gt1_asm_{asm_count}"] = asm
                    asm_count += 1

                for grp_i in sorted(where_gt1_flat, reverse=True):
                    del occ_grps[grp_i]
                leftover_occs = np.delete(leftover_occs, where_gt_1)

            unity_sum_grps = _tuples_sum_to_one(leftover_occs)
            if unity_sum_grps:
                for grps in unity_sum_grps:
                    asm = {
                        en: [leftover[j] for j in occ_grps[grp_i]]
                        for en, (grp_i) in enumerate(grps)
                    }
                    disorder[f"sum_asm_{asm_count}"] = asm
                    asm_count += 1

                all_grp_inds = [i for grp in unity_sum_grps for i in grp]
                for grp_i in sorted(all_grp_inds, reverse=True):
                    del occ_grps[grp_i]
                leftover_occs = np.delete(leftover_occs, all_grp_inds)

            # assemble all remaining groups seperately.
            for grp in occ_grps:
                asm = {None: [leftover[j] for j in grp]}
                disorder[f"asm_{asm_count}"] = asm
                asm_count += 1

    # Check multiplicities ?
    assemblies = []
    for asm_name, asm_ in disorder.items():
        
        if asm_name is None:
            continue

        if len(asm_) < 2 and asym_occs[list(asm_.values())[0][0]] >= 1:
            continue

        # normalise if occupancies sum to > 1
        occ_sum = sum(asym_occs[grp[0]] for grp in asm_.values())
        norm = max(1, occ_sum)
        assemblies.append(DisorderAssembly(
            groups=[
                DisorderGroup(indices=grp, occupancy=asym_occs[grp[0]] / norm)
                for grp in asm_.values()
            ],
            is_substitutional=(asm_name.startswith("sub") and occ_sum >= 1)
        ))

    if not assemblies:
        assemblies = None

    return assemblies


@numba.njit(cache=True)
def _pdist_pbc(points, cell):
    m = len(points)
    out = np.empty((int(m * (m - 1) / 2),), dtype=np.float64)
    ind = 0
    for i in range(m):
        for j in range(i + 1, m):
            diffs1 = np.abs(points[i] - points[j])
            diffs2 = np.abs(diffs1 - 1)
            diff_v = np.minimum(diffs1, diffs2) @ cell
            out[ind] = np.sum(diff_v**2)
            ind += 1
    return np.sqrt(out)


def _tuples_sum_to_one(unqiue_occs):
    """Given a vector ``unqiue_occs`` of numbers between 0 and 1, return
    tuples of indices specifying groups of elements summing to one."""

    ret = []
    done = set()

    def _combs_totalling_one(occs, m):

        def helper(start, index, current_combination):

            if index == m:
                if np.sum(occs[current_combination]) == 1:
                    for i in current_combination:
                        done.add(i)
                    yield current_combination
                return

            for i in range(start, len(occs)):
                if any(current_combination[j] in done for j in range(index)):
                    return
                if i in done:
                    continue
                current_combination[index] = i
                if np.sum(occs[current_combination[: index + 1]]) > 1:
                    return
                yield from helper(i + 1, index + 1, current_combination)

        combination = np.zeros((m,), dtype=np.int64)
        yield from helper(0, 0, combination)

    argsorted = np.argsort(unqiue_occs)
    occs = unqiue_occs[argsorted]

    for asize in range(2, len(occs)):
        for comb in _combs_totalling_one(occs, asize):
            ret.append(argsorted[comb])

    return ret


def _has_disorder(label: str, occupancy) -> bool:
    """Return True if label ends with ? or occupancy is a number < 1."""
    try:
        occupancy = float(occupancy)
    except Exception:
        occupancy = 1
    return (occupancy < 1) or label.endswith("?")


def _atomic_symbols_from_labels(symbols: List[str]) -> List[str]:
    symbols_ = []
    for label in symbols:
        sym = ""
        if label and label not in (".", "?"):
            match = re.search(r"([A-Za-z][A-Za-z]?)", label)
            if match is not None:
                sym = match.group()
            sym = list(sym)
            if len(sym) > 0:
                sym[0] = sym[0].upper()
            if len(sym) > 1:
                sym[1] = sym[1].lower()
            sym = "".join(sym)
        symbols_.append(sym)
    return symbols_


def _get_syms_pymatgen(data: dict) -> Tuple[FloatArray, FloatArray]:
    """Parse symmetry operations given by data = block.data where block
    is a pymatgen CifBlock object. If the symops are not present the
    space group symbol/international number is parsed and symops are
    generated.
    """

    from pymatgen.symmetry.groups import SpaceGroup
    import pymatgen.io.cif

    # Try xyz symmetry operations
    for symmetry_label in _CIF_TAGS["symop"]:
        xyz = data.get(symmetry_label)
        if not xyz:
            continue
        if isinstance(xyz, str):
            xyz = [xyz]
        return _parse_sitesyms(xyz)

    symops = []
    # Try spacegroup symbol
    for symmetry_label in _CIF_TAGS["spacegroup_name"]:
        sg = data.get(symmetry_label)
        if not sg:
            continue
        sg = re.sub(r"[\s_]", "", sg)
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
                if sg == re.sub(r"\s+", "", d["hermann_mauguin"]):
                    return _parse_sitesyms(d["symops"])
        except Exception:  # CHANGE
            continue
        if symops:
            break

    # Try international number
    if not symops:
        for symmetry_label in _CIF_TAGS["spacegroup_number"]:
            num = data.get(symmetry_label)
            if not num:
                continue
            try:
                i = int(str2float(num))
                symops = SpaceGroup.from_int_number(i).symmetry_ops
                break
            except ValueError:
                continue

    if not symops:
        warnings.warn("no symmetry data found, defaulting to P1")
        return _parse_sitesyms(["x,y,z"])

    rotations = [op.rotation_matrix for op in symops]
    translations = [op.translation_vector for op in symops]
    rotations = np.array(rotations, dtype=np.float64)
    translations = np.array(translations, dtype=np.float64)
    return rotations, translations


def _frac_molecular_centres_ccdc(crystal, tol: float) -> FloatArray:
    """Return the geometric centres of molecules in the unit cell.
    Expects a ccdc Crystal object and returns fractional coordiantes.
    """

    frac_centres = []
    for comp in crystal.packing(inclusion="CentroidIncluded").components:
        coords = [a.fractional_coordinates for a in comp.atoms]
        frac_centres.append([sum(ax) / len(coords) for ax in zip(*coords)])
    frac_centres = np.mod(np.array(frac_centres, dtype=np.float64), 1)
    return frac_centres[_unique_sites(frac_centres, tol)]


def _heaviest_component_ccdc(molecule):
    """Remove all but the heaviest component of the asymmetric unit.
    Intended for removing solvents. Expects and returns a ccdc Molecule
    object.
    """

    component_weights = []
    for component in molecule.components:
        weight = 0
        for a in component.atoms:
            try:
                occ = float(a.occupancy)
            except:
                occ = 1
            try:
                weight += float(a.atomic_weight) * occ
            except ValueError:
                pass
        component_weights.append(weight)
    largest_component_ind = np.argmax(np.array(component_weights))
    molecule = molecule.components[largest_component_ind]
    return molecule


def str2float(string):
    """Remove uncertainty brackets from strings and return the float.
    Returns np.nan if given '.' or '?'."""
    try:
        return float(re.sub(r"\(.+\)*", "", string))
    except TypeError:
        if isinstance(string, list) and len(string) == 1:
            return float(re.sub(r"\(.+\)*", "", string[0]))
    except ValueError as e:
        if string.strip() in (".", "?"):
            return np.nan
        raise ParseError("String could not be converted to float")


def _get_cif_tags(block, cif_tags):

    import gemmi

    data = {}
    for tag in cif_tags:
        column = list(block.find_values(tag))

        if len(column) == 0:
            data[tag] = None
        else:
            column_ = []
            for v in column:
                try:
                    v = int(v)
                except ValueError:
                    try:
                        v = str2float(v)
                    except ValueError:
                        v = gemmi.cif.as_string(v)
                column_.append(v)
            if len(column_) == 1:
                data[tag] = column_[0]
            else:
                data[tag] = column_

    return data


def _snap_small_prec_coords(frac_coords: FloatArray, tol: float) -> FloatArray:
    """Find where frac_coords is within 1e-4 of 1/3 or 2/3, change to
    1/3 and 2/3. Recommended by pymatgen's CIF parser.
    """
    frac_coords[np.abs(1 - 3 * frac_coords) < tol] = 1 / 3.0
    frac_coords[np.abs(1 - 3 * frac_coords / 2) < tol] = 2 / 3.0
    return frac_coords


# def periodicset_from_ase_cifblock(
#     block,
#     remove_hydrogens: bool = False,
#     skip_disorder: bool = False,
#     eq_site_tol: float = 1e-3,
# ) -> PeriodicSet:
#     """Convert a :class:`ase.io.cif.CIFBlock` object to a
#     :class:`amd.PeriodicSet <.periodicset.PeriodicSet>`.
#     :class:`ase.io.cif.CIFBlock` is the type returned by
#     :func:`ase.io.cif.parse_cif`.

#     Parameters
#     ----------
#     block : :class:`ase.io.cif.CIFBlock`
#         An ase :class:`ase.io.cif.CIFBlock` object representing a
#         crystal.
#     remove_hydrogens : bool, optional
#         Remove Hydrogens from the crystal.
#     disorder : str, optional
#         Controls how disordered structures are handled. Default is
#         ``skip`` which skips any crystal with disorder, since disorder
#         conflicts with the periodic set model. To read disordered
#         structures anyway, choose either :code:`ordered_sites` to remove
#         atoms with disorder or :code:`all_sites` include all atoms
#         regardless of disorder. Note that :code:`all_sites` has
#         different behaviour than :code:`periodicset_from_gemmi_block`
#         and outputs of this function always have None assigned to the
#         :code:`.occupancies` attribute.

#     Returns
#     -------
#     :class:`amd.PeriodicSet <.periodicset.PeriodicSet>`
#         Represents the crystal as a periodic set, consisting of a finite
#         set of points (motif) and lattice (unit cell). Contains other
#         useful data, e.g. the crystal's name and information about the
#         asymmetric unit for calculation.

#     Raises
#     ------
#     ParseError
#         Raised if the structure fails to be parsed for any of the
#         following: 1. Required data is missing (e.g. cell parameters),
#         2. The motif is empty after removing H or disordered sites,
#         3. :code:``disorder == 'skip'`` and disorder is found on any
#         atom.
#     """

#     import ase
#     import ase.spacegroup

#     # Unit cell
#     cellpar = [str2float(str(block.get(tag))) for tag in _CIF_TAGS["cellpar"]]
#     if None in cellpar:
#         raise ParseError(f"{block.name} has missing cell data")
#     cell = cellpar_to_cell(np.array(cellpar))

#     # Asymmetric unit coordinates. ase removes uncertainty brackets
#     asym_unit = [
#         [str2float(str(n)) for n in block.get(tag)]
#         for tag in _CIF_TAGS["atom_site_fract"]
#     ]
#     if None in asym_unit:
#         asym_unit = [block.get(tag.lower()) for tag in _CIF_TAGS["atom_site_cartn"]]
#         if None in asym_unit:
#             raise ParseError(f"{block.name} has missing coordinates")
#         else:
#             raise ParseError(
#                 f"{block.name} uses _atom_site_Cartn_ tags for coordinates, "
#                 "only _atom_site_fract_ is supported"
#             )
#     asym_unit = list(zip(*asym_unit))

#     # Labels
#     asym_labels = block.get("_atom_site_label")
#     if asym_labels is None:
#         asym_labels = [""] * len(asym_unit)

#     # Atomic types
#     asym_symbols = block.get("_atom_site_type_symbol")
#     if asym_symbols is not None:
#         asym_symbols_ = _atomic_symbols_from_labels(asym_symbols)
#     else:
#         asym_symbols_ = [""] * len(asym_unit)

#     asym_types = []
#     for s in asym_symbols_:
#         if s in ATOMIC_NUMBERS:
#             asym_types.append(ATOMIC_NUMBERS[s])
#         else:
#             asym_types.append(0)

#     # Find where sites have disorder if necassary
#     has_disorder = []
#     occupancies = block.get("_atom_site_occupancy")
#     if occupancies is None:
#         occupancies = [1] * len(asym_unit)
#     for lab, occ in zip(asym_labels, occupancies):
#         has_disorder.append(_has_disorder(lab, occ))

#     # Remove sites with ?, . or other invalid string for coordinates
#     invalid = []
#     for i, xyz in enumerate(asym_unit):
#         if not all(isinstance(coord, (int, float)) for coord in xyz):
#             invalid.append(i)
#     if invalid:
#         warnings.warn("atoms without sites or missing data will be removed")
#         asym_unit = [c for i, c in enumerate(asym_unit) if i not in invalid]
#         asym_types = [t for i, t in enumerate(asym_types) if i not in invalid]
#         has_disorder = [d for i, d in enumerate(has_disorder) if i not in invalid]

#     remove_sites = []

#     if remove_hydrogens:
#         remove_sites.extend(i for i, num in enumerate(asym_types) if num == 1)

#     # Remove atoms with fractional occupancy or raise ParseError
#     for i, dis in enumerate(has_disorder):
#         if i in remove_sites:
#             continue
#         if dis:
#             if skip_disorder:
#                 raise ParseError(
#                     f"{block.name} has disorder, pass "
#                     "disorder='ordered_sites' or 'all_sites' to "
#                     "remove/ignore disorder"
#                 )
#             remove_sites.append(i)

#     # Asymmetric unit
#     asym_unit = [c for i, c in enumerate(asym_unit) if i not in remove_sites]
#     asym_types = [t for i, t in enumerate(asym_types) if i not in remove_sites]
#     if len(asym_unit) == 0:
#         raise ParseError(f"{block.name} has no valid sites")
#     asym_unit = np.mod(np.array(asym_unit), 1)
#     asym_types = np.array(asym_types, dtype=np.uint8)

#     # # recommended by pymatgen
#     # asym_unit = _snap_small_prec_coords(asym_unit, 1e-4)

#     # Get symmetry operations
#     sitesym = block._get_any(_CIF_TAGS["symop"])
#     if sitesym is None:
#         label_or_num = block._get_any([s.lower() for s in _CIF_TAGS["spacegroup_name"]])
#         if label_or_num is None:
#             label_or_num = block._get_any(
#                 [s.lower() for s in _CIF_TAGS["spacegroup_number"]]
#             )
#         if label_or_num is None:
#             warnings.warn("no symmetry data found, defaulting to P1")
#             label_or_num = 1
#         spg = ase.spacegroup.Spacegroup(label_or_num)
#         rot, trans = spg.get_op()
#     else:
#         if isinstance(sitesym, str):
#             sitesym = [sitesym]
#         rot, trans = _parse_sitesyms(sitesym)

#     frac_motif, invs = _expand_asym_unit(asym_unit, rot, trans, eq_site_tol)
#     _, wyc_muls = np.unique(invs, return_counts=True)
#     asym_inds = np.zeros_like(wyc_muls, dtype=np.int64)
#     asym_inds[1:] = np.cumsum(wyc_muls)[:-1]
#     motif = np.matmul(frac_motif, cell)

#     return PeriodicSet(
#         motif=motif,
#         cell=cell,
#         name=block.name,
#         asym_unit=asym_inds,
#         multiplicities=wyc_muls,
#         types=asym_types,
#         occupancies=None,
#     )


# def periodicset_from_pymatgen_cifblock(
#     block,
#     remove_hydrogens: bool = False,
#     skip_disorder: bool = False,
#     eq_site_tol: float = 1e-3,
# ) -> PeriodicSet:
#     """Convert a :class:`pymatgen.io.cif.CifBlock` object to a
#     :class:`amd.PeriodicSet <.periodicset.PeriodicSet>`.
#     :class:`pymatgen.io.cif.CifBlock` is the type returned by
#     :class:`pymatgen.io.cif.CifFile`.

#     Parameters
#     ----------
#     block : :class:`pymatgen.io.cif.CifBlock`
#         A pymatgen CifBlock object representing a crystal.
#     remove_hydrogens : bool, optional
#         Remove Hydrogens from the crystal.
#     disorder : str, optional
#         Controls how disordered structures are handled. Default is
#         ``skip`` which skips any crystal with disorder, since disorder
#         conflicts with the periodic set model. To read disordered
#         structures anyway, choose either :code:`ordered_sites` to remove
#         atoms with disorder or :code:`all_sites` include all atoms
#         regardless of disorder. Note that :code:`all_sites` has
#         different behaviour than :code:`periodicset_from_gemmi_block`
#         and outputs of this function always have None assigned to the
#         :code:`.occupancies` attribute.

#     Returns
#     -------
#     :class:`amd.PeriodicSet <.periodicset.PeriodicSet>`
#         Represents the crystal as a periodic set, consisting of a finite
#         set of points (motif) and lattice (unit cell). Contains other
#         useful data, e.g. the crystal's name and information about the
#         asymmetric unit for calculation.

#     Raises
#     ------
#     ParseError
#         Raised if the structure can/should not be parsed for the
#         following reasons: 1. No sites found or motif is empty after
#         removing Hydrogens & disorder, 2. A site has missing
#         coordinates, 3. :code:``disorder == 'skip'`` and disorder is
#         found on any atom.
#     """

#     odict = block.data

#     # Unit cell
#     cellpar = [odict.get(tag) for tag in _CIF_TAGS["cellpar"]]
#     if any(par in (None, "?", ".") for par in cellpar):
#         raise ParseError(f"{block.header} has missing cell data")

#     try:
#         cellpar = [str2float(v) for v in cellpar]
#     except ValueError:
#         raise ParseError(f"{block.header} could not be parsed")
#     cell = cellpar_to_cell(np.array(cellpar, dtype=np.float64))

#     # Asymmetric unit coordinates
#     asym_unit = [odict.get(tag) for tag in _CIF_TAGS["atom_site_fract"]]
#     # check for . and ?
#     if None in asym_unit:
#         asym_unit = [odict.get(tag) for tag in _CIF_TAGS["atom_site_cartn"]]
#         if None in asym_unit:
#             raise ParseError(f"{block.header} has missing coordinates")
#         else:
#             raise ParseError(
#                 f"{block.header} uses _atom_site_Cartn_ tags for coordinates, "
#                 "only _atom_site_fract_ is supported"
#             )
#     asym_unit = list(zip(*asym_unit))
#     try:
#         asym_unit = [[str2float(coord) for coord in xyz] for xyz in asym_unit]
#     except ValueError:
#         raise ParseError(f"{block.header} could not be parsed")

#     # Labels
#     asym_labels = odict.get("_atom_site_label")
#     if asym_labels is None:
#         asym_labels = [""] * len(asym_unit)

#     # Atomic types
#     asym_symbols = odict.get("_atom_site_type_symbol")
#     if asym_symbols is not None:
#         asym_symbols_ = _atomic_symbols_from_labels(asym_symbols)
#     else:
#         asym_symbols_ = [""] * len(asym_unit)

#     asym_types = []
#     for s in asym_symbols_:
#         if s in ATOMIC_NUMBERS:
#             asym_types.append(ATOMIC_NUMBERS[s])
#         else:
#             asym_types.append(0)

#     # Find where sites have disorder if necassary
#     has_disorder = []
#     occupancies = odict.get("_atom_site_occupancy")
#     if occupancies is None:
#         occupancies = np.ones((len(asym_unit),))
#     else:
#         occupancies = np.array([str2float(occ) for occ in occupancies])
#     labels = odict.get("_atom_site_label")
#     if labels is None:
#         labels = [""] * len(asym_unit)
#     for lab, occ in zip(labels, occupancies):
#         has_disorder.append(_has_disorder(lab, occ))

#     # Remove sites with ?, . or other invalid string for coordinates
#     invalid = []
#     for i, xyz in enumerate(asym_unit):
#         if not all(isinstance(coord, (int, float)) for coord in xyz):
#             invalid.append(i)

#     if invalid:
#         warnings.warn("atoms without sites or missing data will be removed")
#         asym_unit = [c for i, c in enumerate(asym_unit) if i not in invalid]
#         asym_types = [c for i, c in enumerate(asym_types) if i not in invalid]
#         has_disorder = [d for i, d in enumerate(has_disorder) if i not in invalid]

#     remove_sites = []

#     if remove_hydrogens:
#         remove_sites.extend((i for i, n in enumerate(asym_types) if n == 1))

#     # Remove atoms with fractional occupancy or raise ParseError
#     for i, dis in enumerate(has_disorder):
#         if i in remove_sites:
#             continue
#         if dis:
#             if skip_disorder:
#                 raise ParseError(
#                     f"{block.header} has disorder, pass "
#                     "disorder='ordered_sites' or 'all_sites' to "
#                     "remove/ignore disorder"
#                 )
#             remove_sites.append(i)

#     # Asymmetric unit
#     asym_unit = [c for i, c in enumerate(asym_unit) if i not in remove_sites]
#     asym_types = [t for i, t in enumerate(asym_types) if i not in remove_sites]
#     if len(asym_unit) == 0:
#         raise ParseError(f"{block.header} has no valid sites")
#     asym_unit = np.mod(np.array(asym_unit), 1)
#     asym_types = np.array(asym_types, dtype=np.uint8)

#     # recommended by pymatgen
#     # asym_unit = _snap_small_prec_coords(asym_unit, 1e-4)

#     # Apply symmetries to asymmetric unit
#     rot, trans = _get_syms_pymatgen(odict)
#     frac_motif, invs = _expand_asym_unit(asym_unit, rot, trans, eq_site_tol)
#     _, wyc_muls = np.unique(invs, return_counts=True)
#     asym_inds = np.zeros_like(wyc_muls, dtype=np.int64)
#     asym_inds[1:] = np.cumsum(wyc_muls)[:-1]
#     motif = np.matmul(frac_motif, cell)

#     return PeriodicSet(
#         motif=motif,
#         cell=cell,
#         name=block.header,
#         asym_unit=asym_inds,
#         multiplicities=wyc_muls,
#         types=asym_types,
#         occupancies=None,
#     )


# def periodicset_from_ase_atoms(
#     atoms, remove_hydrogens: bool = False, eq_site_tol: float = 1e-3
# ) -> PeriodicSet:
#     """Convert an :class:`ase.atoms.Atoms` object to a
#     :class:`amd.PeriodicSet <.periodicset.PeriodicSet>`. Does not have
#     the option to remove disorder.

#     Parameters
#     ----------
#     atoms : :class:`ase.atoms.Atoms`
#         An ase :class:`ase.atoms.Atoms` object representing a crystal.
#     remove_hydrogens : bool, optional
#         Remove Hydrogens from the crystal.

#     Returns
#     -------
#     :class:`amd.PeriodicSet <.periodicset.PeriodicSet>`
#         Represents the crystal as a periodic set, consisting of a finite
#         set of points (motif) and lattice (unit cell). Contains other
#         useful data, e.g. the crystal's name and information about the
#         asymmetric unit for calculation.

#     Raises
#     ------
#     ParseError
#         Raised if there are no valid sites in atoms.
#     """

#     from ase.spacegroup import get_basis

#     cell = atoms.get_cell().array

#     remove_inds = []
#     if remove_hydrogens:
#         for i in np.where(atoms.get_atomic_numbers() == 1)[0]:
#             remove_inds.append(i)
#     for i in sorted(remove_inds, reverse=True):
#         atoms.pop(i)

#     if len(atoms) == 0:
#         raise ParseError("ase Atoms object has no valid sites")

#     # Symmetry operations from spacegroup
#     spg = None
#     if "spacegroup" in atoms.info:
#         spg = atoms.info["spacegroup"]
#         rot, trans = spg.rotations, spg.translations
#     else:
#         warnings.warn("no symmetry data found, defaulting to P1")
#         rot = np.identity(3)[None, :]
#         trans = np.zeros((1, 3))

#     # Asymmetric unit. ase default tol is 1e-5
#     # do differently! get_basis determines a reduced asym unit from the atoms;
#     # surely this is not needed!
#     asym_unit = get_basis(atoms, spacegroup=spg, tol=eq_site_tol)
#     frac_motif, invs = _expand_asym_unit(asym_unit, rot, trans, eq_site_tol)
#     _, wyc_muls = np.unique(invs, return_counts=True)
#     asym_inds = np.zeros_like(wyc_muls, dtype=np.int64)
#     asym_inds[1:] = np.cumsum(wyc_muls)[:-1]
#     motif = np.matmul(frac_motif, cell)
#     motif_types = atoms.get_atomic_numbers()
#     types = np.array([motif_types[i] for i in asym_inds], dtype=np.uint8)

#     return PeriodicSet(
#         motif=motif,
#         cell=cell,
#         asym_unit=asym_inds,
#         multiplicities=wyc_muls,
#         types=types,
#         occupancies=None,
#     )


def periodicset_from_pymatgen_structure(
    structure,
    remove_hydrogens: bool = False,
    skip_disorder: bool = False,
    missing_coords: str = "warn",
    eq_site_tol: float = 1e-3,
) -> PeriodicSet:
    """Convert a :class:`pymatgen.core.structure.Structure` object to a
    :class:`amd.PeriodicSet <.periodicset.PeriodicSet>`. Does not set
    the name of the periodic set, as pymatgen Structure objects seem to
    have no name attribute.

    Parameters
    ----------
    structure : :class:`pymatgen.core.structure.Structure`
        A pymatgen Structure object representing a crystal.

    Returns
    -------
    :class:`amd.PeriodicSet <.periodicset.PeriodicSet>`
        Represents the crystal as a periodic set, consisting of a finite
        set of points (motif) and lattice (unit cell). Contains other
        useful data, e.g. the crystal's name and information about the
        asymmetric unit for calculation.
    """

    if remove_hydrogens:
        structure.remove_species(["H", "D"])

    # # Disorder
    # if skip_disorder:
    #     if not structure.is_ordered:
    #         raise ParseError("Structure has disorder")
    # else:
    #     remove_inds = []
    #     for i, comp in enumerate(structure.species_and_occu):
    #         if comp.num_atoms < 1:
    #             remove_inds.append(i)
    #     structure.remove_sites(remove_inds)

    motif = structure.cart_coords
    cell = structure.lattice.matrix
    types = np.array(structure.atomic_numbers, dtype=np.uint64)
    # occupancies = np.ones((len(motif), ), dtype=np.float64)

    return PeriodicSet(
        motif=motif,
        cell=cell,
        types=types,
        # occupancies=occupancies,
    )
