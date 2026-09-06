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
from typing import (
    Iterable,
    Iterator,
    NamedTuple,
    Optional,
    Union,
    Callable,
    Tuple,
    List,
)

import numpy as np
import numpy.typing as npt
import numba
import tqdm
from scipy.spatial.distance import pdist

from .utils import cellpar_to_cell, lll_reduce
from .periodicset import DisorderGroup, DisorderAssembly, PeriodicSet
from .calculate import _collapse_into_groups
from .globals_ import (
    SUBS_DISORDER_TOL,
    EXACT_OVERLAP_TOL,
    OCCUPANCY_TOL,
    OCCUPANCY_ROUNDING_TOL,
    SPECIAL_POSITION_OCC_TOL,
    DISORDER_REGION_TOL,
    DISORDER_COMPONENT_TOL,
    MAX_OCCUPANCY_SUBSET_SEARCH,
    ATOMIC_NUMBERS,
)
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
    "_atom_site_symmetry": [
        "_atom_site_site_symmetry_order",
        "_atom_site_symmetry_multiplicity"
    ],
    "symop": [
        "_space_group_symop_operation_xyz",
        "_space_group_symop.operation_xyz",
        "_symmetry_equiv_pos_as_xyz",
    ],
    "spacegroup_name": [
        "_space_group_name_H-M_alt",
        "_symmetry_space_group_name_H-M"
    ],
    "spacegroup_number": [
        "_space_group_IT_number",
        "_symmetry_Int_Tables_number"
    ],
    
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

        while True:

            try:
                item = next(self._iterator)
            except StopIteration:
                if self._progress_bar is not None:
                    self._progress_bar.close()
                raise StopIteration

            error = None
            with warnings.catch_warnings(record=True) as warning_msgs:
                try:
                    periodic_set = self._converter(item)
                except ParseError as err:
                    error = err
                finally:
                    if self._progress_bar is not None:
                        self._progress_bar.update(1)

            if not self.show_warnings:
                if error is not None:
                    continue
                return periodic_set

            if error is not None:
                warnings.warn(str(error))
                continue

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


class CifReader(_Reader, Iterator[PeriodicSet]):
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
    wrap_coords : bool, default True
        Wrap motif points inside the unit cell.
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
        wrap_coords: bool = True,
        show_warnings: bool = True,
        verbose: bool = False,
        heaviest_component: bool = False,
        molecular_centres: bool = False
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
                reduce_cell=reduce_cell,
                wrap_coords=wrap_coords
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
                wrap_coords=wrap_coords,
                reduce_cell=reduce_cell,
                heaviest_component=heaviest_component,
                molecular_centres=molecular_centres
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

    def __iter__(self) -> Iterator[PeriodicSet]:
        return self

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


class CSDReader(_Reader, Iterator[PeriodicSet]):
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
    wrap_coords : bool, default True
        Wrap motif points inside the unit cell.
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
        wrap_coords: bool = True,
        show_warnings: bool = True,
        verbose: bool = False,
        heaviest_component: bool = False,
        molecular_centres: bool = False
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
            wrap_coords=wrap_coords,
            heaviest_component=heaviest_component,
            molecular_centres=molecular_centres,
        )

        entry_reader = ccdc.io.EntryReader("CSD")
        if refcodes is None:
            iterable = entry_reader
        else:
            iterable = map(entry_reader.entry, refcodes)

        super().__init__(iterable, converter, show_warnings, verbose)

    def __iter__(self) -> Iterator[PeriodicSet]:
        return self


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
    reduce_cell: bool = False,
    wrap_coords: bool = True
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
        Raise ParseError for disordered structures.
    missing_coords : str
        Action to take upon finding missing coordinates. "warn" (default) will
        emit a warning when a structure has missing coordinates, and the sites
        will be removed. "skip" will not read structures with missing coords.
    eq_site_tol : float, default 0.001
        Tolerance below which two sites are considered to be at the same
        position. 
    reduce_cell : bool, default False
        Reduce the unit cell (Lenstra-Lenstra-Lovasz lattice basis reduction).
    wrap_coords : bool, default True
        Wrap motif points inside the unit cell.

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

    loop_dict = {tag.lower(): l for tag, l in zip(xyz_loop.tags, tablified_loop)}
    xyz_str = [loop_dict[t] for t in _CIF_TAGS["atom_site_fract"]]
    asym_unit = np.transpose(
        np.array([[str2float(c) for c in xyz] for xyz in xyz_str], dtype=np.float64)
    )
    if wrap_coords:
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
        symbols = [
            _element_symbol(gemmi.cif.as_string(s))
            for s in loop_dict["_atom_site_type_symbol"]
        ]
    else:
        symbols = _atomic_symbols_from_labels(asym_labels)

    asym_types = [
        ATOMIC_NUMBERS[s] if s in ATOMIC_NUMBERS else 0 for s in symbols
    ]

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
            asym_occs.append(occ)
    else:
        asym_occs = [1] * xyz_loop.length()

    # Site symmetry order, the number of symops a site is invariant under.
    # If present, the file keeps multiplicity out of the occupancy column.
    site_sym_orders = None
    for tag in _CIF_TAGS["_atom_site_symmetry"]:
        if tag in loop_dict:
            site_sym_orders = []
            for value in loop_dict[tag]:
                try:
                    site_sym_orders.append(str2float(value))
                except (ValueError, TypeError):
                    site_sym_orders.append(float("nan"))
            break

    if "_atom_site_disorder_group" in loop_dict:
        groups = list(loop_dict["_atom_site_disorder_group"])
    else:
        groups = [None] * len(asym_occs)

    if "_atom_site_disorder_assembly" in loop_dict:
        assemblies = list(loop_dict["_atom_site_disorder_assembly"])
    else:
        assemblies = [None] * len(asym_occs)

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
    if "_atom_site_calc_flag" in loop_dict:
        calc_flags = [gemmi.cif.as_string(f) for f in loop_dict["_atom_site_calc_flag"]]
        remove_sites.extend(i for i, f in enumerate(calc_flags) if f == "dum")

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
                if spg_num is None:
                    continue
                try:
                    spg = gemmi.SpaceGroup(gemmi.cif.as_int(spg_num))
                    break
                except ValueError:
                    continue
            if spg is None:
                warnings.warn(f"{block.name} has no symmetry data, defaulting to P1")
                spg = gemmi.SpaceGroup(1)

        rot = np.array([np.array(o.rot) / o.DEN for o in spg.operations()])
        trans = np.array([np.array(o.tran) / o.DEN for o in spg.operations()])

    frac_motif, invs, kept = _expand_asym_unit(
        asym_unit, rot, trans, eq_site_tol, uniquify_sites=False,
        wrap_coords=wrap_coords
    )
    _, wyc_muls = np.unique(invs, return_counts=True)
    wyc_muls = np.array(wyc_muls, dtype=np.uint64)
    asym_inds = np.zeros_like(wyc_muls, dtype=np.uint64)
    asym_inds[1:] = np.cumsum(wyc_muls, dtype=np.uint64)[:-1]

    motif = np.matmul(frac_motif, cell)

    remove_sites = set(int(i) for i in remove_sites)
    asym_types = [s for i, s in enumerate(asym_types) if i not in remove_sites]
    asym_occs = [s for i, s in enumerate(asym_occs) if i not in remove_sites]
    asym_labels = [s for i, s in enumerate(asym_labels) if i not in remove_sites]
    assemblies = [s for i, s in enumerate(assemblies) if i not in remove_sites]
    groups = [s for i, s in enumerate(groups) if i not in remove_sites]
    if site_sym_orders is not None:
        site_sym_orders = [
            s for i, s in enumerate(site_sym_orders) if i not in remove_sites
        ]

    n_sites = asym_unit.shape[0]
    asym_types, asym_occs, asym_labels, assemblies, groups, site_sym_orders = _keep_sites(
        kept, n_sites,
        asym_types, asym_occs, asym_labels, assemblies, groups, site_sym_orders
    )
    if len(kept) != n_sites:
        asym_unit = asym_unit[np.asarray(kept, dtype=np.intp)]

    disorder = _disorder_assemblies(
        asym_unit,
        wyc_muls,
        len(rot),
        cell,
        asym_types,
        assemblies,
        groups,
        asym_occs,
        site_sym_orders=site_sym_orders,
    )
    if skip_disorder and disorder is not None:
        raise ParseError(f"{block.name} has disorder")
    asym_types = np.array(asym_types, dtype=np.uint64)

    symop_rotations, symop_translations = rot, trans
    if reduce_cell:
        cell = lll_reduce(cell)
        if wrap_coords:
            motif = np.mod(motif @ np.linalg.inv(cell), 1) @ cell
        # Original symops not correct
        symop_rotations, symop_translations = None, None

    return PeriodicSet(
        motif=motif,
        cell=cell,
        name=block.name,
        asym_unit=asym_inds,
        multiplicities=wyc_muls,
        types=asym_types,
        labels=asym_labels,
        disorder=disorder,
        symop_rotations=symop_rotations,
        symop_translations=symop_translations
    )


def periodicset_from_ccdc_entry(
    entry,
    remove_hydrogens: bool = False,
    skip_disorder: bool = False,
    missing_coords: str = "warn",
    eq_site_tol: float = 1e-3,
    reduce_cell: bool = False,
    wrap_coords: bool = True,
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
    wrap_coords : bool, default True
        Wrap motif points inside the unit cell.
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

    if not entry.has_3d_structure:
        raise ParseError(f"{entry.identifier} has no 3D structure")

    if skip_disorder and entry.has_disorder:
        raise ParseError(f"{entry.identifier} has disorder")

    return periodicset_from_ccdc_crystal(
        entry.crystal,
        remove_hydrogens=remove_hydrogens,
        skip_disorder=skip_disorder,
        missing_coords=missing_coords,
        eq_site_tol=eq_site_tol,
        reduce_cell=reduce_cell,
        wrap_coords=wrap_coords,
        heaviest_component=heaviest_component,
        molecular_centres=molecular_centres
    )


def periodicset_from_ccdc_crystal(
    crystal,
    remove_hydrogens: bool = False,
    skip_disorder: bool = False,
    missing_coords: str = "warn",
    eq_site_tol: float = 1e-3,
    reduce_cell: bool = False,
    wrap_coords: bool = True,
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
    wrap_coords : bool, default True
        Wrap motif points inside the unit cell.
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
        1. :code:``skip_disorder is True`` and either ccdc reports
        disorder or any site is partially occupied, 2. The crystal has
        no unit cell, 3. :code:``missing_coords == 'skip'`` and any site
        has no coordinates, 4. The motif is empty after removing H,
        sites without coordinates or solvents.
    """

    molecule = crystal.asymmetric_unit_molecule

    # Disorder
    if skip_disorder:
        if crystal.has_disorder:
            raise ParseError(f"{crystal.identifier} has disorder")

    # molecule.atoms builds a fresh list of wrappers over the C++ layer
    # on every access, so it is fetched once per state of the molecule.
    sites_removed = False
    atoms = molecule.atoms
    if remove_hydrogens:
        hydrogens = [a for a in atoms if a.atomic_symbol in ("H", "D")]
        if hydrogens:
            molecule.remove_atoms(hydrogens)
            atoms = molecule.atoms
            sites_removed = True

    missing = [a for a in atoms if a.fractional_coordinates is None]
    if missing:
        if missing_coords == "skip":
            raise ParseError(f"{crystal.identifier} has missing coordinates")
        elif missing_coords == "warn":
            warnings.warn(f"{crystal.identifier} has missing coordinates")
        molecule.remove_atoms(missing)
        sites_removed = True

    if heaviest_component:
        components = molecule.components
        if len(components) > 1:
            molecule = _heaviest_component_ccdc(components)
            sites_removed = True

    # Unit cell
    cellpar = np.array(crystal.cell_lengths + crystal.cell_angles)
    if None in cellpar:
        raise ParseError(f"{crystal.identifier} has no unit cell")
    cell = cellpar_to_cell(cellpar)

    # Molecular centres
    if molecular_centres:
        frac_centres = _frac_molecular_centres_ccdc(crystal, eq_site_tol)
        mol_centres = np.matmul(frac_centres, cell)
        return PeriodicSet(mol_centres, cell, name=crystal.identifier)

    # Symmetry operations
    sitesym = crystal.symmetry_operators
    if not sitesym:
        warnings.warn(f"{crystal.identifier} has no symmetry data, defaulting to P1")
        sitesym = ["x,y,z"]
    rot, trans = _parse_sitesyms(sitesym)

    asym_atoms = list(molecule.atoms)

    # crystal.disorder is documented to be None when the structure has
    # no disorder, which should agree with has_disorder but is not worth
    # relying on. The model is read once, into plain data that both the
    # restore step and the code assignment below work from.
    disorder_model = crystal.disorder if crystal.has_disorder else None
    model_groups = (
        [] if disorder_model is None
        else _ccdc_disorder_snapshot(disorder_model)
    )

    # ccdc's asymmetric unit describes one disorder configuration
    restored, covered_by_symmetry = _restore_disorder_sites_ccdc(
        model_groups,
        asym_atoms,
        cell,
        rot,
        trans,
        eq_site_tol,
        remove_hydrogens=remove_hydrogens,
        heaviest_component=heaviest_component,
    )
    # Which configuration each site belongs to. The asymmetric unit ccdc
    # built is one self-consistent configuration of the crystal, so the
    # sites read from it are simultaneously present, while the restored
    # ones are its alternatives. The disorder heuristics can use that
    # where occupancies distinguish nothing - every site of a suppressed
    # 0.5:0.5 model carries the same occupancy.
    majority_config = [True] * len(asym_atoms) + [False] * len(restored)
    asym_atoms.extend(restored)

    asym_unit = np.array([tuple(a.fractional_coordinates) for a in asym_atoms], dtype=np.float64)

    if asym_unit.shape[0] == 0:
        raise ParseError(f"{crystal.identifier} has no coordinates")

    if wrap_coords:
        asym_unit = np.mod(asym_unit, 1)

    asym_types = [a.atomic_number for a in asym_atoms]
    asym_labels = [_ccdc_label(a.label) for a in asym_atoms]
    # None where the structure recorded no occupancy, kept apart from a
    # recorded 1 so the disorder model below can fill it in.
    asym_occs = [_ccdc_float(a.occupancy) for a in asym_atoms]

    # crystal.has_disorder is ccdc's own judgement, and a structure can
    # carry a partial occupancy it does not call disorder - an unmodelled
    # half occupied solvent, say. The gemmi backend takes any occupancy
    # below one as disorder, so without this the same structure survives
    # skip_disorder under one reader and not the other.
    if skip_disorder and any(occ is not None and occ < 1 for occ in asym_occs):
        raise ParseError(f"{crystal.identifier} has disorder")

    # Apply symmetries to asymmetric unit. Without uniquify_sites,
    # _expand_asym_unit keeps every site, so the per-site columns built
    # above stay aligned with asym_unit as they are.
    frac_motif, invs, _ = _expand_asym_unit(
        asym_unit, rot, trans, eq_site_tol, wrap_coords=wrap_coords
    )
    _, wyc_muls = np.unique(invs, return_counts=True)
    wyc_muls = np.array(wyc_muls, dtype=np.uint64)
    asym_inds = np.zeros_like(wyc_muls, dtype=np.uint64)
    asym_inds[1:] = np.cumsum(wyc_muls, dtype=np.uint64)[:-1]
    motif = np.matmul(frac_motif, cell)

    # None (rather than ".") so that atoms the ccdc disorder model does
    # not mention still reach the occupancy heuristics; that model can be
    # incomplete, so silence about an atom is not a statement that it is
    # ordered.
    assemblies = [None] * len(asym_unit)
    groups = [None] * len(asym_unit)

    # A suppressed disorder model does not describe the structure's
    # assemblies. Per the ccdc documentation it holds a single assembly
    # of two groups, the first containing every minority occupancy atom
    # in the crystal and the second none of them, so reading its ids as
    # assembly and group codes would declare unrelated regions of
    # disorder to be alternatives of one another. Leave it to the
    # occupancy heuristics.
    if disorder_model is not None and not disorder_model.is_suppressed:
        match_site = _ccdc_site_matcher(
            asym_unit, asym_labels, asym_types, eq_site_tol
        )
        saw_atoms, filled_groups, empty_groups = _assign_ccdc_disorder_codes(
            model_groups, match_site, assemblies, groups, asym_occs,
            covered_by_symmetry,
        )

        # Some atoms of the model not matching is normal - they are the
        # removed ones, or symmetry copies completing a molecule outside
        # the asymmetric unit. None of them matching is not, and means
        # the codes have been lost rather than applied.
        if saw_atoms and not filled_groups:
            warnings.warn(
                f"{crystal.identifier} has a disorder model whose atoms could "
                "not be matched to any site; inferring disorder from "
                "occupancies instead"
            )
        elif empty_groups and filled_groups and not sites_removed:
            # A whole group having no site left, while its siblings do,
            # means one alternative of an assembly is missing from the
            # asymmetric unit rather than merely unmatched. Nothing was
            # removed here, so the sites never arrived - most likely the
            # majority component only.
            warnings.warn(
                f"{crystal.identifier} has {empty_groups} disorder group(s) "
                "with no corresponding site in the asymmetric unit; those "
                "alternatives are missing from the motif"
            )

    asym_occs = [1.0 if occ is None else occ for occ in asym_occs]

    disorder = _disorder_assemblies(
        asym_unit,
        wyc_muls,
        len(rot),
        cell,
        asym_types,
        assemblies,
        groups,
        asym_occs,
        majority_config=majority_config,
    )
    asym_types = np.array(asym_types, dtype=np.uint64)

    if reduce_cell:
        cell = lll_reduce(cell)
        if wrap_coords:
            motif = np.mod(motif @ np.linalg.inv(cell), 1) @ cell
        rot, trans = None, None

    return PeriodicSet(
        motif=motif,
        cell=cell,
        name=crystal.identifier,
        asym_unit=asym_inds,
        multiplicities=wyc_muls,
        types=asym_types,
        labels=asym_labels,
        disorder=disorder,
        symop_rotations=rot,
        symop_translations=trans,
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


def _keep_sites(kept: UIntArray, n_sites: int, *columns):
    """Removes items with index not in kept from columns."""
    if len(kept) == n_sites:
        return columns
    return tuple(
        None if col is None else [col[int(i)] for i in kept] for col in columns
    )


def _expand_asym_unit(
    asym_unit: FloatArray,
    rotations: FloatArray,
    translations: FloatArray,
    tol: float,
    uniquify_sites: bool = False,
    wrap_coords: bool = True,
) -> Tuple[FloatArray, UIntArray, UIntArray]:
    """Expand the asymmetric unit by applying symmetries given by
    ``rotations`` and ``translations`` and removing invariant points.

    Returns ``(frac_motif, inverses, kept)`` where ``inverses`` gives the
    asymmetric unit index each motif point came from and ``kept`` lists
    the asymmetric unit indices which survived. ``kept`` is everything
    unless ``uniquify_sites`` dropped a site for being symmetrically
    equivalent to an earlier one, but it is always returned, so a caller
    can index its own per site data by it unconditionally.

    Points are wrapped into the unit cell unless ``wrap_coords`` is
    False; duplicates are found modulo 1 either way.
    """

    asym_unit = asym_unit.astype(np.float64, copy=False)
    rotations = rotations.astype(np.float64, copy=False)
    translations = translations.astype(np.float64, copy=False)
    expanded_sites = _expand_sites(asym_unit, rotations, translations, wrap_coords)
    frac_motif, invs = _reduce_expanded_sites(expanded_sites, tol)

    if uniquify_sites:
        if not all(_unique_sites(frac_motif, tol)):
            return _reduce_expanded_equiv_sites(expanded_sites, tol)

    kept = np.arange(asym_unit.shape[0], dtype=np.uint64)
    return frac_motif, invs, kept


@numba.njit(cache=True)
def _expand_sites(
    asym_unit: FloatArray,
    rotations: FloatArray,
    translations: FloatArray,
    wrap: bool = True,
) -> FloatArray:
    """Expand the asymmetric unit by applying ``rotations`` and
    ``translations``, without yet removing points duplicated because
    they are invariant under a symmetry. Returns a 3D array shape
    (#points, #syms, dims). Points are wrapped into the unit cell
    unless ``wrap`` is False.
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
    if wrap:
        return np.mod(expanded_sites, 1)
    return expanded_sites


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
) -> Tuple[FloatArray, UIntArray, UIntArray]:
    """Reduce the asymmetric unit after being expended by symmetries by
    removing invariant points. This version also removes symmetrically
    equivalent sites in the asymmetric unit.

    Returns ``(frac_motif, inverses, kept)`` like
    :func:`_expand_asym_unit`, ``kept`` being the asymmetric unit indices
    which were not dropped for duplicating an earlier site.
    """

    sites = expanded_sites[0]
    unique_inds = _unique_sites(sites, tol)
    frac_motif = sites[unique_inds]
    inverses = [0] * len(frac_motif)
    # Site 0 is the one every other site is compared against, so it is
    # always kept and has to be in the list.
    asym_inds = [0]

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
            asym_inds.append(i)

    return (
        frac_motif,
        np.array(inverses, dtype=np.uint64),
        np.array(asym_inds, dtype=np.uint64),
    )


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


class _RawAssembly(NamedTuple):
    """Intermediate assembly, before occupancies are normalised."""

    groups: List[List[int]]
    is_substitutional: bool
    source: str


# _atom_site_disorder_assembly/_atom_site_disorder_group take the CIF
# null values "." (inapplicable, i.e. the atom takes no part in disorder)
# and "?" (unknown). The two are not interchangeable: "." is a statement
# that the atom is ordered, "?" is the absence of a statement.

def _is_code(value) -> bool:
    """True if a value is not . or ? (CIF null/unknown)."""
    return value != "." and value != "?"


def _normalise_disorder_codes(values, n_sites: int) -> list:
    """Normalise a column of disorder assembly or group codes to a list
    of either string codes or the ./? sentinels."""

    if values is None:
        return ["?"] * n_sites

    normalised = []
    for value in values:
        if value is None:
            normalised.append("?")
            continue
        value = str(value).strip().strip("'\"")
        if value == ".":
            normalised.append(".")
        elif value in ("?", ""):
            normalised.append("?")
        else:
            normalised.append(value)
    return normalised


def _declares_site_symmetry(site_sym_orders) -> bool:
    """True if the file gave a usable site symmetry order column."""

    if site_sym_orders is None:
        return False
    values = np.asarray(site_sym_orders, dtype=np.float64)
    return bool(values.size) and bool(np.any(np.isfinite(values) & (values >= 1)))


def _is_special_position_occupancy(
    occ: float, multiplicity: float, n_symmetries: int
) -> bool:
    """True if ``occ`` looks like it accounts for the site multiplicity
    of an atom on a special position rather than indicating disorder.

    An atom lying on a symmetry element is generated fewer times than the
    general multiplicity. Under the CIF core dictionary its occupancy is
    still 1 and the multiplicity is recorded separately, but files which
    omit the multiplicity commonly fold it into the occupancy instead,
    writing 0.5 for an atom on an inversion centre or 1/3 on a threefold
    axis despite the atom being fully ordered.

    Only used when the file does not state the site symmetry order; see
    ``_disorder_assemblies``.
    """

    if not n_symmetries or multiplicity >= n_symmetries:
        return False
    expected = multiplicity / n_symmetries
    return abs(occ - expected) <= SPECIAL_POSITION_OCC_TOL


def _group_indices_by_occupancy(occs: FloatArray, inds: List[int]) -> List[List[int]]:
    """Partition ``inds`` into groups of sites sharing an occupancy."""

    if len(inds) == 1:
        return [list(inds)]
    occ_values = occs[inds]
    # NOTE: pdist expects one row per observation, so the occupancies
    # must form a column. Passing occ_values[None, :] yields a single
    # n-dimensional observation and hence no pairwise distances at all.
    close = pdist(occ_values[:, None], metric="chebyshev") <= OCCUPANCY_TOL
    return [[inds[j] for j in grp] for grp in _collapse_into_groups(close)]


def _overlapping_clusters(
    asym_unit: FloatArray, cell: FloatArray, inds: List[int]
) -> Tuple[List[Tuple[List[int], float]], List[int]]:
    """Cluster ``inds`` into sets of mutually overlapping sites.

    Returns ``(clusters, leftover)`` where each cluster is a
    ``(indices, max_pairwise_distance)`` pair and ``leftover`` holds
    sites belonging to no cluster.

    ``_collapse_into_groups`` is single linkage, so a chain of sites each
    just inside ``SUBS_DISORDER_TOL`` of the next would merge into one
    arbitrarily large cluster. Clusters are therefore re-checked under
    complete linkage and rejected wholesale if they fail, rather than
    split at a boundary the data does not justify.
    """

    if len(inds) < 2:
        return [], list(inds)

    m = len(inds)
    dists = _pdist_pbc(np.mod(asym_unit[inds], 1), cell)

    def _pair_dist(i: int, j: int) -> float:
        return dists[m * i + j - ((i + 2) * (i + 1)) // 2]

    clusters, leftover = [], []
    for grp in _collapse_into_groups(dists <= SUBS_DISORDER_TOL):

        if len(grp) == 1:
            leftover.append(inds[grp[0]])
            continue

        max_d = max(_pair_dist(i, j) for i, j in itertools.combinations(grp, 2))
        if max_d > SUBS_DISORDER_TOL:
            # Single linkage chained sites which are not all mutually
            # close; treat them as unclustered.
            leftover.extend(inds[j] for j in grp)
        else:
            clusters.append(([inds[j] for j in grp], max_d))

    return clusters, leftover


def _sites_sharing_a_position_with_another_element(
    asym_unit: FloatArray, cell: FloatArray, types: list, inds: List[int]
) -> set:
    """Which of ``inds`` sit on the same position as another of ``inds``
    of a different element.

    A site multiplicity cannot account for that: a symmetry element puts
    an atom on top of *itself*, never a Co on top of an Ni. Such a site
    is substitutional disorder whatever its occupancy looks like.
    """

    if types is None or len(inds) < 2:
        return set()

    dists = _pdist_pbc(np.mod(asym_unit[inds], 1), cell)
    close = np.flatnonzero(dists <= EXACT_OVERLAP_TOL)
    row, col = np.triu_indices(len(inds), 1)
    shared = set()
    for a, b in zip(row[close], col[close]):
        if types[inds[a]] != types[inds[b]]:
            shared.add(inds[a])
            shared.add(inds[b])
    return shared


def _classify_overlapping_cluster(
    cluster: List[int], max_d: float, types: list, occs: FloatArray
) -> Optional[_RawAssembly]:
    """Classify a cluster of mutually overlapping partially occupied
    sites as substitutional or positional disorder.

    Substitutional disorder means alternative *chemical identities* on
    one site, so it requires the sites to be of differing element, or to
    be refined at essentially the same position. Overlapping sites of the
    same element pulled apart by a real distance are alternative
    positions, i.e. positional disorder.
    """

    occ_sum = float(np.sum(occs[cluster]))
    if occ_sum > 1 + OCCUPANCY_ROUNDING_TOL:
        # Cannot be alternatives on one site. The tolerance is rounding
        # in the deposited occupancies rather than OCCUPANCY_TOL: CSD
        # LEYQUD gives its mixed Mo/V sites as 0.88 and 0.13, and
        # rejecting those leaves the two atoms unpaired on one position,
        # both present at once, which puts a zero distance into the
        # structure's PDD.
        return None

    cluster_types = [types[i] for i in cluster]
    distinct_types = len(set(cluster_types)) == len(cluster_types)

    substitutional = distinct_types or max_d <= EXACT_OVERLAP_TOL
    return _RawAssembly(
        groups=[[i] for i in cluster],
        is_substitutional=substitutional,
        source="overlapping-sites",
    )


def _disorder_assemblies(
    asym_unit: FloatArray,
    multiplicities,
    n_symmetries: int,
    cell: FloatArray,
    types: list,
    assemblies: list,
    groups: list,
    asym_occs: list,
    site_sym_orders=None,
    majority_config=None,
) -> Optional[List[DisorderAssembly]]:
    """Determine the disorder assemblies of an asymmetric unit.

    An *assembly* is a cluster of atoms whose disorder is coupled; the
    *groups* of an assembly are its mutually exclusive alternative
    configurations. Explicit ``_atom_site_disorder_assembly`` and
    ``_atom_site_disorder_group`` codes are authoritative and are always
    honoured. Sites they do not cover are assigned heuristically from
    occupancies and positions.

    Parameters
    ----------
    asym_unit :
        Fractional coordinates of the asymmetric unit.
    multiplicities :
        Number of distinct symmetry images generated by each site.
    n_symmetries :
        Total number of symmetry operations, i.e. the general
        multiplicity. Used to recognise occupancies which account for a
        special position rather than indicating disorder.
    cell :
        Unit cell, for measuring distances between sites.
    types :
        Atomic numbers of the sites, or None. Used to tell substitutional
        from positional disorder.
    assemblies, groups :
        Disorder assembly and group codes, one per site.
    asym_occs :
        Fractional occupancies, one per site.
    site_sym_orders :
        Values of ``_atom_site_site_symmetry_order`` (or the older
        ``_atom_site_symmetry_multiplicity``) if the file gives them,
        otherwise None. Their presence means the file records site
        multiplicity separately from occupancy, so a fractional
        occupancy can be taken at face value as disorder.
    majority_config :
        One bool per site, True for sites belonging to one
        self-consistent configuration of the crystal (ccdc's asymmetric
        unit) and False for the alternatives restored from its disorder
        model, or None when the source does not distinguish them. Lets
        the heuristics split sites into groups when the occupancies
        cannot, e.g. a suppressed model whose alternatives are all 0.5.

    Returns
    -------
    A list of :class:`DisorderAssembly`, or None if the structure has no
    disorder.
    """

    n_sites = len(asym_occs)
    occs = np.asarray(asym_occs, dtype=np.float64)
    mults = np.asarray(multiplicities, dtype=np.float64)
    assemblies = _normalise_disorder_codes(assemblies, n_sites)
    groups = _normalise_disorder_codes(groups, n_sites)

    raw: List[_RawAssembly] = []

    # 1. Explicit disorder codes. These are a statement by the author
    # about which sites are alternatives to which, and are honoured
    # regardless of occupancy: a group can legitimately contain a site
    # refined at full occupancy, and a CIF can give groups without giving
    # occupancies at all.
    explicit: dict = collections.OrderedDict()
    assembly_only: dict = collections.OrderedDict()
    unassigned: List[int] = []
    any_group_codes = any(_is_code(g) for g in groups)

    for i in range(n_sites):
        asm, grp = assemblies[i], groups[i]

        if _is_code(grp):
            # A group code with no assembly code means the single
            # implicit assembly; _atom_site_disorder_assembly is only
            # required when a structure has more than one assembly.
            key = asm if _is_code(asm) else ("implicit",)
            explicit.setdefault(key, collections.OrderedDict()).setdefault(
                grp, []
            ).append(i)

        elif _is_code(asm):
            assembly_only.setdefault(asm, []).append(i)

        elif (
            grp == "." and any_group_codes and occs[i] >= 1 - OCCUPANCY_TOL
        ):
            # Explicitly excluded from disorder by a CIF which does label
            # its disordered sites, so do not second guess it - but only
            # while the site is fully occupied. A partial occupancy states
            # disorder on its own, and files routinely give group codes for
            # one disordered fragment while leaving a half occupied solvent
            # site marked '.'. COD 1100728 is one: its C50/Cl1/Cl2 fragment
            # carries group codes, and `Cl5 ... 0.50 1` does not, so
            # skipping it here hid the structure's disorder completely.
            # Such sites must reach the occupancy heuristics below.
            continue

        else:
            unassigned.append(i)

    for grps in explicit.values():
        grp_inds = list(grps.values())
        if len(grp_inds) < 2:
            # One group is not a set of alternatives, but the author still
            # labelled these sites as disordered, and at partial occupancy
            # that is a statement in itself: the group is either present or
            # it is not, with the implicit vacancy carrying the rest. That
            # is an assembly with one alternative, and keeping it as one
            # matters - handing these sites to the heuristics instead
            # discards the assembly code, after which sites the file put in
            # *different* assemblies are fused for sharing an occupancy.
            # Only fully occupied sites have nothing to model here.
            if occs[grp_inds[0][0]] < 1 - OCCUPANCY_TOL:
                raw.append(_RawAssembly(grp_inds, False, "cif"))
            else:
                unassigned.extend(itertools.chain.from_iterable(grp_inds))
            continue
        substitutional = _explicit_is_substitutional(
            grp_inds, asym_unit, cell, types, occs
        )
        raw.append(_RawAssembly(grp_inds, substitutional, "cif"))

    # 2. Sites given an assembly but no group. The assembly says they are
    # coupled but not which alternative each belongs to; distinct
    # occupancies are the only available signal.
    for inds in assembly_only.values():
        grp_inds = _group_indices_by_occupancy(occs, inds)
        if len(grp_inds) < 2:
            # One occupancy across the whole assembly, so no alternatives
            # to tell apart - but as above, an assembly code plus partial
            # occupancy is present-or-absent, not silence.
            if occs[grp_inds[0][0]] < 1 - OCCUPANCY_TOL:
                raw.append(_RawAssembly(grp_inds, False, "cif-assembly-only"))
            else:
                unassigned.extend(inds)
            continue
        raw.append(_RawAssembly(grp_inds, False, "cif-assembly-only"))

    # 3. Heuristics, for sites carrying no usable disorder codes. Only
    # partial occupancy indicates disorder here.
    candidates = [i for i in unassigned if occs[i] < 1 - OCCUPANCY_TOL]

    # A file which states site symmetry order is keeping multiplicity out
    # of the occupancy column, so its occupancies mean what they say.
    # Only guess at special positions when it does not.
    if not _declares_site_symmetry(site_sym_orders):
        # Except where two elements share one position, which no site
        # multiplicity explains. CSD KESFOG is one: its Co/Ni site is
        # half occupied and of multiplicity 8 out of 16, so the special
        # position reading drops it and leaves both atoms in the motif,
        # on top of one another and both always present.
        shared = _sites_sharing_a_position_with_another_element(
            asym_unit, cell, types, candidates
        )
        candidates = [
            i
            for i in candidates
            if i in shared
            or not _is_special_position_occupancy(occs[i], mults[i], n_symmetries)
        ]

    if candidates:
        raw.extend(_heuristic_assemblies(
            asym_unit, cell, types, occs, candidates, majority_config
        ))

    # 4. Normalise occupancies and build the result.
    out = []
    for asm in raw:
        occ_sum = sum(float(occs[grp[0]]) for grp in asm.groups)
        norm = max(1.0, occ_sum)
        out.append(
            DisorderAssembly(
                groups=[
                    DisorderGroup(indices=grp, occupancy=float(occs[grp[0]]) / norm)
                    for grp in asm.groups
                ],
                is_substitutional=asm.is_substitutional,
                source=asm.source,
            )
        )

    return out if out else None


def _explicit_is_substitutional(
    grp_inds: List[List[int]],
    asym_unit: FloatArray,
    cell: FloatArray,
    types: list,
    occs: np.ndarray
) -> bool:
    """Decide whether an assembly read from explicit CIF codes describes
    substitutional disorder. The CIF disorder tags do not distinguish the
    two, so this applies the same geometric test used for inferred
    assemblies: every group must be a single site, and all those sites
    must share one position.
    """

    if any(len(grp) != 1 for grp in grp_inds):
        return False
    cluster = [grp[0] for grp in grp_inds]
    clusters, _ = _overlapping_clusters(asym_unit, cell, cluster)
    if len(clusters) != 1 or len(clusters[0][0]) != len(cluster):
        return False
    asm = _classify_overlapping_cluster(cluster, clusters[0][1], types, occs)
    return asm is not None and asm.is_substitutional


def _connected_components(
    linked: npt.NDArray[np.bool_], m: int
) -> List[List[int]]:
    """Connected components of ``m`` items whose pairwise adjacency is
    given as a condensed boolean matrix (the layout of ``pdist``).

    Unlike :func:`_collapse_into_groups <.calculate._collapse_into_groups>`
    this takes the transitive closure: that function assigns each item in
    one pass and never revisits, so a chain a-b-c where b is claimed
    first leaves c stranded. Region clustering needs the closure - a site
    reaches its region through any neighbour.
    """

    parent = list(range(m))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    row, col = np.triu_indices(m, 1)
    for i, j in zip(row[linked], col[linked]):
        ri, rj = find(int(i)), find(int(j))
        if ri != rj:
            parent[rj] = ri

    comps = collections.defaultdict(list)
    for i in range(m):
        comps[find(i)].append(i)
    return list(comps.values())


def _majority_minority_assemblies(
    asym_unit: FloatArray,
    cell: FloatArray,
    occs: FloatArray,
    inds: List[int],
    majority: List[bool],
) -> Tuple[List[_RawAssembly], List[int]]:
    """Pair up disordered sites by which configuration of the crystal
    each belongs to.

    ccdc's asymmetric unit represents one self-consistent configuration:
    the sites read from it are simultaneously present, never alternatives
    of one another, while the minority alternatives restored from the
    disorder model are what replaces them. Within one *region* of
    disorder that is exactly the grouping wanted - and it is the only
    signal that resolves a suppressed 0.5:0.5 model, where every site
    carries the same occupancy and no occupancy pairing can say which
    sites are alternatives of which.

    Sites are first split into regions by distance: an alternative
    position sits within a bond length or so of the sites it replaces,
    while independent regions of disorder are separated by van der Waals
    contacts. Each region containing sites of both configurations
    becomes one assembly, its groups being the majority sites and the
    minority sites split by occupancy, provided the group occupancies
    are consistent with being mutually exclusive. Where several regions
    were fused by proximity (group occupancies sum past one),
    complementary majority/minority pairs are matched within the region
    instead, and anything unresolved is returned in ``leftover`` for the
    occupancy heuristics.

    Returns ``(assemblies, leftover)``.
    """

    if len(inds) < 2 or all(majority[i] for i in inds):
        return [], list(inds)

    raw: List[_RawAssembly] = []
    leftover: List[int] = []
    dists = _pdist_pbc(np.mod(asym_unit[inds], 1), cell)

    for comp in _connected_components(dists <= DISORDER_REGION_TOL, len(inds)):
        comp = [inds[j] for j in comp]
        maj = [i for i in comp if majority[i]]
        mino = [i for i in comp if not majority[i]]
        if not maj or not mino:
            # A region entirely of one configuration has nothing to pair
            # against here; the occupancy heuristics may still place it.
            leftover.extend(comp)
            continue

        maj_groups = _group_indices_by_occupancy(occs, maj)
        min_groups = _group_indices_by_occupancy(occs, mino)
        groups = maj_groups + min_groups
        occ_sum = sum(float(occs[grp[0]]) for grp in groups)

        if occ_sum <= 1 + OCCUPANCY_ROUNDING_TOL:
            raw.append(_RawAssembly(groups, False, "majority-minority"))
            continue

        # Group occupancies past one mean proximity fused several
        # regions. Salvage the unambiguous part: a majority group and a
        # minority group whose occupancies complement to one are a
        # two-way assembly whichever region boundary was missed.
        unused = [True] * len(min_groups)
        for maj_grp in maj_groups:
            for k, min_grp in enumerate(min_groups):
                pair_occ = float(occs[maj_grp[0]]) + float(occs[min_grp[0]])
                if unused[k] and abs(pair_occ - 1) <= OCCUPANCY_ROUNDING_TOL:
                    unused[k] = False
                    raw.append(
                        _RawAssembly([maj_grp, min_grp], False, "majority-minority")
                    )
                    break
            else:
                leftover.extend(maj_grp)
        for k, min_grp in enumerate(min_groups):
            if unused[k]:
                leftover.extend(min_grp)

    return raw, leftover


def _heuristic_assemblies(
    asym_unit: FloatArray,
    cell: FloatArray,
    types: list,
    occs: FloatArray,
    candidates: List[int],
    majority_config=None,
) -> List[_RawAssembly]:
    """Infer assemblies for partially occupied sites which carry no
    disorder codes, from their positions and occupancies, and - when the
    source distinguishes them - which configuration of the crystal each
    site belongs to."""

    raw: List[_RawAssembly] = []

    # Sites sharing a position are alternatives for that position.
    clusters, leftover = _overlapping_clusters(asym_unit, cell, candidates)
    for cluster, max_d in clusters:
        asm = _classify_overlapping_cluster(cluster, max_d, types, occs)
        if asm is None:
            leftover.extend(cluster)
        else:
            raw.append(asm)

    if not leftover:
        return raw

    # Sites split by which configuration they belong to
    if majority_config is not None:
        mm_assemblies, leftover = _majority_minority_assemblies(
            asym_unit, cell, occs, sorted(leftover), majority_config
        )
        raw.extend(mm_assemblies)

    if not leftover:
        return raw

    # Remaining sites are grouped by occupancy, then occupancies summing
    # to one are taken to be alternatives of one assembly.
    occ_grps = _group_indices_by_occupancy(occs, sorted(leftover))
    grp_occs = np.array([occs[grp[0]] for grp in occ_grps])

    usable = [i for i, occ in enumerate(grp_occs) if occ <= 1 + OCCUPANCY_TOL]
    if len(usable) != len(occ_grps):
        warnings.warn(
            f"{len(occ_grps) - len(usable)} site(s) have occupancy > 1; "
            "ignoring them when assigning disorder"
        )

    consumed = set()
    for combination in _tuples_sum_to_one(grp_occs[usable]):
        combination = [usable[j] for j in combination]
        raw.append(
            _RawAssembly(
                groups=[occ_grps[i] for i in combination],
                is_substitutional=False,
                source="occupancies-sum-to-one",
            )
        )
        consumed.update(combination)

    # Anything left is disorder we can see but cannot pair up: a site
    # partially occupied against a vacancy, or one alternative of a set
    # the occupancies do not resolve. Model it as present-or-absent.
    for i in usable:
        if i in consumed:
            continue
        raw.append(
            _RawAssembly(
                groups=[occ_grps[i]],
                is_substitutional=False,
                source="partial-occupancy",
            )
        )

    return raw


@numba.njit(cache=True)
def _pdist_pbc(points, cell):
    """Condensed matrix of distances between the fractional coordinates
    ``points``, which are expected to be wrapped into the unit cell.

    Each distance is measured to the image given by wrapping the signed
    fractional difference into [-1/2, 1/2). That is not always the
    nearest image in a strongly skewed cell, but it is exact for the
    near-coincident pairs this is used to find, and it is an upper bound
    otherwise, so a pair is never reported closer than it really is.
    """

    m = len(points)
    out = np.empty((int(m * (m - 1) / 2),), dtype=np.float64)
    ind = 0
    for i in range(m):
        for j in range(i + 1, m):
            # Wrap the SIGNED difference into [-1/2, 1/2). Taking the
            # absolute value of each fractional component first and then
            # applying the cell only gives the right length when the cell
            # is orthogonal: it turns a difference of a - b into a + b,
            # which is a different vector whenever the axes are not
            # perpendicular.
            diffs = points[i] - points[j]
            diffs = diffs - np.floor(diffs + 0.5)
            diff_v = diffs @ cell
            out[ind] = np.sum(diff_v**2)
            ind += 1
    return np.sqrt(out)


def _tuples_sum_to_one(occupancies) -> List[List[int]]:
    """Given a vector ``occupancies`` of numbers between 0 and 1, return
    disjoint groups of two or more indices whose occupancies sum to one.

    Each index is used at most once, and smaller combinations are found
    first, so that a plain pair is preferred over a larger set. The
    result can still be ambiguous where several subsets sum to one
    (0.3, 0.7, 0.1, 0.2 admits both ``{0.3, 0.7}`` and
    ``{0.7, 0.1, 0.2}``); the pair is returned.
    """

    occs = np.asarray(occupancies, dtype=np.float64)
    n = len(occs)
    if n < 2:
        return []

    if n > MAX_OCCUPANCY_SUBSET_SEARCH:
        warnings.warn(
            f"{n} distinct occupancies is too many to pair up by subset sum; "
            "leaving these sites unassembled"
        )
        return []

    # Search largest first so the running total exceeds one, and pruning
    # takes effect, as early as possible.
    order = np.argsort(-occs, kind="stable")
    sorted_occs = occs[order]
    used: set = set()

    def _find(size: int) -> Optional[List[int]]:
        """Find one unused combination of exactly ``size`` indices whose
        occupancies sum to one."""

        def helper(start: int, chosen: List[int], total: float):
            if len(chosen) == size:
                return list(chosen) if abs(total - 1.0) <= OCCUPANCY_TOL else None
            if total > 1.0 + OCCUPANCY_TOL:
                return None
            for i in range(start, n):
                if i in used:
                    continue
                chosen.append(i)
                result = helper(i + 1, chosen, total + sorted_occs[i])
                if result is not None:
                    return result
                chosen.pop()
            return None

        return helper(0, [], 0.0)

    found = []
    for size in range(2, n + 1):
        while True:
            combination = _find(size)
            if combination is None:
                break
            used.update(combination)
            found.append([int(order[i]) for i in combination])

    return found


def _has_disorder(label: str, occupancy) -> bool:
    """Return True if label ends with ? or occupancy is a number < 1."""
    try:
        occupancy = float(occupancy)
    except Exception:
        occupancy = 1
    return (occupancy < 1) or label.endswith("?")


# Hydrogen isotopes appear as their own symbols in neutron structures.
_ISOTOPE_SYMBOLS = {"D": "H", "T": "H"}


def _element_symbol(text: str) -> str:
    """Element symbol at the start of an atom type symbol or label, or
    "" if there is none. Tries the two letter symbol first and falls
    back to one letter, so ``HA`` is hydrogen but ``Ca1`` is calcium
    (and ``CA``, ambiguous, is read as calcium like ase and pymatgen
    do). D and T are read as hydrogen.
    """

    if not text or text in (".", "?"):
        return ""
    match = re.match(r"([A-Za-z])([A-Za-z]?)", text)
    if match is None:
        return ""
    first, second = match.group(1).upper(), match.group(2).lower()
    if second and first + second in ATOMIC_NUMBERS:
        return first + second
    first = _ISOTOPE_SYMBOLS.get(first, first)
    if first in ATOMIC_NUMBERS:
        return first
    return ""


def _atomic_symbols_from_labels(symbols: List[str]) -> List[str]:
    return [_element_symbol(label) for label in symbols]


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


def _ccdc_float(value, default: Optional[float] = None) -> Optional[float]:
    """Coerce a possibly-null ccdc field to float, or ``default``."""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return default if math.isnan(value) else value


def _ccdc_id_str(value, fallback: str) -> str:
    """Stringify the id of a ccdc disorder assembly or group, falling
    back to ``fallback`` when it says nothing.
    """

    if value is None:
        return fallback
    value = str(value).strip()
    return value if value and _is_code(value) else fallback


def _ccdc_label(label) -> str:
    """Name of a ccdc atom, without the ``?`` ccdc suffixes onto the
    label of an atom which is not part of the structure's majority
    configuration. Stripping it gives a site one name whether it was read
    from the asymmetric unit or from the disorder model.
    """

    label = "" if label is None else str(label)
    return label[:-1] if label.endswith("?") else label


class _CcdcModelAtom(NamedTuple):
    """One atom of a ccdc disorder model, read once into plain data.
    ``atom`` is the underlying ccdc Atom, kept so a restored alternative
    can join the asymmetric unit; ``label`` has the ``?`` minority
    marker stripped so a site has one name whichever way it was read.
    """

    atom: object
    label: str
    atomic_number: int
    atomic_symbol: str
    occupancy: Optional[float]
    frac: Optional[tuple]


class _CcdcModelGroup(NamedTuple):
    """One group of a ccdc disorder model, with its ids resolved."""

    assembly_id: str
    group_id: str
    occupancy: Optional[float]
    atoms: List[_CcdcModelAtom]


def _ccdc_disorder_snapshot(disorder_model) -> List[_CcdcModelGroup]:
    """Read a ccdc disorder model into plain data in one traversal.

    Every property access on a ccdc object crosses into its C++ layer
    and builds fresh wrappers, so the model is read exactly once, and
    the two consumers - restoring the absent alternatives, and applying
    the assembly/group codes - work from the same snapshot rather than
    walking the model separately.

    Assembly and group ids are resolved here. An id is not guaranteed
    to be a code: it can be empty or None, and ccdc numbers groups
    within their assembly rather than across the structure. Two
    assemblies both giving nothing for an id would then hand their
    group 1s the same (assembly, group) key downstream, fusing sites
    from independent regions of disorder into one set of alternatives.
    Fall back to the position in the model, which is unique by
    construction.
    """

    groups = []
    for k, asm in enumerate(disorder_model.assemblies):
        asm_id = _ccdc_id_str(asm.id, f"ccdc-{k}")
        for j, grp in enumerate(asm.groups):
            groups.append(
                _CcdcModelGroup(
                    assembly_id=asm_id,
                    group_id=_ccdc_id_str(grp.id, f"ccdc-{j}"),
                    occupancy=_ccdc_float(getattr(grp, "occupancy", None)),
                    atoms=[
                        _CcdcModelAtom(
                            atom=a,
                            label=_ccdc_label(a.label),
                            atomic_number=a.atomic_number,
                            atomic_symbol=a.atomic_symbol,
                            occupancy=_ccdc_float(a.occupancy),
                            frac=(
                                None if a.fractional_coordinates is None
                                else tuple(a.fractional_coordinates)
                            ),
                        )
                        for a in grp.atoms
                    ],
                )
            )
    return groups


def _symmetry_images(
    frac: FloatArray,
    types: FloatArray,
    rotations: FloatArray,
    translations: FloatArray,
) -> Tuple[FloatArray, FloatArray]:
    """Every symmetry image of ``frac``, wrapped into the unit cell, with
    the atomic number of the site each image came from. Images are in
    site-major order: the rows of site ``i`` are ``i*n_syms`` to
    ``(i+1)*n_syms``.
    """

    frac = np.asarray(frac, dtype=np.float64).reshape(-1, 3)
    images = _expand_sites(frac, rotations, translations).reshape(-1, 3)
    return images, np.repeat(types, len(rotations))


def _restore_disorder_sites_ccdc(
    model_groups: List[_CcdcModelGroup],
    asym_atoms: list,
    cell: FloatArray,
    rotations: FloatArray,
    translations: FloatArray,
    tol: float,
    remove_hydrogens: bool = False,
    heaviest_component: bool = False,
) -> Tuple[list, set]:
    """Return the atoms of a ccdc disorder model which are absent from
    ``asym_atoms``, i.e. the minority alternatives ccdc leaves out of
    :attr:`ccdc.crystal.Crystal.asymmetric_unit_molecule`, together with
    the labels of those it left out because a site already read covers
    them.

    The model describes the whole disordered molecule, which can reach
    beyond one asymmetric unit, so some of its atoms are symmetry copies
    of sites already read rather than alternatives to them. Each
    candidate is therefore compared against every symmetry image of the
    accepted sites: a copy restored as a site of its own lands exactly
    on top of an existing motif point, putting a zero into the
    structure's PDD.
    """

    if not model_groups:
        return [], set()

    present = {_ccdc_label(a.label) for a in asym_atoms}
    kept = [a for a in asym_atoms if a.fractional_coordinates is not None]
    n_syms = len(rotations)

    # One row per accepted site (the asymmetric unit, then each restored
    # alternative in turn), alongside a preallocated table of their
    # symmetry images in site-major blocks of n_syms rows - so an
    # image's owning site is its row // n_syms, with no bookkeeping.
    site_types = [a.atomic_number for a in kept]
    site_occs = [_ccdc_float(a.occupancy, 1.0) for a in kept]
    kept_frac = np.mod(
        np.array(
            [tuple(a.fractional_coordinates) for a in kept], dtype=np.float64
        ).reshape(-1, 3),
        1,
    )
    n_model_atoms = sum(len(grp.atoms) for grp in model_groups)
    images = np.empty(((len(kept) + n_model_atoms) * n_syms, 3))
    n_img = len(kept) * n_syms
    images[:n_img] = _expand_sites(kept_frac, rotations, translations).reshape(
        -1, 3
    )

    extra = []
    # Labels of model atoms left out because a site already read covers
    # them, so a group built only of these is accounted for rather than
    # missing.
    covered = set()
    for grp in model_groups:
        for matom in grp.atoms:
            if matom.label in present or matom.frac is None:
                continue
            if remove_hydrogens and matom.atomic_symbol in ("H", "D"):
                continue

            point = np.mod(np.array(matom.frac, dtype=np.float64), 1)
            # Signed minimum-image displacement to every image, shared
            # by all three tests below.
            d = images[:n_img] - point
            d -= np.round(d)
            coincident = np.all(np.abs(d) <= tol, axis=-1)
            owners = np.unique(np.flatnonzero(coincident) // n_syms)

            # A site of the same element in the same place is the same
            # site under another name, not an alternative to it - and
            # not a loss: symmetry expansion regenerates this position
            # from the site it duplicates. Substitutional partners do
            # coincide, so position alone is not enough to call this a
            # duplicate.
            if any(site_types[o] == matom.atomic_number for o in owners):
                covered.add(matom.label)
                continue

            # Sharing a position with sites of other elements is
            # substitution, but only if the occupancies allow it. Where
            # they do not the deposited data contradicts itself - CSD
            # MOQKOS gives a mixed C/N site as two atoms of occupancy 1
            # - and restoring the alternative would put a second point
            # on an occupied position, always present alongside the
            # first.
            if owners.size:
                occ = 1.0 if matom.occupancy is None else matom.occupancy
                total = occ + sum(site_occs[o] for o in owners)
                if total > 1 + OCCUPANCY_ROUNDING_TOL:
                    continue

            if heaviest_component and n_img:
                # Everything but one component has been discarded, and
                # the disorder model still describes the whole
                # asymmetric unit. Only alternatives of a site which
                # survived belong in the motif.
                dists = np.linalg.norm(d @ cell, axis=-1)
                if dists.min() > DISORDER_COMPONENT_TOL:
                    continue

            present.add(matom.label)
            extra.append(matom.atom)
            site_types.append(matom.atomic_number)
            site_occs.append(1.0 if matom.occupancy is None else matom.occupancy)
            # A later alternative can equally be a symmetry copy of this
            # one, so its images join the table.
            images[n_img:n_img + n_syms] = np.mod(
                rotations @ point + translations, 1
            )
            n_img += n_syms

    return extra, covered


def _assign_ccdc_disorder_codes(
    model_groups: List[_CcdcModelGroup],
    match_site: Callable,
    assemblies: list,
    groups: list,
    asym_occs: list,
    covered_by_symmetry: set,
) -> Tuple[bool, int, int]:
    """Write a ccdc disorder model's assembly and group codes onto the
    per-site ``assemblies`` and ``groups`` columns, in place.

    Returns ``(saw_atoms, filled_groups, empty_groups)``: whether the
    model named any atom at all, and how many of its groups did or did
    not land a code on at least one site, for the caller's diagnostics.
    A group whose atoms are all covered by symmetry images of existing
    sites counts as neither - nothing of it is missing from the motif,
    since disorder across a symmetry element is one crystallographic
    site at partial occupancy.
    """

    saw_atoms = False
    filled_groups, empty_groups = 0, 0
    for grp in model_groups:
        matched = covered = False
        for matom in grp.atoms:
            saw_atoms = True
            i = match_site(matom)
            if i is None:
                covered = covered or matom.label in covered_by_symmetry
                continue
            matched = True
            assemblies[i] = grp.assembly_id
            groups[i] = grp.group_id
            # ccdc states the occupancy of an alternative on the group.
            # Only used for sites whose own occupancy was not recorded:
            # a group is otherwise built of atoms which already carry
            # it, and the group value is a summary of theirs rather than
            # a correction to it.
            if asym_occs[i] is None and grp.occupancy is not None:
                asym_occs[i] = grp.occupancy
        if grp.atoms:
            if matched:
                filled_groups += 1
            elif not covered:
                empty_groups += 1
    return saw_atoms, filled_groups, empty_groups


def _ccdc_site_matcher(
    asym_unit: FloatArray,
    asym_labels: List[str],
    asym_types: List[int],
    tol: float,
) -> Callable:
    """Return a function mapping a :class:`_CcdcModelAtom` onto its
    index in the asymmetric unit, or None if it is not one of those
    sites.

    ``atom.index`` cannot be used: it is a position in the molecule the
    disorder model was built from, not in the asymmetric unit, which has
    had hydrogens, sites without coordinates and lighter components taken
    out of it. Indexing by it puts the codes on the wrong sites as soon
    as any of those removals happened.

    Labels are tried first as they identify a site and survive the
    removals, but ccdc does not guarantee they are unique - the same
    label can appear on two components - so ambiguous ones fall back to
    matching on position. A label naming a different element is not that
    site either: CSD KAKMIX has a V atom labelled Co1 alongside the Co
    atom of the same name, and taking the label at its word puts the
    codes of a mixed metal site on whichever it happened to hit.
    """

    label_counts = collections.Counter(asym_labels)
    pos_by_label = {
        lab: i
        for i, lab in enumerate(asym_labels)
        if lab and label_counts[lab] == 1
    }
    wrapped = np.mod(np.asarray(asym_unit, dtype=np.float64), 1)
    types = np.asarray(asym_types, dtype=np.int64)

    def match(matom: _CcdcModelAtom) -> Optional[int]:
        i = pos_by_label.get(matom.label)
        if i is not None and types[i] == matom.atomic_number:
            return i

        if matom.frac is None:
            return None
        point = np.mod(np.array(matom.frac, dtype=np.float64), 1)
        d = wrapped - point
        d -= np.round(d)
        hits = np.flatnonzero(np.all(np.abs(d) <= tol, axis=-1))
        hits = hits[types[hits] == matom.atomic_number]
        if hits.size != 1:
            # Either nothing of the asymmetric unit is there (a symmetry
            # copy completing a molecule, or a site removed above), or
            # several same-element sites are, which positional disorder
            # makes routine. Guessing which of those the model meant
            # would be worse than leaving them to the heuristics.
            return None
        return int(hits[0])

    return match


def _frac_molecular_centres_ccdc(
    crystal, tol: float, min_atoms: int = 2
) -> FloatArray:
    """Return the geometric centres of molecules in the unit cell.
    Expects a ccdc Crystal object and returns fractional coordiantes.
    """
    frac_centres = []
    for comp in crystal.packing(inclusion="AnyAtomIncluded").components:
        if len(comp.atoms) < min_atoms:  # Temporary: delete single atoms
            continue
        coords = [a.fractional_coordinates for a in comp.atoms]
        frac_centres.append([sum(ax) / len(coords) for ax in zip(*coords)])
    
    if not frac_centres:
        raise ValueError("No molecules with > 1 atom found")
    
    frac_centres = np.mod(np.array(frac_centres, dtype=np.float64), 1)
    return frac_centres[_unique_sites(frac_centres, tol)]


def _heaviest_component_ccdc(components):
    """The heaviest of a molecule's components, by occupancy-weighted
    atomic weight. Intended for removing solvents. Takes the value of
    :attr:`ccdc.molecule.Molecule.components` (fetching that property
    builds every component afresh, so the caller passes it in) and
    returns one ccdc Molecule.
    """

    component_weights = []
    for component in components:
        weight = 0
        for a in component.atoms:
            occ = _ccdc_float(a.occupancy, 1.0)
            try:
                weight += float(a.atomic_weight) * occ
            except (TypeError, ValueError):
                pass
        component_weights.append(weight)
    return components[np.argmax(np.array(component_weights))]


def str2float(string):
    """Remove uncertainty brackets from strings and return the float.
    Returns np.nan if given '.' or '?'."""
    try:
        return float(re.sub(r"\(.+\)*", "", string))
    except TypeError:
        # A one element list is the shape a looped tag comes back as;
        # anything else is not a number and must not fall through as None.
        if isinstance(string, list) and len(string) == 1:
            return str2float(string[0])
        raise ParseError(
            f"{string.__class__.__name__} could not be converted to float"
        )
    except ValueError:
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


class _PymatgenSites(NamedTuple):
    """Per (site, species) entry read from a pymatgen Structure."""

    site_inds: List[int]
    frac: FloatArray
    types: List[int]
    occs: List[float]
    labels: Optional[List[str]]
    assemblies: List[str]
    groups: List[str]


def _pymatgen_atomic_number(species) -> int:
    """Atomic number of a pymatgen ``Element`` or ``Species``, or 0, this
    module's code for an unrecognised element, where it has none.
    """

    for obj in (species, getattr(species, "element", None)):
        if obj is None:
            continue
        try:
            return int(obj.Z)
        except Exception:
            continue
    return 0


def _pymatgen_symbol(species) -> str:
    """Element symbol of a pymatgen ``Element`` or ``Species``."""

    symbol = getattr(species, "symbol", None)
    if isinstance(symbol, str) and symbol:
        return symbol
    return str(species)


def _pymatgen_name(structure, name: Optional[str]) -> Optional[str]:
    """Name to give the periodic set.

    A pymatgen Structure has no name of its own, so fall back to any
    identifier its ``properties`` dict carries - the Materials Project
    puts ``material_id`` there - and then to the reduced formula, which
    at least tells one structure from another in a warning message.
    """

    if name is not None:
        return name

    # properties was added to Structure in 2023; older releases have no
    # such attribute.
    properties = getattr(structure, "properties", None)
    if isinstance(properties, dict):
        for key in ("name", "id", "material_id"):
            value = properties.get(key)
            if value:
                return str(value)

    try:
        return structure.composition.reduced_formula
    except Exception:
        return None


def _split_pymatgen_sites(structure) -> _PymatgenSites:
    """Read a pymatgen Structure into one entry per (site, species) pair.

    A pymatgen site holds a ``Composition``, so one site can carry
    several partially occupied species. A periodic set has one type per
    point, so such a site becomes one entry per species, all at the same
    position - which is also how a CIF writes substitutional disorder,
    as several ``_atom_site`` rows sharing coordinates.

    Species sharing a site are alternatives to one another, and that is
    something pymatgen states rather than something to be guessed, so
    they are given a shared disorder assembly code and one group code
    each. A site with a single species gets no codes, leaving it to the
    occupancy heuristics of :func:`_disorder_assemblies`, which can link
    a partially occupied site to its neighbours - a relationship a
    Structure does not record.
    """

    site_inds: List[int] = []
    frac: List[FloatArray] = []
    types: List[int] = []
    occs: List[float] = []
    labels: List[Optional[str]] = []
    symbols: List[str] = []
    assemblies: List[str] = []
    groups: List[str] = []
    any_labels = False

    for site_ind, site in enumerate(structure):
        species = site.species

        # Composition is a mapping of species to occupancy, so .items()
        # covers the ordered and the disordered site in one branch.
        try:
            items = list(species.items())
        except AttributeError:
            items = [(species, 1.0)]

        # PeriodicSite.label defaults to the species string, which is a
        # formula and not a label. Only a label differing from it was
        # given by whoever built the structure. Both label and
        # species_string may be absent on older pymatgen releases.
        default_label = getattr(site, "species_string", None)
        if default_label is None:
            default_label = str(species)
        label = getattr(site, "label", None)
        if label is not None and str(label) != str(default_label):
            label = str(label)
            any_labels = True
        else:
            label = None

        coords = np.asarray(site.frac_coords, dtype=np.float64)
        is_mixed = len(items) > 1

        for grp_ind, (sp, occ) in enumerate(items):
            symbol = _pymatgen_symbol(sp)
            site_inds.append(site_ind)
            frac.append(coords)
            types.append(_pymatgen_atomic_number(sp))
            occs.append(float(occ))
            symbols.append(symbol)
            if is_mixed:
                assemblies.append(f"pmg{site_ind}")
                groups.append(str(grp_ind + 1))
                # One label per species, so the column stays unique.
                labels.append(None if label is None else f"{label}_{symbol}")
            else:
                assemblies.append("?")
                groups.append("?")
                labels.append(label)

    if frac:
        frac_arr = np.array(frac, dtype=np.float64)
    else:
        frac_arr = np.empty((0, 3), dtype=np.float64)

    final_labels = None
    if any_labels:
        # Some sites were labelled and some were not; name the rest after
        # their element, as PeriodicSet.site_labels() would have done,
        # skipping any name already taken so the column stays unique.
        taken = set(lab for lab in labels if lab is not None)
        counts: dict = {}
        final_labels = []
        for lab, symbol in zip(labels, symbols):
            if lab is not None:
                final_labels.append(lab)
                continue
            while True:
                counts[symbol] = counts.get(symbol, 0) + 1
                generated = f"{symbol}{counts[symbol]}"
                if generated not in taken:
                    break
            taken.add(generated)
            final_labels.append(generated)

    return _PymatgenSites(
        site_inds, frac_arr, types, occs, final_labels, assemblies, groups
    )


def _pymatgen_symmetry(structure, symprec: float, name: Optional[str]):
    """Detect the symmetry of a pymatgen Structure with spglib, through
    pymatgen's :class:`SpacegroupAnalyzer`.

    Returns ``(rotations, translations, orbits)``, ``orbits`` being lists
    of the indices of symmetrically equivalent sites, or None if the
    analysis failed. A Structure records no symmetry of its own, so this
    infers it from the geometry and can disagree with the space group the
    structure was built from; ``symprec`` is the tolerance it uses.
    """

    try:
        from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

        sga = SpacegroupAnalyzer(structure, symprec=symprec)
        # get_symmetry_operations and get_symmetrized_structure are
        # preferred over get_symmetry_dataset, which recent versions of
        # spglib return as a dataclass rather than a dict.
        ops = sga.get_symmetry_operations(cartesian=False)
        orbits = [
            list(orbit)
            for orbit in sga.get_symmetrized_structure().equivalent_indices
        ]
    except Exception as e:
        warnings.warn(f"{name} could not be symmetrised ({e!r}), reading it in P1")
        return None

    if not ops or not orbits:
        return None

    rot = np.array(
        [np.asarray(op.rotation_matrix, dtype=np.float64) for op in ops],
        dtype=np.float64,
    )
    trans = np.array(
        [np.asarray(op.translation_vector, dtype=np.float64) for op in ops],
        dtype=np.float64,
    )
    return rot, trans, orbits


def _p1_symmetry() -> Tuple[FloatArray, FloatArray]:
    """The identity operation alone, as rotation and translation arrays."""

    return (
        np.identity(3, dtype=np.float64)[None, :, :],
        np.zeros((1, 3), dtype=np.float64),
    )


def periodicset_from_pymatgen_structure(
    structure,
    remove_hydrogens: bool = False,
    skip_disorder: bool = False,
    missing_coords: str = "warn",
    eq_site_tol: float = 1e-3,
    reduce_cell: bool = False,
    wrap_coords: bool = True,
    name: Optional[str] = None,
    symprec: Optional[float] = None,
) -> PeriodicSet:
    """Convert a :class:`pymatgen.core.structure.Structure` object to a
    :class:`amd.PeriodicSet <.periodicset.PeriodicSet>`.

    A Structure is a P1 listing of a whole unit cell which records
    neither a space group nor an asymmetric unit, so unlike the CIF and
    ccdc readers this one has no symmetry to read. Pass ``symprec`` to
    have it inferred from the geometry with spglib instead.

    Fractional occupancies are read: a site holding several species
    becomes one point per species sharing a position, and both that and
    a site occupied below 1 become disorder assemblies of the periodic
    set.

    Parameters
    ----------
    structure : :class:`pymatgen.core.structure.Structure`
        A pymatgen Structure object representing a crystal.
    remove_hydrogens : bool, optional
        Remove Hydrogens from the crystal.
    skip_disorder : bool
        Do not read structures with any partially occupied site.
    missing_coords : str
        Action to take upon finding missing coordinates. "warn" (default)
        will emit a warning when a structure has missing coordinates, and
        the sites will be removed. "skip" will not read structures with
        missing coords.
    eq_site_tol : float, default 0.001
        Tolerance below which two sites are considered to be at the same
        position.
    reduce_cell : bool, default False
        Reduce the unit cell (Lenstra-Lenstra-Lovasz lattice basis
        reduction).
    wrap_coords : bool, default True
        Wrap motif points inside the unit cell.
    name : str, optional
        Name of the periodic set. A Structure has no name of its own, so
        without this one is taken from ``structure.properties`` if it
        holds a name, id or material_id, and otherwise from the reduced
        formula.
    symprec : float, optional
        If given, the symmetry of the structure is found with spglib
        (through pymatgen's :class:`SpacegroupAnalyzer`) at this
        tolerance, and the resulting periodic set has an asymmetric unit,
        Wyckoff multiplicities and symmetry operations. Without it the
        set is P1, one point of the asymmetric unit per site. The
        symmetry is inferred from the geometry, so it can differ from the
        space group the structure was built from. The motif is then
        rebuilt from the asymmetric unit, which idealises a structure
        only approximately symmetric: points can move by up to about
        ``symprec``, so an experimental structure comes back slightly
        changed.

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
        1. :code:``skip_disorder is True`` and any site is partially
        occupied, 2. :code:``missing_coords == 'skip'`` and any site has
        no coordinates, 3. The motif is empty, either to begin with or
        after removing Hydrogens and sites without coordinates.
    """

    name = _pymatgen_name(structure, name)
    cell = np.array(structure.lattice.matrix, dtype=np.float64)

    if skip_disorder and not structure.is_ordered:
        raise ParseError(f"{name} has disorder")

    sites = _split_pymatgen_sites(structure)
    n_entries = len(sites.types)
    if n_entries == 0:
        raise ParseError(f"{name} has no coordinates")

    keep = np.ones((n_entries,), dtype=np.bool_)

    # Missing coordinates. A pymatgen site always has coordinates, so
    # this guards against NaNs arriving from elsewhere rather than being
    # a common path, but it makes missing_coords mean here what it means
    # in the other converters.
    where_missing = np.isnan(sites.frac).any(axis=-1)
    if np.any(where_missing):
        if missing_coords == "skip":
            raise ParseError(f"{name} has missing coordinates")
        elif missing_coords == "warn":
            warnings.warn(f"{name} has missing coordinates")
        keep &= ~where_missing

    # Hydrogens go by atomic number rather than through
    # Structure.remove_species(["H", "D"]): that depends on which
    # pymatgen version names the isotopes as elements, and cannot remove
    # a hydrogen which is one species of a mixed site.
    if remove_hydrogens:
        keep &= np.array([t != 1 for t in sites.types], dtype=np.bool_)

    kept_entries = [i for i in range(n_entries) if keep[i]]
    if not kept_entries:
        raise ParseError(f"{name} has no coordinates")

    rot, trans, asym_entries = None, None, None

    if symprec is not None:
        found = _pymatgen_symmetry(structure, symprec, name)
        if found is not None:
            detected_rot, detected_trans, orbits = found
            kept_sites = set(sites.site_inds[i] for i in kept_entries)

            # A removed site has to take its whole orbit with it. If it
            # does not, the surviving sites are not the images of the
            # representatives and the asymmetric unit would be wrong.
            reps, usable = [], True
            for orbit in orbits:
                inside = [s for s in orbit if s in kept_sites]
                if not inside:
                    continue
                if len(inside) != len(orbit):
                    usable = False
                    break
                reps.append(orbit[0])

            if usable and reps:
                rep_sites = set(reps)
                asym_entries = [
                    i for i in kept_entries if sites.site_inds[i] in rep_sites
                ]
                rot, trans = detected_rot, detected_trans
            else:
                warnings.warn(
                    f"{name} has symmetrically equivalent sites of which only "
                    "some were read, so it is read in P1"
                )

    used_symmetry = rot is not None
    if not used_symmetry:
        rot, trans = _p1_symmetry()
        asym_entries = kept_entries

    def _expand(entries):
        asym_frac = sites.frac[np.asarray(entries, dtype=np.intp)]
        if wrap_coords:
            asym_frac = np.mod(asym_frac, 1)
        frac_motif, invs, _ = _expand_asym_unit(
            asym_frac, rot, trans, eq_site_tol,
            uniquify_sites=False, wrap_coords=wrap_coords,
        )
        return asym_frac, frac_motif, invs

    asym_frac, frac_motif, invs = _expand(asym_entries)

    # The operations must regenerate exactly the sites the structure
    # listed. Where they do not, spglib described a cell other than the
    # one in hand and the motif would disagree with the input, so fall
    # back to reading every site as its own.
    if used_symmetry and frac_motif.shape[0] != len(kept_entries):
        warnings.warn(
            f"{name}: the symmetry found generates {frac_motif.shape[0]} sites "
            f"where the structure has {len(kept_entries)}, so it is read in P1"
        )
        rot, trans = _p1_symmetry()
        asym_entries = kept_entries
        asym_frac, frac_motif, invs = _expand(asym_entries)

    _, wyc_muls = np.unique(invs, return_counts=True)
    wyc_muls = np.array(wyc_muls, dtype=np.uint64)
    asym_inds = np.zeros_like(wyc_muls, dtype=np.uint64)
    asym_inds[1:] = np.cumsum(wyc_muls, dtype=np.uint64)[:-1]
    motif = np.matmul(frac_motif, cell)

    asym_types = [sites.types[i] for i in asym_entries]
    asym_occs = [sites.occs[i] for i in asym_entries]
    assemblies = [sites.assemblies[i] for i in asym_entries]
    groups = [sites.groups[i] for i in asym_entries]
    asym_labels = None
    if sites.labels is not None:
        asym_labels = [sites.labels[i] for i in asym_entries]

    disorder = _disorder_assemblies(
        asym_frac,
        wyc_muls,
        len(rot),
        cell,
        asym_types,
        assemblies,
        groups,
        asym_occs,
    )
    asym_types = np.array(asym_types, dtype=np.uint64)

    symop_rotations, symop_translations = rot, trans
    if reduce_cell:
        cell = lll_reduce(cell)
        if wrap_coords:
            motif = np.mod(motif @ np.linalg.inv(cell), 1) @ cell
        # The operations are expressed in the basis of the old cell.
        symop_rotations, symop_translations = None, None

    return PeriodicSet(
        motif=motif,
        cell=cell,
        name=name,
        asym_unit=asym_inds,
        multiplicities=wyc_muls,
        types=asym_types,
        labels=asym_labels,
        disorder=disorder,
        symop_rotations=symop_rotations,
        symop_translations=symop_translations,
    )
