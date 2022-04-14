import warnings
from typing import Callable, Iterable, Sequence, Tuple

import numpy as np
import ase.spacegroup.spacegroup    # parse_sitesym
import ase.io.cif

from .periodicset import PeriodicSet
from .utils import cellpar_to_cell


def _warning(message, category, filename, lineno, *args, **kwargs):
    return f'{filename}:{lineno}: {category.__name__}: {message}\n'

warnings.formatwarning = _warning


def _atom_has_disorder(label, occupancy):
    return label.endswith('?') or (np.isscalar(occupancy) and occupancy < 1)

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
    reserved_tags = {
        'motif',
        'cell',
        'name',
        'asymmetric_unit',
        'wyckoff_multiplicities',
        'types',
        'filename',}
    atom_site_fract_tags = [
        '_atom_site_fract_x',
        '_atom_site_fract_y',
        '_atom_site_fract_z',]
    atom_site_cartn_tags = [
        '_atom_site_cartn_x',
        '_atom_site_cartn_y',
        '_atom_site_cartn_z',]
    symop_tags = [
        '_space_group_symop_operation_xyz',
        '_space_group_symop.operation_xyz',
        '_symmetry_equiv_pos_as_xyz',]

    equiv_site_tol = 1e-3

    def __init__(
            self,
            remove_hydrogens=False,
            disorder='skip',
            heaviest_component=False,
            show_warnings=True,
            extract_data=None,
            include_if=None):

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
                    raise ValueError(f'extract_data includes reserved key {key}')

        if include_if:
            for func in include_if:
                if not callable(func):
                    raise ValueError('include_if must be a list of callables')

        self.remove_hydrogens = remove_hydrogens
        self.disorder = disorder
        self.heaviest_component = heaviest_component
        self.extract_data = extract_data
        self.include_if = include_if
        self.show_warnings = show_warnings
        self.current_identifier = None
        self.current_filename = None
        self._generator = []

    def __iter__(self):
        yield from self._generator

    def read_one(self):
        """Read the next (or first) item."""
        return next(iter(self._generator))

    # basically the builtin map, but skips items if the function returned None.
    # The object returned by this function (Iterable of PeriodicSets) is set to
    # self._generator; then iterating over the Reader iterates over
    # self._generator.
    @staticmethod
    def _map(func: Callable, iterable: Iterable) -> Iterable[PeriodicSet]:
        """Iterates over iterable, passing items through parser and
        yielding the result if it is not None.
        """
        
        for item in iterable:
            res = func(item)
            if res is not None:
                yield res

    def _CIFBlock_to_PeriodicSet(self, block) -> PeriodicSet:
        """ase.io.cif.CIFBlock --> PeriodicSet. Returns None for a "bad" set."""

        # skip if structure does not pass checks in include_if
        if self.include_if:
            if not all(check(block) for check in self.include_if):
                return None

        # read name, cell, asym motif and atomic symbols
        self.current_identifier = block.name
        cell = block.get_cell().array
        asym_frac_motif = [block.get(name) for name in _Reader.atom_site_fract_tags]
        if None in asym_frac_motif:
            asym_motif = [block.get(name) for name in _Reader.atom_site_cartn_tags]
            if None in asym_motif:
                if self.show_warnings:
                    warnings.warn(
                        f'Skipping {self.current_identifier} as coordinates were not found')
                return None
            asym_frac_motif = np.array(asym_motif) @ np.linalg.inv(cell)
        asym_frac_motif = np.array(asym_frac_motif).T

        try:
            asym_symbols = block.get_symbols()
        except ase.io.cif.NoStructureData as _:
            asym_symbols = ['Unknown' for _ in range(len(asym_frac_motif))]

        # indices of sites to remove
        remove = []
        if self.remove_hydrogens:
            remove.extend((i for i, sym in enumerate(asym_symbols) if sym in 'HD'))

        # find disordered sites
        asym_is_disordered = []
        occupancies = block.get('_atom_site_occupancy')
        labels = block.get('_atom_site_label')
        if occupancies is not None:
            disordered = []     # indices where there is disorder
            for i, (occ, label) in enumerate(zip(occupancies, labels)):
                if _atom_has_disorder(label, occ):
                    if i not in remove:
                        disordered.append(i)
                        asym_is_disordered.append(True)
                else:
                    asym_is_disordered.append(False)

            if self.disorder == 'skip' and len(disordered) > 0:
                if self.show_warnings:
                    warnings.warn(
                        f'Skipping {self.current_identifier} as structure is disordered')
                return None

            if self.disorder == 'ordered_sites':
                remove.extend(disordered)

        # remove sites
        asym_frac_motif = np.mod(np.delete(asym_frac_motif, remove, axis=0), 1)
        asym_symbols = [s for i, s in enumerate(asym_symbols) if i not in remove]
        asym_is_disordered = [v for i, v in enumerate(asym_is_disordered) if i not in remove]

        keep_sites = self._validate_sites(asym_frac_motif, asym_is_disordered)
        if keep_sites is not None:
            asym_frac_motif = asym_frac_motif[keep_sites]
            asym_symbols = [sym for sym, keep in zip(asym_symbols, keep_sites) if keep]

        if self._has_no_valid_sites(asym_frac_motif):
            return None

        sitesym = ['x,y,z', ]
        for tag in _Reader.symop_tags:
            if tag in block:
                sitesym = block[tag]
                break

        if isinstance(sitesym, str):
            sitesym = [sitesym]
        
        return self._construct_periodic_set(block, asym_frac_motif, asym_symbols, sitesym, cell)


    def _Entry_to_PeriodicSet(self, entry) -> PeriodicSet:
        """ccdc.entry.Entry --> PeriodicSet. Returns None for a "bad" set."""

        # skip if structure does not pass checks in include_if
        if self.include_if:
            if not all(check(entry) for check in self.include_if):
                return None

        self.current_identifier = entry.identifier
        # structure must pass this test
        if not entry.has_3d_structure:
            if self.show_warnings:
                warnings.warn(
                    f'Skipping {self.current_identifier} as entry has no 3D structure')
            return None

        crystal = entry.crystal

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
                if _atom_has_disorder(a.label, occ):
                    may_have_disorder = True
                    break

            if not may_have_disorder:
                if crystal.has_disorder or entry.has_disorder:
                    if self.show_warnings:
                        warnings.warn(f'Skipping {self.current_identifier} as structure is disordered')
                    return None

        if self.remove_hydrogens:
            molecule.remove_atoms(a for a in molecule.atoms if a.atomic_symbol in 'HD')

        if self.heaviest_component:
            molecule = _Reader._heaviest_component(molecule)

        crystal.molecule = molecule

        # by here all atoms to be removed have been (except via ordered_sites).
        # If disorder == 'skip' and there were atom(s) with occ < 1 found
        # eariler, we check if all such atoms were removed. If not, skip.
        if self.disorder == 'skip' and may_have_disorder:
            for a in crystal.disordered_molecule.atoms:
                occ = a.occupancy
                if _atom_has_disorder(a.label, occ):
                    if self.show_warnings:
                        warnings.warn(
                            f'Skipping {self.current_identifier} as structure is disordered')
                    return None

        # if disorder is all_sites, we need to know where disorder is to ignore overlaps
        asym_is_disordered = []     # True/False list same length as asym unit
        if self.disorder == 'all_sites':
            for a in crystal.asymmetric_unit_molecule.atoms:
                occ = a.occupancy
                if _atom_has_disorder(a.label, occ):
                    asym_is_disordered.append(True)
                else:
                    asym_is_disordered.append(False)

        # check all atoms have coords. option/default remove unknown sites?
        if not molecule.all_atoms_have_sites or \
           any(a.fractional_coordinates is None for a in molecule.atoms):
            if self.show_warnings:
                warnings.warn(
                    f'Skipping {self.current_identifier} as some atoms do not have sites')
            return None

        # get cell & asymmetric unit
        cell = cellpar_to_cell(*crystal.cell_lengths, *crystal.cell_angles)
        asym_frac_motif = np.array([tuple(a.fractional_coordinates)
                                    for a in crystal.asymmetric_unit_molecule.atoms])
        asym_frac_motif = np.mod(asym_frac_motif, 1)
        asym_symbols = [a.atomic_symbol for a in crystal.asymmetric_unit_molecule.atoms]

        # remove overlapping sites, check sites exist
        keep_sites = self._validate_sites(asym_frac_motif, asym_is_disordered)
        if keep_sites is not None:
            asym_frac_motif = asym_frac_motif[keep_sites]
            asym_symbols = [sym for sym, keep in zip(asym_symbols, keep_sites) if keep]

        if self._has_no_valid_sites(asym_frac_motif):
            return None

        sitesym = crystal.symmetry_operators
        if not sitesym:
            sitesym = ['x,y,z', ]

        entry.crystal.molecule = crystal.disordered_molecule    # for extract_data. remove?

        return self._construct_periodic_set(entry, asym_frac_motif, asym_symbols, sitesym, cell)

    def expand(
            self,
            asym_frac_motif: np.ndarray,
            sitesym: Sequence[str]
    ) -> Tuple[np.ndarray, ...]:
        """
        Asymmetric unit's fractional coords + sitesyms (as strings)
        -->
        frac_motif, asym_unit, multiplicities, inverses
        """
        
        rotations, translations = ase.spacegroup.spacegroup.parse_sitesym(sitesym)
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

                if not self._is_site_overlapping(site_, all_sites, inverses, inv):
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

    def _is_site_overlapping(self, new_site, all_sites, inverses, inv):
        """Return True (and warn) if new_site overlaps with a site in all_sites."""
        diffs1 = np.abs(new_site - all_sites)
        diffs2 = np.abs(diffs1 - 1)
        mask = np.all(np.logical_or(diffs1 <= _Reader.equiv_site_tol,
                                    diffs2 <= _Reader.equiv_site_tol),
                        axis=-1)

        if np.any(mask):
            where_equal = np.argwhere(mask).flatten()
            for ind in where_equal:
                if inverses[ind] == inv:
                    pass
                else:
                    if self.show_warnings:
                        warnings.warn(
                            f'{self.current_identifier} has equivalent positions {inverses[ind]} and {inv}')
            return True
        else:
            return False

    def _validate_sites(self, asym_frac_motif, asym_is_disordered):
        site_diffs1 = np.abs(asym_frac_motif[:, None] - asym_frac_motif)
        site_diffs2 = np.abs(site_diffs1 - 1)
        overlapping = np.triu(np.all(
            (site_diffs1 <= _Reader.equiv_site_tol) |
            (site_diffs2 <= _Reader.equiv_site_tol),
            axis=-1), 1)

        if self.disorder == 'all_sites':
            for i, j in np.argwhere(overlapping):
                if asym_is_disordered[i] or asym_is_disordered[j]:
                    overlapping[i, j] = False

        if overlapping.any():
            if self.show_warnings:
                warnings.warn(
                    f'{self.current_identifier} may have overlapping sites; duplicates will be removed')
            keep_sites = ~overlapping.any(0)
            return keep_sites

    def _has_no_valid_sites(self, motif):
        if motif.shape[0] == 0:
            if self.show_warnings:
                warnings.warn(
                    f'Skipping {self.current_identifier} as there are no sites with coordinates')
            return True
        return False

    def _construct_periodic_set(self, raw_item, asym_frac_motif, asym_symbols, sitesym, cell):
        frac_motif, asym_unit, multiplicities, inverses = self.expand(asym_frac_motif, sitesym)
        full_types = [asym_symbols[i] for i in inverses]
        motif = frac_motif @ cell

        kwargs = {
            'name': self.current_identifier,
            'asymmetric_unit': asym_unit,
            'wyckoff_multiplicities': multiplicities,
            'types': full_types,
        }

        if self.current_filename:
            kwargs['filename'] = self.current_filename

        if self.extract_data is not None:
            for key in self.extract_data:
                kwargs[key] = self.extract_data[key](raw_item)

        return PeriodicSet(motif, cell, **kwargs)

    def _heaviest_component(molecule):
        """Heaviest component (removes all but the heaviest component of the asym unit).
        Intended for removing solvents. Probably doesn't play well with disorder"""
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
            largest_component_arg = np.argmax(np.array(component_weights))
            molecule = molecule.components[largest_component_arg]

        return molecule
