"""Contains base reader class for the io module. 
This class implements the converters from CifBlock, Entry to PeriodicSets.
"""

import warnings
from typing import Callable, Iterable, Sequence, Tuple

import numpy as np
import ase.spacegroup.spacegroup    # parse_sitesym
import ase.io.cif

from .periodicset import PeriodicSet
from .utils import cellpar_to_cell


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

        if disorder not in _Reader.disorder_options:
            raise ValueError(f'disorder parameter {disorder} must be one of {_Reader.disorder_options}')

        if extract_data is None:
            self.extract_data = {}
        else:
            _validate_extract_data(extract_data)
            self.extract_data = extract_data

        if include_if is None:
            self.include_if = ()
        elif not all(callable(func) for func in include_if):
            raise ValueError('include_if must be a list of callables')
        else:
            self.include_if = include_if
        
        self.remove_hydrogens = remove_hydrogens
        self.disorder = disorder
        self.heaviest_component = heaviest_component
        self.show_warnings = show_warnings
        self.current_name = None
        self.current_filename = None
        self._generator = []

    def __iter__(self):
        yield from self._generator

    def read_one(self):
        """Read the next (or first) item."""
        return next(iter(self._generator))

    def _map(self, func: Callable, iterable: Iterable) -> Iterable[PeriodicSet]:
        """Iterates over iterable, passing items through parser and yielding the 
        result if it is not None. Applies warning and include_if filter.
        """

        with warnings.catch_warnings():
            if not self.show_warnings:
                warnings.simplefilter('ignore')
            for item in iterable:
                if any(not check(item) for check in self.include_if):
                    continue
                res = func(item)
                if res is not None:
                    yield res

    def _cifblock_to_periodicset(self, block) -> PeriodicSet:
        """ase.io.cif.CIFBlock --> PeriodicSet. Returns None for a "bad" set."""

        self.current_name = block.name
        asym_unit, asym_symbols, sitesym, cell = self._cifblock_to_asym_unit(block)

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
                warnings.warn(f'Skipping {self.current_name} as structure is disordered')
                return None

            if self.disorder == 'ordered_sites':
                remove.extend(disordered)

        # remove sites
        asym_unit = np.delete(asym_unit, remove, axis=0)
        asym_symbols = [s for i, s in enumerate(asym_symbols) if i not in remove]
        asym_is_disordered = [v for i, v in enumerate(asym_is_disordered) if i not in remove]

        keep_sites = self._validate_sites(asym_unit, asym_is_disordered)
        if keep_sites is not None:
            asym_unit = asym_unit[keep_sites]
            asym_symbols = [sym for sym, keep in zip(asym_symbols, keep_sites) if keep]

        if self._has_no_valid_sites(asym_unit):
            return None

        data = {key: func(block) for key, func in self.extract_data.items()}
        periodic_set = self._construct_periodic_set(asym_unit, asym_symbols, sitesym, cell, **data)
        return periodic_set

    def _cifblock_to_asym_unit(self, block):
        """ase.io.cif.CIFBlock --> 
        asymmetric unit (frac coords), asym_symbols, cell, symops (as strings)"""
        
        cell = block.get_cell().array
        asym_unit = [block.get(name) for name in _Reader.atom_site_fract_tags]
        if None in asym_unit:
            asym_motif = [block.get(name) for name in _Reader.atom_site_cartn_tags]
            if None in asym_motif:
                warnings.warn(f'Skipping {self.current_name} as coordinates were not found')
                return None
            asym_unit = np.array(asym_motif) @ np.linalg.inv(cell)
        asym_unit = np.mod(np.array(asym_unit).T, 1)

        try:
            asym_symbols = block.get_symbols()
        except ase.io.cif.NoStructureData as _:
            asym_symbols = ['Unknown' for _ in range(len(asym_unit))]

        sitesym = ['x,y,z', ]
        for tag in _Reader.symop_tags:
            if tag in block:
                sitesym = block[tag]
                break

        if isinstance(sitesym, str):
            sitesym = [sitesym]

        return asym_unit, asym_symbols, sitesym, cell

    def _entry_to_periodicset(self, entry) -> PeriodicSet:
        """ccdc.entry.Entry --> PeriodicSet. Returns None for a "bad" set."""

        self.current_name = entry.identifier
        crystal = entry.crystal

        if not entry.has_3d_structure:
            warnings.warn(f'Skipping {self.current_name} as entry has no 3D structure')
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
                if _atom_has_disorder(a.label, occ):
                    may_have_disorder = True
                    break

            if not may_have_disorder:
                if crystal.has_disorder or entry.has_disorder:
                    warnings.warn(f'Skipping {self.current_name} as structure is disordered')
                    return None

        # make same as cifblock version??
        if self.remove_hydrogens:
            molecule.remove_atoms(a for a in molecule.atoms if a.atomic_symbol in 'HD')

        if self.heaviest_component and len(molecule.components) > 1:
            molecule = _heaviest_component(molecule)

        crystal.molecule = molecule

        # by here all atoms to be removed have been (except via ordered_sites).
        # If disorder == 'skip' and there were atom(s) with occ < 1 found
        # eariler, we check if all such atoms were removed. If not, skip.
        if self.disorder == 'skip' and may_have_disorder:
            for a in crystal.disordered_molecule.atoms:
                occ = a.occupancy
                if _atom_has_disorder(a.label, occ):
                    warnings.warn(f'Skipping {self.current_name} as structure is disordered')
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
            warnings.warn(f'Skipping {self.current_name} as some atoms do not have sites')
            return None

        asym_unit, asym_symbols, sitesym, cell = self._crystal_to_asym_unit(crystal)

        # remove overlapping sites, check sites exist
        keep_sites = self._validate_sites(asym_unit, asym_is_disordered)
        if keep_sites is not None:
            asym_unit = asym_unit[keep_sites]
            asym_symbols = [sym for sym, keep in zip(asym_symbols, keep_sites) if keep]

        if self._has_no_valid_sites(asym_unit):
            return None

        entry.crystal.molecule = crystal.disordered_molecule
        data = {key: func(entry) for key, func in self.extract_data.items()}
        periodic_set = self._construct_periodic_set(asym_unit, asym_symbols, sitesym, cell, **data)
        return periodic_set

    def _crystal_to_asym_unit(self, crystal):
        """ase.io.cif.CIFBlock -->
        asymmetric unit (frac coords), asym_symbols, symops, cell"""

        asym_atoms = crystal.asymmetric_unit_molecule.atoms
        asym_unit = np.array([tuple(a.fractional_coordinates) for a in asym_atoms])
        asym_unit = np.mod(asym_unit, 1)
        asym_symbols = [a.atomic_symbol for a in asym_atoms]
        cell = cellpar_to_cell(*crystal.cell_lengths, *crystal.cell_angles)
        sitesym = crystal.symmetry_operators
        if not sitesym:
            sitesym = ['x,y,z', ]
        return asym_unit, asym_symbols, sitesym, cell

    def _is_site_overlapping(self, new_site, all_sites, inverses, inv):
        """Return True (and warn) if new_site overlaps with a site in all_sites."""
        diffs1 = np.abs(new_site - all_sites)
        diffs2 = np.abs(diffs1 - 1)
        mask = np.all((diffs1 <= _Reader.equiv_site_tol) |
                      (diffs2 <= _Reader.equiv_site_tol),
                    axis=-1)

        if np.any(mask):
            where_equal = np.argwhere(mask).flatten()
            for ind in where_equal:
                if inverses[ind] == inv:
                    pass
                else:
                    warnings.warn(
                        f'{self.current_name} has equivalent positions {inverses[ind]} and {inv}')
            return True
        else:
            return False

    def _validate_sites(self, asym_unit, asym_is_disordered):
        site_diffs1 = np.abs(asym_unit[:, None] - asym_unit)
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
            warnings.warn(
                f'{self.current_name} may have overlapping sites; duplicates will be removed')
            keep_sites = ~overlapping.any(0)
            return keep_sites

    def _has_no_valid_sites(self, motif):
        if motif.shape[0] == 0:
            warnings.warn(
                f'Skipping {self.current_name} as there are no sites with coordinates')
            return True
        return False

    def _construct_periodic_set(self, asym_unit, asym_symbols, sitesym, cell, **kwargs):
        """Asym motif + symbols + sitesym + cell (+kwargs) --> PeriodicSet"""
        frac_motif, asym_inds, multiplicities, inverses = self.expand(asym_unit, sitesym)
        full_types = [asym_symbols[i] for i in inverses]
        motif = frac_motif @ cell

        tags = {
            'name': self.current_name,
            'asymmetric_unit': asym_inds,
            'wyckoff_multiplicities': multiplicities,
            'types': full_types,
            **kwargs
        }

        if self.current_filename:
            tags['filename'] = self.current_filename

        return PeriodicSet(motif, cell, **tags)

    def expand(self, asym_unit: np.ndarray, sitesym: Sequence[str]) -> Tuple[np.ndarray, ...]:
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

                if not self._is_site_overlapping(site_, all_sites, inverses, inv):
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
    return label.endswith('?') or (np.isscalar(occupancy) and occupancy < 1)

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
    largest_component_arg = np.argmax(np.array(component_weights))
    molecule = molecule.components[largest_component_arg]
    return molecule

def _validate_extract_data(extract_data):
    if not isinstance(extract_data, dict):
        raise ValueError('extract_data must be a dict of callables')
    for key in extract_data:
        if not callable(extract_data[key]):
            raise ValueError('extract_data must be a dict of callables')
        if key in _Reader.reserved_tags:
            raise ValueError(f'extract_data includes reserved key {key}')
