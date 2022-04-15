"""Contains I/O tools, including a .CIF reader and CSD reader
(``csd-python-api`` only) to extract periodic set representations
of crystals which can be passed to :func:`.calculate.AMD` and :func:`.calculate.PDD`.

These intermediate :class:`.periodicset.PeriodicSet` representations can be written
to a .hdf5 file with :class:`SetWriter`, which can be read back with :class:`SetReader`.
This is much faster than rereading a .CIF and recomputing invariants.
"""

import os
import warnings

import numpy as np
import ase.io.cif

from .periodicset import PeriodicSet
from .utils import _extend_signature, cellpar_to_cell
from ._reader import _Reader

try:
    import ccdc.io       # EntryReader
    import ccdc.search   # TextNumericSearch
    _CCDC_ENABLED = True
except (ImportError, RuntimeError) as _:
    _CCDC_ENABLED = False


def _warning(message, category, filename, lineno, *args, **kwargs):
    return f'{filename}:{lineno}: {category.__name__}: {message}\n'

warnings.formatwarning = _warning


class CifReader(_Reader):
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

    @_extend_signature(_Reader.__init__)
    def __init__(self, path, reader='ase', folder=False, **kwargs):

        super().__init__(**kwargs)

        if reader not in ('ase', 'ccdc'):
            raise ValueError(f'Invalid reader {reader}; must be ase or ccdc.')

        if reader == 'ase' and self.heaviest_component:
            raise NotImplementedError('Parameter heaviest_component not implimented for ase, only ccdc.')

        if reader == 'ase':
            extensions = {'cif'}
            file_parser = ase.io.cif.parse_cif
            pset_converter = self._cifblock_to_periodicset

        elif reader == 'ccdc':
            if not _CCDC_ENABLED:
                raise ImportError("Failed to import csd-python-api; check it is installed and licensed.")
            extensions = ccdc.io.EntryReader.known_suffixes
            file_parser = ccdc.io.EntryReader
            pset_converter = self._entry_to_periodicset

        if folder:
            generator = self._folder_generator(path, file_parser, extensions)
        else:
            generator = file_parser(path)

        self._generator = self._map(pset_converter, generator)

    def _folder_generator(self, path, file_parser, extensions):
        for file in os.listdir(path):
            suff = os.path.splitext(file)[1][1:]
            if suff.lower() in extensions:
                self.current_filename = file
                yield from file_parser(os.path.join(path, file))


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

    @_extend_signature(_Reader.__init__)
    def __init__(self, refcodes=None, families=False, **kwargs):

        if not _CCDC_ENABLED:
            raise ImportError("Failed to import csd-python-api; check it is installed and licensed.")

        super().__init__(**kwargs)

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
        self._generator = self._map(
            self._entry_to_periodicset,
            self._ccdc_generator(refcodes))

    def _ccdc_generator(self, refcodes):
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
                    warnings.warn(
                        f'Identifier {refcode} not found in database')

    def entry(self, refcode: str) -> PeriodicSet:
        """Read a PeriodicSet given any CSD refcode."""

        entry = self._entry_reader.entry(refcode)
        periodic_set = self._entry_to_periodicset(entry)
        return periodic_set


def crystal_to_periodicset(crystal):
    """ccdc.crystal.Crystal --> amd.periodicset.PeriodicSet.
    Ignores disorder, missing sites/coords, checks & no options.
    Is a stripped-down version of the function used in CifReader."""

    cell = cellpar_to_cell(*crystal.cell_lengths, *crystal.cell_angles)

    # asymmetric unit fractional coordinates
    asym_frac_motif = np.array([tuple(a.fractional_coordinates)
                                for a in crystal.asymmetric_unit_molecule.atoms])
    asym_frac_motif = np.mod(asym_frac_motif, 1)

    # if the above removed everything, skip this structure
    if asym_frac_motif.shape[0] == 0:
        raise ValueError(f'{crystal.identifier} has no coordinates')

    sitesym = crystal.symmetry_operators
    if not sitesym:
        sitesym = ('x,y,z', )
    r = _Reader()
    r.current_identifier = crystal.identifier
    frac_motif, asym_unit, multiplicities, _ = r.expand(asym_frac_motif, sitesym)
    motif = frac_motif @ cell

    kwargs = {
        'name': crystal.identifier,
        'asymmetric_unit': asym_unit,
        'wyckoff_multiplicities': multiplicities,
    }

    periodic_set = PeriodicSet(motif, cell, **kwargs)
    return periodic_set


def cifblock_to_periodicset(block):
    """ase.io.cif.CIFBlock --> amd.periodicset.PeriodicSet.
    Ignores disorder, missing sites/coords, checks & no options.
    Is a stripped-down version of the function used in CifReader."""

    cell = block.get_cell().array
    asym_frac_motif = [block.get(name) for name in _Reader.atom_site_fract_tags]

    if None in asym_frac_motif:
        asym_motif = [block.get(name) for name in _Reader.atom_site_cartn_tags]
        if None in asym_motif:
            warnings.warn(
                f'Skipping {block.name} as coordinates were not found')
            return None

        asym_frac_motif = np.array(asym_motif) @ np.linalg.inv(cell)

    asym_frac_motif = np.mod(np.array(asym_frac_motif).T, 1)

    if asym_frac_motif.shape[0] == 0:
        raise ValueError(f'{block.name} has no coordinates')

    sitesym = ('x,y,z', )
    for tag in _Reader.symop_tags:
        if tag in block:
            sitesym = block[tag]
            break

    if isinstance(sitesym, str):
        sitesym = [sitesym]

    dummy_reader = _Reader()
    dummy_reader.current_identifier = block.name
    frac_motif, asym_unit, multiplicities, _ = dummy_reader.expand(asym_frac_motif, sitesym)
    motif = frac_motif @ cell

    kwargs = {
        'name': block.name,
        'asymmetric_unit': asym_unit,
        'wyckoff_multiplicities': multiplicities
    }

    periodic_set = PeriodicSet(motif, cell, **kwargs)
    return periodic_set
