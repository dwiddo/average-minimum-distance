"""Contains I/O tools for storing :class:`.periodicset.PeriodicSet` representations 
of crystals, their invariants and other data, which is faster than .CIF or CSD reads.
"""

from typing import Iterable, Optional

import numpy as np
import h5py

from .periodicset import PeriodicSet


class SetWriter:
    """Write several :class:`.periodicset.PeriodicSet` objects to a .hdf5 file.
    Reading the .hdf5 is much faster than parsing a .CIF file.

    Examples:

        Write the crystals in mycif.cif to a .hdf5 file::

            with amd.SetWriter('crystals.hdf5') as writer:

                for periodic_set in amd.CifReader('mycif.cif'):
                    writer.write(periodic_set)

                # use iwrite to write straight from an iterator
                # below is equivalent to the above loop
                writer.iwrite(amd.CifReader('mycif.cif'))

    Read the crystals back from the file with :class:`SetReader`.
    """

    def __init__(self, filename: str):

        self.file = h5py.File(filename, 'w', track_order=True)

    def write(self, periodic_set: PeriodicSet, name: Optional[str] = None):
        """Write a PeriodicSet object to file."""

        if not isinstance(periodic_set, PeriodicSet):
            raise ValueError(
                f'Object type {periodic_set.__class__.__name__} cannot be written with SetWriter')

        # need a name to store or you can't access items by key
        if name is None:
            if periodic_set.name is None:
                raise ValueError(
                    'Periodic set must have a name to be written. Either set the name '
                    'attribute of the PeriodicSet or pass a name to SetWriter.write()')
            name = periodic_set.name

        # this group is the PeriodicSet
        group = self.file.create_group(name)

        # datasets in the group for motif and cell
        group.create_dataset('motif', data=periodic_set.motif)
        group.create_dataset('cell', data=periodic_set.cell)

        if periodic_set.tags:
            # a subgroup contains tags that are lists or ndarrays
            tags_group = group.create_group('tags')

            for tag in periodic_set.tags:
                data = periodic_set.tags[tag]

                if data is None:               # nonce to handle None
                    tags_group.attrs[tag] = '__None'
                elif np.isscalar(data):        # scalars (nums and strs) stored as attrs
                    tags_group.attrs[tag] = data
                elif isinstance(data, np.ndarray):
                    tags_group.create_dataset(tag, data=data)
                elif isinstance(data, list):
                    # lists of strings stored as special type for some reason
                    if any(isinstance(d, str) for d in data):
                        data = [str(d) for d in data]
                        tags_group.create_dataset(tag,
                                                  data=data,
                                                  dtype=h5py.vlen_dtype(str))
                    else:    # other lists must be castable to ndarray
                        data = np.asarray(data)
                        tags_group.create_dataset(tag, data=np.array(data))
                else:
                    raise ValueError(
                        f'Cannot store tag of type {type(data)} with SetWriter')

    def iwrite(self, periodic_sets: Iterable[PeriodicSet]):
        """Write :class:`.periodicset.PeriodicSet` objects from an iterable to file."""
        for periodic_set in periodic_sets:
            self.write(periodic_set)

    def close(self):
        """Close the :class:`SetWriter`."""
        self.file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, tb):
        self.file.close()


class SetReader:
    """Read :class:`.periodicset.PeriodicSet` objects from a .hdf5 file written
    with :class:`SetWriter`. Acts like a read-only dict that can be iterated
    over (preserves write order).

    Examples:

        Get PDDs (k=100) of crystals in crystals.hdf5::

            pdds = []
            with amd.SetReader('crystals.hdf5') as reader:
                for periodic_set in reader:
                    pdds.append(amd.PDD(periodic_set, 100))

            # above is equivalent to:
            pdds = [amd.PDD(pset, 100) for pset in amd.SetReader('crystals.hdf5')]
    """

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
        """Yield any :class:`.periodicset.PeriodicSet` whose name starts with
        input refcode."""
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
        return bool(item in self.keys())

    def keys(self):
        """Yield names of items in the :class:`SetReader`."""
        return self.file['/'].keys()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, tb):
        self.file.close()
