"""Implements the :class:`PeriodicSet` class representing a periodic set,
defined by a motif and unit cell. This models a crystal with a point at the
center of each atom.

This is the object type yielded by the readers :class:`.io.CifReader` and
:class:`.io.CSDReader`. The :class:`PeriodicSet` can be passed as the first argument
to :func:`.calculate.AMD` or :func:`.calculate.PDD` to calculate its invariants.
"""

from typing import Optional
import numpy as np


class PeriodicSet:
    """A periodic set is the mathematical representation of a crystal by putting a
    single point in the center of every atom. A periodic set is defined by a basis
    (unit cell) and collection of points (motif) which repeats according to the basis.

    :class:`PeriodicSet` s are returned by the readers in the :mod:`.io` module.
    Instances of this object can be passed to :func:`.calculate.AMD` or
    :func:`.calculate.PDD` to calculate the invariant.
    """

    def __init__(
            self,
            motif: np.ndarray,
            cell: np.ndarray,
            name: Optional[str] = None,
            **kwargs):

        self.motif = motif
        self.cell = cell
        self.name = name
        self.tags = kwargs

    def __str__(self):

        m, dims = self.motif.shape
        return f"PeriodicSet({self.name}: {m} motif points in {dims} dimensions)"

    def __repr__(self):
        return f"PeriodicSet(name: {self.name}, cell: {self.cell}, motif: {self.motif}, tags: {self.tags})"

    # used for debugging, checks if the motif/cell agree point for point
    # (disregarding order), NOT up to isometry.
    def __eq__(self, other):

        if self.cell.shape != other.cell.shape or self.motif.shape != other.motif.shape:
            return False

        if not np.allclose(self.cell, other.cell):
            return False

        # this is the part that'd be confused by overlapping sites
        # just checks every motif point in either have a neighbour in the other
        diffs = np.amax(np.abs(other.motif[:, None] - self.motif), axis=-1)
        if not np.all((np.amin(diffs, axis=0) <= 1e-6) & (np.amin(diffs, axis=-1) <= 1e-6)):
            return False

        shared_tags = set(self.tags.keys()).intersection(set(other.tags.keys()))
        for tag in shared_tags:
            if np.isscalar(self.tags[tag]):
                if self.tags[tag] != other.tags[tag]:
                    return False
            elif isinstance(self.tags[tag], np.ndarray):
                if not np.allclose(self.tags[tag], other.tags[tag]):
                    return False
            elif isinstance(self.tags[tag], list):
                for i, i_ in zip(self.tags[tag], other.tags[tag]):
                    if i != i_:
                        return False

        return True

    def __getattr__(self, attr):

        if 'tags' not in self.__dict__:
            self.tags = {}
        if attr in self.tags:
            return self.tags[attr]

        raise AttributeError(f"{self.__class__.__name__} object has no attribute or tag {attr}")

    def __ne__(self, other):
        return not self.__eq__(other)
