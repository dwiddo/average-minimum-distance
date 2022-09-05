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
    single point in the center of every atom. It is defined by a basis (unit cell) 
    and collection of points (motif) which repeats according to the basis.

    :class:`PeriodicSet` s are returned by the readers in the :mod:`.io` module.
    Instances of this object can be passed to :func:`.calculate.AMD` or
    :func:`.calculate.PDD` to calculate the invariant.

    Parameters
    ----------
    motif : numpy.ndarray
        Cartesian (orthogonal) coordinates of the motif, shape (no points, dims).
    cell : numpy.ndarray
        Cartesian (orthogonal) square array representing the unit cell, shape (dims, dims).
        Use :func:`.utils.cellpar_to_cell` to convert 6 cell parameters to an orthogonal square matrix.
    name : str, optional
        Name of the periodic set.
    asymmetric_unit : numpy.ndarray, optional
        Indices for the asymmetric unit, pointing to the motif.
    wyckoff_multiplicities : numpy.ndarray, optional
        Wyckoff multiplicities of each atom in the asymmetric unit
        (number of unique sites generated under all symmetries).
    types : numpy.ndarray, optional
        Array of atomic numbers of motif points.
    """

    def __init__(
            self,
            motif: np.ndarray,
            cell: np.ndarray,
            name: Optional[str] = None,
            asymmetric_unit: Optional[np.ndarray] = None,
            wyckoff_multiplicities: Optional[np.ndarray] = None,
            types : Optional[np.ndarray] = None
    ):

        self.motif = motif
        self.cell = cell
        self.name = name
        self.asymmetric_unit = asymmetric_unit
        self.wyckoff_multiplicities = wyckoff_multiplicities
        self.types = types

    def __str__(self):
        return repr(self)

    def __repr__(self):
        cell_volume = round(np.linalg.det(self.cell), 1)
        return f'PeriodicSet(name={self.name}; {len(self.motif)} motif points, cell volume {cell_volume})'
 
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

        return True

    def __ne__(self, other):
        return not self.__eq__(other)
