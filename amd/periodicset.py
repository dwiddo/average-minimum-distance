"""Implements the :class:`PeriodicSet` class representing a periodic set,
defined by a motif and unit cell. This models a crystal with a point at the
center of each atom.

This is the object type yielded by the readers :class:`.io.CifReader` and
:class:`.io.CSDReader`. The :class:`PeriodicSet` can be passed as the first argument
to :func:`.calculate.AMD` or :func:`.calculate.PDD` to calculate its invariants.
"""

from typing import Optional
import numpy as np

from .utils import cell_to_cellpar, cell_to_cellpar_2D, cellpar_to_cell, cellpar_to_cell_2D, random_cell


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

        if self.asymmetric_unit is None:
            n_asym_sites = len(self.motif)
        else:
            n_asym_sites = len(self.asymmetric_unit)

        s = f'PeriodicSet(name={self.name}, ' \
            f'motif {self.motif.shape}, ' \
            f'{n_asym_sites} asym sites'

        if self.cell.shape[0] == 2:
            cellpar = np.round(cell_to_cellpar_2D(self.cell), 2)
            cell_str = f'a={cellpar[0]}, b={cellpar[1]}, α={cellpar[2]}'
            s += ', ' + cell_str
        elif self.cell.shape[0] == 3:
            cellpar = np.round(cell_to_cellpar(self.cell), 2)
            cell_str = f'a={cellpar[0]}, b={cellpar[1]}, c={cellpar[2]}, ' \
                       f'α={cellpar[3]}, β={cellpar[4]}, γ={cellpar[5]}'
            s += ', ' + cell_str

        s += ')'

        return s
 
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


def lattice_cubic(scale=1, dims=3):
    """Return a pair ``(motif, cell)`` representing a hexagonal lattice, passable to
    ``amd.AMD()`` or ``amd.PDD()``. """

    return PeriodicSet(np.zeros((1, dims)), np.identity(dims) * scale)

def lattice_hexagonal(scale=1, dims=3):
    """Dimension 3 only. Return a pair ``(motif, cell)`` representing a cubic lattice, 
    passable to ``amd.AMD()`` or ``amd.PDD()``."""

    if dims == 3:
        return PeriodicSet(np.zeros((1, 3)), cellpar_to_cell(scale, scale, scale, 90, 90, 120))
    elif dims == 2:
        return PeriodicSet(np.zeros((1, 2)), cellpar_to_cell_2D(scale, scale, 60))
    else:
        raise ValueError(f'lattice_hexagonal only implemented for dimensions 2 and 3 (passed {dims})')

def random_periodicset(n_points, length_bounds=(1, 2), angle_bounds=(60, 120), dims=3):
    
    cell = random_cell(length_bounds=length_bounds, angle_bounds=angle_bounds, dims=dims)
    frac_motif = np.random.uniform(size=(n_points, dims))
    motif = frac_motif @ cell
    return PeriodicSet(motif, cell)
