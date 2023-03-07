"""Implements the :class:`PeriodicSet` class representing a periodic
set, defined by a motif and unit cell. This models a crystal with a
point at the center of each atom.

This is the type yielded by :class:`amd.CifReader <.io.CifReader>` and
:class:`amd.CSDReader <.io.CSDReader>`. A :class:`PeriodicSet` can be
passed as the first argument to :func:`amd.AMD() <.calculate.AMD>` or
:func:`amd.PDD() <.calculate.PDD>` to calculate its invariants.
"""

from typing import Optional, Union, Tuple

import numpy as np
import numpy.typing as npt

from .utils import (
    cellpar_to_cell,
    cellpar_to_cell_2D,
    cell_to_cellpar,
    cell_to_cellpar_2D,
    random_cell,
)


class PeriodicSet:
    """A periodic set is a collection of points (motif) which
    periodically repeats according to a lattice (unit cell), often
    representing a crystal.

    :class:`PeriodicSet` s are returned by the readers in the
    :mod:`.io` module. They can be passed to
    :func:`amd.AMD() <.calculate.AMD>` or
    :func:`amd.PDD() <.calculate.PDD>` to calculate their invariants.

    Parameters
    ----------
    motif : :class:`numpy.ndarray`
        Cartesian (orthogonal) coordinates of the motif, shape (no
        points, dims).
    cell : :class:`numpy.ndarray`
        Cartesian (orthogonal) square array representing the unit cell,
        shape (dims, dims). Use
        :func:`amd.cellpar_to_cell <.utils.cellpar_to_cell>` to convert
        6 cell parameters to an orthogonal square matrix.
    name : str, optional
        Name of the periodic set.
    asymmetric_unit : :class:`numpy.ndarray`, optional
        Indices for the asymmetric unit, pointing to the motif. Useful
        in invariant calculations.
    wyckoff_multiplicities : :class:`numpy.ndarray`, optional
        Wyckoff multiplicities of each atom in the asymmetric unit
        (number of unique sites generated under all symmetries). Useful
        in invariant calculations.
    types : :class:`numpy.ndarray`, optional
        Array of atomic numbers of motif points.
    """

    def __init__(
            self,
            motif: npt.NDArray,
            cell: npt.NDArray,
            name: Optional[str] = None,
            asymmetric_unit: Optional[npt.NDArray] = None,
            wyckoff_multiplicities: Optional[npt.NDArray] = None,
            types: Optional[npt.NDArray] = None
    ):

        self.motif = motif
        self.cell = cell
        self.name = name
        self.asymmetric_unit = asymmetric_unit
        self.wyckoff_multiplicities = wyckoff_multiplicities
        self.types = types

    @property
    def ndim(self) -> int:
        return self.cell.shape[0]

    def __str__(self):
        """Returns a string representation of the PeriodicSet:
        PeriodicSet(name, motif (m, d), t asym sites, abcαβγ=cellpar).
        """

        if self.asymmetric_unit is None:
            n_asym_sites = len(self.motif)
        else:
            n_asym_sites = len(self.asymmetric_unit)

        cellpar_str = None
        if self.ndim == 1:
            cellpar_str = f', cell={self.cell[0][0]})'
        if self.ndim == 2:
            cellpar = np.round(cell_to_cellpar_2D(self.cell), 2)
            cellpar_str = ','.join(str(round(p, 2)) for p in cellpar)
            cellpar_str = f', abα={cellpar_str})'
        elif self.ndim == 3:
            cellpar = np.round(cell_to_cellpar(self.cell), 2)
            cellpar_str = ','.join(str(round(p, 2)) for p in cellpar)
            cellpar_str = f', abcαβγ={cellpar_str})'
        else:
            cellpar_str = ')'

        return (
            f'PeriodicSet(name={self.name}, motif shape {self.motif.shape} '
            f'({n_asym_sites} asym sites){cellpar_str}'
        )

    def __repr__(self):
        return (
            f'PeriodicSet(name={self.name}, '
            f'motif={self.motif}, cell={self.cell}, '
            f'asymmetric_unit={self.asymmetric_unit}, '
            f'wyckoff_multiplicities={self.wyckoff_multiplicities}, '
            f'types={self.types})'
        )

    def __eq__(self, other):
        """Used for debugging/tests. True if both 1. the unit cells are
        (close to) identical, and 2. the motifs are the same shape, and
        every point in one motif has a (close to) identical point
        somewhere in the other, accounting for pbc.
        """

        tol = 1e-8
        if self.cell.shape != other.cell.shape or \
           self.motif.shape != other.motif.shape or \
           not np.allclose(self.cell, other.cell):
            return False

        fm1 = np.mod(self.motif @ np.linalg.inv(self.cell), 1)
        fm2 = np.mod(other.motif @ np.linalg.inv(other.cell), 1)
        d1 = np.abs(fm2[:, None] - fm1)
        d2 = np.abs(d1 - 1)
        diffs = np.amax(np.minimum(d1, d2), axis=-1)

        if not np.all(
            (np.amin(diffs, axis=0) <= tol) | (np.amin(diffs, axis=-1) <= tol)
        ):
            return False

        return True

    def __ne__(self, other):
        return not self.__eq__(other)

    @staticmethod
    def cubic(scale: float = 1.0, dims: int = 3):
        """Returns a :class:`PeriodicSet` representing a cubic lattice.
        """
        return PeriodicSet(np.zeros((1, dims)), np.identity(dims) * scale)

    @staticmethod
    def hexagonal(scale: float = 1.0, dims: int = 3):
        """Dimensions 2 and 3 only. Return a :class:`PeriodicSet`
        representing a hexagonal lattice.
        """

        if dims == 3:
            cellpar = np.array([scale, scale, scale, 90., 90., 120.])
            cell = cellpar_to_cell(cellpar)
        elif dims == 2:
            cell = cellpar_to_cell_2D(np.array([scale, scale, 60.]))
        else:
            raise NotImplementedError(
                'amd.PeriodicSet.hexagonal() only implemented for dimensions '
                f'2 and 3, passed {dims}'
            )
        return PeriodicSet(np.zeros((1, dims)), cell)

    @staticmethod
    def _random(
        n_points: int,
        length_bounds: tuple = (1.0, 2.0),
        angle_bounds: tuple = (60.0, 120.0),
        dims: int = 3
    ):
        """Dimensions 2 and 3 only. Return a :class:`PeriodicSet` with a
        chosen number of randomly placed points, in random cell with
        edges between length_bounds and angles between angle_bounds.
        """

        cell = random_cell(
            length_bounds=length_bounds,
            angle_bounds=angle_bounds,
            dims=dims
        )
        frac_motif = np.random.uniform(size=(n_points, dims))
        return PeriodicSet(frac_motif @ cell, cell)


PeriodicSetType = Union[PeriodicSet, Tuple[npt.NDArray, npt.NDArray]]
