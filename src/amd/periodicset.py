"""Implements the :class:`PeriodicSet` class representing a periodic
set, defined by a motif (collection of points) and unit cell (basis).
This models a crystal with a point at the center of each atom.

This is the type yielded by :class:`amd.CifReader <.io.CifReader>` and
:class:`amd.CSDReader <.io.CSDReader>`. A :class:`PeriodicSet` can be
passed as the first argument to :func:`amd.AMD() <.calculate.AMD>` or
:func:`amd.PDD() <.calculate.PDD>` to calculate its invariants.
"""

from __future__ import annotations
from typing import Optional

import numpy as np
import numpy.typing as npt

from .utils import (
    cellpar_to_cell,
    cellpar_to_cell_2D,
    random_cell,
)

FloatArray = npt.NDArray[np.floating]
IntArray = npt.NDArray[np.integer]

__all__ = ['PeriodicSet']


class PeriodicSet:
    """A periodic set is a collection of points (motif) which
    periodically repeats according to a lattice (unit cell), often
    representing a crystal.

    :class:`PeriodicSet` s are returned by the readers in the
    :mod:`.io` module (:class:`amd.CifReader <.io.CifReader>` and
    :class:`amd.CSDReader <.io.CSDReader>`). They can be passed to
    :func:`amd.AMD() <.calculate.AMD>` or
    :func:`amd.PDD() <.calculate.PDD>` to calculate their invariants.

    Parameters
    ----------
    motif : :class:`numpy.ndarray`
        Collection of motif points in Cartesian (orthogonal)
        coordinates, shape (no points, ``dims``).
    cell : :class:`numpy.ndarray`
        Square array representing the unit cell in Cartesian
        (orthogonal) coordinates, shape (``dims``, ``dims``). Use
        :func:`amd.cellpar_to_cell <.utils.cellpar_to_cell>` to convert
        a vector of 6 conventional cell parameters a,b,c,α,β,γ to a 3x3
        matrix.
    name : str, optional
        Name of the periodic set.
    asym_unit : :class:`numpy.ndarray`, optional
        Indices of motif points constituting an asymmetric unit, so that
        ``motif[asym_unit]`` are points in an asymmetric unit. Used in
        AMD/PDD calculation.
    multiplicities : :class:`numpy.ndarray`, optional
        Wyckoff multiplicities of points in the asymmetric unit (the
        number of unique sites generated under symmetries). Used in
        AMD/PDD calculation.
    types : :class:`numpy.ndarray`, optional
        Atomic numbers of motif points.
    """

    def __init__(
            self,
            motif: FloatArray,
            cell: FloatArray,
            name: Optional[str] = None,
            asym_unit: Optional[IntArray] = None,
            multiplicities: Optional[FloatArray] = None,
            types: Optional[IntArray] = None
    ):
        self.motif = motif
        self.cell = cell
        self.name = name
        self.asym_unit = asym_unit
        self.multiplicities = multiplicities
        self.types = types

    @property
    def ndim(self) -> int:
        return self.cell.shape[0]

    def __str__(self) -> str:
        m, n = self.motif.shape
        m_pl = '' if m == 1 else 's'
        n_pl = '' if n == 1 else 's'
        name_str = f'{self.name}: ' if self.name is not None else ''
        return f'PeriodicSet({name_str}{m} point{m_pl} in {n} dimension{n_pl})'

    def __repr__(self) -> str:
        motif_str = str(self.motif).replace('\n ', '\n' + ' ' * 11)
        cell_str = str(self.cell).replace('\n ', '\n' + ' ' * 10)
        optional_attrs = []
        for attr in ('asym_unit', 'multiplicities', 'types'):
            val = getattr(self, attr)
            if val is not None:
                st = str(val).replace('\n ', '\n' + ' ' * (len(attr) + 6))
                optional_attrs.append(f'{attr}={st}')
        optional_attrs_str = ',\n    ' if optional_attrs else ''
        optional_attrs_str += ',\n    '.join(optional_attrs)

        return (
            f'PeriodicSet(name={self.name},\n'
            f'    motif={motif_str},\n'
            f'    cell={cell_str}{optional_attrs_str})'
        )

    def _equal_cell_and_motif(self, other: PeriodicSet) -> bool:
        """Used for debugging/tests. True if both 1) the unit cells are
        (close to) identical, and 2) the motifs are the same shape, and
        every point in one motif has a (close to) identical point
        somewhere in the other, accounting for pbc.
        """

        tol = 1e-8
        if self.cell.shape != other.cell.shape or \
           self.motif.shape != other.motif.shape or \
           not np.allclose(self.cell, other.cell):
            return False

        cell_inv = np.linalg.inv(self.cell)
        fm1 = np.mod(self.motif @ cell_inv, 1)
        fm2 = np.mod(other.motif @ cell_inv, 1)
        d1 = np.abs(fm2[:, None] - fm1)
        d2 = np.abs(d1 - 1)
        diffs = np.amax(np.minimum(d1, d2), axis=-1)

        if not np.all(
            (np.amin(diffs, axis=0) <= tol) | (np.amin(diffs, axis=-1) <= tol)
        ):
            return False
        return True

    @staticmethod
    def cubic(scale: float = 1.0, dims: int = 3) -> PeriodicSet:
        """Return a :class:`PeriodicSet` representing a cubic lattice."""
        return PeriodicSet(np.zeros((1, dims)), np.identity(dims) * scale)

    @staticmethod
    def hexagonal(scale: float = 1.0, dims: int = 3) -> PeriodicSet:
        """Return a :class:`PeriodicSet` representing a hexagonal
        lattice. Implemented for dimensions 2 and 3 only.
        """

        if dims == 3:
            cellpar = np.array([scale, scale, scale, 90.0, 90.0, 120.0])
            cell = cellpar_to_cell(cellpar)
        elif dims == 2:
            cell = cellpar_to_cell_2D(np.array([scale, scale, 60.0]))
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
    ) -> PeriodicSet:
        """Return a :class:`PeriodicSet` with a chosen number of
        randomly placed points, in a random cell with edges between
        ``length_bounds`` and angles between ``angle_bounds``.
        Implemented for dimensions 2 and 3 only.
        """

        if dims not in (2, 3):
            raise NotImplementedError(
            'amd.PeriodicSet._random only implemented for dimensions 2 '
            f'and 3, passed {dims}'
        )

        cell = random_cell(
            length_bounds=length_bounds, angle_bounds=angle_bounds, dims=dims
        )
        frac_motif = np.random.uniform(size=(n_points, dims))
        return PeriodicSet(frac_motif @ cell, cell)
