"""Implements the :class:`PeriodicSet` class representing a periodic
set, defined by a motif and unit cell. This models a crystal with a
point at the center of each atom.

This is the type yielded by the readers 
:class:`amd.CifReader <.io.CifReader>` and 
:class:`amd.CSDReader <.io.CSDReader>`. A :class:`PeriodicSet` can be
passed as the first argument to :func:`amd.AMD() <.calculate.AMD>` or
:func:`amd.PDD() <.calculate.PDD>` to calculate its invariants.
"""

from typing import Optional
import numpy as np

from .utils import (
    cellpar_to_cell,
    cellpar_to_cell_2D,
    cell_to_cellpar,
    cell_to_cellpar_2D,
    random_cell,
)

from ase.data import atomic_masses


class PeriodicSet:
    """A periodic set representats a crystal by putting a point in the
    center of each atom. Mathematically it's defined by a basis (unit 
    cell) and collection of points (motif) which repeats according to
    the basis. Periodic sets are defined for any dimension.

    :class:`PeriodicSet` s are returned by the readers in the
    :mod:`.io` module. They can be passed to
    :func:`amd.AMD() <.calculate.AMD>` or 
    :func:`amd.PDD() <.calculate.PDD>` to calculate the invariants.

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
            motif: np.ndarray,
            cell: np.ndarray,
            name: Optional[str] = None,
            asymmetric_unit: Optional[np.ndarray] = None,
            wyckoff_multiplicities: Optional[np.ndarray] = None,
            types: Optional[np.ndarray] = None
    ):

        self.motif = motif
        self.cell = cell
        self.name = name
        self.asymmetric_unit = asymmetric_unit
        self.wyckoff_multiplicities = wyckoff_multiplicities
        self.types = types

    @property
    def ndim(self):
        return self.cell.shape[0]

    @property
    def density(self):
        if self.types is None:
            raise ValueError('')
        cell_mass = sum(atomic_masses[n] for n in self.types)
        cell_vol = np.linalg.det(self.cell)
        return cell_mass / cell_vol

    def __str__(self):
        """Returns a string representation of the format
        PeriodicSet(name, motif (m, d), t asym sites, 
        abcαβγ=cellpar).
        """

        if self.asymmetric_unit is None:
            n_asym_sites = len(self.motif)
        else:
            n_asym_sites = len(self.asymmetric_unit)

        cellpar_str = None
        if self.ndim == 1:
            cellpar_str = f', cell={self.cell[0][0]}'
        if self.ndim == 2:
            cellpar = np.round(cell_to_cellpar_2D(self.cell), 2)
            cellpar_str = ','.join(str(round(p, 2)) for p in cellpar)
            cellpar_str = f', abα={cellpar_str}'
        elif self.ndim == 3:
            cellpar = np.round(cell_to_cellpar(self.cell), 2)
            cellpar_str = ','.join(str(round(p, 2)) for p in cellpar)
            cellpar_str = f', abcαβγ={cellpar_str}'
        else:
            cellpar_str = f''

        s = f'PeriodicSet(name={self.name}, ' \
            f'motif {self.motif.shape} ({n_asym_sites} asym sites)' \
            f'{cellpar_str})'

        return s

    def __repr__(self):

        if self.asymmetric_unit is None:
            n_asym_sites = len(self.motif)
        else:
            n_asym_sites = len(self.asymmetric_unit)

        s = f'PeriodicSet(name={self.name}, ' \
            f'motif={self.motif} ({n_asym_sites} asym sites), ' \
            f'cell={self.cell})'

        return s

    
    def __eq__(self, other):
        """Used for debugging/tests. Returns True if:
        1. The unit cells are identical (element for element within tol)
        2. The motifs are the same shape, and when transformed into
        fractional coordinates every point in one motif has an identical
        (within tol) point somewhere in the other, accounting for pbc.
        """

        if self.cell.shape != other.cell.shape or \
           self.motif.shape != other.motif.shape or \
           not np.allclose(self.cell, other.cell):
            return False

        fm1 = np.mod(self.motif @ np.linalg.inv(self.cell), 1)
        fm2 = np.mod(other.motif @ np.linalg.inv(other.cell), 1)
        d1 = np.abs(fm2[:, None] - fm1)
        d2 = np.abs(d1 - 1)
        diffs = np.amax(np.minimum(d1, d2), axis=-1)

        if not np.all((np.amin(diffs, axis=0) <= 1e-8) | (np.amin(diffs, axis=-1) <= 1e-8)):
            return False

        return True

    def __ne__(self, other):
        return not self.__eq__(other)

    @staticmethod
    def cubic(scale=1, dims=3):
        """Return a :class:`PeriodicSet` representing a cubic lattice.
        """

        cell = np.identity(dims) * scale
        return PeriodicSet(np.zeros((1, dims)), cell)

    @staticmethod
    def hexagonal(scale=1, dims=3):
        """Dimensions 2 and 3 only. Return a :class:`PeriodicSet`
        representing a hexagonal lattice.
        """

        if dims == 3:
            cell = cellpar_to_cell(scale, scale, scale, 90, 90, 120)
        elif dims == 2:
            cell = cellpar_to_cell_2D(scale, scale, 60)
        else:
            msg = 'PeriodicSet.hexagonal() only implemented for dimensions ' \
                  f'2 and 3 (passed {dims})'
            raise NotImplementedError(msg)

        return PeriodicSet(np.zeros((1, dims)), cell)

    @staticmethod
    def _random(
        n_points,
        length_bounds=(1, 2),
        angle_bounds=(60, 120),
        dims=3
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
