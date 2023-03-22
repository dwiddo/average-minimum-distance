"""Miscellaneous utility functions.
"""

from typing import Tuple

import numpy as np
import numpy.typing as npt
import numba
from scipy.spatial.distance import squareform


@numba.njit()
def diameter(cell: npt.NDArray) -> float:
    """Return the diameter of a unit cell (given as a square matrix in
    orthogonal coordinates) in 3 or fewer dimensions. The diameter is
    the maxiumum distance between any two points in the unit cell.

    Parameters
    ----------
    cell : :class:`numpy.ndarray`
        Unit cell represented as a square matrix (in orthogonal
        coordinates).
    Returns
    -------
    diameter : float
        Diameter of the unit cell.
    """

    dims = cell.shape[0]
    if dims == 1:
        return cell[0][0]
    if dims == 2:
        diagonals = np.empty((2, 2))
        diagonals[0] = (cell[0] + cell[1]) ** 2
        diagonals[1] = (cell[0] - cell[1]) ** 2
    elif dims == 3:
        diagonals = np.empty((4, 3))
        diagonals[0] = ( cell[0] + cell[1] + cell[2]) ** 2
        diagonals[1] = ( cell[0] + cell[1] - cell[2]) ** 2
        diagonals[2] = ( cell[0] - cell[1] + cell[2]) ** 2
        diagonals[3] = (-cell[0] + cell[1] + cell[2]) ** 2
    else:
        raise ValueError('amd.diameter() only implemented for dimensions <= 3')

    return np.sqrt(np.amax(np.sum(diagonals, axis=-1)))


@numba.njit()
def cellpar_to_cell(cellpar: npt.NDArray) -> npt.NDArray[np.float64]:
    """Convert canonical 3D unit cell parameters a,b,c,α,β,γ into a 3x3
    :class:`numpy.ndarray` representing the unit cell in orthogonal
    coordinates. Numba-accelerated version of function from
    :mod:`ase.geometry` of the same name.

    Parameters
    ----------
    cellpar : :class:`numpy.ndarray`
        Vector of six unit cell parameters in the canonical order
        a,b,c,α,β,γ.

    Returns
    -------
    cell : :class:`numpy.ndarray`
        Unit cell represented as a 3x3 matrix in orthogonal coordinates.
    """

    a, b, c, alpha, beta, gamma = cellpar
    eps = 2 * np.spacing(90)

    if abs(abs(alpha) - 90.0) < eps:
        cos_alpha = 0.0
    else:
        cos_alpha = np.cos(alpha * np.pi / 180.0)

    if abs(abs(beta) - 90.0) < eps:
        cos_beta = 0.0
    else:
        cos_beta = np.cos(beta * np.pi / 180.0)

    if abs(abs(gamma) - 90.0) < eps:
        cos_gamma = 0.0
    else:
        cos_gamma = np.cos(gamma * np.pi / 180.0)

    if abs(gamma - 90.0) < eps:
        sin_gamma = 1.0
    elif abs(gamma + 90.0) < eps:
        sin_gamma = -1.0
    else:
        sin_gamma = np.sin(gamma * np.pi / 180.0)

    cy = (cos_alpha - cos_beta * cos_gamma) / sin_gamma
    cz_sqr = 1.0 - cos_beta ** 2 - cy ** 2
    if cz_sqr < 0:
        raise ValueError('Could not create unit cell from given parameters.')

    cell = np.zeros((3, 3), dtype=np.float64)
    cell[0, 0] = a
    cell[1, 0] = b * cos_gamma
    cell[1, 1] = b * sin_gamma
    cell[2, 0] = c * cos_beta
    cell[2, 1] = c * cy
    cell[2, 2] = c * np.sqrt(cz_sqr)
    return cell


@numba.njit()
def cellpar_to_cell_2D(cellpar: npt.NDArray) -> npt.NDArray[np.float64]:
    """Convert 3 parameters defining a 2D unit cell a,b,α into a 2x2
    :class:`numpy.ndarray` representing the unit cell in orthogonal
    coordinates.

    Parameters
    ----------
    cellpar : :class:`numpy.ndarray`
        Vector of three unit cell parameters a,b,α, where a and b are
        lengths of sides of the unit cell and α is the angle between the
        sides.
    Returns
    -------
    cell : :class:`numpy.ndarray`
        Unit cell represented as a 2x2 matrix in orthogonal coordinates.
    """

    a, b, alpha = cellpar
    cell = np.zeros((2, 2), dtype=np.float64)
    ang = alpha * np.pi / 180.0
    cell[0, 0] = a
    cell[1, 0] = b * np.cos(ang)
    cell[1, 1] = b * np.sin(ang)
    return cell


@numba.njit()
def cell_to_cellpar(cell: npt.NDArray) -> npt.NDArray[np.float64]:
    """Convert a 3x3 :class:`numpy.ndarray` representing a unit cell in
    orthogonal coordinates (as returned by
    :func:`cellpar_to_cell() <.utils.cellpar_to_cell>`) into a list of 3
    lengths and 3 angles representing the unit cell.

    Parameters
    ----------
    cell : :class:`numpy.ndarray`
        Unit cell as a 3x3 matrix in orthogonal coordinates.

    Returns
    -------
    cellpar : :class:`numpy.ndarray`
        Vector of 6 unit cell parameters in the canonical order
        a,b,c,α,β,γ.
    """

    cellpar = np.empty((6, ), dtype=np.float64)
    cellpar[0] = np.linalg.norm(cell[0])
    cellpar[1] = np.linalg.norm(cell[1])
    cellpar[2] = np.linalg.norm(cell[2])
    cellpar[3] = np.rad2deg(np.arccos(
        np.dot(cell[1], cell[2]) / (cellpar[1] * cellpar[2])
    ))
    cellpar[4] = np.rad2deg(np.arccos(
        np.dot(cell[0], cell[2]) / (cellpar[0] * cellpar[2])
    ))
    cellpar[5] = np.rad2deg(np.arccos(
        np.dot(cell[0], cell[1]) / (cellpar[0] * cellpar[1])
    ))
    return cellpar


@numba.njit()
def cell_to_cellpar_2D(cell: npt.NDArray) -> npt.NDArray[np.float64]:
    """Convert a 2x2 :class:`numpy.ndarray` representing a unit cell in
    orthogonal coordinates (as returned by
    :func:`cellpar_to_cell_2D() <.utils.cellpar_to_cell_2D>`) into a
    list of 2 lengths and 1 angle representing the unit cell.

    Parameters
    ----------
    cell : :class:`numpy.ndarray`
        Unit cell as a 2x2 matrix in orthogonal coordinates.

    Returns
    -------
    cellpar : :class:`numpy.ndarray`
        Vector with 3 elements, containing 3 unit cell parameters.
    """

    cellpar = np.empty((3, ), dtype=np.float64)
    cellpar[0] = np.linalg.norm(cell[0])
    cellpar[1] = np.linalg.norm(cell[1])
    cellpar[2] = np.rad2deg(np.arccos(
        np.dot(cell[0], cell[1]) / (cellpar[0] * cellpar[1])
    ))
    return cellpar


def neighbours_from_distance_matrix(
        n: int,
        dm: npt.NDArray
) -> Tuple[npt.NDArray, npt.NDArray[np.int32]]:
    """Given a distance matrix, find the n nearest neighbours of each
    item.

    Parameters
    ----------
    n : int
        Number of nearest neighbours to find for each item.
    dm : :class:`numpy.ndarray`
        2D distance matrix or 1D condensed distance matrix.

    Returns
    -------
    (nn_dm, inds) : tuple of :class:`numpy.ndarray` s
        ``nn_dm[i][j]`` is the distance from item :math:`i` to its
        :math:`j+1` st nearest neighbour, and ``inds[i][j]`` is the
        index of this neighbour (:math:`j+1` since index 0 is the first
        nearest neighbour).
    """

    inds = None

    if len(dm.shape) == 2:
        inds = np.array([
            np.argpartition(row, n)[:n] for row in dm
        ], dtype=np.int32)
    elif len(dm.shape) == 1:
        dm = squareform(dm)
        inds = []
        for i, row in enumerate(dm):
            inds_row = np.argpartition(row, n+1)[:n+1]
            inds_row = inds_row[inds_row != i][:n]
            inds.append(inds_row)
        inds = np.array(inds, dtype=np.int32)
    else:
        ValueError(
            'amd.neighbours_from_distance_matrix() accepts a distance matrix, '
            'either a 2D distance matrix or a condensed distance matrix as '
            'returned by scipy.spatial.distance.pdist()'
        )

    # inds are the indexes of nns: inds[i, j] is the j-th nn to point i
    nn_dm = np.take_along_axis(dm, inds, axis=-1)
    sorted_inds = np.argsort(nn_dm, axis=-1)
    inds = np.take_along_axis(inds, sorted_inds, axis=-1)
    nn_dm = np.take_along_axis(nn_dm, sorted_inds, axis=-1)
    return nn_dm, inds


def random_cell(
        length_bounds: Tuple = (1.0, 2.0),
        angle_bounds: Tuple = (60.0, 120.0),
        dims: int = 3
) -> npt.NDArray[np.float64]:
    """Return a random unit cell with uniformally chosen length and
    angle parameters between bounds. Dimensions 2 and 3 only.
    """

    ll, lu = length_bounds
    al, au = angle_bounds

    if dims == 3:
        while True:
            lengths = [np.random.uniform(low=ll, high=lu) for _ in range(dims)]
            angles = [np.random.uniform(low=al, high=au) for _ in range(dims)]
            cellpar = np.array(lengths + angles)
            try:
                cell = cellpar_to_cell(cellpar)
                break
            except RuntimeError:
                continue
    elif dims == 2:
        cellpar = [np.random.uniform(low=ll, high=lu) for _ in range(2)]
        cellpar.append(np.random.uniform(low=al, high=au))
        cell = cellpar_to_cell_2D(np.array(cellpar))
    else:
        raise NotImplementedError(
            'amd.random_cell() only implemented for dimensions 2 and 3, '
            f'passed {dims}'
        )
    return cell
