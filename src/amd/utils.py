"""Miscellaneous utility functions.
"""

from typing import Tuple

import numpy as np
import numba
from scipy.spatial.distance import squareform


def diameter(cell):
    """Diameter of a unit cell (given as a square matrix in orthogonal 
    coordinates) in 3 or fewer dimensions. The diameter is the maxiumum
    distance between any two points in the unit cell.

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
        diagonals = np.array([
            cell[0] + cell[1], 
            cell[0] - cell[1]
        ])
    elif dims == 3:
        diagonals = np.array([
            cell[0] + cell[1] + cell[2],
            cell[0] + cell[1] - cell[2],
            cell[0] - cell[1] + cell[2],
            - cell[0] + cell[1] + cell[2]
        ])
    else:
        msg = 'diameter() not implemented for dimensions > 3 (passed cell ' \
              f'shape {cell.shape}).'
        raise NotImplementedError(msg)

    return np.amax(np.linalg.norm(diagonals, axis=-1))


@numba.njit()
def cellpar_to_cell(a, b, c, alpha, beta, gamma):
    """Numba-accelerated version of function from :mod:`ase.geometry` of
    the same name. Converts canonical 3D unit cell parameters
    a,b,c,α,β,γ into a 3x3 :class:`numpy.ndarray` representing the unit
    cell (in orthogonal coordinates).

    Parameters
    ----------
    a : float
        Unit cell side length a.
    b : float
        Unit cell side length b.
    c : float
        Unit cell side length c.
    alpha : float
        Unit cell angle alpha (degrees).
    beta : float
        Unit cell angle beta (degrees).
    gamma : float
        Unit cell angle gamma (degrees).

    Returns
    -------
    cell : :class:`numpy.ndarray`
        Unit cell represented as a 3x3 matrix in orthogonal coordinates.
    """

    eps = 2 * np.spacing(90) # ~1.4e-14

    if abs(abs(alpha) - 90) < eps:
        cos_alpha = 0
    else:
        cos_alpha = np.cos(alpha * np.pi / 180)

    if abs(abs(beta) - 90) < eps:
        cos_beta = 0
    else:
        cos_beta = np.cos(beta * np.pi / 180)

    if abs(abs(gamma) - 90) < eps:
        cos_gamma = 0
    else:
        cos_gamma = np.cos(gamma * np.pi / 180)

    if abs(gamma - 90) < eps:
        sin_gamma = 1
    elif abs(gamma + 90) < eps:
        sin_gamma = -1
    else:
        sin_gamma = np.sin(gamma * np.pi / 180.)

    cy = (cos_alpha - cos_beta * cos_gamma) / sin_gamma
    cz_sqr = 1 - cos_beta ** 2 - cy ** 2
    if cz_sqr < 0:
        raise ValueError('Could not create unit cell from given parameters.')

    cell = np.zeros((3, 3))
    cell[0, 0] = a
    cell[1, 0] = b * cos_gamma
    cell[1, 1] = b * sin_gamma
    cell[2, 0] = c * cos_beta
    cell[2, 1] = c * cy
    cell[2, 2] = c * np.sqrt(cz_sqr)
    return cell


@numba.njit()
def cellpar_to_cell_2D(a, b, alpha):
    """Converts 3 parameters defining a 2D unit cell a,b,α into a 2x2
    :class:`numpy.ndarray` representing the unit cell in orthogonal
    coordinates.

    Parameters
    ----------
    a : float
        Unit cell side length a.
    b : float
        Unit cell side length b.
    alpha : float
        Unit cell angle alpha (degrees).

    Returns
    -------
    cell : :class:`numpy.ndarray`
        Unit cell represented as a 2x2 matrix in orthogonal coordinates.
    """

    cell = np.zeros((2, 2))
    ang = alpha * np.pi / 180.
    cell[0, 0] = a
    cell[1, 0] = b * np.cos(ang)
    cell[1, 1] = b * np.sin(ang)
    return cell


@numba.njit()
def cell_to_cellpar(cell):
    """Converts a 3x3 :class:`numpy.ndarray` representing a unit cell in
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

    cellpar = np.zeros((6, ))
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
def cell_to_cellpar_2D(cell):
    """Converts a 2x2 :class:`numpy.ndarray` representing a unit cell in
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

    cellpar = np.zeros((3, ))
    cellpar[0] = np.linalg.norm(cell[0])
    cellpar[1] = np.linalg.norm(cell[1])
    cellpar[2] = np.rad2deg(np.arccos(
        np.dot(cell[0], cell[1]) / (cellpar[0] * cellpar[1])
    ))
    return cellpar


def neighbours_from_distance_matrix(
        n: int,
        dm: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
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

    if len(dm.shape) == 2: # 2D distance matrix
        inds = np.array([np.argpartition(row, n)[:n] for row in dm])
    elif len(dm.shape) == 1: # 1D condensed distance vector
        dm = squareform(dm)
        inds = []
        for i, row in enumerate(dm):
            inds_row = np.argpartition(row, n+1)[:n+1]
            inds_row = inds_row[inds_row != i][:n]
            inds.append(inds_row)
        inds = np.array(inds)
    else:
        msg = "Input must be a NumPy ndarray, either a 2D distance matrix " \
              "or a condensed distance matrix (as returned by SciPy's pdist)."
        ValueError(msg)

    # inds are the indexes of nns: inds[i,j] is the j-th nn to point i
    nn_dm = np.take_along_axis(dm, inds, axis=-1)
    sorted_inds = np.argsort(nn_dm, axis=-1)
    inds = np.take_along_axis(inds, sorted_inds, axis=-1)
    nn_dm = np.take_along_axis(nn_dm, sorted_inds, axis=-1)
    return nn_dm, inds


def random_cell(length_bounds=(1, 2), angle_bounds=(60, 120), dims=3):
    """Dimensions 2 and 3 only. Random unit cell with uniformally chosen
    length and angle parameters between bounds.
    """

    ll, lu = length_bounds
    al, au = angle_bounds

    if dims == 3:
        while True:
            lengths = [np.random.uniform(low=ll, high=lu) for _ in range(dims)]
            angles = [np.random.uniform(low=al, high=au) for _ in range(dims)]

            try:
                cell = cellpar_to_cell(*lengths, *angles)
                break
            except RuntimeError:
                continue

    elif dims == 2:
        lengths = [np.random.uniform(low=ll, high=lu) for _ in range(dims)]
        alpha = np.random.uniform(low=al, high=au)
        cell = cellpar_to_cell_2D(*lengths, alpha)

    else:
        msg = 'random_cell only implimented for dimensions 2 and 3 (passed ' \
              f'{dims})'
        raise NotImplementedError(msg)

    return cell
