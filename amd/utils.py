"""General utility functions."""

from typing import Tuple

import numpy as np
import numba
from scipy.spatial.distance import squareform

from .periodicset import PeriodicSet


def diameter(cell):
    """Diameter of a unit cell (as a square matrix in Cartesian/Orthogonal form)
    in 3 or fewer dimensions."""

    dims = cell.shape[0]
    if dims == 1:
        return cell[0][0]
    if dims == 2:
        d = np.amax(np.linalg.norm(np.array([cell[0] + cell[1], cell[0] - cell[1]]), axis=-1))
    elif dims == 3:
        diams = np.array([
            cell[0] + cell[1] + cell[2],
            cell[0] + cell[1] - cell[2],
            cell[0] - cell[1] + cell[2],
            -cell[0] + cell[1] + cell[2]
        ])
        d =  np.amax(np.linalg.norm(diams, axis=-1))
    else:
        raise ValueError(f'diameter only implimented for dimensions <= 3.')
    return d


@numba.njit()
def cellpar_to_cell(a, b, c, alpha, beta, gamma):
    """Simplified version of function from :mod:`ase.geometry` of the same name.
    3D unit cell parameters a,b,c,α,β,γ --> cell as 3x3 NumPy array.
    """

    eps = 2 * np.spacing(90.0)

    cos_alpha = 0. if abs(abs(alpha) - 90.) < eps else np.cos(alpha * np.pi / 180.)
    cos_beta = 0. if abs(abs(beta) - 90.) < eps else np.cos(beta * np.pi / 180.)
    cos_gamma = 0. if abs(abs(gamma) - 90.) < eps else np.cos(gamma * np.pi / 180.)

    if abs(gamma - 90) < eps:
        sin_gamma = 1.
    elif abs(gamma + 90) < eps:
        sin_gamma = -1.
    else:
        sin_gamma = np.sin(gamma * np.pi / 180.)

    cy = (cos_alpha - cos_beta * cos_gamma) / sin_gamma
    cz_sqr = 1. - cos_beta ** 2 - cy ** 2
    if cz_sqr < 0:
        raise RuntimeError('Could not create unit cell from given parameters.')

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
    """2D unit cell parameters a,b,α --> cell as 2x2 ndarray."""

    cell = np.zeros((2, 2))
    cell[0, 0] = a
    cell[1, 0] = b * np.cos(alpha * np.pi / 180.)
    cell[1, 1] = b * np.sin(alpha * np.pi / 180.)

    return cell


def neighbours_from_distance_matrix(
        n: int,
        dm: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Given a distance matrix, find the n nearest neighbours of each item.

    Parameters
    ----------
    n : int
        Number of nearest neighbours to find for each item.
    dm : numpy.ndarray
        2D distance matrix or 1D condensed distance matrix.

    Returns
    -------
    nn_dm, inds : Tuple[numpy.ndarray, numpy.ndarray] 
        ``nn_dm[i][j]`` is the distance from item ``i`` to its ``j+1`` st
        nearest neighbour, and ``inds[i][j]`` is the index of this neighbour 
        (``j+1`` since index 0 is the first nearest neighbour).
    """

    inds = None

    # 2D distance matrix
    if len(dm.shape) == 2:
        inds = np.array([np.argpartition(row, n)[:n] for row in dm])

    # 1D condensed distance vector
    elif len(dm.shape) == 1:
        dm = squareform(dm)
        inds = []
        for i, row in enumerate(dm):
            inds_row = np.argpartition(row, n+1)[:n+1]
            inds_row = inds_row[inds_row != i][:n]
            inds.append(inds_row)
        inds = np.array(inds)

    else:
        ValueError(
            'Input must be an ndarray, either a 2D distance matrix '
            'or a condensed distance matrix (returned by pdist).')

    # inds are the indexes of nns: inds[i,j] is the j-th nn to point i
    nn_dm = np.take_along_axis(dm, inds, axis=-1)
    sorted_inds = np.argsort(nn_dm, axis=-1)
    inds = np.take_along_axis(inds, sorted_inds, axis=-1)
    nn_dm = np.take_along_axis(nn_dm, sorted_inds, axis=-1)
    return nn_dm, inds


def lattice_cubic(scale=1, dims=3):
    """Return a pair ``(motif, cell)`` representing a hexagonal lattice, passable to
    ``amd.AMD()`` or ``amd.PDD()``. """

    return PeriodicSet(np.zeros((1, dims)), np.identity(dims) * scale)


def lattice_hexagonal(scale=1):
    """Dimension 3 only. Return a pair ``(motif, cell)`` representing a cubic lattice, 
    passable to ``amd.AMD()`` or ``amd.PDD()``."""

    return PeriodicSet(np.zeros((1, 3)), cellpar_to_cell(scale, scale, scale, 90, 90, 120))


def random_cell(length_bounds=(1, 2), angle_bounds=(60, 120), dims=3):
    """Dimensions 2 and 3 only. Random unit cell with uniformally chosen length and 
    angle parameters between bounds."""

    if dims == 3:
        while True:
            lengths = [np.random.uniform(low=length_bounds[0],
                                         high=length_bounds[1])
                       for _ in range(dims)]
            angles = [np.random.uniform(low=angle_bounds[0],
                                        high=length_bounds[1])
                      for _ in range(dims)]

            try:
                cell = cellpar_to_cell(*lengths, *angles)
                break
            except RuntimeError:
                continue

    elif dims == 2:
        lengths = [np.random.uniform(low=length_bounds[0],
                                     high=length_bounds[1])
                       for _ in range(dims)]
        alpha = np.random.uniform(low=angle_bounds[0],
                                  high=length_bounds[1])
        cell = cellpar_to_cell_2D(*lengths, alpha)

    else:
        raise ValueError(f'random_cell only implimented for dimensions 2 and 3 (passed {dims})')

    return cell
