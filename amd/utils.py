"""Helpful utility functions, e.g. unit cell diameter, converting
cell parameters to Cartesian form, and an ETA class."""

from typing import Tuple
import time
import datetime

import numpy as np
import scipy.spatial


def diameter(cell):
    """Diameter of a unit cell (as a square matrix in Cartesian form) 
    in 3 or fewer dimensions."""

    dims = cell.shape[0]
    if dims == 1:
        return cell[0][0]
    if dims == 2:
        d = np.amax(np.linalg.norm(np.array([cell[0] + cell[1], cell[0] - cell[1]]), axis=-1))
    elif dims == 3:
        d = np.amax(np.array([
            np.linalg.norm(cell[0] + cell[1] + cell[2]),
            np.linalg.norm(cell[0] + cell[1] - cell[2]),
            np.linalg.norm(cell[0] - cell[1] + cell[2]),
            np.linalg.norm(-cell[0] + cell[1] + cell[2])
        ]))
    else:
        raise ValueError(f'diameter only implimented for dimensions <= 3 (passed {dims})')
    return d


def cellpar_to_cell(a, b, c, alpha, beta, gamma):
    """Simplified version of function from ase.geometry of the same name.
    3D unit cell parameters a,b,c,α,β,γ --> cell as 3x3 NumPy array.
    """

    # Handle orthorhombic cells separately to avoid rounding errors
    eps = 2 * np.spacing(90.0, dtype=np.float64)  # around 1.4e-14

    cos_alpha = 0. if abs(abs(alpha) - 90.) < eps else np.cos(alpha * np.pi / 180.)
    cos_beta  = 0. if abs(abs(beta)  - 90.) < eps else np.cos(beta * np.pi / 180.)
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
        raise RuntimeError('Could not create unit cell from parameters ' + \
                           f'a={a},b={b},c={c},α={alpha},β={beta},γ={gamma}')

    return np.array([[a, 0, 0],
                     [b*cos_gamma, b*sin_gamma, 0],
                     [c*cos_beta, c*cy, c*np.sqrt(cz_sqr)]])


def cellpar_to_cell_2D(a, b, alpha):
    """2D unit cell parameters a,b,α --> cell as 2x2 ndarray."""

    return np.array([[a, 0],
                     [b * np.cos(alpha * np.pi / 180.), b * np.sin(alpha * np.pi / 180.)]])


def neighbours_from_distance_matrix(
        n: int,
        dm: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Given a distance matrix, find the ``n`` nearest neighbours of each item.

    Parameters
    ----------
    n : int
        Number of nearest neighbours to find for each item.
    dm : ndarray
        2D distance matrix or 1D condensed distance matrix.

    Returns
    -------
    tuple of numpy ndarrays (nn_dm, inds)
        For item ``i``, ``nn_dm[i][j]`` is the distance from item ``i`` to its ``j+1`` st
        nearest neighbour, and ``inds[i][j]`` is the index of this neighbour (``j+1`` since
        index 0 is the first nearest neighbour).
    """

    inds = None

    # 2D distance matrix
    if len(dm.shape) == 2:
        inds = np.array([np.argpartition(row, n)[:n] for row in dm])

    # 1D condensed distance vector
    elif len(dm.shape) == 1:
        dm = scipy.spatial.distance.squareform(dm)
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
    """Return a pair (motif, cell) representing a cubic lattice, passable to
    ``amd.AMD()`` or ``amd.PDD()``."""

    return (np.zeros((1, dims)), np.identity(dims) * scale)


def random_cell(length_bounds=(1, 2), angle_bounds=(60, 120), dims=3):
    """Random unit cell with uniformally chosen length and angle parameters
    between bounds."""

    lengths = [np.random.uniform(low=length_bounds[0],
                                 high=length_bounds[1]) 
               for _ in range(dims)]

    if dims == 3:
        angles = [np.random.uniform(low=angle_bounds[0],
                                    high=length_bounds[1]) 
                  for _ in range(dims)]
        return cellpar_to_cell(*lengths, *angles)

    if dims == 2:
        alpha = np.random.uniform(low=angle_bounds[0],
                                   high=length_bounds[1])
        return cellpar_to_cell_2D(*lengths, alpha)

    raise ValueError(f'random_cell only implimented for dimensions 2 and 3 (passed {dims})')


class ETA:
    """Pass total amount to do, then call .update() on every loop.
    This object will estimate an ETA and print it to the terminal."""

    # epochtime_{n+1} = factor * epochtime + (1-factor) * epochtime_{n}
    _moving_average_factor = 0.3

    def __init__(self, to_do, update_rate=100):
        self.to_do = to_do
        self.update_rate = update_rate
        self.counter = 0
        self.start_time = time.perf_counter()
        self.tic = self.start_time
        self.time_per_epoch = None
        self.done = False

    def update(self):
        """Call when one item is finished."""

        self.counter += 1

        if self.counter == self.to_do:
            msg = self._finished()
            print(msg, end='\r\n')
            self.done = True
            return

        if self.counter > self.to_do:
            return

        if not self.counter % self.update_rate:
            msg = self._end_epoch()
            print(msg, end='\r')

    def _end_epoch(self):
        toc = time.perf_counter()
        epoch_time = toc - self.tic
        if self.time_per_epoch is None:
            self.time_per_epoch = epoch_time
        else:
            self.time_per_epoch = ETA._moving_average_factor * epoch_time + \
                                  (1 - ETA._moving_average_factor) * self.time_per_epoch

        percent = round(100 * self.counter / self.to_do, 2)
        percent = '{:.2f}'.format(percent)
        remaining = int(((self.to_do - self.counter) / self.update_rate) * self.time_per_epoch)
        eta = str(datetime.timedelta(seconds=remaining))
        self.tic = toc
        return f'{percent}%, ETA {eta}' + ' ' * 30

    def _finished(self):
        total = time.perf_counter() - self.start_time
        msg = f'Total time: {round(total, 2)}s, ' \
              f'n passes: {self.counter} ' \
              f'({round(self.to_do/total, 2)} passes/second)'
        return msg
