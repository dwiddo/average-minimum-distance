"""Implements core function nearest_neighbours used for AMD and PDD
calculations.
"""

from typing import Tuple, Iterable
from itertools import product

import numba
import numpy as np
from scipy.spatial import KDTree
from scipy.spatial.distance import cdist

__all__ = [
    'nearest_neighbours', 'nearest_neighbours_minval',
    'generate_concentric_cloud'
]


def nearest_neighbours(
        motif: np.ndarray,
        cell: np.ndarray,
        x: np.ndarray,
        k: int
) -> Tuple[np.ndarray, ...]:
    """Find the ``k`` nearest neighbours in a periodic set for points in
    ``x``.

    Given a periodic set described by ``motif`` and ``cell``, a query
    set of points ``x`` and an integer ``k``, find the ``k`` nearest
    neighbours in the periodic set for all points in ``x``. Return
    an array of distances to neighbours, the point cloud generated
    during the search and the indices of which points in the cloud are
    the neighbours of points in ``x``.

    Parameters
    ----------
    motif : :class:`numpy.ndarray`
        Cartesian coordinates of the motif, shape (no points, dims).
    cell : :class:`numpy.ndarray`
        The unit cell as a square array, shape (dims, dims).
    x : :class:`numpy.ndarray`
        Array of points to query for neighbours. For AMD/PDD invariants
        this is the motif, or more commonly an asymmetric unit of it.
    k : int
        Number of nearest neighbours to find for each point in ``x``.

    Returns
    -------
    dists : numpy.ndarray
        Array shape ``(x.shape[0], k)`` of distances from points in
        ``x`` to their ``k`` nearest neighbours in the periodic set in
        order, e.g. ``dists[m][n]`` is the distance from ``x[m]`` to its
        n-th nearest neighbour in the periodic set.
    cloud : numpy.ndarray
        Collection of points in the periodic set that were generated
        during the search.
    inds : numpy.ndarray
        Array shape ``(x.shape[0], k)`` containing the indices of
        nearest neighbours in ``cloud``, e.g. the n-th nearest neighbour
        to ``x[m]`` is ``cloud[inds[m][n]]``.
    """

    m, dims = motif.shape

    # Generate a cloud of lattice points such that lattice + motif has k points
    if dims == 3:
        int_lat_generator = _generate_integer_lattice_3D()
    else:
        int_lat_generator = _generate_integer_lattice(dims)

    # we want the initial cloud to be a decent estimate set.
    # get largest ratio of cell side lengths, round down and add this many layers

    int_lat_generator = iter(int_lat_generator)
    n_points = 0
    int_lat_cloud = []
    while n_points <= k:
        layer = next(int_lat_generator)
        n_points += layer.shape[0] * m
        int_lat_cloud.append(layer)

    # Add one layer to the lattice, on average this is faster
    int_lat_cloud.append(next(int_lat_generator))
    cloud = _int_lattice_to_cloud(motif, cell, np.concatenate(int_lat_cloud))

    # Find k neighbours in the point cloud for points in x
    dists, inds = KDTree(
        cloud, leafsize=30, compact_nodes=False, balanced_tree=False
    ).query(x, k=k)

    # Generate layers of lattice points until they are too far away to give
    # nearer neighbours than have already been found. For a lattice point l,
    # points in l + motif are further away from x than |l| - max|p-p'| (where
    # p in x, p' in motif), used to check if l is too far away.
    motif_diameter = float(np.amax(cdist(x, motif)))
    lattice_layers = []
    while True:
        lattice = _close_lattice_points(
            next(int_lat_generator), cell, dists[:, -1], motif_diameter
        )
        if lattice.size == 0:
            break
        lattice_layers.append(lattice)

    if lattice_layers:
        lattice_layers = np.concatenate(lattice_layers)
        cloud = np.vstack((
            cloud[np.unique(inds)], _lattice_to_cloud(motif, lattice_layers)
        ))
        dists, inds = KDTree(
            cloud, leafsize=30, compact_nodes=False, balanced_tree=False
        ).query(x, k=k)

    return dists, cloud, inds


def _generate_integer_lattice(dims: int) -> Iterable[np.ndarray[np.float64]]:
    """Generate batches of integer lattice points. Each yield gives all
    points (that have not already been yielded) inside a sphere centered
    at the origin with radius d; d starts at 0 and increments by 1 on
    each loop.

    Parameters
    ----------
    dims : int
        The dimension of Euclidean space the lattice is in.

    Yields
    -------
    :class:`numpy.ndarray`
        Yields arrays of integer points in `dims`-dimensional Euclidean
        space.
    """

    d = 0

    if dims == 1:
        yield np.zeros((1, 1), dtype=np.float64)
        while True:
            d += 1
            yield np.array([[-d], [d]], dtype=np.float64)

    ymax = {}
    while True:
        positive_int_lattice = []
        while True:
            batch = False
            for xy in product(range(d + 1), repeat=dims-1):
                if xy not in ymax:
                    ymax[xy] = 0
                if _in_sphere(xy, ymax[xy], d):
                    positive_int_lattice.append((*xy, ymax[xy]))
                    batch = True
                    ymax[xy] += 1
            if not batch:
                break
        yield _reflect_positive_integer_lattice(np.array(positive_int_lattice))
        d += 1


@numba.njit(cache=True)
def _generate_integer_lattice_3D() -> Iterable[np.ndarray[np.float64]]:
    """3D specific version of _generate_integer_lattice() which is
    accelerated by numba. 3D is the most common case and the function is
    difficult to generalise to any dimension with numba.

    Yields
    -------
    :class:`numpy.ndarray`
        Yields arrays of integer points in 3-dimensional Euclidean
        space.
    """

    d = 0
    ymax = {}
    while True:
        positive_int_lattice = []
        while True:
            batch = False
            for x in range(d + 1):
                for y in range(d + 1):
                    xy = (x, y)
                    if xy not in ymax:
                        ymax[xy] = 0
                    if x**2 + y**2 + ymax[xy]**2 <= d**2:
                        positive_int_lattice.append((x, y, ymax[xy]))
                        batch = True
                        ymax[xy] += 1
            if not batch:
                break
        yield _reflect_positive_integer_lattice(np.array(positive_int_lattice))
        d += 1


@numba.njit(cache=True)
def _reflect_positive_integer_lattice(
        positive_int_lattice: np.ndarray
) -> np.ndarray[np.float64]:
    """Reflect points in the positive quadrant across all combinations
    of axes, without duplicating points that are invariant under
    reflections.
    """

    dims = positive_int_lattice.shape[-1]
    batches = []
    batches.extend(positive_int_lattice)

    for n_reflections in range(1, dims + 1):

        axes = np.arange(n_reflections)
        batches.extend(_reflect_in_axes(positive_int_lattice, axes))

        while True:
            i = n_reflections - 1
            for _ in range(n_reflections):
                if axes[i] != i + dims - n_reflections:
                    break
                i -= 1
            else:
                break
            axes[i] += 1
            for j in range(i + 1, n_reflections):
                axes[j] = axes[j-1] + 1
            batches.extend(_reflect_in_axes(positive_int_lattice, axes))

    int_lattice = np.empty(shape=(len(batches), dims), dtype=np.float64)
    for i in range(len(batches)):
        int_lattice[i] = batches[i]

    return int_lattice


@numba.njit(cache=True)
def _reflect_in_axes(
        positive_int_lattice: np.ndarray,
        axes: np.ndarray
) -> np.ndarray:
    """Reflect points in `positive_int_lattice` in the axes described by
    `axes`, without duplicating invariant points.
    """

    not_on_axes = (positive_int_lattice[:, axes] == 0).sum(axis=-1) == 0
    int_lattice = positive_int_lattice[not_on_axes]
    int_lattice[:, axes] *= -1
    return int_lattice


@numba.njit(cache=True)
def _close_lattice_points(
        int_lattice: np.ndarray,
        cell: np.ndarray,
        max_nn_dists: np.ndarray,
        max_cdist: float
) -> np.ndarray[np.float64]:
    """Given integer lattice points, a unit cell, ``max_cdist`` (max of
    cdist(x, motif)) and ``max_nn_dist`` (max of the dists to k-th
    nearest neighbours found so far), return lattice points which are
    close enough such that the corresponding motif copy could contain
    nearest neighbours.
    """

    lattice = int_lattice @ cell
    bound = np.amax(max_nn_dists) + max_cdist
    return lattice[np.sqrt(np.sum(lattice ** 2, axis=-1)) < bound]


@numba.njit(cache=True)
def _lattice_to_cloud(
        motif: np.ndarray,
        lattice: np.ndarray
) -> np.ndarray[np.float64]:
    """Transform a batch of non-integer lattice points (generated by
    _generate_integer_lattice then mutliplied by the cell) into a cloud
    of points from a periodic set with the motif and cell.
    """

    m = len(motif)
    layer = np.empty((m * len(lattice), motif.shape[-1]), dtype=np.float64)
    i1 = 0
    for translation in lattice:
        i2 = i1 + m
        layer[i1:i2] = motif + translation
        i1 = i2
    return layer


@numba.njit(cache=True)
def _int_lattice_to_cloud(
        motif: np.ndarray,
        cell: np.ndarray,
        int_lattice: np.ndarray
) -> np.ndarray[np.float64]:
    """Transform a batch of integer lattice points (generated by
    _generate_integer_lattice) into a cloud of points from a periodic
    set with the motif and cell.
    """
    return _lattice_to_cloud(motif, int_lattice @ cell)


@numba.njit(cache=True)
def _in_sphere(xy: Tuple[float, ...], z: float, d: float) -> bool:
    """True if sum(i^2 for i in xy) + z^2 <= d^2."""

    s = z ** 2
    for val in xy:
        s += val ** 2
    return s <= d ** 2


def nearest_neighbours_minval(
        motif: np.ndarray,
        cell: np.ndarray,
        min_val: float
) -> Tuple[np.ndarray, ...]:
    """Return the same ``dists``/PDD matrix as ``nearest_neighbours``,
    but with enough columns such that all values in the last column are
    at least ``min_val``. Unlike ``nearest_neighbours``, does not take a
    query array ``x`` but only finds neighbours to motif points, and
    does not return the point cloud or indices of the nearest
    neighbours. Used in ``PDD_reconstructable``.
    """

    # Generate initial cloud of points from the periodic set
    int_lat_generator = _generate_integer_lattice(cell.shape[0])
    int_lat_generator = iter(int_lat_generator)
    cloud = []
    for _ in range(3):
        cloud.append(_lattice_to_cloud(motif, next(int_lat_generator) @ cell))
    cloud = np.concatenate(cloud)

    # Find k neighbours in the point cloud for points in motif
    dists_, inds = KDTree(
        cloud, leafsize=30, compact_nodes=False, balanced_tree=False
    ).query(motif, k=cloud.shape[0])
    dists = np.zeros_like(dists_, dtype=np.float64)

    # Add layers & find k nearest neighbours until all distances smaller than
    # min_val don't change
    max_cdist = np.amax(cdist(motif, motif))
    while True:
        if np.all(dists_[:, -1] >= min_val):
            col = np.argwhere(np.all(dists_ >= min_val, axis=0))[0][0] + 1
            if np.array_equal(dists[:, :col], dists_[:, :col]):
                break
        dists = dists_
        lattice = next(int_lat_generator) @ cell
        closest_dist_bound = np.linalg.norm(lattice, axis=-1) - max_cdist
        is_close = closest_dist_bound <= np.amax(dists_[:, -1])
        if not np.any(is_close):
            break
        cloud = np.vstack((cloud, _lattice_to_cloud(motif, lattice[is_close])))
        dists_, inds = KDTree(
            cloud, leafsize=30, compact_nodes=False, balanced_tree=False
        ).query(motif, k=cloud.shape[0])

    k = np.argwhere(np.all(dists >= min_val, axis=0))[0][0]
    return dists_[:, 1:k+1], cloud, inds


def generate_concentric_cloud(motif, cell):
    """Generates batches of points from a periodic set given by (motif,
    cell) which get successively further away from the origin.

    Each yield gives all points (that have not already been yielded)
    which lie in a unit cell whose corner lattice point was generated by
    ``generate_integer_lattice(motif.shape[1])``.

    Parameters
    ----------
    motif : :class:`numpy.ndarray`
        Cartesian representation of the motif, shape (no points, dims).
    cell : :class:`numpy.ndarray`
        Cartesian representation of the unit cell, shape (dims, dims).

    Yields
    -------
    :class:`numpy.ndarray`
        Yields arrays of points from the periodic set.
    """

    int_lat_generator = _generate_integer_lattice(cell.shape[0])
    for layer in int_lat_generator:
        yield _lattice_to_cloud(motif, layer @ cell)
