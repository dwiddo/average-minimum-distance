"""Implements core function nearest_neighbours used for AMD and PDD
calculations.
"""

from typing import Tuple, Iterable
from itertools import product, tee
import functools

import numba
import numpy as np
from scipy.spatial import KDTree
from scipy.spatial.distance import cdist

__all__ = [
    'nearest_neighbours',
    'nearest_neighbours_data',
    'nearest_neighbours_minval',
    'generate_concentric_cloud'
]


def nearest_neighbours(
        motif: np.ndarray, cell: np.ndarray, x: np.ndarray, k: int
) -> np.ndarray:
    """Find distances to ``k`` nearest neighbours in a periodic set for
    each point in ``x``.

    Given a periodic set described by ``motif`` and ``cell``, a query
    set of points ``x`` and an integer ``k``, find distances to the
    ``k`` nearest neighbours in the periodic set for all points in
    ``x``. Returns an array with shape (x.shape[0], k) of distances to
    the neighbours. This function only returns distances, see the
    function nearest_neighbours_data() to also get the point cloud and
    indices of the points which are neighbours.

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
    """

    m, dims = motif.shape
    # Get an initial collection of lattice points + a generator for more
    int_lat_cloud, int_lat_generator = _get_integer_lattice(dims, m, k)
    cloud = _int_lattice_to_cloud(motif, cell, int_lat_cloud)

    # Squared distances to k nearest neighbours
    sqdists = _cdist_sqeuclidean(x, cloud)
    motif_diam = np.sqrt(_max_in_columns(sqdists, m))
    sqdists.partition(k - 1)
    sqdists = sqdists[:, :k]
    sqdists.sort()

    # Generate layers of lattice until they are too far away to give
    # nearer neighbours. For a lattice point l, points in l + motif are
    # further away from x than |l| - max|p-p'| (p in x, p' in motif),
    # giving a bound we can use to rule out distant lattice points
    max_sqd = np.amax(sqdists[:, -1])
    bound = (np.sqrt(max_sqd) + motif_diam) ** 2

    while True:

        # Get next layer of lattice
        lattice = _close_lattice_points(next(int_lat_generator), cell, bound)
        if lattice.size == 0:  # None are close enough
            break

        # Squared distances to new points
        sqdists_ = _cdist_sqeuclidean(x, _lattice_to_cloud(motif, lattice))
        close = sqdists_ < max_sqd
        if not np.any(close):  # None are close enough
            break

        # Squared distances to up to k nearest new points
        sqdists_ = sqdists_[:, np.any(close, axis=0)]
        if sqdists_.shape[-1] > k:
            sqdists_.partition(k - 1)
        sqdists_ = sqdists_[:, :k]
        sqdists_.sort()

        # Merge existing and new distances
        sqdists = _merge_sorted_arrays(sqdists, sqdists_)
        max_sqd = np.amax(sqdists[:, -1])
        bound = (np.sqrt(max_sqd) + motif_diam) ** 2

    return np.sqrt(sqdists)


def nearest_neighbours_data(
        motif: np.ndarray, cell: np.ndarray, x: np.ndarray, k: int
) -> np.ndarray:
    """Find the ``k`` nearest neighbours in a periodic set for each
    point in ``x``.

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
    int_lat, int_lat_gen = _get_integer_lattice(dims, m, k)
    cloud = _int_lattice_to_cloud(motif, cell, int_lat)
    dists = cdist(x, cloud)
    motif_diam = _max_in_columns(dists, m)
    inds = np.argsort(dists)[:, :k]
    dists = np.take_along_axis(dists, inds, -1)
    b = (np.amax(dists[:, -1]) + motif_diam) ** 2

    while True:

        lattice = _close_lattice_points(next(int_lat_gen), cell, b)
        if lattice.size == 0:
            break

        cloud = np.concatenate((cloud, _lattice_to_cloud(motif, lattice)))
        dists = cdist(x, cloud)
        inds = np.argsort(dists)[:, :k]
        dists = np.take_along_axis(dists, inds, -1)
        b = (np.amax(dists[:, -1]) + motif_diam) ** 2

    return dists, cloud, inds


def _get_integer_lattice_cache(f):
    """Specialised cache for ``_get_integer_lattice()``."""

    cache = {}
    num_points_cache = {}

    @functools.wraps(f)
    def wrapper(dims, m, k):

        if dims not in num_points_cache:
            num_points_cache[dims] = []

        n_points = 0
        n_layers = 0
        within_cache = False
        for num_p in num_points_cache[dims]:
            if n_points > k / m:
                within_cache = True
                break
            n_points += num_p
            n_layers += 1
        n_layers += 1

        if not (within_cache and (dims, n_layers) in cache):
            layers, int_lat_generator = f(dims, m, k)
            n_layers = len(layers)
            if len(num_points_cache[dims]) < n_layers:
                num_points_cache[dims] = [len(i) for i in layers]
            layers = np.concatenate(layers)
            cache[(dims, n_layers)] = [layers, int_lat_generator]

        arr, g = cache[(dims, n_layers)]
        cache[(dims, n_layers)][1], r = tee(g)
        return arr, r

    return wrapper


@_get_integer_lattice_cache
def _get_integer_lattice(dims, m, k):
    """Return an initial batch of integer lattice points (number
    according to m and k) and a generator for more distant points.

    Parameters
    ----------
    dims : int
        The dimension of Euclidean space the lattice is in.
    m : int
        Number of motif points.
    k : int
        Number of nearest neighbours to find (parameter of
        nearest_neighbours).

    Returns
    -------
    initial_integer_lattice : :class:`numpy.ndarray`
        A collection of integer lattice points. Consists of the first
        few layers generated by ``integer_lattice_generator`` (number of
        layers depends on m, k).
    integer_lattice_generator
        A generator for integer lattice points more distant than those
        in ``initial_integer_lattice``.
    """

    g = iter(_generate_integer_lattice(dims))
    layers = [next(g)]
    n_points = 1
    while n_points <= k / m:
        layer = next(g)
        n_points += layer.shape[0]
        layers.append(layer)
    layers.append(next(g))
    return layers, g


def memoized_generator(f):
    """Caches results of a generator."""
    cache = {}
    @functools.wraps(f)
    def wrapper(*args):
        if args not in cache:
            cache[args] = f(*args)
        cache[args], r = tee(cache[args])
        return r
    return wrapper


@memoized_generator
def _generate_integer_lattice(dims: int) -> Iterable[np.ndarray]:
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
                if sum(i**2 for i in xy) + ymax[xy]**2 <= d**2:
                    positive_int_lattice.append((*xy, ymax[xy]))
                    batch = True
                    ymax[xy] += 1
            if not batch:
                break
        pos_int_lat = np.array(positive_int_lattice, dtype=np.float64)
        yield _reflect_positive_integer_lattice(pos_int_lat)
        d += 1


@numba.njit(cache=True, fastmath=True)
def _reflect_positive_integer_lattice(
        positive_int_lattice: np.ndarray
) -> np.ndarray:
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


@numba.njit(cache=True, fastmath=True)
def _reflect_in_axes(
        positive_int_lattice: np.ndarray, axes: np.ndarray
) -> np.ndarray:
    """Reflect points in `positive_int_lattice` in the axes described by
    `axes`, without duplicating invariant points.
    """
    not_on_axes = (positive_int_lattice[:, axes] == 0).sum(axis=-1) == 0
    int_lattice = positive_int_lattice[not_on_axes]
    int_lattice[:, axes] *= -1
    return int_lattice


@numba.njit(cache=True, fastmath=True)
def _close_lattice_points(
        int_lattice: np.ndarray, cell: np.ndarray, bound: float
) -> np.ndarray:
    """Given integer lattice points, a unit cell and ``bound``, return
    lattice points which are close enough such that the corresponding
    motif copy could contain nearest neighbours. ``bound`` should be
    equal to (max_d + motif_diam) ** 2, where max_d is the maximum
    k-th nearest neighbour distance found so far and motif_diam is the
    largest distance between any point in the query set and motif.
    """

    lattice = int_lattice @ cell
    inds = []
    for i in range(len(lattice)):
        s = 0
        for xyz in lattice[i]:
            s += xyz ** 2
        if s < bound:
            inds.append(i)
    ret = np.empty((len(inds), lattice.shape[-1]), dtype=np.float64)
    for i in range(len(inds)):
        ret[i] = lattice[inds[i]]
    return ret


@numba.njit(cache=True, fastmath=True)
def _lattice_to_cloud(motif: np.ndarray, lattice: np.ndarray) -> np.ndarray:
    """Transform a batch of lattice points (generated by
    _generate_integer_lattice then mutliplied by the cell) into a cloud
    of points from a periodic set.
    """

    m = len(motif)
    layer = np.empty((m * len(lattice), motif.shape[-1]), dtype=np.float64)
    i1 = 0
    for translation in lattice:
        i2 = i1 + m
        layer[i1:i2] = motif + translation
        i1 = i2
    return layer


@numba.njit(cache=True, fastmath=True)
def _int_lattice_to_cloud(
        motif: np.ndarray, cell: np.ndarray, int_lattice: np.ndarray
) -> np.ndarray:
    """Transform a batch of integer lattice points (generated by
    _generate_integer_lattice) into a cloud of points from a periodic
    set.
    """
    return _lattice_to_cloud(motif, int_lattice @ cell)


@numba.njit(cache=True, fastmath=True)
def _cdist_sqeuclidean(arr1, arr2):
    """Squared Euclidean distance between points in ``arr1`` and
    ``arr2``."""
    n1, n2 = arr1.shape[0], arr2.shape[0]
    res = np.empty((n1, n2), dtype=np.float64)
    for i in range(n1):
        for j in range(n2):
            s = 0.
            for n in range(arr1.shape[-1]):
                s += (arr1[i, n] - arr2[j, n]) ** 2
            res[i, j] = s
    return res


@numba.njit(cache=True, fastmath=True)
def _max_in_column(arr, col):
    """Return maximum value in chosen column col of array arr."""
    ret = 0
    for i in range(arr.shape[0]):
        v = arr[i, col]
        if v > ret:
            ret = v
    return ret


@numba.njit(cache=True, fastmath=True)
def _max_in_columns(arr, max_col):
    """Return maximum value in all columns up to a chosen column col of
    array arr."""
    ret = 0
    for col in range(max_col):
        v = _max_in_column(arr, col)
        if v > ret:
            ret = v
    return ret


@numba.njit(cache=True, fastmath=True)
def _merge_sorted_arrays(dists, dists_):
    """Merge two 2D arrays sorted along last axis into one sorted array
    with same number of columns as ``dists``. Optimised for the distance
    arrays in nearest_neighbours, where ``dists`` will contain most of
    the smallest elements and only a few values in later columns will
    need to be replaced with values in ``dists_``.
    """

    m, n_new_points = dists_.shape
    ret = np.copy(dists)

    for i in range(m):
        # Traverse row backwards until value smaller than dists_[i, 0]
        j = 0
        dp_ = 0
        d_ = dists_[i, dp_]
        while True:
            j -= 1
            if dists[i, j] <= d_:
                j += 1
                break

        if j == 0:  # If dists_[i, 0] >= dists[i, -1], no need to insert
            continue

        # dp points to dists[i], dp_ points to dists_[i].
        # fill ret with the larger dist, then increment pointers and repeat.
        dp = j
        d = dists[i, dp]

        while j < 0:
            if d <= d_:
                ret[i, j] = d
                dp += 1
                d = dists[i, dp]
            else:
                ret[i, j] = d_
                dp_ += 1
                if dp_ < n_new_points:
                    d_ = dists_[i, dp_]
                else:  # ran out of points in dists_
                    d_ = np.inf
            j += 1

    return ret


def nearest_neighbours_minval(
        motif: np.ndarray, cell: np.ndarray, min_val: float
) -> Tuple[np.ndarray, ...]:
    """Return the same ``dists``/PDD matrix as ``nearest_neighbours``,
    but with enough columns such that all values in the last column are
    at least ``min_val``. Unlike ``nearest_neighbours``, does not take a
    query array ``x`` but only finds neighbours to motif points, and
    does not return the point cloud or indices of the nearest
    neighbours. Used in ``PDD_reconstructable``.
    
    TODO: this function should be updated in line with
    nearest_neighbours.
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
