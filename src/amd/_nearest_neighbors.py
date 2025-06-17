"""Implements core function nearest_neighbors used for AMD and PDD
calculations.
"""

from typing import Tuple, Iterable, Iterator, Callable, List
from itertools import product, tee, accumulate
import functools

import numba
import numpy as np

from ._types import FloatArray, UIntArray

__all__ = ["nearest_neighbors", "nearest_neighbors_data", "nearest_neighbors_minval"]


def nearest_neighbors(
    motif: FloatArray, cell: FloatArray, x: FloatArray, k: int
) -> FloatArray:
    """Find distances to ``k`` nearest neighbors in a periodic set for
    each point in ``x``.

    Given a periodic set described by ``motif`` and ``cell``, a query
    set of points ``x`` and an integer ``k``, find distances to the
    ``k`` nearest neighbors in the periodic set for all points in
    ``x``. Returns an array with shape ``(x.shape[0], k)`` of distances
    to the neighbors. This function only returns distances, see also
    ``nearest_neighbors_data()`` which also returns indices of those
    points which are neighbors.

    Parameters
    ----------
    motif : :class:`numpy.ndarray`
        Cartesian coordinates of the motif, shape (no points, dims).
    cell : :class:`numpy.ndarray`
        The unit cell as a square array, shape (dims, dims).
    x : :class:`numpy.ndarray`
        Array of points to query for neighbors. For isometry invariants
        of crystals this is typically the asymmetric unit.
    k : int
        Number of nearest neighbors to find for each point in ``x``.

    Returns
    -------
    dists : numpy.ndarray
        Array shape ``(x.shape[0], k)`` of distances from points in
        ``x`` to their ``k`` nearest neighbors in the periodic set in
        order, e.g. ``dists[m][n]`` is the distance from ``x[m]`` to its
        n-th nearest neighbor in the periodic set.
    """

    m, dims = motif.shape

    # Get an initial collection of points and a generator for more
    int_lattice, int_lat_generator = _integer_lattice_batches(dims, m, k)
    cloud = _lattice_to_cloud(motif, int_lattice @ cell)

    # Squared distances to k nearest neighbors
    sqdists = _cdist_sqeulcidean(x, cloud)
    motif_diam = np.sqrt(sqdists[:, :m].max())
    sqdists.partition(k - 1)
    sqdists = sqdists[:, :k]
    sqdists.sort()

    # If a lattice point l has |l| >= max|p-p'| + max_d for p ∈ x, p' ∈ motif
    # then |p-(l+p')| >= max_d for all p, p' and l can be discarded.
    max_sqd = sqdists[:, -1].max()
    bound = (np.sqrt(max_sqd) + motif_diam) ** 2

    while True:
        # Get next layer of lattice and prune distant lattice points
        lattice = next(int_lat_generator) @ cell

        lattice = lattice[_where_sq_sum_lt_bound(lattice, bound)]
        if lattice.size == 0:  # None are close enough
            break

        # Squared distances to new points
        cloud = _lattice_to_cloud(motif, lattice)
        sqdists_ = _cdist_sqeulcidean(x, cloud)
        where_close = (sqdists_ < max_sqd).any(axis=0)
        if not np.any(where_close):  # None are close enough
            break

        # Squared distances to up to k nearest new points
        sqdists_ = sqdists_[:, where_close]
        if sqdists_.shape[-1] > k:
            sqdists_.partition(k - 1)
            sqdists_ = sqdists_[:, :k]
        sqdists_.sort()

        # Merge existing and new distances
        sqdists = _merge_sorted_arrays(sqdists, sqdists_)
        max_sqd = sqdists[:, -1].max()
        bound = (np.sqrt(max_sqd) + motif_diam) ** 2

    return np.sqrt(sqdists)


def nearest_neighbors_data(
    motif: FloatArray, cell: FloatArray, x: FloatArray, k: int
) -> Tuple[FloatArray, FloatArray, UIntArray]:
    """Find the ``k`` nearest neighbors in a periodic set for each
    point in ``x``, along with the point cloud generated during the
    search and indices of the nearest neighbors in the cloud.

    Note: the points in ``x`` are

    Given a periodic set described by ``motif`` and ``cell``, a query
    set of points ``x`` and an integer ``k``, find the ``k`` nearest
    neighbors in the periodic set for all points in ``x``. Return
    an array of distances to neighbors, the point cloud generated
    during the search and the indices of which points in the cloud are
    the neighbors of points in ``x``.

    Note: the point ``cloud[i]`` in the periodic set comes from (is
    translationally equivalent to) the motif point
    ``motif[i % len(motif)]``, because points are added to ``cloud`` in
    batches of whole unit cells and not rearranged.

    Parameters
    ----------
    motif : :class:`numpy.ndarray`
        Cartesian coordinates of the motif, shape (no points, dims).
    cell : :class:`numpy.ndarray`
        The unit cell as a square array, shape (dims, dims).
    x : :class:`numpy.ndarray`
        Array of points to query for neighbors. For AMD/PDD invariants
        this is the motif, or more commonly an asymmetric unit of it.
    k : int
        Number of nearest neighbors to find for each point in ``x``.

    Returns
    -------
    dists : numpy.ndarray
        Array shape ``(x.shape[0], k)`` of distances from points in
        ``x`` to their ``k`` nearest neighbors in the periodic set in
        order, e.g. ``dists[m][n]`` is the distance from ``x[m]`` to its
        n-th nearest neighbor in the periodic set.
    cloud : numpy.ndarray
        Collection of points in the periodic set that were generated
        during the search. Arranged such that cloud[i] comes from the
        motif point motif[i % len(motif)] by translation.
    inds : numpy.ndarray
        Array shape ``(x.shape[0], k)`` containing the indices of
        nearest neighbors in ``cloud``, e.g. the n-th nearest neighbor
        to ``x[m]`` is ``cloud[inds[m][n]]``.
    """

    full_cloud = []
    m, dims = motif.shape

    int_lattice, int_lat_generator = _integer_lattice_batches(dims, m, k)
    cloud = _lattice_to_cloud(motif, int_lattice @ cell)
    full_cloud.append(cloud)
    cloud_ind_offset = len(cloud)

    sqdists = _cdist_sqeulcidean(x, cloud)
    motif_diam = np.sqrt(sqdists[:, :m].max())
    part_inds = np.argpartition(sqdists, k - 1)[:, :k]
    part_sqdists = np.take_along_axis(sqdists, part_inds, axis=-1)[:, :k]
    part_sort_inds = np.argsort(part_sqdists)
    inds = np.take_along_axis(part_inds, part_sort_inds, axis=-1)
    sqdists = np.take_along_axis(part_sqdists, part_sort_inds, axis=-1)

    max_sqd = sqdists[:, -1].max()
    bound = (np.sqrt(max_sqd) + motif_diam) ** 2

    while True:
        lattice = next(int_lat_generator) @ cell
        lattice = lattice[np.einsum("ij,ij->i", lattice, lattice) < bound]
        if lattice.size == 0:
            break

        cloud = _lattice_to_cloud(motif, lattice)
        sqdists_ = _cdist_sqeulcidean(x, cloud)
        close = sqdists_ < max_sqd
        if not np.any(close):
            break

        argpart_upto = min(k - 1, sqdists_.shape[-1] - 1)
        part_inds = np.argpartition(sqdists_, argpart_upto)[:, :k]
        part_sqdists = np.take_along_axis(sqdists_, part_inds, axis=-1)[:, :k]
        part_sort_inds = np.argsort(part_sqdists)
        inds_ = np.take_along_axis(part_inds, part_sort_inds, axis=-1)
        sqdists_ = np.take_along_axis(part_sqdists, part_sort_inds, axis=-1)
        inds_ += cloud_ind_offset
        cloud_ind_offset += len(cloud)
        full_cloud.append(cloud)

        sqdists, inds = _merge_sorted_arrays_and_inds(sqdists, sqdists_, inds, inds_)
        max_sqd = sqdists[:, -1].max()
        bound = (np.sqrt(max_sqd) + motif_diam) ** 2

    return np.sqrt(sqdists), np.concatenate(full_cloud), inds


def _integer_lattice_batches_cache(func: Callable) -> Callable:
    """Specialised cache for ``_integer_lattice_batches``. Stores layers
    of integer lattice points from ``_integer_lattice_generator``.
    How many layers are needed depends on the ratio of k to m, so this
    is passed to the cache. One cache is kept for each dimension.
    """

    cache = {}  # (dims, n_layers): (layers, generator)
    npoints_cache = {}  # dims: [num points accumulated in layers 1,2,...]

    @functools.wraps(func)
    def wrapper(
        dims: int, m: int, k: int
    ) -> Tuple[List[FloatArray], Iterator[FloatArray]]:
        if dims not in npoints_cache:
            npoints_cache[dims] = []

        # Use npoints_cache to get how many layers of the lattice are needed
        # Delicate code; change in accordance with _integer_lattice_batches
        n_layers = 1
        for n_points in npoints_cache[dims]:
            if n_points * m > k:
                break
            n_layers += 1
        n_layers += 1

        # Update cache and possibly npoints_cache
        if (dims, n_layers) not in cache:
            layers, generator = func(dims, m, k)
            n_layers = len(layers)
            # If more layers were made than seen so far, update npoints_cache
            if len(npoints_cache[dims]) < n_layers:
                npoints_cache[dims] = list(accumulate(len(i) for i in layers))
            cache[(dims, n_layers)] = [np.concatenate(layers), generator]

        layers, generator = cache[(dims, n_layers)]
        cache[(dims, n_layers)][1], ret_generator = tee(generator)
        return layers, ret_generator

    return wrapper


@_integer_lattice_batches_cache
def _integer_lattice_batches(
    dims: int, m: int, k: int
) -> Tuple[List[FloatArray], Iterator[FloatArray]]:
    """Return an initial batch of integer lattice points (number
    according to k & m) and a generator for more distant points.

    Parameters
    ----------
    dims : int
        The dimension of Euclidean space the lattice is in.
    m : int
        Number of motif points. Used to determine how many layers of
        lattice points to put into the initial batch.
    k : int
        Parameter of ``nearest_neighbors``, the number of neighbors to
        find for each point. Used to determine how many layers of
        lattice points to put into the initial batch.

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

    int_lattice_generator = iter(_integer_lattice_generator(dims))
    layers = [next(int_lattice_generator)]
    n_points = 1
    while n_points * m <= k:
        layer = next(int_lattice_generator)
        n_points += layer.shape[0]
        layers.append(layer)
    layers.append(next(int_lattice_generator))
    return layers, int_lattice_generator


def _memoized_generator(generator_function: Callable) -> Callable:
    """Caches results of a generator."""
    cache = {}

    @functools.wraps(generator_function)
    def wrapper(*args) -> Iterable:
        if args not in cache:
            cache[args] = generator_function(*args)
        cache[args], r = tee(cache[args])
        return r

    return wrapper


@_memoized_generator
def _integer_lattice_generator(dims: int) -> Iterable[FloatArray]:
    """Generate batches of integer lattice points. Each yield gives all
    points (that have not been yielded) in a sphere radius d centered at
    0; d starts at 0 and increments by 1 on each yield.

    Parameters
    ----------
    dims : int
        The dimension of Euclidean space of the lattice.

    Yields
    -------
    :class:`numpy.ndarray`
        Yields arrays of integer lattice points in `dims`-dimensional
        Euclidean space.
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
            for xy in product(range(d + 1), repeat=dims - 1):
                if xy not in ymax:
                    ymax[xy] = 0
                if sum(i**2 for i in xy) + ymax[xy] ** 2 <= d**2:
                    positive_int_lattice.append((*xy, ymax[xy]))
                    batch = True
                    ymax[xy] += 1
            if not batch:
                break
        pos_int_lat = np.array(positive_int_lattice, dtype=np.float64)
        yield _reflect_positive_integer_lattice(pos_int_lat)
        d += 1


@numba.njit(cache=True, fastmath=True)
def _reflect_positive_integer_lattice(positive_int_lattice: FloatArray) -> FloatArray:
    """Reflect points in the positive quadrant across all combinations
    of axes without duplicating points that are invariant under
    reflections.
    """

    dims = positive_int_lattice.shape[-1]
    batches = []
    batches.extend(positive_int_lattice)

    for n_reflections in range(1, dims + 1):
        axes = np.arange(n_reflections, dtype=np.int64)
        off_axes = (positive_int_lattice[:, axes] == 0).sum(axis=-1) == 0
        int_lattice = positive_int_lattice[off_axes]
        int_lattice[:, axes] *= -1
        batches.extend(int_lattice)

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
                axes[j] = axes[j - 1] + 1

            off_axes = (positive_int_lattice[:, axes] == 0).sum(axis=-1) == 0
            int_lattice = positive_int_lattice[off_axes]
            int_lattice[:, axes] *= -1
            batches.extend(int_lattice)

    int_lattice = np.empty(shape=(len(batches), dims), dtype=np.float64)
    for i in range(len(batches)):
        int_lattice[i] = batches[i]
    return int_lattice


@numba.njit(cache=True, fastmath=True)
def _lattice_to_cloud(motif: FloatArray, lattice: FloatArray) -> FloatArray:
    """Create a cloud of points from a periodic set with ``motif``,
    and a collection of lattice points ``lattice``.
    """
    m, dims = motif.shape
    n_lat_points = lattice.shape[0]
    layer = np.empty((m * n_lat_points, dims), dtype=np.float64)
    i = 0
    for lat_i in range(n_lat_points):
        for mot_i in range(m):
            for dim in range(dims):
                layer[i, dim] = motif[mot_i, dim] + lattice[lat_i, dim]
            i += 1
    return layer


@numba.njit(cache=True, fastmath=True)
def _cdist_sqeulcidean(XA, XB):
    mA, dims = XA.shape
    mB = XB.shape[0]
    dm = np.empty((mA, mB), np.float64)
    for i in range(mA):
        for j in range(mB):
            v = 0.0
            for d in range(dims):
                v += (XA[i, d] - XB[j, d]) ** 2
            dm[i, j] = v
    return dm


@numba.njit(cache=True, fastmath=True)
def _where_sq_sum_lt_bound(XA, bound):
    """Returns an array of bools such that
    XA[_where_sq_sum_lt_bound(XA, bound)] contains only points whose
    squared sum is < bound, equivalent to
    XA[(lattice ** 2).sum(axis=-1) < bound]."""
    m, dims = XA.shape
    ret = np.full((m,), fill_value=True, dtype=np.bool_)
    for i in range(m):
        v = 0.0
        for dim in range(dims):
            v += XA[i, dim] ** 2
        if v >= bound:
            ret[i] = False
    return ret


@numba.njit(cache=True, fastmath=True)
def _merge_sorted_arrays(dists: FloatArray, dists_: FloatArray) -> FloatArray:
    """Merge two 2D arrays with sorted rows into one sorted array with
    same number of columns as ``dists``. Optimised assuming that only a
    few elements at the end of each row of ``dists`` needs to be
    changed.
    """

    m, n_new_points = dists_.shape
    ret = np.copy(dists)

    for i in range(m):
        # Traverse dists[i] backwards until value smaller than dists_[i, 0]
        j = -1
        dp_ = 0
        dist_ = dists_[i, dp_]
        while True:
            if dists[i, j] <= dist_:
                j += 1
                break
            j -= 1

        # If dists_[i, 0] >= dists[i, -1], no need to insert
        if j == 0:
            continue

        # dp points to dists[i], dp_ points to dists_[i], j points to ret[i]
        # Fill ret with the larger value, increment pointers and repeat
        dp = j
        dist = dists[i, dp]

        while j < 0:
            if dist <= dist_:
                ret[i, j] = dist
                dp += 1
                dist = dists[i, dp]
            else:
                ret[i, j] = dist_
                dp_ += 1
                if dp_ < n_new_points:
                    dist_ = dists_[i, dp_]
                else:  # ran out of points in dists_
                    dist_ = np.inf
            j += 1

    return ret


@numba.njit(cache=True, fastmath=True)
def _merge_sorted_arrays_and_inds(
    dists: FloatArray, dists_: FloatArray, inds: UIntArray, inds_: UIntArray
) -> Tuple[FloatArray, UIntArray]:
    """Similar to _merge_sorted_arrays, but also merges two arrays
    ``inds`` and ``inds_`` with the same pattern for
    ``nearest_neighbors_data``.
    """

    m, n_new_points = dists_.shape
    ret_dists = np.copy(dists)
    ret_inds = np.copy(inds)

    for i in range(m):
        j = -1
        dp_ = 0
        dist_ = dists_[i, dp_]
        p_ = inds_[i, dp_]

        while True:
            if dists[i, j] <= dist_:
                j += 1
                break
            j -= 1

        if j == 0:
            continue

        dp = j
        dist = dists[i, dp]
        p = inds[i, dp]

        while j < 0:
            if dist <= dist_:
                ret_dists[i, j] = dist
                ret_inds[i, j] = p
                dp += 1
                dist = dists[i, dp]
                p = inds[i, dp]
            else:
                ret_dists[i, j] = dist_
                ret_inds[i, j] = p_
                dp_ += 1
                if dp_ < n_new_points:
                    dist_ = dists_[i, dp_]
                    p_ = inds_[i, dp_]
                else:
                    dist_ = np.inf
            j += 1

    return ret_dists, ret_inds


def nearest_neighbors_minval(
    motif: FloatArray, cell: FloatArray, x: FloatArray, min_val: float
) -> Tuple[FloatArray, FloatArray, UIntArray]:
    """Return the same ``dists``/PDD matrix as ``nearest_neighbors``,
    but with enough columns such that all values in the last column are
    at least ``min_val``.
    """

    full_cloud, all_sqdists = [], []
    m, dims = motif.shape

    int_lattice, int_lat_generator = _integer_lattice_batches(dims, m, 1)
    cloud = _lattice_to_cloud(motif, int_lattice @ cell)
    full_cloud.append(cloud)
    sqdists = _cdist_sqeulcidean(x, cloud)
    motif_diam = np.sqrt(sqdists[:, :m].max())
    all_sqdists.append(sqdists)
    max_sqd = min_val**2
    bound = (np.sqrt(max_sqd) + motif_diam) ** 2

    while True:
        lattice = next(int_lat_generator) @ cell
        lattice = lattice[np.einsum("ij,ij->i", lattice, lattice) < bound]
        if lattice.size == 0:
            break

        cloud = _lattice_to_cloud(motif, lattice)
        sqdists = _cdist_sqeulcidean(x, cloud)
        close = sqdists <= max_sqd
        if not np.any(close):
            break

        all_sqdists.append(sqdists)
        full_cloud.append(cloud)

    sqdists = np.hstack(all_sqdists)
    inds = np.argsort(sqdists)
    sqdists = np.take_along_axis(sqdists, inds, axis=-1)

    i = np.argmax(np.all(sqdists >= max_sqd, axis=0))
    sqdists = sqdists[:, : i + 1]
    inds = inds[:, : i + 1]

    return np.sqrt(sqdists), np.concatenate(full_cloud), inds


def generate_concentric_cloud(
    motif: FloatArray, cell: FloatArray
) -> Iterable[FloatArray]:
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

    int_lat_generator = _integer_lattice_generator(cell.shape[0])
    for layer in int_lat_generator:
        yield _lattice_to_cloud(motif, layer @ cell)
