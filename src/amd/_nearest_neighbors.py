"""Implements core function nearest_neighbors used for AMD and PDD
calculations.
"""

from typing import Tuple, Iterable, Iterator, Callable, List
from itertools import product, tee, accumulate
import functools

import numba
import numpy as np

from ._types import FloatArray, UIntArray
from .utils import lll_reduce

__all__ = ["nearest_neighbors", "nearest_neighbors_data", "nearest_neighbors_minval"]

# Value used to pad the buffers of squared distances below before they
# have been filled. It stands in for infinity, but is finite because the
# kernels are compiled with fastmath (which allows assuming operands are
# not infinite) and is far larger than any squared distance that can
# arise, including after a square root and squaring again.
_SQDIST_MAX = 1e150


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

    red_cell = _reduced_cell(cell)
    if red_cell is not cell:
        motif = _wrap_into_cell(motif, red_cell)
        x = _wrap_into_cell(x, red_cell)
        cell = red_cell

    m, dims = motif.shape

    # Get an initial collection of points and a generator for more
    int_lattice, lat_seed, int_lat_generator = _integer_lattice_batches(dims, m, k)
    lat_seed = _seed_translates(lat_seed, len(int_lattice), m, k)
    layer = _n_layers(int_lattice)
    layer_dist = _layer_dist_factor(cell)
    xc, motif_radius = _motif_bounding_sphere(motif, x)
    motif_t = np.ascontiguousarray(motif.T)
    motif_diam = np.sqrt(_cdist_sqeulcidean(x, motif).max())

    # Squared distances to the k nearest neighbors of each point of x.
    # The points of the periodic set are never materialised: each batch
    # of lattice points is folded straight into the buffers.
    buf, thresh = _sqdist_buffers(x.shape[0], k)
    init_lattice = int_lattice @ cell
    _init_sqdist_buffers(
        buf, thresh, x, _cloud_transposed(motif, init_lattice[:lat_seed]), k
    )
    _update_sqdist_buffers(
        buf, thresh, x, xc, motif_t, init_lattice, motif_radius, k, lat_seed
    )

    # If a lattice point l has |l| >= max|p-p'| + max_d for p ∈ x, p' ∈ motif
    # then |p-(l+p')| >= max_d for all p, p' and l can be discarded.
    radius = np.sqrt(thresh.max()) + motif_diam

    # Every point not yet generated lies in layer ``layer`` or later, so
    # has norm > layer - 1 in the integer lattice and is further than
    # layer_dist * (layer - 1) from the origin. Once that reaches the
    # radius above, no remaining lattice point can give a neighbor. An
    # individual layer being empty does not imply this, since layers of
    # the integer lattice need not grow outwards steadily once mapped
    # into Cartesian space.
    while layer_dist * (layer - 1) < radius:
        # Get next layer of lattice and prune distant lattice points.
        # This bound holds for every point of x at once, so it is worth
        # applying before the sharper per-point one inside the kernel.
        lattice = next(int_lat_generator) @ cell
        layer += 1

        lattice = lattice[_where_sq_sum_lt_bound(lattice, radius**2)]
        if lattice.size == 0:  # None are close enough
            continue

        _update_sqdist_buffers(
            buf, thresh, x, xc, motif_t, lattice, motif_radius, k, 0
        )
        radius = np.sqrt(thresh.max()) + motif_diam

    sqdists = buf[:, :k]
    sqdists.sort()
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

    # Reduce the basis of the lattice (see nearest_neighbors()). The
    # motif and x are left alone so that the returned cloud and indices
    # still refer to neighbors of the points passed in.
    cell = _reduced_cell(cell)

    int_lattice, lat_seed, int_lat_generator = _integer_lattice_batches(dims, m, k)
    lat_seed = _seed_translates(lat_seed, len(int_lattice), m, k)
    layer = _n_layers(int_lattice)
    layer_dist = _layer_dist_factor(cell)
    xc, motif_radius = _motif_bounding_sphere(motif, x)
    motif_t = np.ascontiguousarray(motif.T)
    motif_diam = np.sqrt(_cdist_sqeulcidean(x, motif).max())

    lattice = int_lattice @ cell
    cloud = _lattice_to_cloud(motif, lattice)
    full_cloud.append(cloud)
    cloud_ind_offset = len(cloud)

    # Buffers of squared distances per point of x (see _sqdist_buffers),
    # with the index in the cloud each distance came from alongside
    buf, thresh = _sqdist_buffers(x.shape[0], k)
    buf_inds = np.zeros(buf.shape, dtype=np.int64)
    _init_sqdist_buffers_with_inds(
        buf, buf_inds, thresh, x, _cloud_transposed(motif, lattice[:lat_seed]), k
    )
    _update_sqdist_buffers_with_inds(
        buf, buf_inds, thresh, x, xc, motif_t, lattice, motif_radius, k,
        lat_seed, 0,
    )

    max_sqd = thresh.max()
    radius = np.sqrt(max_sqd) + motif_diam

    while layer_dist * (layer - 1) < radius:
        lattice = next(int_lat_generator) @ cell
        layer += 1
        lattice = lattice[np.einsum("ij,ij->i", lattice, lattice) < radius**2]
        if lattice.size == 0:
            continue

        cloud = _lattice_to_cloud(motif, lattice)
        n_kept = _update_sqdist_buffers_with_inds(
            buf, buf_inds, thresh, x, xc, motif_t, lattice, motif_radius, k,
            0, cloud_ind_offset,
        )
        if n_kept == 0:  # No point of this batch is a neighbor
            continue

        cloud_ind_offset += len(cloud)
        full_cloud.append(cloud)
        max_sqd = thresh.max()
        radius = np.sqrt(max_sqd) + motif_diam

    # The buffers hold the right k distances, but in no particular order
    sqdists, inds = buf[:, :k], buf_inds[:, :k]
    order = np.argsort(sqdists, axis=-1)
    sqdists = np.take_along_axis(sqdists, order, axis=-1)
    inds = np.take_along_axis(inds, order, axis=-1)

    return np.sqrt(sqdists), np.concatenate(full_cloud), inds


def _reduced_cell(cell: FloatArray) -> FloatArray:
    """Return an LLL-reduced basis of the lattice given by ``cell``.

    The searches below expand the periodic set in layers of the integer
    lattice, so small integer combinations of the basis vectors must
    correspond to short vectors; that only holds for a reduced basis.
    Returns ``cell`` itself when it is already reduced (so the search is
    left untouched) or if reduction fails (e.g. a degenerate cell).
    """

    red_cell = lll_reduce(np.asarray(cell, dtype=np.float64))
    if not np.all(np.isfinite(red_cell)) or np.array_equal(red_cell, cell):
        return cell
    return red_cell


def _wrap_into_cell(points: FloatArray, cell: FloatArray) -> FloatArray:
    """Translate ``points`` by lattice vectors of ``cell`` so they lie
    in the unit cell. Distances from/to the periodic set are unchanged.
    """
    return np.mod(points @ np.linalg.inv(cell), 1) @ cell


def _n_layers(int_lattice: FloatArray) -> int:
    """Number of layers of ``_integer_lattice_generator()`` contained in
    ``int_lattice``. Layer d holds the points with norm in (d-1, d], so
    a batch of layers 0, ..., d has maximum norm d.
    """
    sqnorms = np.einsum("ij,ij->i", int_lattice, int_lattice)
    return int(round(float(np.sqrt(sqnorms.max())))) + 1


def _layer_dist_factor(cell: FloatArray) -> float:
    """Smallest singular value of ``cell``, so that any point of the
    integer lattice with norm n is at least ``n`` times this value away
    from the origin once mapped into the periodic set by ``cell``. Used
    to know when no further layer can hold a neighbor.
    """
    return float(np.linalg.svd(cell, compute_uv=False).min())


def _motif_bounding_sphere(
    motif: FloatArray, x: FloatArray
) -> Tuple[FloatArray, float]:
    """Return ``x`` shifted so the motif's centre is the origin, and a
    radius ``r`` with ``|p - centre| <= r`` for every motif point ``p``.

    The searches use this to skip a lattice point ``l`` outright: every
    point of the translated motif ``l + motif`` is at least
    ``|(x - centre) - l| - r`` away from ``x``, so if that already
    exceeds the current worst neighbor distance, none of the ``m``
    points behind ``l`` can be a neighbor and all ``m`` distance
    computations can be skipped.
    """

    centre = motif.mean(axis=0)
    offsets = motif - centre
    radius = float(np.sqrt(np.einsum("ij,ij->i", offsets, offsets).max()))
    return x - centre, radius


def _seed_translates(lat_seed: int, n_lat: int, m: int, k: int) -> int:
    """How many translates of the motif to measure in full before the
    rest of the first batch is scanned with a threshold to prune by.

    ``lat_seed`` translates (all layers of the batch but the last) hold
    more than k points, but a threshold taken from barely more than k
    points sits at the edge of what it was drawn from and rules almost
    nothing out. Scanning the rest of the batch against such a threshold
    costs more than selecting from the whole batch in one go, so the
    batch is only split when the seed holds at least twice the k asked
    for. (Widening the seed instead, to force a split, measures worse:
    what is left is then pruned well but a translate at a time, and for
    a small motif those passes are too short to be efficient.)
    """

    return lat_seed if lat_seed * m >= 2 * k else n_lat


def _integer_lattice_batches_cache(func: Callable) -> Callable:
    """Specialised cache for ``_integer_lattice_batches``. Stores layers
    of integer lattice points from ``_integer_lattice_generator``.
    How many layers are needed depends on the ratio of k to m, so this
    is passed to the cache. One cache is kept for each dimension.
    """

    cache = {}  # (dims, n_layers): [int_lattice, generator, n_seed]
    npoints_cache = {}  # dims: [num points accumulated in layers 1,2,...]

    @functools.wraps(func)
    def wrapper(
        dims: int, m: int, k: int
    ) -> Tuple[FloatArray, int, Iterator[FloatArray]]:
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
            # Points of every layer but the last, which is the extra
            # one _integer_lattice_batches appends; depends only on dims
            # and n_layers, so it is shared by every entry of the cache
            n_seed = sum(len(i) for i in layers[:-1])
            cache[(dims, n_layers)] = [np.concatenate(layers), generator, n_seed]

        int_lattice, generator, n_seed = cache[(dims, n_layers)]
        cache[(dims, n_layers)][1], ret_generator = tee(generator)
        return int_lattice, n_seed, ret_generator

    return wrapper


@_integer_lattice_batches_cache
def _integer_lattice_batches(
    dims: int, m: int, k: int
) -> Tuple[List[FloatArray], Iterator[FloatArray]]:
    """Return an initial batch of integer lattice points (number
    according to k & m) and a generator for more distant points.

    Decorated by ``_integer_lattice_batches_cache``, so the batch comes
    back as one array, along with how many of its points come from all
    but the last layer. Those alone already carry more than k points of
    the periodic set, which the searches use to get a threshold before
    scanning the rest.

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
    n_seed : int
        How many of those points come from all but the last layer.
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
def _select_kth(a: FloatArray, n: int, kth: int) -> float:
    """Reorder ``a[:n]`` in place so that ``a[:kth + 1]`` holds its
    ``kth + 1`` smallest values and ``a[kth]`` is the largest of them,
    which is returned. Quickselect with a median-of-three pivot.
    """

    lo, hi = 0, n - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if a[mid] < a[lo]:
            a[mid], a[lo] = a[lo], a[mid]
        if a[hi] < a[lo]:
            a[hi], a[lo] = a[lo], a[hi]
        if a[hi] < a[mid]:
            a[hi], a[mid] = a[mid], a[hi]
        pivot = a[mid]

        i, j = lo, hi
        while i <= j:
            while a[i] < pivot:
                i += 1
            while a[j] > pivot:
                j -= 1
            if i <= j:
                a[i], a[j] = a[j], a[i]
                i += 1
                j -= 1

        if kth <= j:
            hi = j
        elif kth >= i:
            lo = i
        else:
            return a[kth]

    return a[kth]


@numba.njit(cache=True, fastmath=True)
def _select_kth_with_inds(
    a: FloatArray, inds: UIntArray, n: int, kth: int
) -> float:
    """``_select_kth`` which permutes ``inds[:n]`` alongside ``a[:n]``."""

    lo, hi = 0, n - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if a[mid] < a[lo]:
            a[mid], a[lo] = a[lo], a[mid]
            inds[mid], inds[lo] = inds[lo], inds[mid]
        if a[hi] < a[lo]:
            a[hi], a[lo] = a[lo], a[hi]
            inds[hi], inds[lo] = inds[lo], inds[hi]
        if a[hi] < a[mid]:
            a[hi], a[mid] = a[mid], a[hi]
            inds[hi], inds[mid] = inds[mid], inds[hi]
        pivot = a[mid]

        i, j = lo, hi
        while i <= j:
            while a[i] < pivot:
                i += 1
            while a[j] > pivot:
                j -= 1
            if i <= j:
                a[i], a[j] = a[j], a[i]
                inds[i], inds[j] = inds[j], inds[i]
                i += 1
                j -= 1

        if kth <= j:
            hi = j
        elif kth >= i:
            lo = i
        else:
            return a[kth]

    return a[kth]


def _sqdist_buffers(n_x: int, k: int) -> Tuple[FloatArray, FloatArray]:
    """Allocate the state the searches below keep for each point of
    ``x``: a buffer of squared distances and the current threshold.

    Each row of ``buf`` holds the k smallest squared distances found so
    far in ``buf[i, :k]``, followed by room to append new candidates.
    ``thresh[i]`` is the largest of the k, so a point is only worth
    keeping if it is closer than that. Both start out as ``_SQDIST_MAX``
    so that every point is a candidate until k of them are found.
    """

    # Candidates are appended rather than inserted, and the k smallest
    # re-selected only once the room runs out, which costs O(k) per
    # ``k // 8`` candidates instead of O(log k) per candidate. The
    # threshold then lags by at most that many candidates, which only
    # weakens the pruning below slightly and never the result.
    buf = np.full((n_x, k + max(16, k // 8)), _SQDIST_MAX)
    return buf, np.full(n_x, _SQDIST_MAX)


@numba.njit(cache=True, fastmath=True)
def _cloud_transposed(motif: FloatArray, lattice: FloatArray) -> FloatArray:
    """``_lattice_to_cloud`` with one row per dimension rather than one
    per point, so that measuring a point of ``x`` against the whole
    cloud is one contiguous pass per dimension.

    Column ``lat_i * len(motif) + mot_i`` is the motif point ``mot_i``
    translated by ``lattice[lat_i]``, matching the column order of
    ``_lattice_to_cloud``.
    """

    m, dims = motif.shape
    n_lat = lattice.shape[0]
    cloud_t = np.empty((dims, m * n_lat), dtype=np.float64)
    for lat_i in range(n_lat):
        off = lat_i * m
        for d in range(dims):
            lat_d = lattice[lat_i, d]
            for mot_i in range(m):
                cloud_t[d, off + mot_i] = motif[mot_i, d] + lat_d
    return cloud_t


@numba.njit(cache=True, fastmath=True)
def _init_sqdist_buffers(
    buf: FloatArray,
    thresh: FloatArray,
    x: FloatArray,
    cloud_t: FloatArray,
    k: int,
) -> None:
    """Seed the buffers of ``_sqdist_buffers`` with the k points of
    ``cloud_t`` nearest each point of ``x``. ``cloud_t`` comes from
    ``_cloud_transposed`` and holds more than k points.

    Until k points have been found there is no threshold to prune with,
    and a running threshold would be beaten by most of them anyway. So
    the whole seed cloud is measured into a scratch array, one point of
    ``x`` at a time, and the k smallest selected in a single pass; the
    caller folds in the remaining translates, which a now exact
    threshold prunes properly. Only the scratch array is materialised,
    not the (len(x), seed size) matrix of all the distances.

    The distances are accumulated one dimension at a time over the whole
    cloud, which keeps the loops long and contiguous however few points
    the motif has.
    """

    n_x, dims = x.shape
    n_pairs = cloud_t.shape[-1]
    n = max(n_pairs, k)
    scratch = np.empty(n, dtype=np.float64)

    for i in range(n_x):
        x_d = x[i, 0]
        for j in range(n_pairs):
            diff = x_d - cloud_t[0, j]
            scratch[j] = diff * diff
        for d in range(1, dims):
            x_d = x[i, d]
            for j in range(n_pairs):
                diff = x_d - cloud_t[d, j]
                scratch[j] += diff * diff

        # Only if the whole batch holds fewer than k points, which the
        # sizing in _integer_lattice_batches rules out for a periodic set
        for j in range(n_pairs, n):
            scratch[j] = _SQDIST_MAX

        thresh[i] = _select_kth(scratch, n, k - 1)
        for j in range(k):
            buf[i, j] = scratch[j]


@numba.njit(cache=True, fastmath=True)
def _init_sqdist_buffers_with_inds(
    buf: FloatArray,
    buf_inds: UIntArray,
    thresh: FloatArray,
    x: FloatArray,
    cloud_t: FloatArray,
    k: int,
) -> None:
    """``_init_sqdist_buffers`` which also records which point of the
    cloud each distance came from. Column ``j`` of ``cloud_t`` is point
    ``j`` of ``_lattice_to_cloud(motif, lattice)``.
    """

    n_x, dims = x.shape
    n_pairs = cloud_t.shape[-1]
    n = max(n_pairs, k)
    scratch = np.empty(n, dtype=np.float64)
    scratch_inds = np.empty(n, dtype=np.int64)

    for i in range(n_x):
        x_d = x[i, 0]
        for j in range(n_pairs):
            diff = x_d - cloud_t[0, j]
            scratch[j] = diff * diff
        for d in range(1, dims):
            x_d = x[i, d]
            for j in range(n_pairs):
                diff = x_d - cloud_t[d, j]
                scratch[j] += diff * diff

        for j in range(n_pairs, n):
            scratch[j] = _SQDIST_MAX
        # Reset every row: the selection below permutes both arrays
        for j in range(n):
            scratch_inds[j] = j

        thresh[i] = _select_kth_with_inds(scratch, scratch_inds, n, k - 1)
        for j in range(k):
            buf[i, j] = scratch[j]
            buf_inds[i, j] = scratch_inds[j]


@numba.njit(cache=True, fastmath=True)
def _update_sqdist_buffers(
    buf: FloatArray,
    thresh: FloatArray,
    x: FloatArray,
    xc: FloatArray,
    motif_t: FloatArray,
    lattice: FloatArray,
    motif_radius: float,
    k: int,
    lat_start: int,
) -> None:
    """Offer every point of ``lattice[lat_start:] + motif`` to the
    buffers of the k smallest squared distances from each point of ``x``
    (see ``_sqdist_buffers``). ``motif_t`` is the motif transposed, one
    row per dimension.

    The points are never materialised: the translates of the motif are
    walked one lattice point at a time, and a whole translate is
    rejected by a single distance computation whenever its bounding
    sphere lies further away than the current threshold (see
    ``_motif_bounding_sphere``). On return each ``thresh[i]`` is again
    exactly the k-th smallest squared distance to ``x[i]`` found so far.
    """

    n_x, dims = x.shape
    m = motif_t.shape[-1]
    n_lat = lattice.shape[0]
    cap = buf.shape[-1]
    xl = np.empty(dims, dtype=np.float64)
    sqdists = np.empty(m, dtype=np.float64)

    for i in range(n_x):
        row = buf[i]
        t = thresh[i]
        n_buf = 0
        bound = np.sqrt(t) + motif_radius
        bound_sq = bound * bound

        for lat_i in range(lat_start, n_lat):
            # Distance to the centre of this translate of the motif; no
            # point of it is within sqrt(t) of x[i] if this is too big
            sq = 0.0
            for d in range(dims):
                diff = xc[i, d] - lattice[lat_i, d]
                sq += diff * diff
            if sq > bound_sq:
                continue

            for d in range(dims):
                xl[d] = x[i, d] - lattice[lat_i, d]

            # Squared distances to the whole translate, accumulated one
            # dimension at a time. Kept apart from the comparisons below
            # so these loops stay contiguous, branchless and vectorised,
            # and so ``sqdists`` stays small enough to live in cache.
            xl_d = xl[0]
            for mot_i in range(m):
                diff = xl_d - motif_t[0, mot_i]
                sqdists[mot_i] = diff * diff
            for d in range(1, dims):
                xl_d = xl[d]
                for mot_i in range(m):
                    diff = xl_d - motif_t[d, mot_i]
                    sqdists[mot_i] += diff * diff

            for mot_i in range(m):
                sq = sqdists[mot_i]
                if sq < t:
                    row[k + n_buf] = sq
                    n_buf += 1
                    if k + n_buf == cap:  # Out of room, re-select
                        t = _select_kth(row, cap, k - 1)
                        n_buf = 0
                        bound = np.sqrt(t) + motif_radius
                        bound_sq = bound * bound

        if n_buf > 0:  # Leave the threshold exact for the next batch
            t = _select_kth(row, k + n_buf, k - 1)
        thresh[i] = t


@numba.njit(cache=True, fastmath=True)
def _update_sqdist_buffers_with_inds(
    buf: FloatArray,
    buf_inds: UIntArray,
    thresh: FloatArray,
    x: FloatArray,
    xc: FloatArray,
    motif_t: FloatArray,
    lattice: FloatArray,
    motif_radius: float,
    k: int,
    lat_start: int,
    ind_offset: int,
) -> int:
    """``_update_sqdist_buffers`` which also records which point of the
    cloud each distance came from.

    ``buf_inds`` is permuted alongside ``buf`` and holds indices into
    the cloud ``_lattice_to_cloud(motif, lattice)`` of this batch,
    shifted by ``ind_offset``. Returns how many points of this batch
    were kept as candidates, which is 0 exactly when none of them is a
    neighbor of any point of ``x``.
    """

    n_x, dims = x.shape
    m = motif_t.shape[-1]
    n_lat = lattice.shape[0]
    cap = buf.shape[-1]
    xl = np.empty(dims, dtype=np.float64)
    sqdists = np.empty(m, dtype=np.float64)
    n_kept = 0

    for i in range(n_x):
        row = buf[i]
        row_inds = buf_inds[i]
        t = thresh[i]
        n_buf = 0
        bound = np.sqrt(t) + motif_radius
        bound_sq = bound * bound

        for lat_i in range(lat_start, n_lat):
            sq = 0.0
            for d in range(dims):
                diff = xc[i, d] - lattice[lat_i, d]
                sq += diff * diff
            if sq > bound_sq:
                continue

            for d in range(dims):
                xl[d] = x[i, d] - lattice[lat_i, d]
            cloud_ind = ind_offset + lat_i * m

            xl_d = xl[0]
            for mot_i in range(m):
                diff = xl_d - motif_t[0, mot_i]
                sqdists[mot_i] = diff * diff
            for d in range(1, dims):
                xl_d = xl[d]
                for mot_i in range(m):
                    diff = xl_d - motif_t[d, mot_i]
                    sqdists[mot_i] += diff * diff

            for mot_i in range(m):
                sq = sqdists[mot_i]
                if sq < t:
                    row[k + n_buf] = sq
                    row_inds[k + n_buf] = cloud_ind + mot_i
                    n_buf += 1
                    n_kept += 1
                    if k + n_buf == cap:
                        t = _select_kth_with_inds(row, row_inds, cap, k - 1)
                        n_buf = 0
                        bound = np.sqrt(t) + motif_radius
                        bound_sq = bound * bound

        if n_buf > 0:
            t = _select_kth_with_inds(row, row_inds, k + n_buf, k - 1)
        thresh[i] = t

    return n_kept


@numba.njit(cache=True, fastmath=True)
def _count_within_sqdist(
    n_closer: UIntArray,
    x: FloatArray,
    xc: FloatArray,
    motif_t: FloatArray,
    lattice: FloatArray,
    motif_radius: float,
    min_sqval: float,
) -> None:
    """Add to ``n_closer[i]`` the number of points of
    ``lattice + motif`` whose squared distance to ``x[i]`` is at most
    ``min_sqval``. ``motif_t`` is the motif transposed, one row per
    dimension.
    """

    n_x, dims = x.shape
    m = motif_t.shape[-1]
    n_lat = lattice.shape[0]
    xl = np.empty(dims, dtype=np.float64)
    sqdists = np.empty(m, dtype=np.float64)
    bound = np.sqrt(min_sqval) + motif_radius
    bound_sq = bound * bound

    for i in range(n_x):
        count = 0
        for lat_i in range(n_lat):
            sq = 0.0
            for d in range(dims):
                diff = xc[i, d] - lattice[lat_i, d]
                sq += diff * diff
            if sq > bound_sq:
                continue

            for d in range(dims):
                xl[d] = x[i, d] - lattice[lat_i, d]

            xl_d = xl[0]
            for mot_i in range(m):
                diff = xl_d - motif_t[0, mot_i]
                sqdists[mot_i] = diff * diff
            for d in range(1, dims):
                xl_d = xl[d]
                for mot_i in range(m):
                    diff = xl_d - motif_t[d, mot_i]
                    sqdists[mot_i] += diff * diff

            for mot_i in range(m):
                if sqdists[mot_i] <= min_sqval:
                    count += 1

        n_closer[i] += count


def nearest_neighbors_minval(
    motif: FloatArray, cell: FloatArray, x: FloatArray, min_val: float
) -> Tuple[FloatArray, FloatArray, UIntArray]:
    """Return the same ``dists``, ``cloud`` and ``inds`` as
    ``nearest_neighbors_data``, but with enough columns such that all
    values in the last column are at least ``min_val``.
    """

    m, dims = motif.shape

    # Reduce the basis of the lattice (see nearest_neighbors()). The
    # motif and x are left alone so that the returned cloud and indices
    # still refer to neighbors of the points passed in.
    cell = _reduced_cell(cell)
    min_sqval = min_val**2

    # A point of the periodic set within min_val of a point p ∈ x comes
    # from a lattice point l with |l| < min_val + max|p-p'| (p' ∈ motif),
    # so this pass counts every neighbor closer than min_val exactly.
    int_lattice, _, int_lat_generator = _integer_lattice_batches(dims, m, 1)
    layer = _n_layers(int_lattice)
    layer_dist = _layer_dist_factor(cell)
    xc, motif_radius = _motif_bounding_sphere(motif, x)
    motif_t = np.ascontiguousarray(motif.T)
    motif_diam = np.sqrt(_cdist_sqeulcidean(x, motif).max())
    # Counted and pruned with <= so that neighbors landing exactly on
    # min_val cannot be missed, at the cost of an occasional extra column
    n_closer = np.zeros(x.shape[0], dtype=np.int64)
    _count_within_sqdist(
        n_closer, x, xc, motif_t, int_lattice @ cell, motif_radius, min_sqval
    )
    radius = min_val + motif_diam

    while layer_dist * (layer - 1) <= radius:
        lattice = next(int_lat_generator) @ cell
        layer += 1
        lattice = lattice[np.einsum("ij,ij->i", lattice, lattice) <= radius**2]
        if lattice.size == 0:
            continue

        _count_within_sqdist(
            n_closer, x, xc, motif_t, lattice, motif_radius, min_sqval
        )

    # Column n_closer[i] is the first of row i which is at least min_val,
    # so this is the smallest number of columns with a last column all
    # at least min_val. Distances themselves come from the k nearest
    # neighbors search, which gets the last column right as well.
    k = int(n_closer.max()) + 1
    return nearest_neighbors_data(motif, cell, x, k)


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
