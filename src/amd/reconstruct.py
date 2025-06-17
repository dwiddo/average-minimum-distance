"""Functions for resconstructing a periodic set up to isometry from its
PDD. This is possible 'in a general position', see our papers for more.
"""

from itertools import combinations, permutations, product

import numpy as np
import numba
from scipy.spatial.distance import cdist
from scipy.spatial import KDTree

from ._nearest_neighbors import generate_concentric_cloud
from .utils import diameter

from ._types import FloatArray

__all__ = ["reconstruct"]


def reconstruct(pdd: FloatArray, cell: FloatArray) -> FloatArray:
    """Reconstruct a motif from a PDD and unit cell. This function will
    only work if ``pdd`` has enough columns, such that the last column
    has all values larger than 2 times the diameter of the unit cell. It
    also expects an uncollapsed PDD with no weights column. Do not use
    ``amd.PDD`` to compute the PDD for this function, instead use
    ``amd.PDD_reconstructable`` which returns a version of the PDD which
    is passable to this function. This function is quite slow and run
    time may vary a lot arbitrarily depending on input.

    Parameters
    ----------
    pdd : :class:`numpy.ndarray`
        The PDD of the periodic set to reconstruct. Needs `k` at least
        large enough so all values in the last column of pdd are greater
        than :code:`2 * diameter(cell)`, and needs to be uncollapsed
        without weights. Use amd.PDD_reconstructable to get a PDD which
        is acceptable for this argument.
    cell : :class:`numpy.ndarray`
        Unit cell of the periodic set to reconstruct.

    Returns
    -------
    :class:`numpy.ndarray`
        The reconstructed motif of the periodic set.
    """

    # TODO: get a more reduced neighbor set
    # TODO: find all shared distances in a big operation at the start
    # TODO: move PREC variable to its proper place
    PREC = 1e-10

    dims = cell.shape[0]
    if dims not in (2, 3):
        raise ValueError(
            "Reconstructing from PDD only implemented for 2 and 3 dimensions"
        )
    diam = diameter(cell)
    motif = [np.zeros((dims,))]
    if pdd.shape[0] == 1:
        return np.array(motif)

    # finding lattice distances so we can ignore them
    cloud_generator = iter(generate_concentric_cloud(np.array(motif), cell))
    next(cloud_generator)  # the origin
    cloud = []
    layer = next(cloud_generator)
    while np.any(np.linalg.norm(layer, axis=-1) <= diam):
        cloud.append(layer)
        layer = next(cloud_generator)
    cloud = np.concatenate(cloud)

    # is (a superset of) lattice points close enough to Voronoi domain
    nn_set = _neighbor_set(cell, PREC)
    lattice_dists = np.linalg.norm(cloud, axis=-1)
    lattice_dists = lattice_dists[lattice_dists <= diam]
    lattice_dists = _unique_within_tol(lattice_dists, PREC)

    # remove lattice distances from first and second rows
    row1_reduced = _remove_vals(pdd[0], lattice_dists, PREC)
    row2_reduced = _remove_vals(pdd[1], lattice_dists, PREC)
    # get shared dists between first and second rows
    shared_dists = _shared_vals(row1_reduced, row2_reduced, PREC)
    shared_dists = _unique_within_tol(shared_dists, PREC)

    # all combinations of vecs in neighbor set forming a basis
    bases = []
    for vecs in combinations(nn_set, dims):
        vecs = np.asarray(vecs)
        if np.abs(np.linalg.det(vecs)) > PREC:
            bases.extend(basis for basis in permutations(vecs, dims))

    q = _find_second_point(shared_dists, bases, cloud, PREC)
    if q is None:
        raise RuntimeError("Second point of motif could not be found.")
    motif.append(q)

    if pdd.shape[0] == 2:
        return np.array(motif)

    for row in pdd[2:, :]:
        row_reduced = _remove_vals(row, lattice_dists, PREC)
        shared_dists1 = _shared_vals(row1_reduced, row_reduced, PREC)
        shared_dists2 = _shared_vals(row2_reduced, row_reduced, PREC)
        shared_dists1 = _unique_within_tol(shared_dists1, PREC)
        shared_dists2 = _unique_within_tol(shared_dists2, PREC)
        q_ = _find_further_point(shared_dists1, shared_dists2, bases, cloud, q, PREC)
        if q_ is None:
            raise RuntimeError("Further point of motif could not be found.")
        motif.append(q_)

    motif = np.array(motif)
    motif = np.mod(motif @ np.linalg.inv(cell), 1) @ cell
    return motif


def _find_second_point(shared_dists, bases, cloud, prec):
    dims = cloud.shape[-1]
    abs_q = shared_dists[0]
    sphere_intersect_func = _trilaterate if dims == 3 else _bilaterate

    for distance_tup in combinations(shared_dists[1:], dims):
        for basis in bases:
            res = sphere_intersect_func(*basis, *distance_tup, abs_q, prec)
            if res is None:
                continue
            cloud_res_dists = np.linalg.norm(cloud - res, axis=-1)
            if np.all(cloud_res_dists - abs_q + prec > 0):
                return res


def _find_further_point(shared_dists1, shared_dists2, bases, cloud, q, prec):
    # distance from origin (first motif point) to further point
    dims = cloud.shape[-1]
    abs_q_ = shared_dists1[0]

    # try all ordered subsequences of distances shared between first and
    # further row, with all combinations of the vectors in the neighbor set
    # forming a basis, see if spheres centered at the vectors with the shared
    # distances as radii intersect at 4 (3 dims) points.
    sphere_intersect_func = _trilaterate if dims == 3 else _bilaterate
    for distance_tup in combinations(shared_dists1[1:], dims):
        for basis in bases:
            res = sphere_intersect_func(*basis, *distance_tup, abs_q_, prec)
            if res is None:
                continue
            # check point is in the voronoi domain
            cloud_res_dists = np.linalg.norm(cloud - res, axis=-1)
            if not np.all(cloud_res_dists - abs_q_ + prec > 0):
                continue
            # check |p - point| is among the row's shared distances
            dist_diff = np.abs(shared_dists2 - np.linalg.norm(q - res))
            if np.any(dist_diff < prec):
                return res


def _neighbor_set(cell, prec):
    """(A superset of) the neighbor set of origin for a lattice."""

    k_ = 5
    coeffs = np.array(list(product((-1, 0, 1), repeat=cell.shape[0])))
    coeffs = coeffs[coeffs.any(axis=-1)]  # remove (0,0,0)

    # half of all combinations of basis vectors
    vecs = []
    for c in coeffs:
        vecs.append(np.sum(cell * c[None, :].T, axis=0) / 2)
    vecs = np.array(vecs)

    origin = np.zeros((1, cell.shape[0]))
    cloud_generator = iter(generate_concentric_cloud(origin, cell))
    cloud = np.concatenate((next(cloud_generator), next(cloud_generator)))
    tree = KDTree(cloud, compact_nodes=False, balanced_tree=False)
    dists, inds = tree.query(vecs, k=k_)
    dists_ = np.empty_like(dists)

    while not np.allclose(dists, dists_, atol=0, rtol=1e-12):
        dists = dists_
        cloud = np.vstack((cloud, next(cloud_generator), next(cloud_generator)))
        tree = KDTree(cloud, compact_nodes=False, balanced_tree=False)
        dists_, inds = tree.query(vecs, k=k_)

    tmp_inds = np.unique(inds[:, 1:].flatten())
    tmp_inds = tmp_inds[tmp_inds != 0]
    neighbor_set = cloud[tmp_inds]

    # reduce neighbor set
    # half the lattice points and find their nearest neighbors in the lattice
    neighbor_set_half = neighbor_set / 2
    # for each of these vectors, check if 0 is a nearest neighbor.
    # so, check if the dist to 0 is leq (within tol) than dist to all other
    # lattice points.
    nn_norms = np.linalg.norm(neighbor_set, axis=-1)
    halves_norms = nn_norms / 2
    halves_to_lattice_dists = cdist(neighbor_set_half, neighbor_set)

    # Do I need to + PREC?
    # inds of voronoi neighbors in cloud
    voronoi_neighbors = np.all(
        halves_to_lattice_dists - halves_norms + prec >= 0, axis=-1
    )
    neighbor_set = neighbor_set[voronoi_neighbors]
    return neighbor_set


@numba.njit(cache=True, fastmath=True)
def _bilaterate(p1, p2, r1, r2, abs_val, prec):
    """Return the intersection of three circles."""

    d = np.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)
    v = (p2 - p1) / d

    if d > r1 + r2:
        return None
    if d < abs(r1 - r2):
        return None
    if d == 0 and r1 == r2:
        return None

    a = (r1**2 - r2**2 + d**2) / (2 * d)
    h = np.sqrt(r1**2 - a**2)
    x2 = p1[0] + a * v[0]
    y2 = p1[1] + a * v[1]
    x3 = x2 + h * v[1]
    y3 = y2 - h * v[0]
    x4 = x2 - h * v[1]
    y4 = y2 + h * v[0]
    q1 = np.array((x3, y3))
    q2 = np.array((x4, y4))

    if np.abs(np.sqrt(x3**2 + y3**2) - abs_val) < prec:
        return q1
    if np.abs(np.sqrt(x4**2 + y4**2) - abs_val) < prec:
        return q2
    return None


@numba.njit(cache=True)
def _trilaterate(p1, p2, p3, r1, r2, r3, abs_val, prec):
    """Return the intersection of four spheres."""

    if np.linalg.norm(p1) > abs_val + r1 - prec:
        return None
    if np.linalg.norm(p2) > abs_val + r2 - prec:
        return None
    if np.linalg.norm(p3) > abs_val + r3 - prec:
        return None
    if np.linalg.norm(p1 - p2) > r1 + r2 - prec:
        return None
    if np.linalg.norm(p1 - p3) > r1 + r3 - prec:
        return None
    if np.linalg.norm(p2 - p3) > r2 + r3 - prec:
        return None

    temp1 = p2 - p1
    d = np.linalg.norm(temp1)
    e_x = temp1 / d
    temp2 = p3 - p1
    i = np.dot(e_x, temp2)
    temp3 = temp2 - i * e_x
    e_y = temp3 / np.linalg.norm(temp3)

    j = np.dot(e_y, temp2)
    x = (r1 * r1 - r2 * r2 + d * d) / (2 * d)
    y = (r1 * r1 - r3 * r3 - 2 * i * x + i * i + j * j) / (2 * j)
    temp4 = r1 * r1 - x * x - y * y

    if temp4 < 0:
        return None

    e_z = np.cross(e_x, e_y)
    z = np.sqrt(temp4)
    p_12_a = p1 + x * e_x + y * e_y + z * e_z
    p_12_b = p1 + x * e_x + y * e_y - z * e_z

    if np.abs(np.linalg.norm(p_12_a) - abs_val) < prec:
        return p_12_a
    if np.abs(np.linalg.norm(p_12_b) - abs_val) < prec:
        return p_12_b
    return None


def _unique_within_tol(arr, prec):
    """Return only unique values in a vector within ``prec``."""
    return arr[~np.any(np.triu(np.abs(arr[:, None] - arr) < prec, 1), axis=0)]


def _remove_vals(vec, vals_to_remove, prec):
    """Remove specified values in vec, within ``prec``."""
    return vec[~np.any(np.abs(vec[:, None] - vals_to_remove) < prec, axis=-1)]


def _shared_vals(v1, v2, prec):
    """Return values shared between v1, v2 within ``prec``."""
    return v1[np.argwhere(np.abs(v1[:, None] - v2) < prec)[:, 0]]
