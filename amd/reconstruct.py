"""Functions for resconstructing a periodic set up to isometry from its
PDD. This is possible 'in a general position', see our papers for more.
"""

import itertools

import numpy as np
import numba
from scipy.spatial.distance import cdist
from scipy.spatial import KDTree

from ._nns import generate_concentric_cloud
from .utils import diameter


def reconstruct(pdd, cell):
    """Reconstruct a motif from a PDD and unit cell. This function will
    only work if pdd has enough columns, such that the last column has
    all values larger than 2 times the diameter of the unit cell. It
    also expects an uncollapsed PDD with no weights column. Do not use
    amd.PDD to compute the PDD for this function, instead use
    amd.PDD_reconstructable which returns a version of the PDD which is
    passable to this function. Currently quite slow and run time varys a
    lot depending on input.

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

    # TODO: get a more reduced neighbour set
    # TODO: find all shared distances in a big operation at the start? could be faster
    # TODO: move PREC variable to its proper place
    PREC = 1e-10

    dims = cell.shape[0]
    if dims not in (2, 3):
        msg = 'Reconstructing from PDD only implemented for 2 and 3 dimensions'
        raise ValueError(msg)

    diam = diameter(cell)
    motif = [np.zeros((dims, ))] # set first point as origin wlog, return if 1 motif point

    if pdd.shape[0] == 1:
        motif = np.array(motif)
        return motif

    # finding lattice distances so we can ignore them
    cloud_generator = generate_concentric_cloud(np.array(motif), cell)
    cloud = []
    next(cloud_generator)
    l = next(cloud_generator)
    while np.any(np.linalg.norm(l, axis=-1) <= diam):
        cloud.append(l)
        l = next(cloud_generator)
    cloud = np.concatenate(cloud)

    # neighbour set: (a superset of) the lattice points close enough to the Voronoi domain
    nn_set = _neighbour_set(cell, PREC)
    lattice_dists = np.linalg.norm(cloud, axis=-1)
    lattice_dists = lattice_dists[lattice_dists <= diam]
    lattice_dists = _unique_within_tol(lattice_dists, PREC)

    # remove lattice distances from first and second rows
    row1_reduced = _remove_values_within_tol(pdd[0], lattice_dists, PREC)
    row2_reduced = _remove_values_within_tol(pdd[1], lattice_dists, PREC)
    # get shared dists between first and second rows
    shared_dists = _shared_values_within_tol(row1_reduced, row2_reduced, PREC)
    shared_dists = _unique_within_tol(shared_dists, PREC)

    # all combinations of vecs in neighbour set forming a basis
    bases = []
    for vecs in itertools.combinations(nn_set, dims):
        vecs = np.asarray(vecs)
        if np.abs(np.linalg.det(vecs)) > PREC:
            bases.extend(basis for basis in itertools.permutations(vecs, dims))

    q = _find_second_point(shared_dists, bases, cloud, PREC)

    if q is None:
        raise RuntimeError('Second point of motif could not be found.')

    motif.append(q)

    if pdd.shape[0] == 2:
        motif = np.array(motif)
        return motif

    for row in pdd[2:, :]:
        row_reduced = _remove_values_within_tol(row, lattice_dists, PREC)
        shared_dists1 = _shared_values_within_tol(row1_reduced, row_reduced, PREC)
        shared_dists2 = _shared_values_within_tol(row2_reduced, row_reduced, PREC)
        shared_dists1 = _unique_within_tol(shared_dists1, PREC)
        shared_dists2 = _unique_within_tol(shared_dists2, PREC)
        q_ = _find_further_point(shared_dists1, shared_dists2, bases, cloud, q, PREC)

        if q_ is None:
            raise RuntimeError('Further point of motif could not be found.')

        motif.append(q_)

    motif = np.array(motif)
    motif = np.mod(motif @ np.linalg.inv(cell), 1) @ cell
    return motif


def _find_second_point(shared_dists, bases, cloud, prec):
    dims = cloud.shape[-1]
    abs_q = shared_dists[0]

    for distance_tup in itertools.combinations(shared_dists[1:], dims):
        for basis in bases:
            res = None
            if dims == 2:
                res = _bilaterate(*basis, *distance_tup, abs_q, prec)
            elif dims == 3:
                if not _four_sphere_pairwise_intersecion(*basis, *distance_tup, abs_q, prec):
                    continue
                res = _trilaterate(*basis, *distance_tup, abs_q, prec)

            if res is not None:
                if np.all(np.linalg.norm(cloud - res, axis=-1) - abs_q + prec > 0):
                    return res


def _find_further_point(shared_dists1, shared_dists2, bases, cloud, q, prec):
    # distance from origin (first motif point) to further point
    dims = cloud.shape[-1]
    abs_q_ = shared_dists1[0]

    # try all ordered subsequences of distances shared between first and further row,
    # with all combinations of the vectors in the neighbour set forming a basis
    for distance_tup in itertools.combinations(shared_dists1[1:], dims):
        for basis in bases:
            res = None
            if dims == 2:
                res = _bilaterate(*basis, *distance_tup, abs_q_, prec)
            elif dims == 3:
                if not _four_sphere_pairwise_intersecion(*basis, *distance_tup, abs_q_, prec):
                    continue
                res = _trilaterate(*basis, *distance_tup, abs_q_, prec)

            if res is not None:
                # check point is in the voronoi domain
                if np.all(np.linalg.norm(cloud - res, axis=-1) - abs_q_ + prec > 0):
                    # check |p-point| is among the row's shared distances
                    if np.any(np.abs(shared_dists2 - np.linalg.norm(q - res)) < prec):
                        return res


def _neighbour_set(cell, prec):
    """(superset of) the neighbour set of origin for a lattice.
    """

    k_ = 5
    coeffs = np.array(list(itertools.product((-1, 0, 1), repeat=cell.shape[0])))
    coeffs = coeffs[coeffs.any(axis=-1)]    # remove (0,0,0)

    # half of all combinations of basis vectors
    vecs = []
    for c in coeffs:
        vecs.append(np.sum(cell * c[None, :].T, axis=0) / 2)
    vecs = np.array(vecs)

    origin = np.zeros((1, cell.shape[0]))
    cloud_generator = generate_concentric_cloud(origin, cell)
    cloud = np.concatenate((next(cloud_generator), next(cloud_generator)))
    tree = KDTree(cloud, compact_nodes=False, balanced_tree=False)
    dists, inds = tree.query(vecs, k=k_, workers=-1)
    dists_ = np.empty_like(dists)

    while not np.allclose(dists, dists_, atol=0, rtol=1e-12):
        dists = dists_
        cloud = np.vstack((cloud,
                           next(cloud_generator),
                           next(cloud_generator)))
        tree = KDTree(cloud, compact_nodes=False, balanced_tree=False)
        dists_, inds = tree.query(vecs, k=k_, workers=-1)

    tmp_inds = np.unique(inds[:, 1:].flatten())
    tmp_inds = tmp_inds[tmp_inds != 0]
    neighbour_set = cloud[tmp_inds]

    # reduce neighbour set
    # half the lattice points and find their nearest neighbours in the lattice
    neighbour_set_half = neighbour_set / 2
    # for each of these vectors, check if 0 is A nearest neighbour.
    # so, check if the dist to 0 is leq (within tol) than dist to all other lattice points
    nn_norms = np.linalg.norm(neighbour_set, axis=-1)
    halves_norms = nn_norms / 2
    halves_to_lattice_dists = cdist(neighbour_set_half, neighbour_set)

    # Do I need to + PREC?
    # inds of voronoi neighbours in cloud
    voronoi_neighbours = np.all(halves_to_lattice_dists - halves_norms + prec >= 0, axis=-1)
    neighbour_set = neighbour_set[voronoi_neighbours]
    return neighbour_set


def _four_sphere_pairwise_intersecion(p1, p2, p3, r1, r2, r3, abs_val, prec):
    # True/False intersection of four spheres, one centered at the origin (radius abs_val)
    flags = [False] * 6
    if np.linalg.norm(p1) > abs_val + r1 - prec:
        flags[0] = True
    if np.linalg.norm(p2) > abs_val + r2 - prec:
        flags[1] = True
    if np.linalg.norm(p3) > abs_val + r3 - prec:
        flags[2] = True
    if np.linalg.norm(p1 - p2) > r1 + r2 - prec:
        flags[3] = True
    if np.linalg.norm(p1 - p3) > r1 + r3 - prec:
        flags[4] = True
    if np.linalg.norm(p2 - p3) > r2 + r3 - prec:
        flags[5] = True

    if any(flags):
        return False
    return True


@numba.njit()
def _bilaterate(p1, p2, r1, r2, abs_val, prec):
    # Intersection of three circles, one centered at the origin (radius abs_val)
    d = np.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)
    v = (p2 - p1) / d

    if d > r1 + r2: # non intersecting
        return None
    if d < abs(r1 - r2): # One circle within other
        return None
    if d == 0 and r1 == r2: # coincident circles
        return None

    a = (r1 ** 2 - r2 ** 2 + d ** 2) / (2 * d)
    h = np.sqrt(r1 ** 2 - a ** 2)
    x2 = p1[0] + a * v[0]
    y2 = p1[1] + a * v[1]
    x3 = x2 + h * v[1]
    y3 = y2 - h * v[0]
    x4 = x2 - h * v[1]
    y4 = y2 + h * v[0]
    q1 = np.array((x3, y3))
    q2 = np.array((x4, y4))

    if np.abs(np.sqrt(x3 ** 2 + y3 ** 2) - abs_val) < prec:
        return q1
    if np.abs(np.sqrt(x4 ** 2 + y4 ** 2) - abs_val) < prec:
        return q2
    return None


@numba.njit()
def _trilaterate(p1, p2, p3, r1, r2, r3, abs_val, prec):
    # Intersection of four spheres, one centered at the origin (radius abs_val)
    temp1 = p2 - p1
    d = np.linalg.norm(temp1)
    e_x = temp1 / d
    temp2 = p3 - p1
    i = np.core.dot(e_x, temp2)
    temp3 = temp2 - i * e_x
    e_y = temp3 / np.linalg.norm(temp3)

    j = np.core.dot(e_y, temp2)
    x = (r1 * r1 - r2 * r2 + d * d) / (2 * d)
    y = (r1 * r1 - r3 * r3 -2 * i * x + i * i + j * j) / (2 * j)
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
    # get only unique values in a vector within a tolerance
    return arr[~np.any(np.triu(np.abs(arr[:, None] - arr) < prec, 1), axis=0)]


def _remove_values_within_tol(vec, values_to_remove, prec):
    # remove specified values in vec, within tol
    return vec[~np.any(np.abs(vec[:, None] - values_to_remove) < prec, axis=-1)]


def _shared_values_within_tol(v1, v2, prec):
    # get values shared between v1, v2 within tol
    return v1[np.argwhere(np.abs(v1[:, None] - v2) < prec)[:, 0]]
