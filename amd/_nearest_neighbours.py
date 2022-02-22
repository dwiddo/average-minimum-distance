"""Implements core function nearest_neighbours used for AMD and PDD calculations."""

from typing import Iterable, Optional
from itertools import product, combinations
from collections import defaultdict
import numpy as np
from numba import njit
from scipy.spatial import cKDTree

@njit()
def _dist(xy, z):
    s = 0
    for val in xy:
        s += val ** 2
    s += z ** 2
    return s

@njit()
def _distkey(pt):
    s = 0
    for val in pt:
        s += val ** 2
    return s

def generate_integer_lattice(dims: int) -> Iterable[np.ndarray]:
    """Generates batches of integer lattice points.
    
    Each yield gives all points (that have not already been yielded) 
    inside a sphere centered at the origin with radius d. d starts at 0 
    and increments by 1 on each loop. 
    
    Parameters
    ----------
    dims : int
        The dimension of Euclidean space the lattice is in.
        
    Yields
    -------
    ndarray 
        Yields arrays of integer points in dims dimensional Euclidean space.
    """
    
    ymax = defaultdict(int)
    d = 0
    
    while True:
        
        # get integer lattice points in +ve directions
        positive_int_lattice = []
        while True:
            batch = []
            for xy in product(range(d+1), repeat=dims-1):
                if _dist(xy, ymax[xy]) <= d**2:
                    batch.append((*xy, ymax[xy]))
                    ymax[xy] += 1
            if not batch:
                break
            positive_int_lattice += batch
        positive_int_lattice.sort(key=_distkey)

        # expand +ve integer lattice to full lattice with reflections
        int_lattice = []
        for p in positive_int_lattice:
            int_lattice.append(p)
            for n_reflections in range(1, dims+1):
                for indexes in combinations(range(dims), n_reflections):
                    if all((p[i] for i in indexes)):
                        p_ = list(p)
                        for i in indexes:
                            p_[i] *= -1
                        int_lattice.append(p_)
        
        yield np.array(int_lattice)
        d += 1

def generate_concentric_cloud(motif: np.ndarray, cell: np.ndarray) -> Iterable[np.ndarray]:
    """
    Generates batches of points from a periodic set given by (motif, cell)
    which are roughly successively further away from the origin.
    
    Each yield gives all points (that have not already been yielded) which
    lie in a unit cell whose corner lattice point was generated by 
    _generate_integer_lattice(motif.shape[1]). 
    
    Parameters
    ----------
    motif : ndarray
        Cartesian representation of the motif, shape (no points, dims).
    cell : ndarray
        Cartesian representation of the unit cell, shape (dims, dims).
        
    Yields
    -------
    ndarray 
        Yields arrays of points from the periodic set.
    """
    
    int_lattice_generator = generate_integer_lattice(cell.shape[0])
    
    while True:
        int_lattice = next(int_lattice_generator) @ cell
        yield np.concatenate([motif + translation for translation in int_lattice])

def nearest_neighbours(motif: np.ndarray, 
                       cell: np.ndarray, 
                       k: int, 
                       asymmetric_unit: Optional[np.ndarray] = None):
    """
    Given a periodic set represented by (motif, cell) and an integer k, find 
    the k nearest neighbours of the motif points in the periodic set.
    
    Note that cloud and inds are not used yet but may be in the future.
    
    Parameters
    ----------
    motif : ndarray
        Cartesian coords of the full motif, shape (no points, dims).
    cell : ndarray
        Cartesian coords of the unit cell, shape (dims, dims).
    k : int
        Number of nearest neighbours to find for each motif point.
    asymmetric_unit : ndarray, optional
        Indices pointing to an asymmetric unit in motif.
    
    Returns
    -------
    pdd : ndarray
        An array shape (motif.shape[0], k) of distances from each motif 
        point to its k nearest neighbours in order. Points do not count 
        as their own nearest neighbour. E.g. the distance to the n-th 
        nearest neighbour of the m-th motif point is pdd[m][n].
    cloud : ndarray 
        The collection of points in the periodic set that were generated
        during the nearest neighbour search.
    inds : ndarray
        An array shape (motif.shape[0], k) containing the indices of  
        nearest neighbours in cloud. E.g. the n-th nearest neighbour to 
        the m-th motif point is cloud[inds[m][n]].
    """
    
    if asymmetric_unit is not None:
        asym_unit = motif[asymmetric_unit]
    else:
        asym_unit = motif
    
    cloud_generator = generate_concentric_cloud(motif, cell)

    n_points = 0
    cloud = []
    while n_points <= k:
        l = next(cloud_generator)
        n_points += l.shape[0]
        cloud.append(l)
    cloud.append(next(cloud_generator))
    cloud = np.concatenate(cloud)

    tree = cKDTree(cloud, compact_nodes=False, balanced_tree=False)
    pdd_, inds = tree.query(asym_unit, k=k+1, workers=-1)
    pdd = np.empty_like(pdd_)

    while not np.array_equal(pdd, pdd_):
        pdd = pdd_
        cloud = np.append(cloud, next(cloud_generator), axis=0)
        cloud = np.append(cloud, next(cloud_generator), axis=0)
        tree = cKDTree(cloud, compact_nodes=False, balanced_tree=False)
        pdd_, inds = tree.query(asym_unit, k=k+1, workers=-1)

    return pdd_[:, 1:], cloud, inds[:, 1:]

def nearest_neighbours_minval(motif, cell, min_val):
    """PDD large enough to be reconstructed from 
    (such that last col values all > 2 * diam(cell))."""
    
    cloud_generator = generate_concentric_cloud(motif, cell)
    
    cloud = []
    for _ in range(3):
        cloud.append(next(cloud_generator))
        
    cloud = np.concatenate(cloud)
    tree = cKDTree(cloud, compact_nodes=False, balanced_tree=False)
    pdd_, _ = tree.query(motif, k=cloud.shape[0], workers=-1)
    pdd = np.empty_like(pdd_)

    while True:
        if np.all(pdd[:, -1] >= min_val):
            col_where = np.argwhere(np.all(pdd >= min_val, axis=0))[0][0]
            if np.array_equal(pdd[:, :col_where+1], pdd_[:, :col_where+1]):
                break
            
        pdd = pdd_
        cloud = np.vstack((cloud, next(cloud_generator), next(cloud_generator)))
        tree = cKDTree(cloud, compact_nodes=False, balanced_tree=False)
        pdd_, _ = tree.query(motif, k=cloud.shape[0], workers=-1)
    
    k = np.argwhere(np.all(pdd >= min_val, axis=0))[0][0]

    return pdd[:, 1:k+1]

if __name__ == '__main__':
    
    motif = np.array([[0,0,0],[0.1,0.1,0.1]])
    cell = np.identity(3)
    cloud, pdd, inds = nearest_neighbours(motif, cell, 10)
    print(pdd)