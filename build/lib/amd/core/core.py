# Contains core functions for AMD/PDD calculations.

from itertools import product, combinations
from collections import defaultdict
import numpy as np
from scipy.spatial import cKDTree

def _dist(p):
    return sum(x**2 for x in p)

def _generate_integer_lattice(dims):
    ymax = defaultdict(int)
    d = 0
    while True:
        positive_int_lattice = []
        while True:
            batch = []
            for x in product(range(d+1), repeat=dims-1):
                pt = [*x, ymax[x]]
                if _dist(pt) <= d**2:
                    batch.append(pt)
                    ymax[x] += 1
            if not batch:
                break
            positive_int_lattice += batch
        positive_int_lattice.sort(key=_dist)

        # expand positive_int_lattice to int_lattice with reflections
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
        
        yield int_lattice
        d += 1

def _generate_concentric_cloud(motif, cell):
    
    lattice_generator = _generate_integer_lattice(motif.shape[1])
    while True:
        int_lattice = next(lattice_generator)
        lattice = np.array(int_lattice) @ cell
        yield np.concatenate([motif + translation for translation in lattice])


def _nearest_neighbours(motif, cell, k):
    
    cloud_generator = _generate_concentric_cloud(motif, cell)
    
    n_points = 0
    cloud = []
    while n_points <= k:
        l = next(cloud_generator)
        n_points += l.shape[0]
        cloud.append(l)
    cloud.append(next(cloud_generator))
    cloud = np.concatenate(cloud)

    tree = cKDTree(cloud, compact_nodes=False, balanced_tree=False)
    pdd_, inds = tree.query(motif, k=k+1, workers=-1)
    pdd = np.zeros_like(pdd_)

    while not np.array_equal(pdd, pdd_):
        pdd = np.copy(pdd_)
        cloud = np.append(cloud, next(cloud_generator), axis=0)
        tree = cKDTree(cloud, compact_nodes=False, balanced_tree=False)
        pdd_, inds = tree.query(motif, k=k+1, workers=-1)

    return cloud, pdd_[:, 1:], inds[:, 1:]