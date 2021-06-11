import numpy as np
from .core.core import _nearest_neighbours

def amd(periodic_set, k):
    '''PeriodicSet as given by amd.io or (motif, cell) --> AMD_k'''
    
    _, pdd, _ = _nearest_neighbours(periodic_set[0], periodic_set[1], k)
    return np.average(pdd, axis=0)

def pdd(periodic_set, k, order=True):
    '''PeriodicSet as given by amd.io or (motif, cell) --> PDD_k'''
    
    _, distance_matrix, _ = _nearest_neighbours(periodic_set[0], periodic_set[1], k)
    m = distance_matrix.shape[0]
    
    dists, weights = np.unique(distance_matrix, axis=0, return_counts=True)
    pdd = np.hstack((weights[:, np.newaxis].transpose() / m, dists))
    
    if order:
        inds = np.lexsort(np.rot90(dists))
        pdd = pdd[inds]

    return pdd
    

def ppc(periodic_set):
    m, n = periodic_set[0].shape
    det = np.linalg.det(periodic_set[1])
    t = (n - n % 2) / 2
    if n % 2 == 0:
        V = (np.pi ** t) / np.math.factorial(t)
    else:
        V = (2 * np.math.factorial(t) * (4 * np.pi) ** t) / np.math.factorial(n)
    return (det / (m * V)) ** (1./n)

def amd_estimate(periodic_set, k):
    n = periodic_set[0].shape[1]
    c = ppc(periodic_set[0], periodic_set[1])
    return np.array([(x ** (1. / n)) * c for x in range(1, k+1)])


if __name__ == "__main__":
    pass