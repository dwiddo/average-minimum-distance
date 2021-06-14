import numpy as np
from .core.core import nearest_neighbours

def amd(periodic_set, k):
    """
    Computes an AMD vector up to k from a periodic set.
    
    Parameters
    ----------
    periodic_set : amd.PeriodicSet or tuple of ndarrays (motif, cell)
        Representation of the periodic set, in Cartesian coordinates.
        amd.CifReader yields PeriodicSets which can be given here. 
        Otherwise pass a tuple of arrays (motif, cell) in Cartesian form. 
    k : int
        A :math:`m_b` by k PDD matrix (with weights in the first column).
        
    Returns
    -------
    ndarray
        AMD of periodic_set up to k. 
    """
    
    pdd, _, _ = nearest_neighbours(periodic_set[0], periodic_set[1], k)
    
    return np.average(pdd, axis=0)

def pdd(periodic_set, k, order=True, collapse=True):
    """
    Computes a PDD up to k from a periodic set.
    
    Parameters
    ----------
    periodic_set : amd.PeriodicSet or tuple of ndarrays (motif, cell)
        Representation of the periodic set, in Cartesian coordinates.
        amd.CifReader yields PeriodicSets which can be given here. 
        Otherwise pass a tuple of arrays (motif, cell) in Cartesian form. 
    k : int
        A :math:`m_b` by k PDD matrix (with weights in the first column).
    order : bool, optional
        Whether or not to lexicographically order the rows. Default True.
    collapse: bool, optional
        Whether or not to collapse identical rows. Default True.

    Returns
    -------
    ndarray
        PDD of periodic_set up to k. 
    """
    
    pdd, _, _ = nearest_neighbours(periodic_set[0], periodic_set[1], k)
    m = pdd.shape[0]
    
    if collapse:
        dists, weights = np.unique(pdd, axis=0, return_counts=True)
        pdd = np.hstack((weights[:, np.newaxis] / m, dists))
    else:
        dists = pdd
        pdd = np.hstack(np.full((m,1), 1/m), dists)
    
    if order:
        inds = np.lexsort(np.rot90(dists))
        pdd = pdd[inds]

    return pdd


def ppc(periodic_set):
    """
    Calculate the point packing coefficient (ppc) of periodic_set.
    The ppc is a constant of any periodic set determining the 
    asymptotic behaviour of its AMD/PDD as k -> infinity. 
    
    As k -> infinity, the ratio AMD_k / (n-th root of k) approaches
    the ppc (as does any row of a PDD). 
    
    For a unit cell U and m motif points in n dimensions,
        ppc = nth_root(Vol[U] / (m * V_n))
    where V_n is the volume of a unit sphere in n dimensions.
    """
    m, n = periodic_set[0].shape
    det = np.linalg.det(periodic_set[1])
    t = (n - n % 2) / 2
    if n % 2 == 0:
        V = (np.pi ** t) / np.math.factorial(t)
    else:
        V = (2 * np.math.factorial(t) * (4 * np.pi) ** t) / np.math.factorial(n)
    return (det / (m * V)) ** (1./n)

def amd_estimate(periodic_set, k):
    """
    Calculates an estimate of AMD_k (or PDD) based on the 
    point packing coefficient (ppc), which determines the
    asymptotic behaviour. 
    """
    n = periodic_set[0].shape[1]
    c = ppc(periodic_set[0], periodic_set[1])
    return np.array([(x ** (1. / n)) * c for x in range(1, k+1)])


if __name__ == "__main__":
    pass