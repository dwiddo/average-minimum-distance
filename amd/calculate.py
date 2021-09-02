import warnings
import numpy as np
from .core.nearest_neighbours import nearest_neighbours
from .core.PeriodicSet import PeriodicSet

def AMD(periodic_set, k):
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
    
    asymmetric_unit, multiplicities = None, None

    if isinstance(periodic_set, PeriodicSet):
        
        motif, cell = periodic_set.motif, periodic_set.cell
        
        if 'asymmetric_unit' in periodic_set.tags and 'wyckoff_multiplicities' in periodic_set.tags:
            
            asymmetric_unit = periodic_set.asymmetric_unit
            multiplicities = periodic_set.wyckoff_multiplicities
            
    else:
        motif, cell = periodic_set[0], periodic_set[1]
        
    pdd, _, _ = nearest_neighbours(motif, cell, k, asymmetric_unit=asymmetric_unit)
    
    return np.average(pdd, axis=0, weights=multiplicities)


def amd(*args, **kwargs):
    warnings.warn("amd.amd() and amd.pdd() are deprecated; use amd.AMD() and amd.PDD() instead.", DeprecationWarning)
    return AMD(*args, **kwargs)


def PDD(periodic_set, k, order=True, collapse=True, collapse_tol=1e-4):
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
        Whether or not to collapse identical rows (within a tolerance). Default True.
    collapse_tol: float
        If two rows have all entries closer than collapse_tol, they get collapsed.
        Default 0.0001.

    Returns
    -------
    ndarray
        PDD of periodic_set up to k. 
    """
    
    asymmetric_unit, multiplicities = None, None
    
    if isinstance(periodic_set, PeriodicSet):
        
        motif, cell = periodic_set.motif, periodic_set.cell
        
        if 'asymmetric_unit' in periodic_set.tags and 'wyckoff_multiplicities' in periodic_set.tags:

            asymmetric_unit = periodic_set.asymmetric_unit
            multiplicities = periodic_set.wyckoff_multiplicities
            
    else:
        
        motif, cell = periodic_set[0], periodic_set[1]
        
    dists, _, _ = nearest_neighbours(motif, cell, k, asymmetric_unit=asymmetric_unit)
    
    if multiplicities is None:
        multiplicities = np.ones((motif.shape[0], ))
        
    m = np.sum(multiplicities)
    weights = multiplicities / m
    
    if collapse:
        
        diffs = np.abs(dists[:, None] - dists)
        overlapping = np.triu(np.all(diffs <= collapse_tol, axis=-1), 1)
        
        if overlapping.any():
            keep = ~overlapping.any(0)
            dists = dists[keep]
            np.fill_diagonal(overlapping, True)
            ind_groups, mul_inds = np.argwhere(overlapping[keep]).T

            weights_ = []
            for i in range(len(dists)):
                weights_.append(np.sum(weights[mul_inds[ind_groups == i]]))
            weights = np.array(weights_)
    
    pdd = np.hstack((weights[:, None], dists))
    
    if order:
        pdd = pdd[np.lexsort(np.rot90(dists))]
    
    return pdd


def pdd(*args, **kwargs):
    warnings.warn("amd.amd() and amd.pdd() are deprecated; use amd.AMD() and amd.PDD() instead.", DeprecationWarning)
    return PDD(*args, **kwargs)

def PDD_to_AMD(pdd):
    """Calculates AMD from a PDD."""
    return np.average(pdd[:, 1:], weights=pdd[:, 0], axis=0)

def PPC(periodic_set):
    
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
    
    if isinstance(periodic_set, PeriodicSet):
        motif, cell = periodic_set.motif, periodic_set.cell
    else:
        motif, cell = periodic_set[0], periodic_set[1]

    m, n = motif.shape
    det = np.linalg.det(cell)
    t = (n - n % 2) / 2
    if n % 2 == 0:
        V = (np.pi ** t) / np.math.factorial(t)
    else:
        V = (2 * np.math.factorial(t) * (4 * np.pi) ** t) / np.math.factorial(n)
    return (det / (m * V)) ** (1./n)

def ppc(*args, **kwargs):
    warnings.warn("amd.ppc() is deprecated; use amd.PPC() instead.", DeprecationWarning)
    return PPC(*args, **kwargs)

def AMD_estimate(periodic_set, k):
    """
    Calculates an estimate of AMD_k (or PDD) based on the 
    point packing coefficient (ppc), which determines the
    asymptotic behaviour. 
    """
    if isinstance(periodic_set, PeriodicSet):
        motif, cell = periodic_set.motif, periodic_set.cell
    else:
        motif, cell = periodic_set[0], periodic_set[1]
        
    n = motif.shape[1]
    c = PPC(motif, cell)
    return np.array([(x ** (1. / n)) * c for x in range(1, k + 1)])

def amd_estimate(*args, **kwargs):
    warnings.warn("amd.amd_estimate() is deprecated; use amd.AMD_estimate() instead.", DeprecationWarning)
    return AMD_estimate(*args, **kwargs)


if __name__ == "__main__":
    pass