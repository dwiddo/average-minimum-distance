"""Functions to calculate AMDs and PDDs.
"""

from typing import Union, Tuple
from collections import defaultdict

import numpy as np
from scipy.spatial.distance import pdist, squareform

from ._nearest_neighbours import nearest_neighbours
from .periodicset import PeriodicSet

def _extract_motif_cell(periodic_set):
    """`periodic_set` is either a :class:`.periodicset.PeriodicSet` object, or
    a tuple of ndarrays (motif, cell). If possible, extracts the asymmetric unit 
    and wyckoff multiplicities and returns them, otherwise returns None.
    """
    
    asymmetric_unit, multiplicities = None, None

    if isinstance(periodic_set, PeriodicSet):
        
        motif, cell = periodic_set.motif, periodic_set.cell
        
        if 'asymmetric_unit' in periodic_set.tags and 'wyckoff_multiplicities' in periodic_set.tags:
            
            asymmetric_unit = periodic_set.asymmetric_unit
            multiplicities = periodic_set.wyckoff_multiplicities
            
    else:
        motif, cell = periodic_set[0], periodic_set[1]
        
    return motif, cell, asymmetric_unit, multiplicities

def _collapse_rows(weights, dists, collapse_tol):
    """Given a vector `weights`, matrix `dists` and tolerance `collapse_tol`, collapse 
    the identical rows of dists (if all entries in a row are within  `collapse_tol`) 
    and collapse the same entires of `weights` (adding entries that merge).
    """
    
    diffs = np.abs(dists[:, None] - dists)
    overlapping = np.all(diffs <= collapse_tol, axis=-1)
    
    # I hate this solution, it's a hotfix.
    # But I can't seem to make anything cleverer work.
    
    if np.triu(overlapping, 1).any():
        groups = {}
        group = 0
        for i, row in enumerate(overlapping):
            
            if i not in groups:
                groups[i] = group
                group += 1
            
            for j in np.argwhere(row).T[0]:
                groups[j] = groups[i]

        groupings = defaultdict(list)
        for key, val in sorted(groups.items()):
            groupings[val].append(key)

        weights_ = []
        keep = []
        for inds in groupings.values():
            keep.append(inds[0])
            weights_.append(np.sum(weights[inds]))
        weights = np.array(weights_)
        dists = dists[keep]
        
    return weights, dists
    

def AMD(periodic_set: Union[PeriodicSet, Tuple[np.ndarray, np.ndarray]], k: int) -> np.ndarray:
    """Computes an AMD vector up to `k` from a periodic set.
    
    Parameters
    ----------
    periodic_set : :class:`.periodicset.PeriodicSet` or tuple of ndarrays
        A periodic set represented by a :class:`.periodicset.PeriodicSet` object or 
        by a tuple (motif, cell) with coordinates in Cartesian form.
    k : int
        Length of AMD returned.
        
    Returns
    -------
    ndarray
        An ndarray of shape (k,), the AMD of ``periodic_set`` up to `k`.
        
    Examples
    --------
    Make list of AMDs with ``k=100`` for crystals in mycif.cif::

        amds = []
        for periodic_set in amd.CifReader('mycif.cif'):
            amds.append(amd.AMD(periodic_set, 100))
            
    Make list of AMDs with ``k=10`` for crystals in these CSD refcode families::
    
        amds = []
        for periodic_set in amd.CSDReader(['HXACAN', 'ACSALA'], families=True):
            amds.append(amd.AMD(periodic_set, 10))
            
    Manually pass a periodic set as a tuple (motif, cell)::

        # simple cubic lattice
        motif = np.array([[0,0,0]])
        cell = np.array([[1,0,0], [0,1,0], [0,0,1]])
        cubic_amd = amd.AMD((motif, cell), 100)
    """

    motif, cell, asymmetric_unit, multiplicities = _extract_motif_cell(periodic_set)
    
    pdd, _, _ = nearest_neighbours(motif, cell, k, asymmetric_unit=asymmetric_unit)
    
    return np.average(pdd, axis=0, weights=multiplicities)

def PDD(periodic_set: Union[PeriodicSet, Tuple[np.ndarray, np.ndarray]], k: int, 
        order: bool = True, collapse: bool = True, collapse_tol: float = 1e-4) -> np.ndarray:
    """Computes a PDD up to k from a periodic set.
    
    Parameters
    ----------
    periodic_set : :class:`.periodicset.PeriodicSet` or tuple of ndarrays
        A periodic set represented by a :class:`.periodicset.PeriodicSet` object or 
        by a tuple (motif, cell) with coordinates in Cartesian form.
    k : int
        Number of columns in the PDD, plus one for the first column of weights.
    order : bool, optional
        Whether or not to lexicographically order the rows. Default True.
    collapse: bool, optional
        Whether or not to collapse identical rows (within a tolerance). Default True.
    collapse_tol: float
        If two rows have all entries closer than collapse_tol, they get collapsed.
        Default is 1e-4.

    Returns
    -------
    ndarray
        An ndarray with k+1 columns, the PDD of ``periodic_set`` up to `k`.
        
    Examples
    --------
    Make list of PDDs with ``k=100`` for crystals in mycif.cif::

        pdds = []
        for periodic_set in amd.CifReader('mycif.cif'):
            pdds.append(amd.PDD(periodic_set, 100, order=False))    # do not lexicographically order rows
            
    Make list of PDDs with ``k=10`` for crystals in these CSD refcode families::
    
        pdds = []
        for periodic_set in amd.CSDReader(['HXACAN', 'ACSALA'], families=True):
            pdds.append(amd.PDD(periodic_set, 10, collapse=False))  # do not collapse rows
            
    Manually pass a periodic set as a tuple (motif, cell)::

        # simple cubic lattice
        motif = np.array([[0,0,0]])
        cell = np.array([[1,0,0], [0,1,0], [0,0,1]])
        cubic_amd = amd.PDD((motif, cell), 100)
    """
  
    motif, cell, asymmetric_unit, multiplicities = _extract_motif_cell(periodic_set)
        
    dists, _, _ = nearest_neighbours(motif, cell, k, asymmetric_unit=asymmetric_unit)
    
    if multiplicities is None:
        multiplicities = np.ones((motif.shape[0], ))
        
    m = np.sum(multiplicities)
    weights = multiplicities / m
    
    if collapse:
        weights, dists = _collapse_rows(weights, dists, collapse_tol)
        
    pdd = np.hstack((weights[:, None], dists))
    
    if order:
        pdd = pdd[np.lexsort(np.rot90(dists))]
    
    return pdd

def finite_AMD(motif: np.ndarray):
    """Computes the AMD of a finite point set (up to k = `len(motif) - 1`).
    
    Parameters
    ----------
    motif : ndarray
        Cartesian coordinates of points in a set. Shape (n_points, dimensions)

    Returns
    -------
    ndarray
        An vector length len(motif) - 1, the AMD of ``motif``.
        
    Examples
    --------
    Find AMD distance between finite trapezium and kite point sets::

        trapezium = np.array([[0,0],[1,1],[3,1],[4,0]])
        kite      = np.array([[0,0],[1,1],[1,-1],[4,0]])
        
        trap_amd = amd.finite_AMD(trapezium)
        kite_amd = amd.finite_AMD(kite)
    
        dist = amd.AMD_pdist(trap_amd, kite_amd)
    """
    
    dm = squareform(pdist(motif))
    dm = np.sort(dm, axis=-1)[:, 1:]
    return np.average(dm, axis=0)

def finite_PDD(motif: np.ndarray, 
               order: bool = True, collapse: bool = True, collapse_tol: float = 1e-4):
    """Computes the PDD of a finite point set (up to k = `len(motif) - 1`).
    
    Parameters
    ----------
    motif : ndarray
        Cartesian coordinates of points in a set. Shape (n_points, dimensions)
    order : bool, optional
        Whether or not to lexicographically order the rows. Default True.
    collapse: bool, optional
        Whether or not to collapse identical rows (within a tolerance). Default True.
    collapse_tol: float
        If two rows have all entries closer than collapse_tol, they get collapsed.
        Default is 1e-4.

    Returns
    -------
    ndarray
        An ndarray with len(motif) columns, the PDD of ``motif``.
        
    Examples
    --------
    Find PDD distance between finite trapezium and kite point sets::

        trapezium = np.array([[0,0],[1,1],[3,1],[4,0]])
        kite      = np.array([[0,0],[1,1],[1,-1],[4,0]])
        
        trap_pdd = amd.finite_PDD(trapezium)
        kite_pdd = amd.finite_PDD(kite)
    
        dist = amd.emd(trap_pdd, kite_pdd)
    """
    
    dm = squareform(pdist(motif))
    dists = np.sort(dm, axis=-1)[:, 1:]
    
    weights = np.full((motif.shape[0], ), 1 / motif.shape[0])
    
    if collapse:
        weights, dists = _collapse_rows(weights, dists, collapse_tol)
    
    pdd = np.hstack((weights[:, None], dists))
    
    if order:
        pdd = pdd[np.lexsort(np.rot90(dists))]
        
    return pdd

def PDD_to_AMD(pdd: np.ndarray) -> np.ndarray:
    """Calculates AMD from a PDD. Faster than computing both from scratch.
    
    Parameters
    ----------
    pdd : np.ndarray
        The PDD of a periodic set. 

    Returns
    -------
    ndarray
        The AMD of the periodic set.
    """
    return np.average(pdd[:, 1:], weights=pdd[:, 0], axis=0)

def PPC(periodic_set: Union[PeriodicSet, Tuple[np.ndarray, np.ndarray]]) -> float:
    r"""Calculate the point packing coefficient (PPC) of ``periodic_set``.
    
    The PPC is a constant of any periodic set determining the 
    asymptotic behaviour of its AMD or PDD as :math:`k \rightarrow \infty`. 
    
    As :math:`k \rightarrow \infty`, the ratio :math:`\text{AMD}_k / \sqrt[n]{k}`
    approaches the PPC (as does any row of its PDD). 
    
    For a unit cell :math:`U` and :math:`m` motif points in :math:`n` dimensions,
    
    .. math::
    
        \text{PPC} = \sqrt[n]{\frac{\text{Vol}[U]}{m V_n}}
        
    where :math:`V_n` is the volume of a unit sphere in :math:`n` dimensions.
    
    Parameters
    ----------
    periodic_set : :class:`.periodicset.PeriodicSet` or tuple of 
        ndarrays (motif, cell) representing the periodic set in Cartesian form.

    Returns
    -------
    float
        The PPC of ``periodic_set``.
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

def AMD_estimate(periodic_set: Union[PeriodicSet, Tuple[np.ndarray, np.ndarray]], k: int) -> np.ndarray:
    r"""Calculates an estimate of AMD based on the PPC, using the fact that
    
    .. math::
    
        \lim_{k\rightarrow\infty}\frac{\text{AMD}_k}{\sqrt[n]{k}} = \sqrt[n]{\frac{\text{Vol}[U]}{m V_n}}
    
    where :math:`U` is the unit cell, :math:`m` is the number of motif points and
    :math:`V_n` is the volume of a unit sphere in :math:`n`-dimensional space.
    """
    
    if isinstance(periodic_set, PeriodicSet):
        motif, cell = periodic_set.motif, periodic_set.cell
    else:
        motif, cell = periodic_set[0], periodic_set[1]
        
    n = motif.shape[1]
    c = PPC(motif, cell)
    return np.array([(x ** (1. / n)) * c for x in range(1, k + 1)])

if __name__ == "__main__":
    pass