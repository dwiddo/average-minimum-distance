import warnings
import numpy as np
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from .core.Wasserstein import wasserstein
from .utils import ETA

def linf(v, v_):
    """l-infinity distance between vectors (AMDs)."""
    return np.amax(np.abs(v-v_))

def emd(pdd, pdd_, metric='chebyshev', **kwargs):
    """
    Earth mover's distance between two PDDs.
    
    Parameters
    ----------
    pdd : ndarray
        A :math:`m_a` by k PDD matrix (with weights in the first column).
    pdd_ : ndarray
        A :math:`m_b` by k PDD matrix (with weights in the first column).
    metric : str or callable, optional
        Usually rows are compared with the Chebyshev/l-infinity distance.
        Can take any metric + kwargs accepted by scipy.spatial.distance.cdist.
        
    Returns
    -------
    float
        Earth mover's distance between PDDs, where rows of the PDDs are
        compared with metric.

    Raises
    ------
    ValueError
        Thrown if reference and comparison do not have 
        the same number of columns.
    
    """
    
    dm = cdist(pdd[:, 1:], pdd_[:, 1:], metric=metric, **kwargs)
    
    return wasserstein(pdd[:, 0], pdd_[:, 0], dm)

"""
Main interface for comparisons. May change around in the future.

Functions AMD_* are for AMD comparisons, PDD_* for PDD comparisons.
Functions *_cdist are for comparing one set against another, *_pdist
are for comparing everything in one set pairwise.
The filter function takes PDDs but uses both AMDs and PDDs to find 
nearest neighbours.
*_mst is for creating minimum spanning trees. 
"""


def AMD_cdist(amds, amds_, 
              k=None,
              low_memory=False,
              metric='chebyshev',
              **kwargs):
    
    """
    Compare two sets of AMDs with each other. Returns a distance matrix.
    
    Parameters
    ----------
    amds : ndarray or list of ndarrays
        An :math:`m_a` by n array (of :math:`m_a` vectors/AMDs).
    amds_ : ndarray or list of ndarrays
        An :math:`m_b` by n array (of :math:`m_b` vectors/AMDs).
    k : int, optional
        If None, compare whole AMDs (largest k). Set k to an int to compare 
        for a specific k (less than the maximum).
    low_memory : bool, optional
        Optionally use a slightly slower but more memory efficient method for
        large collections of AMDs (Chebyshev/l-inf distance only).
    metric : str or callable, optional
        Usually AMDs are compared with the Chebyshev/l-infinity distance.
        Can take any metric + kwargs accepted by scipy.spatial.distance.cdist.
        
    Returns
    -------
    ndarray
        Returns a :math:`m_a` by :math:`m_b` distance matrix.
        The :math:`ij` th entry is the distance between amds[i]
        and amds_[j] given by the metric.

    """
    
    amds = np.asarray(amds)
    amds_ = np.asarray(amds_)
    
    if len(amds.shape) == 1:
        amds = np.array([amds])
    if len(amds_.shape) == 1:
        amds_ = np.array([amds_]) 
    
    if low_memory:
        if not metric == 'chebyshev':
            warnings.warn("Only metric 'chebyshev' implimented for low_memory AMD_cdist(). Using chebyshev",
                            UserWarning)
        dm = np.empty((len(amds), len(amds_)))
        for i in range(len(amds)):
            dm[i] = np.amax(np.abs(amds_ - amds[i]), axis=-1)
    else:
        dm = cdist(amds[:,:k], amds_[:,:k], metric=metric, **kwargs)

    return dm

def AMD_pdist(amds,
              k=None,
              low_memory=False,
              metric='chebyshev',
              **kwargs):
    
    """
    Do a pairwise comparison on one set of AMDs.
    
    Parameters
    ----------
    amds : ndarray or list of ndarrays
        An :math:`m_a` by n array (of :math:`m_a` vectors/AMDs).
    k : int, optional
        If None, compare whole AMDs (largest k). Set k to an int to compare 
        for a specific k (less than the maximum). 
    low_memory : bool, optional
        Optionally use a slightly slower but more memory efficient method for
        large collections of AMDs (Chebyshev/l-inf distance only).
    metric : str or callable, optional
        Usually AMDs are compared with the Chebyshev/l-infinity distance.
        Can take any metric + kwargs accepted by scipy.spatial.distance.cdist.
        
    Returns
    -------
    ndarray
        Returns a condensed distance matrix. Collapses a square 
        distance matrix into a vector just keeping the upper traingle.
        scipy's squareform will convert to a square distance matrix.
    """
    
    amds = np.asarray(amds)
    
    if len(amds.shape) == 1:
        amds = np.array([amds])
    
    if low_memory:
        m = len(amds)
        if not metric == 'chebyshev':
            warnings.warn("Only metric 'chebyshev' implimented for low_memory AMD_pdist(). Using chebyshev",
                            UserWarning)
        cdm = np.empty((m * (m - 1)) // 2, dtype=np.double)
        ind = 0
        for i in range(m):
            ind_ = ind + m - i - 1
            cdm[ind:ind_] = np.amax(np.abs(amds[i+1:] - amds[i]), axis=-1)
            ind = ind_
    else:
        cdm = pdist(amds[:,:k], metric=metric, **kwargs)

    
    return cdm

def PDD_cdist(pdds, pdds_, 
              k=None,
              metric='chebyshev',
              verbose=False,
              **kwargs):
    
    """
    Compare two sets of PDDs with each other. Returns a distance matrix.
    
    Parameters
    ----------
    pdds : list of ndarrays
        A list of PDDs (ndarrays whose last/second dimension agree). If a 2D 
        array, is interpreted as one PDD.
    pdds_ : ndarray or list of ndarrays
        A list of PDDs (ndarrays whose last/second dimension agree). If a 2D 
        array, is interpreted as one PDD.
    k : int, optional
        If None, compare whole PDDs (largest k). Set k to an int to compare 
        for a specific k (less than the maximum). 
    metric : str or callable, optional
        Usually PDD rows are compared with the Chebyshev/l-infinity distance.
        Can take any metric + kwargs accepted by scipy.spatial.distance.cdist.
    verbose : bool, optional
        Optionally print progress as for large sets this can be long.
        
    Returns
    -------
    ndarray
        Returns a :math:`m_a` by :math:`m_b` distance matrix.
        The :math:`ij` th entry is the distance between pdds[i]
        and pdds_[j] given by earth mover's distance.
    """
    
    if isinstance(pdds, np.ndarray):
        if len(pdds.shape) == 2:
            pdds = [pdds]
    
    if isinstance(pdds_, np.ndarray):
        if len(pdds_.shape) == 2:
            pdds_ = [pdds_]
    
    n, m = len(pdds), len(pdds_)
    
    t = None if k is None else k + 1
    
    dm = np.empty((n, m))
    if verbose: eta = ETA(n * m)
    
    for i in range(n):
        pdd = pdds[i]
        for j in range(m):
            dm[i, j] = emd(pdd[:,:t], pdds_[j][:,:t], metric=metric, **kwargs) 
            if verbose: eta.update()

    return dm

def PDD_pdist(pdds,
              k=None,
              metric='chebyshev',
              verbose=False,
              **kwargs):
    """
    Do a pairwise comparison on one set of PDDs.
    
    Parameters
    ----------
    pdds : list of ndarrays
        A list of PDDs (ndarrays whose last/second dimension agree). If a 2D 
        array, is interpreted as one PDD.
    k : int, optional
        If None, compare whole PDDs (largest k). Set k to an int to compare 
        for a specific k (less than the maximum). 
    metric : str or callable, optional
        Usually PDD rows are compared with the Chebyshev/l-infinity distance.
        Can take any metric + kwargs accepted by scipy.spatial.distance.cdist.
    verbose : bool, optional
        Optionally print progress as for large sets this can be long.
    
    Returns
    -------
    ndarray
        Returns a condensed distance matrix. Collapses a square 
        distance matrix into a vector just keeping the upper traingle.
        scipy's squareform will convert to a square distance matrix.

    """
    
    if isinstance(pdds, np.ndarray):
        if len(pdds.shape) == 2:
            pdds = [pdds]
    
    m = len(pdds)
    
    t = None if k is None else k + 1
    
    cdm = np.empty((m * (m - 1)) // 2, dtype=np.double)
    
    if verbose: eta = ETA((m * (m - 1)) // 2)
        
    inds = ((i,j) for i in range(0, m - 1) for j in range(i + 1, m))

    for r, (i, j) in enumerate(inds):
        
        cdm[r] = emd(pdds[i][:,:t], pdds[j][:,:t], metric=metric, **kwargs)
        
        if verbose: eta.update()
    
    return cdm

def filter(n, pdds, pdds_=None,
           k=None,
           low_memory=False,
           metric='chebyshev',
           verbose=False,
           **kwargs):
    
    """
    For each item in pdds, get the n nearest items in pdds_ by AMD,
    then compare references to these nearest items with PDDs.
    Tries to comprimise between the speed of AMDs and the accuracy of PDDs.
    
    If pdds_ is None, essentially sets pdds_ = pdds, i.e. do an
    'AMD neighbourhood graph' for one set whose weights are PDD distances.
    
    If n is smaller than the comparison set (pdds_ if it is not None and pdds
    if pdds_ is None), the AMD filter doesn't happen, this is essentially the
    same behaviour as pdd_cdist or pdd_pdist.
    
    Parameters
    ----------
    n : int 
        Number of nearest neighbours to find.
    pdds : ndarray or list of ndarrays
        A list of PDDs (ndarrays whose last/second dimension agree). If a 2D 
        array, is interpreted as one PDD.
    pdds_ : ndarray or list of ndarrays
        A list of PDDs (ndarrays whose last/second dimension agree). If a 2D 
        array, is interpreted as one PDD.
    k : int, optional
        If None, compare whole PDDs (largest k). Set k to an int to compare 
        for a specific k (less than the maximum). 
    low_memory : bool, optional
        Optionally use a slightly slower but more memory efficient method for
        large collections of AMDs (Chebyshev/l-inf distance only).
    metric : str or callable, optional
        Usually PDD rows are compared with the Chebyshev/l-infinity distance.
        Can take any metric + kwargs accepted by scipy.spatial.distance.cdist.
    verbose : bool, optional
        Optionally print progress as for large sets this can be long.
        
    Returns
    -------
    tuple of ndarrays (distance_matrix, indices)
        For the i-th item in reference and some j < n, distance_matrix[i][j] is 
        the distance from reference i to its j-th nearest neighbour in comparison
        (after the AMD filter). indices[i][j] is the index of said neighbour in
        comparison. 
    """
    
    metric_kwargs = {'metric': metric, **kwargs}
    kwargs = {'k': k, 'metric': metric, **kwargs}
    
    comparison_set_size = len(pdds) if pdds_ is None else len(pdds_)
    
    if n >= comparison_set_size:
        if pdds_ is None:
            pdd_cdm = PDD_pdist(pdds, verbose=verbose, **kwargs)
            dm = squareform(pdd_cdm)
        else:
            dm = PDD_cdist(pdds, pdds_, verbose=verbose, **kwargs)
        
        inds = np.argsort(dm, axis=-1)
        dm = np.take_along_axis(dm, inds, axis=-1)
        return dm, inds
    
    amds = np.array([np.average(pdd[:, 1:], weights=pdd[:, 0], axis=0) for pdd in pdds])
    
    # one set, pairwise
    if pdds_ is None:
        
        amd_cdm = AMD_pdist(amds, low_memory=low_memory, **kwargs)
        # kinda annoying I use squareform here. The alternative was so much worse...
        amd_dm = squareform(amd_cdm)
        inds = []
        for i, row in enumerate(amd_dm):
            inds_row = np.argpartition(row, n+1)[:n+1]
            inds_row = inds_row[inds_row != i][:n]
            inds.append(inds_row)
        inds = np.array(inds)
        
        pdds_ = pdds
    
    # one set v another
    else:
        
        amds_ = np.array([np.average(pdd[:,1:], weights=pdd[:,0], axis=0) for pdd in pdds_])
        amd_dm = AMD_cdist(amds, amds_, low_memory=low_memory, **kwargs)
    
        inds = np.array([np.argpartition(row, n)[:n] for row in amd_dm])
    
    dm = np.empty(inds.shape)
    
    if verbose: eta = ETA(inds.shape[0] * inds.shape[1])
    
    for i, row in enumerate(inds):
        for i_, j in enumerate(row):
            dm[i, i_] = emd(pdds[i][:,:k+1], pdds_[j][:,:k+1], **metric_kwargs)
            
            if verbose: eta.update()

    sorted_inds = np.argsort(dm, axis=-1)
    inds = np.take_along_axis(inds, sorted_inds, axis=-1)
    dm = np.take_along_axis(dm, sorted_inds, axis=-1)
    
    return dm, inds

def AMD_mst(amds,
            k=None,
            low_memory=False,
            metric='chebyshev',
            **kwargs):
    
    """
    Return list of edges in a minimum spanning tree based on AMDs.
    
    Parameters
    ----------
    amds : ndarray or list of ndarrays
        An :math:`m_a` by n array (of :math:`m_a` vectors/AMDs).
    k : int, optional
        If None, compare whole PDDs (largest k). Set k to an int to compare 
        for a specific k (less than the maximum). 
    metric : str or callable, optional
        Usually PDD rows are compared with the Chebyshev/l-infinity distance.
        Can take any metric + kwargs accepted by scipy.spatial.distance.cdist.
        
    Returns
    -------
    list of tuples
        Each tuple (i,j,w) is an edge in the mimimum spanning tree, where
        i and j are the indices of nodes and w is the weight on the edge
        connecting them.
    """
    
    amds = np.asarray(amds)
    
    m = len(amds)
    cdm = AMD_pdist(amds, k=k, metric=metric, low_memory=low_memory, **kwargs)
    dm = squareform(cdm)
    lt_inds = np.tril_indices(m)
    dm[lt_inds] = 0
    
    csr_tree = minimum_spanning_tree(csr_matrix(dm))
    tree = csr_tree.toarray()
    
    edge_list = []
    i1, i2 = np.nonzero(tree)
    for i, j in zip(i1, i2):
        edge_list.append((int(i), int(j), dm[i][j]))
        
    return edge_list

def PDD_mst(pdds,
            amd_filter_cutoff=None,
            k=None,
            metric='chebyshev',
            verbose=False,
            **kwargs):
    """
    Return list of edges in a minimum spanning tree based on PDDs.
    
    Parameters
    ----------
    pdds : ndarray or list of ndarrays
        A list of PDDs (ndarrays whose last/second dimension agree). If a 2D 
        array, is interpreted as one PDD.
    amd_filter_cutoff : int, optional
        If specified, apply the AMD filter behaviour of the filter() function.
        The int specified is the n passed to filter(), the number of neighbours
        to connect in the neighbourhood graph.
    k : int, optional
        If None, compare whole PDDs (largest k). Set k to an int to compare 
        for a specific k (less than the maximum).
    metric : str or callable, optional
        Usually PDD rows are compared with the Chebyshev/l-infinity distance.
        Can take any metric + kwargs accepted by scipy.spatial.distance.cdist.
    verbose : bool, optional
        Optionally print progress as for large sets this can be long.
    
    Returns
    -------
    list of tuples
        Each tuple (i,j,w) is an edge in the mimimum spanning tree, where
        i and j are the indices of nodes and w is the weight on the edge
        connecting them.
    """
    
    if isinstance(pdds, np.ndarray):
        if len(pdds.shape) == 2:
            pdds = [pdds]
    
    m = len(pdds)
    
    kwargs = {'k': k, 'metric': metric, 'verbose': verbose, **kwargs}
    
    if amd_filter_cutoff is None:
        cdm = PDD_pdist(pdds, **kwargs)
        dm = squareform(cdm)
        lt_inds = np.tril_indices(m)
        dm[lt_inds] = 0
        
    else:
        dists, inds = filter(amd_filter_cutoff, pdds, **kwargs)
        dm = np.empty((m, m))
        for i, row in enumerate(inds):
            for i_, j in enumerate(row):
                dm[i, j] = dists[i][i_]
    
    csr_tree = minimum_spanning_tree(csr_matrix(dm))
    tree = csr_tree.toarray()
    
    edge_list = []
    i1, i2 = np.nonzero(tree)
    for i, j in zip(i1, i2):
        edge_list.append((int(i), int(j), dm[i][j]))
        
    return edge_list

def neighbours_from_distance_matrix(n, dm):
    
    # 2D distance matrix
    if len(dm.shape) == 2:
        
        inds = np.array([np.argpartition(row, n)[:n] for row in dm])
    
    # 1D condensed distance vector
    elif len(dm.shape) == 1:
        dm = squareform(dm)
        inds = []
        for i, row in enumerate(dm):
            inds_row = np.argpartition(row, n+1)[:n+1]
            inds_row = inds_row[inds_row != i][:n]
            inds.append(inds_row)
        inds = np.array(inds)
    
    else:
        ValueError('Input must be an ndarray of either a 2D distance matrix or condensed distance matrix')

    # inds are the indexes of nns: inds[i,j] is the j-th nn to point i
    nn_dm = np.take_along_axis(dm, inds, axis=-1)
    sorted_inds = np.argsort(nn_dm, axis=-1)
    inds = np.take_along_axis(inds, sorted_inds, axis=-1)
    nn_dm = np.take_along_axis(nn_dm, sorted_inds, axis=-1)
    
    return nn_dm, inds

if __name__ == '__main__':
    pass