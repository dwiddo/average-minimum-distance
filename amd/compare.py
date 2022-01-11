"""Functions for comparing AMDs or PDDs, finding nearest neighbours 
and minimum spanning trees.
"""

from typing import List, Tuple, Optional, Union
import warnings
import numpy as np
import numpy.typing as npt
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from ._network_simplex import network_simplex
from .utils import ETA

def emd(pdd, pdd_, metric='chebyshev', return_transport=False, **kwargs):
    r"""Earth mover's distance between two PDDs.
    
    Parameters
    ----------
    pdd : ndarray
        A PDD given by :func:`.calculate.PDD`.
    pdd\_ : ndarray
        A PDD given by :func:`.calculate.PDD`.
    metric : str or callable, optional
        Usually rows are compared with the Chebyshev/l-infinity distance.
        Can take any metric + kwargs accepted by ``scipy.spatial.distance.cdist``.
        
    Returns
    -------
    float
        Earth mover's distance between PDDs, where rows of the PDDs are
        compared with ``metric``.

    Raises
    ------
    ValueError
        Thrown if reference and comparison do not have 
        the same number of columns.
    """
    
    dm = cdist(pdd[:, 1:], pdd_[:, 1:], metric=metric, **kwargs)
    emd_dist, transport_plan = network_simplex(pdd[:, 0], pdd_[:, 0], dm)
    
    if return_transport:
        return emd_dist, transport_plan.reshape(dm.shape)
    else:
        return emd_dist

def AMD_cdist(amds: Union[np.ndarray, List[np.ndarray]], 
              amds_: Union[np.ndarray, List[np.ndarray]],
              k: Optional[int] = None,
              metric: str = 'chebyshev',
              low_memory: bool = False,
              **kwargs) -> np.ndarray:
    r"""Compare two sets of AMDs with each other, returning a distance matrix.
    
    Parameters
    ----------
    amds : array_like
        An array/list of AMDs.
    amds\_ : array_like
        An array/list of AMDs.
    k : int, optional
        If :const:`None`, compare entire AMDs. Set ``k`` to an int to compare 
        for a specific ``k`` (less than the maximum).
    low_memory : bool, optional
        Optionally use a slower but more memory efficient method for
        large collections of AMDs (Chebyshev/l-inf distance only).
    metric : str or callable, optional
        Usually AMDs are compared with the Chebyshev/l-infinity distance.
        Can take any metric + kwargs accepted by ``scipy.spatial.distance.cdist``.
        
    Returns
    -------
    ndarray
        Returns a distance matrix shape ``(len(amds), len(amds_))``. 
        The :math:`ij` th entry is the distance between ``amds[i]``
        and ``amds[j]`` given by the ``metric``.
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
        dm = cdist(amds[:, :k], amds_[:, :k], metric=metric, **kwargs)

    return dm

def AMD_pdist(amds: Union[np.ndarray, List[np.ndarray]],
              k: Optional[int] = None,
              low_memory: bool = False,
              metric: str = 'chebyshev',
              **kwargs) -> np.ndarray:
    """Compare a set of AMDs pairwise, returning a condensed distance matrix.
    
    Parameters
    ----------
    amds : array_like
        An array/list of AMDs.
    k : int, optional
        If :const:`None`, compare whole AMDs (largest ``k``). Set ``k`` 
        to an int to compare for a specific ``k`` (less than the maximum). 
    low_memory : bool, optional
        Optionally use a slightly slower but more memory efficient method for
        large collections of AMDs (Chebyshev/l-inf distance only).
    metric : str or callable, optional
        Usually AMDs are compared with the Chebyshev/l-infinity distance.
        Can take any metric + kwargs accepted by ``scipy.spatial.distance.cdist``.
        
    Returns
    -------
    ndarray
        Returns a condensed distance matrix. Collapses a square distance 
        matrix into a vector just keeping the upper half. Use
        ``scipy.spatial.distance.squareform`` to convert to a square distance matrix.
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
        cdm = pdist(amds[:, :k], metric=metric, **kwargs)

    return cdm

def PDD_cdist(pdds: List[np.ndarray], pdds_: List[np.ndarray], 
              k: Optional[int] = None,
              metric: str = 'chebyshev',
              verbose: bool = False,
              **kwargs) -> np.ndarray:
    r"""Compare two sets of PDDs with each other, returning a distance matrix.
    
    Parameters
    ----------
    pdds : list of ndarrays
        A list of PDDs.
    pdds\_ : list of ndarrays
        A list of PDDs. 
    k : int, optional
        If :const:`None`, compare whole PDDs (largest ``k``). Set ``k`` to an int to 
        compare for a specific ``k`` (less than the maximum). 
    metric : str or callable, optional
        Usually PDD rows are compared with the Chebyshev/l-infinity distance.
        Can take any metric + kwargs accepted by ``scipy.spatial.distance.cdist``.
    verbose : bool, optional
        Optionally print an ETA to terminal as large collections can take some time.
        
    Returns
    -------
    ndarray
        Returns a distance matrix shape ``(len(pdds), len(pdds_))``.
        The :math:`ij` th entry is the distance between ``pdds[i]``
        and ``pdds_[j]`` given by Earth mover's distance.
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
            dm[i, j] = emd(pdd[:, :t], pdds_[j][:, :t], metric=metric, **kwargs) 
            if verbose: eta.update()

    return dm

def PDD_pdist(pdds: List[np.ndarray],
              k: Optional[int] = None,
              metric: str = 'chebyshev',
              verbose: bool = False,
              **kwargs) -> np.ndarray:
    """Compare a set of PDDs pairwise, returning a condensed distance matrix.
    
    Parameters
    ----------
    pdds : list of ndarrays
        A list of PDDs.
    k : int, optional
        If :const:`None`, compare whole PDDs (largest ``k``). Set ``k`` to an int 
        to compare for a specific ``k`` (less than the maximum). 
    metric : str or callable, optional
        Usually PDD rows are compared with the Chebyshev/l-infinity distance.
        Can take any metric + kwargs accepted by ``scipy.spatial.distance.cdist``.
    verbose : bool, optional
        Optionally print an ETA to terminal as large collections can take some time.
    
    Returns
    -------
    ndarray
        Returns a condensed distance matrix. Collapses a square 
        distance matrix into a vector just keeping the upper half. Use
        ``scipy.spatial.distance.squareform`` to convert to a square distance matrix.

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
        cdm[r] = emd(pdds[i][:, :t], pdds[j][:, :t], metric=metric, **kwargs)
        if verbose: eta.update()
    
    return cdm

def filter(n: int, pdds: List[np.ndarray], pdds_: Optional[List[np.ndarray]] = None,
           k: Optional[int] = None,
           low_memory: bool = False,
           metric: str = 'chebyshev',
           verbose: bool = False,
           **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    r"""For each item in ``pdds``, get the ``n`` nearest items in ``pdds_`` by AMD,
    then compare references to these nearest items with PDDs.
    Tries to comprimise between the speed of AMDs and the accuracy of PDDs.
    
    If ``pdds_`` is :const:`None`, this essentially sets ``pdds_ = pdds``, i.e. do an
    'AMD neighbourhood graph' for one set whose weights are PDD distances.
    
    Parameters
    ----------
    n : int 
        Number of nearest neighbours to find.
    pdds : list of ndarrays
        A list of PDDs.
    pdds\_ : list of ndarrays, optional
        A list of PDDs.
    k : int, optional
        If :const:`None`, compare entire PDDs. Set ``k`` to an int 
        to compare for a specific ``k`` (less than the maximum). 
    low_memory : bool, optional
        Optionally use a slightly slower but more memory efficient method for
        large collections of AMDs (Chebyshev/l-inf distance only).
    metric : str or callable, optional
        Usually PDD rows are compared with the Chebyshev/l-infinity distance.
        Can take any metric + kwargs accepted by ``scipy.spatial.distance.cdist``.
    verbose : bool, optional
        Optionally print an ETA to terminal as large collections can take some time.
        
    Returns
    -------
    tuple of ndarrays (distance_matrix, indices)
        For the :math:`i` th item in reference and some :math:`j<n`, ``distance_matrix[i][j]`` 
        is the distance from reference i to its j-th nearest neighbour in comparison
        (after the AMD filter). ``indices[i][j]`` is the index of said neighbour in
        ``pdds_``. 
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
    t = None if k is None else k + 1
    
    for i, row in enumerate(inds):
        for i_, j in enumerate(row):
            dm[i, i_] = emd(pdds[i][:,:t], pdds_[j][:,:t], **metric_kwargs)
            if verbose: eta.update()

    sorted_inds = np.argsort(dm, axis=-1)
    inds = np.take_along_axis(inds, sorted_inds, axis=-1)
    dm = np.take_along_axis(dm, sorted_inds, axis=-1)
    
    return dm, inds

def AMD_mst(amds: npt.ArrayLike,
            k: Optional[int] = None,
            low_memory: bool = False,
            metric: str = 'chebyshev',
            **kwargs) -> List[Tuple[int, int, float]]:
    """Return list of edges in a minimum spanning tree based on AMDs.
    
    Parameters
    ----------
    amds : ndarray or list of ndarrays
        An array/list of AMDs.
    k : int, optional
        If :const:`None`, compare whole PDDs (largest ``k``). Set ``k`` to an int 
        to compare for a specific ``k`` (less than the maximum). 
    metric : str or callable, optional
        Usually PDD rows are compared with the Chebyshev/l-infinity distance.
        Can take any metric + kwargs accepted by ``scipy.spatial.distance.cdist``.
        
    Returns
    -------
    list of tuples
        Each tuple ``(i,j,w)`` is an edge in the mimimum spanning tree, where
        ``i`` and ``j`` are the indices of nodes and ``w`` is the AMD distance.
    """
    
    amds = np.asarray(amds)
    
    m = len(amds)
    cdm = AMD_pdist(amds, k=k, metric=metric, low_memory=low_memory, **kwargs)
    dm = squareform(cdm)
    lt_inds = np.tril_indices(m)
    dm[lt_inds] = 0
    
    return _mst_from_distance_matrix(dm)

def PDD_mst(pdds: List[np.ndarray],
            amd_filter_cutoff: Optional[int] = None,
            k: Optional[int] = None,
            metric: str = 'chebyshev',
            verbose: bool = False,
            **kwargs) -> List[Tuple[int, int, float]]:
    """Return list of edges in a minimum spanning tree based on PDDs.
    
    Parameters
    ----------
    pdds : list of ndarrays
        A list of PDDs.
    amd_filter_cutoff : int, optional
        If specified, apply the AMD filter behaviour of :func:`filter`.
        This is the ``n`` passed to :func:`filter`, the number of neighbours
        to connect in the neighbourhood graph.
    k : int, optional
        If :const:`None`, compare whole PDDs (largest ``k``). Set ``k`` to an int 
        to compare for a specific ``k`` (less than the maximum). 
    metric : str or callable, optional
        Usually PDD rows are compared with the Chebyshev/l-infinity distance.
        Can take any metric + kwargs accepted by ``scipy.spatial.distance.cdist``.
    verbose : bool, optional
        Optionally print an ETA to terminal as large collections can take some time.
    
    Returns
    -------
    list of tuples
        Each tuple ``(i,j,w)`` is an edge in the mimimum spanning tree, where
        ``i`` and ``j`` are the indices of nodes and ``w`` is the PDD distance.
    """
    
    kwargs = {'k': k, 'metric': metric, 'verbose': verbose, **kwargs}
    
    if isinstance(pdds, np.ndarray):
        if len(pdds.shape) == 2:
            pdds = [pdds]
    
    m = len(pdds)
    
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
    
    return _mst_from_distance_matrix(dm)

def neighbours_from_distance_matrix(n: int, dm: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Given a distance matrix, find the ``n`` nearest neighbours of each item.
    
    Parameters
    ----------
    n : int
        Number of nearest neighbours to find for each item.
    dm : ndarray
        2D distance matrix or 1D condensed distance matrix.
   
    Returns
    -------
    tuple of ndarrays (nn_dm, inds)
        For item ``i``, ``nn_dm[i][j]`` is the distance from item ``i`` to its ``j+1`` st 
        nearest neighbour, and ``inds[i][j]`` is the index of this neighbour (``j+1`` since
        index 0 is the first nearest neighbour).
    """
    
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

def _mst_from_distance_matrix(dm):
    
    csr_tree = minimum_spanning_tree(csr_matrix(dm))
    tree = csr_tree.toarray()
    
    edge_list = []
    i1, i2 = np.nonzero(tree)
    for i, j in zip(i1, i2):
        edge_list.append((int(i), int(j), dm[i][j]))
        
    return edge_list
