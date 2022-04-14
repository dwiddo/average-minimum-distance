"""Functions for comparing AMDs and PDDs of crystals.
"""

from typing import List, Tuple, Optional, Union
import warnings

import numpy as np
import scipy.spatial    # cdist, pdist, squareform
import scipy.optimize   # linear_sum_assignment

from ._network_simplex import network_simplex
from .utils import ETA


def EMD(
        pdd:  np.ndarray,
        pdd_: np.ndarray,
        metric: Optional[str] = 'chebyshev',
        return_transport: Optional[bool] = False,
        **kwargs):
    r"""Earth mover's distance (EMD) between two PDDs, also known as
    the Wasserstein metric.

    Parameters
    ----------
    pdd : ndarray
        PDD of a crystal.
    pdd\_ : ndarray
        PDD of a crystal.
    metric : str or callable, optional
        EMD between PDDs requires defining a distance between rows of two PDDs.
        By default, Chebyshev/l-infinity distance is chosen as with AMDs.
        Can take any metric + ``kwargs`` accepted by 
        ``scipy.spatial.distance.cdist``.
    return_transport: bool, optional
        Return a tuple (distance, transport_plan) with the optimal transport.

    Returns
    -------
    float
        Earth mover's distance between PDDs.

    Raises
    ------
    ValueError
        Thrown if the two PDDs do not have the
        same number of columns (``k`` value).
    """

    dm = scipy.spatial.distance.cdist(pdd[:, 1:], pdd_[:, 1:], metric=metric, **kwargs)
    emd_dist, transport_plan = network_simplex(pdd[:, 0], pdd_[:, 0], dm)

    if return_transport:
        return emd_dist, transport_plan.reshape(dm.shape)
    else:
        return emd_dist


def SDD_EMD(sdd, sdd_, return_transport: Optional[bool] = False):
    r"""Earth mover's distance (EMD) between two SDDs.

    Parameters
    ----------
    sdd : tuple of ndarrays
        SDD of a crystal.
    sdd\_ : tuple of ndarrays
        SDD of a crystal.
    return_transport: bool, optional
        Return a tuple (distance, transport_plan) with the optimal transport.

    Returns
    -------
    float
        Earth mover's distance between SDDs.

    Raises
    ------
    ValueError
        Thrown if the two SDDs are not of the same order or do not have the 
        same number of columns (``k`` value).
    """

    dists, dists_ = sdd[2], sdd_[2]

    # first order SDD, equivalent to PDD
    if dists.ndim == 2 and dists_.ndim == 2:
        dm = scipy.spatial.distance.cdist(dists, dists_, metric='chebyshev')
        emd_dist, transport_plan = network_simplex(sdd[0], sdd_[0], dm)

        if return_transport:
            return emd_dist, transport_plan.reshape(dm.shape)
        else:
            return emd_dist

    order = dists.shape[-1]
    n, m = len(sdd[0]), len(sdd_[0])

    dist_cdist = None
    if order == 2:
        dist_cdist = np.abs(sdd[1][:, None] - sdd_[1])
    else:
        dist, dist_ = sdd[1], sdd_[1]

        # take EMDs between finite PDDs in dist column
        """
        weights = np.full((order, ), 1 / order)
        dist_cdist = np.empty((n, m), dtype=np.float64)
        for i in range(n):
            for j in range(m):
                finite_pdd_dm = scipy.spatial.distance.cdist(dist[i], dist_[j], metric='chebyshev')
                dists_emd, _ = network_simplex(weights, weights, finite_pdd_dm)
                dist_cdist[i, j] = dists_emd
        """

        # flatten and compare by linf
        flat_dist  = dist.reshape((n, order * (order - 1)))
        flat_dist_ = dist_.reshape((m, order * (order - 1)))
        flat_dist  = np.sort(flat_dist, axis=-1)
        flat_dist_ = np.sort(flat_dist_, axis=-1)
        dist_cdist = scipy.spatial.distance.cdist(flat_dist, flat_dist_, metric='chebyshev')

    dm = np.empty((n, m), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            cost_matrix = scipy.spatial.distance.cdist(dists[i], dists_[j], metric='chebyshev')
            row_ind, col_ind = scipy.optimize.linear_sum_assignment(cost_matrix)
            dm[i, j] = max(np.amax(cost_matrix[row_ind, col_ind]), dist_cdist[i, j])

    emd_dist, transport_plan = network_simplex(sdd[0], sdd_[0], dm)

    if return_transport:
        return emd_dist, transport_plan.reshape(dm.shape)
    else:
        return emd_dist


def AMD_cdist(
        amds:  Union[np.ndarray, List[np.ndarray]], 
        amds_: Union[np.ndarray, List[np.ndarray]],
        k: Optional[int] = None,
        metric: str = 'chebyshev',
        low_memory: bool = False,
        **kwargs
) -> np.ndarray:
    r"""Compare two sets of AMDs with each other, returning a distance matrix.

    Parameters
    ----------
    amds : array_like
        A list of AMDs.
    amds\_ : array_like
        A list of AMDs.
    k : int, optional
        Truncate the AMDs to this value of ``k`` before comparing.
    low_memory : bool, optional
        Use a slower but more memory efficient method for
        large collections of AMDs (Chebyshev/l-inf distance only).
    metric : str or callable, optional
        Usually AMDs are compared with the Chebyshev/l-infinity distance.
        Can take any metric + kwargs accepted by ``scipy.spatial.distance.cdist``.

    Returns
    -------
    ndarray
        A distance matrix shape ``(len(amds), len(amds_))``. 
        The :math:`ij` th entry is the distance between ``amds[i]``
        and ``amds[j]`` given by the ``metric``.
    """

    amds  = np.asarray(amds)
    amds_ = np.asarray(amds_)

    if len(amds.shape) == 1:
        amds = np.array([amds])
    if len(amds_.shape) == 1:
        amds_ = np.array([amds_]) 

    if low_memory:
        if not metric == 'chebyshev':
            warnings.warn(
                "Using only allowed metric 'chebyshev' for low_memory", 
                UserWarning)

        dm = np.empty((len(amds), len(amds_)))
        for i in range(len(amds)):
            dm[i] = np.amax(np.abs(amds_ - amds[i]), axis=-1)
    else:
        dm = scipy.spatial.distance.cdist(amds[:, :k], amds_[:, :k], metric=metric, **kwargs)

    return dm


def AMD_pdist(
        amds: Union[np.ndarray, List[np.ndarray]],
        k: Optional[int] = None,
        low_memory: bool = False,
        metric: str = 'chebyshev',
        **kwargs
) -> np.ndarray:
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
            warnings.warn(
                "Using only allowed metric 'chebyshev' for low_memory", 
                UserWarning)
        cdm = np.empty((m * (m - 1)) // 2, dtype=np.double)
        ind = 0
        for i in range(m):
            ind_ = ind + m - i - 1
            cdm[ind:ind_] = np.amax(np.abs(amds[i+1:] - amds[i]), axis=-1)
            ind = ind_
    else:
        cdm = scipy.spatial.distance.pdist(amds[:, :k], metric=metric, **kwargs)

    return cdm


def PDD_cdist(
        pdds:  List[np.ndarray], 
        pdds_: List[np.ndarray], 
        k: Optional[int] = None,
        metric: str = 'chebyshev',
        verbose: bool = False,
        **kwargs
) -> np.ndarray:
    r"""Compare two sets of PDDs with each other, returning a distance matrix.

    Parameters
    ----------
    pdds : list of ndarrays
        A list of PDDs.
    pdds\_ : list of ndarrays
        A list of PDDs. 
    k : int, optional
        If :const:`None`, compare whole PDDs (largest ``k``). Set ``k`` to 
        an int to compare for a specific ``k`` (less than the maximum). 
    metric : str or callable, optional
        Usually PDD rows are compared with the Chebyshev/l-infinity distance.
        Can take any metric + kwargs accepted by 
        ``scipy.spatial.distance.cdist``.
    verbose : bool, optional
        Optionally print an ETA to terminal as large collections can take 
        some time.

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
            dm[i, j] = EMD(pdd[:, :t], pdds_[j][:, :t], metric=metric, **kwargs)
            if verbose: eta.update()

    return dm


def PDD_pdist(
        pdds: List[np.ndarray],
        k: Optional[int] = None,
        metric: str = 'chebyshev',
        verbose: bool = False,
        **kwargs
) -> np.ndarray:
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
        Optionally print an ETA to terminal as large collections can take
        some time.

    Returns
    -------
    ndarray
        Returns a condensed distance matrix. Collapses a square 
        distance matrix into a vector just keeping the upper half. Use
        ``scipy.spatial.distance.squareform`` to convert to a square 
        distance matrix.
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
        cdm[r] = EMD(pdds[i][:, :t], pdds[j][:, :t], metric=metric, **kwargs)
        if verbose: eta.update()

    return cdm


def emd(
        pdd:  np.ndarray,
        pdd_: np.ndarray, 
        metric: Optional[str] = 'chebyshev',
        return_transport: Optional[bool] = False, 
        **kwargs):
    """Alias for amd.emd()."""
    return EMD(pdd, pdd_, metric=metric, return_transport=return_transport, **kwargs)


def PDD_cdist_AMD_filter(
        n: int, 
        pdds: List[np.ndarray],
        pdds_: Optional[List[np.ndarray]] = None,
        k: Optional[int] = None,
        low_memory: bool = False,
        metric: str = 'chebyshev',
        verbose: bool = False,
        **kwargs
) -> Tuple[np.ndarray, np.ndarray]:
    r"""For each item in ``pdds``, get the ``n`` nearest items in ``pdds_`` by AMD,
    then compare references to these nearest items with PDDs.
    Tries to comprimise between the speed of AMDs and the accuracy of PDDs.

    If ``pdds_`` is :const:`None`, this essentially sets ``pdds_ = pdds``, i.e. 
    do an 'AMD neighbourhood graph' for one set whose weights are PDD distances.

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
        For the :math:`i` th item in reference and some :math:`j<n`,
        ``distance_matrix[i][j]`` is the distance from reference i to its j-th
        nearest neighbour in comparison (after the AMD filter).
        ``indices[i][j]`` is the index of said neighbour in ``pdds_``.
    """

    metric_kwargs = {'metric': metric, **kwargs}
    kwargs = {'k': k, 'metric': metric, **kwargs}

    comparison_set_size = len(pdds) if pdds_ is None else len(pdds_)

    if n >= comparison_set_size:

        if pdds_ is None:
            pdd_cdm = PDD_pdist(pdds, verbose=verbose, **kwargs)
            dm = scipy.spatial.distance.squareform(pdd_cdm)
        else:
            dm = PDD_cdist(pdds, pdds_, verbose=verbose, **kwargs)

        inds = np.argsort(dm, axis=-1)
        dm = np.take_along_axis(dm, inds, axis=-1)
        return dm, inds

    amds = np.array([np.average(pdd[:, 1:], weights=pdd[:, 0], axis=0)[:k]
                     for pdd in pdds])

    # one set, pairwise
    if pdds_ is None:
        pdds_ = pdds
        if low_memory:
            if not metric == 'chebyshev':
                warnings.warn(
                    "Using only allowed metric 'chebyshev' for low_memory",
                    UserWarning)
            if verbose: eta = ETA(len(amds))
            inds = []
            for i in range(len(amds)):
                dists = np.amax(np.abs(amds - amds[i]), axis=-1)
                inds_row = np.argpartition(dists, n+1)[:n+1]
                inds_row = inds_row[inds_row != i][:n]
                inds.append(inds_row)
                if verbose: eta.update()
            inds = np.array(inds)
        else:
            amd_cdm = AMD_pdist(amds, **kwargs)
            amd_dm = scipy.spatial.distance.squareform(amd_cdm)
            inds = []
            for i, row in enumerate(amd_dm):
                inds_row = np.argpartition(row, n+1)[:n+1]
                inds_row = inds_row[inds_row != i][:n]
                inds.append(inds_row)
            inds = np.array(inds)

    # one set v another
    else:
        amds_ = np.array([np.average(pdd[:,1:], weights=pdd[:,0], axis=0)[:k]
                          for pdd in pdds_])
        if low_memory:
            if not metric == 'chebyshev':
                warnings.warn(
                    "Using only allowed metric 'chebyshev' for low_memory",
                    UserWarning)
            if verbose: eta = ETA(len(amds) * len(amds_))
            inds = []
            for i in range(len(amds)):
                row = np.amax(np.abs(amds_ - amds[i]), axis=-1)
                inds.append(np.argpartition(row, n)[:n])
                if verbose: eta.update()
        else:
            amd_dm = AMD_cdist(amds, amds_, low_memory=low_memory, **kwargs)
            inds = np.array([np.argpartition(row, n)[:n] for row in amd_dm])

    dm = np.empty(inds.shape)
    if verbose: eta = ETA(inds.shape[0] * inds.shape[1])
    t = None if k is None else k + 1

    for i, row in enumerate(inds):
        for i_, j in enumerate(row):
            dm[i, i_] = EMD(pdds[i][:,:t], pdds_[j][:,:t], **metric_kwargs)
            if verbose: eta.update()

    sorted_inds = np.argsort(dm, axis=-1)
    inds = np.take_along_axis(inds, sorted_inds, axis=-1)
    dm = np.take_along_axis(dm, sorted_inds, axis=-1)
    
    return dm, inds
