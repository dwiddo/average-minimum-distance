"""Functions for comparing AMDs and PDDs of crystals.
"""

from typing import List, Tuple, Optional, Union
import warnings

import numpy as np
import scipy.spatial    # cdist, pdist, squareform
import scipy.optimize   # linear_sum_assignment

from ._network_simplex import network_simplex
from .utils import ETA


_VERBOSE = False
_VERBOSE_UPDATE_RATE = 100

def set_verbose(setting, update_rate=100):
    """Pass True/False to turn on/off an ETA where relevant."""
    global _VERBOSE
    global _VERBOSE_UPDATE_RATE
    _VERBOSE = setting
    _VERBOSE_UPDATE_RATE = update_rate

set_verbose(False)


def EMD(
        pdd: np.ndarray,
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

        return emd_dist

    order = dists.shape[-1]
    n, m = len(sdd[0]), len(sdd_[0])

    dist_cdist = None
    if order == 2:
        dist_cdist = np.abs(sdd[1][:, None] - sdd_[1])
    else:
        dist, dist_ = sdd[1], sdd_[1]

        # take EMDs between finite PDDs in dist column
        weights = np.full((order, ), 1 / order)
        dist_cdist = np.empty((n, m), dtype=np.float64)
        for i in range(n):
            for j in range(m):
                finite_pdd_dm = scipy.spatial.distance.cdist(dist[i], dist_[j], metric='chebyshev')
                dists_emd, _ = network_simplex(weights, weights, finite_pdd_dm)
                dist_cdist[i, j] = dists_emd

        # flatten and compare by linf
        # flat_dist = dist.reshape((n, order * (order - 1)))
        # flat_dist_ = dist_.reshape((m, order * (order - 1)))
        # flat_dist = np.sort(flat_dist, axis=-1)
        # flat_dist_ = np.sort(flat_dist_, axis=-1)
        # dist_cdist = scipy.spatial.distance.cdist(flat_dist, flat_dist_, metric='chebyshev')

    dm = np.empty((n, m), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            cost_matrix = scipy.spatial.distance.cdist(dists[i], dists_[j], metric='chebyshev')
            row_ind, col_ind = scipy.optimize.linear_sum_assignment(cost_matrix)
            dm[i, j] = max(np.amax(cost_matrix[row_ind, col_ind]), dist_cdist[i, j])

    emd_dist, transport_plan = network_simplex(sdd[0], sdd_[0], dm)

    if return_transport:
        return emd_dist, transport_plan.reshape(dm.shape)

    return emd_dist


def AMD_cdist(
        amds: Union[np.ndarray, List[np.ndarray]],
        amds_: Union[np.ndarray, List[np.ndarray]],
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
    metric : str or callable, optional
        Usually AMDs are compared with the Chebyshev/l-infinity distance.
        Can take any metric + kwargs accepted by ``scipy.spatial.distance.cdist``.
    low_memory : bool, optional
        Use a slower but more memory efficient method for
        large collections of AMDs (Chebyshev/l-inf distance only).

    Returns
    -------
    ndarray
        A distance matrix shape ``(len(amds), len(amds_))``.
        The :math:`ij` th entry is the distance between ``amds[i]``
        and ``amds[j]`` given by the ``metric``.
    """

    amds, amds_ = np.asarray(amds), np.asarray(amds_)

    if len(amds.shape) == 1:
        amds = np.array([amds])
    if len(amds_.shape) == 1:
        amds_ = np.array([amds_])

    if low_memory:
        if metric != 'chebyshev':
            warnings.warn("Using only allowed metric 'chebyshev' for low_memory", UserWarning)

        dm = np.empty((len(amds), len(amds_)))
        for i, amd_vec in enumerate(amds):
            dm[i] = np.amax(np.abs(amds_ - amd_vec), axis=-1)
    else:
        dm = scipy.spatial.distance.cdist(amds, amds_, metric=metric, **kwargs)

    return dm


def AMD_pdist(
        amds: Union[np.ndarray, List[np.ndarray]],
        metric: str = 'chebyshev',
        low_memory: bool = False,
        **kwargs
) -> np.ndarray:
    """Compare a set of AMDs pairwise, returning a condensed distance matrix.

    Parameters
    ----------
    amds : array_like
        An array/list of AMDs.
    metric : str or callable, optional
        Usually AMDs are compared with the Chebyshev/l-infinity distance.
        Can take any metric + kwargs accepted by ``scipy.spatial.distance.cdist``.
    low_memory : bool, optional
        Optionally use a slightly slower but more memory efficient method for
        large collections of AMDs (Chebyshev/l-inf distance only).

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
        if metric != 'chebyshev':
            warnings.warn("Using only allowed metric 'chebyshev' for low_memory", UserWarning)
        cdm = np.empty((m * (m - 1)) // 2, dtype=np.double)
        ind = 0
        for i in range(m):
            ind_ = ind + m - i - 1
            cdm[ind:ind_] = np.amax(np.abs(amds[i+1:] - amds[i]), axis=-1)
            ind = ind_
    else:
        cdm = scipy.spatial.distance.pdist(amds, metric=metric, **kwargs)

    return cdm


def PDD_cdist(
        pdds: List[np.ndarray],
        pdds_: List[np.ndarray],
        metric: str = 'chebyshev',
        **kwargs
) -> np.ndarray:
    r"""Compare two sets of PDDs with each other, returning a distance matrix.

    Parameters
    ----------
    pdds : list of ndarrays
        A list of PDDs.
    pdds\_ : list of ndarrays
        A list of PDDs.
    metric : str or callable, optional
        Usually PDD rows are compared with the Chebyshev/l-infinity distance.
        Can take any metric + kwargs accepted by
        ``scipy.spatial.distance.cdist``.

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
    dm = np.empty((n, m))
    if _VERBOSE:
        eta = ETA(n * m, update_rate=_VERBOSE_UPDATE_RATE)

    for i in range(n):
        pdd = pdds[i]
        for j in range(m):
            dm[i, j] = EMD(pdd, pdds_[j], metric=metric, **kwargs)
            if _VERBOSE:
                eta.update()

    return dm


def PDD_pdist(
        pdds: List[np.ndarray],
        metric: str = 'chebyshev',
        **kwargs
) -> np.ndarray:
    """Compare a set of PDDs pairwise, returning a condensed distance matrix.

    Parameters
    ----------
    pdds : list of ndarrays
        A list of PDDs.
    metric : str or callable, optional
        Usually PDD rows are compared with the Chebyshev/l-infinity distance.
        Can take any metric + kwargs accepted by ``scipy.spatial.distance.cdist``.

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
    cdm = np.empty((m * (m - 1)) // 2, dtype=np.double)
    if _VERBOSE:
        eta = ETA((m * (m - 1)) // 2, update_rate=_VERBOSE_UPDATE_RATE)
    inds = ((i, j) for i in range(0, m - 1) for j in range(i + 1, m))

    for r, (i, j) in enumerate(inds):
        cdm[r] = EMD(pdds[i], pdds[j], metric=metric, **kwargs)
        if _VERBOSE:
            eta.update()

    return cdm


def emd(
        pdd: np.ndarray,
        pdd_: np.ndarray,
        metric: Optional[str] = 'chebyshev',
        return_transport: Optional[bool] = False,
        **kwargs):
    """Alias for amd.emd()."""
    return EMD(pdd, pdd_, metric=metric, return_transport=return_transport, **kwargs)
