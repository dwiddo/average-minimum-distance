"""Functions for comparing AMDs and PDDs of crystals.
"""

from typing import List, Optional, Union
import warnings

import numpy as np
import scipy.spatial
import scipy.optimize

from . import _network_simplex
from . import utils


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
    # put kwargs back! make sure it works with amd_extras, everywhere EMD is used!
    dm = scipy.spatial.distance.cdist(pdd[:, 1:], pdd_[:, 1:], metric=metric, **kwargs)
    emd_dist, transport_plan = _network_simplex.network_simplex(pdd[:, 0], pdd_[:, 0], dm)

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
        verbose=False,
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

    kwargs.pop('return_transport', None)

    n, m = len(pdds), len(pdds_)
    dm = np.empty((n, m))
    if verbose:
        update_rate = (n * m) // 10000
        eta = utils.ETA(n * m, update_rate=update_rate)

    for i in range(n):
        pdd = pdds[i]
        for j in range(m):
            dm[i, j] = EMD(pdd, pdds_[j], metric=metric, **kwargs)
            if verbose:
                eta.update()

    return dm


def PDD_pdist(
        pdds: List[np.ndarray],
        metric: str = 'chebyshev',
        verbose=False,
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

    kwargs.pop('return_transport', None)

    m = len(pdds)
    cdm_len = (m * (m - 1)) // 2
    cdm = np.empty(cdm_len, dtype=np.double)
    if verbose:
        update_rate = cdm_len // 10000
        eta = utils.ETA(cdm_len, update_rate=update_rate)
    inds = ((i, j) for i in range(0, m - 1) for j in range(i + 1, m))

    for r, (i, j) in enumerate(inds):
        cdm[r] = EMD(pdds[i], pdds[j], metric=metric, **kwargs)
        if verbose:
            eta.update()

    return cdm


def emd(
        pdd: np.ndarray,
        pdd_: np.ndarray,
        metric: Optional[str] = 'chebyshev',
        return_transport: Optional[bool] = False,
        **kwargs):
    """Alias for amd.EMD()."""
    return EMD(pdd, pdd_, metric=metric, return_transport=return_transport, **kwargs)
