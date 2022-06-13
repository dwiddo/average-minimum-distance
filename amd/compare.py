"""Functions for comparing AMDs and PDDs of crystals.
"""

from typing import List, Optional, Union
import warnings

import numpy as np
from scipy.spatial.distance import cdist, pdist
from progressbar import ProgressBar

from ._network_simplex import network_simplex


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
    pdd : numpy.ndarray
        PDD of a crystal.
    pdd\_ : numpy.ndarray
        PDD of a crystal.
    metric : str or callable, optional
        EMD between PDDs requires defining a distance between PDD rows.
        By default, Chebyshev (L-infinity) distance is chosen as with AMDs.
        Accepts any metric accepted by :func:`scipy.spatial.distance.cdist`.
    return_transport: bool, optional
        Return a tuple ``(distance, transport_plan)`` with the optimal transport.

    Returns
    -------
    float
        Earth mover's distance between two PDDs.

    Raises
    ------
    ValueError
        Thrown if ``pdd`` and ``pdd_`` do not have the same number of 
        columns (``k`` value).
    """
    # put kwargs back! make sure it works with amd_extras, everywhere EMD is used!
    dm = cdist(pdd[:, 1:], pdd_[:, 1:], metric=metric, **kwargs)
    emd_dist, transport_plan = network_simplex(pdd[:, 0], pdd_[:, 0], dm)

    if return_transport:
        return emd_dist, transport_plan

    return emd_dist


def AMD_cdist(
        amds: Union[np.ndarray, List[np.ndarray]],
        amds_: Union[np.ndarray, List[np.ndarray]],
        metric: str = 'chebyshev',
        low_memory: bool = False,
        **kwargs
) -> np.ndarray:
    r"""Compare two sets of AMDs with each other, returning a distance matrix.
    This function is essentially identical to :func:`scipy.spatial.distance.cdist`
    with the default metric ``chebyshev``.

    Parameters
    ----------
    amds : array_like
        A list of AMDs.
    amds\_ : array_like
        A list of AMDs.
    metric : str or callable, optional
        Usually AMDs are compared with the Chebyshev (L-infinitys) distance.
        Can take any metric accepted by :func:`scipy.spatial.distance.cdist`.
    low_memory : bool, optional
        Use a slower but more memory efficient method for
        large collections of AMDs (Chebyshev metric only).

    Returns
    -------
    dm : :class:`numpy.ndarray`
        A distance matrix shape ``(len(amds), len(amds_))``.
        ``dm[ij]`` is the distance (given by ``metric``) 
        between ``amds[i]`` and ``amds[j]``.
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
        dm = cdist(amds, amds_, metric=metric, **kwargs)

    return dm


def AMD_pdist(
        amds: Union[np.ndarray, List[np.ndarray]],
        metric: str = 'chebyshev',
        low_memory: bool = False,
        **kwargs
) -> np.ndarray:
    """Compare a set of AMDs pairwise, returning a condensed distance matrix.
    This function is essentially identical to :func:`scipy.spatial.distance.pdist`
    with the default metric ``chebyshev``.

    Parameters
    ----------
    amds : array_like
        An array/list of AMDs.
    metric : str or callable, optional
        Usually AMDs are compared with the Chebyshev (L-infinity) distance.
        Can take any metric accepted by :func:`scipy.spatial.distance.pdist`.
    low_memory : bool, optional
        Optionally use a slightly slower but more memory efficient method for
        large collections of AMDs (Chebyshev metric only).

    Returns
    -------
    :class:`numpy.ndarray`
        Returns a condensed distance matrix. Collapses a square distance
        matrix into a vector, just keeping the upper half. See
        :func:`scipy.spatial.distance.squareform` to convert to a square 
        distance matrix or for more on condensed distance matrices.
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
        cdm = pdist(amds, metric=metric, **kwargs)

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
    pdds : List[numpy.ndarray]
        A list of PDDs.
    pdds\_ : List[numpy.ndarray]
        A list of PDDs.
    metric : str or callable, optional
        Usually PDD rows are compared with the Chebyshev/l-infinity distance.
        Can take any metric accepted by :func:`scipy.spatial.distance.cdist`.

    Returns
    -------
    :class:`numpy.ndarray`
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
        bar = ProgressBar(max_value=n * m)

    for i in range(n):
        for j in range(m):
            dm[i, j] = EMD(pdds[i], pdds_[j], metric=metric, **kwargs)
            if verbose:
                bar.update(m * i + j)

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
    pdds : List[numpy.ndarray]
        A list of PDDs.
    metric : str or callable, optional
        Usually PDD rows are compared with the Chebyshev/l-infinity distance.
        Can take any metric accepted by :func:`scipy.spatial.distance.pdist`.

    Returns
    -------
    :class:`numpy.ndarray`
        Returns a condensed distance matrix. Collapses a square
        distance matrix into a vector just keeping the upper half. See
        :func:`scipy.spatial.distance.squareform` to convert to a square 
        distance matrix or for more on condensed distance matrices.
    """

    if isinstance(pdds, np.ndarray):
        if len(pdds.shape) == 2:
            pdds = [pdds]

    kwargs.pop('return_transport', None)

    m = len(pdds)
    cdm_len = (m * (m - 1)) // 2
    cdm = np.empty(cdm_len, dtype=np.double)
    if verbose:
        bar = ProgressBar(max_value=cdm_len)

    inds = ((i, j) for i in range(0, m - 1) for j in range(i + 1, m))

    for r, (i, j) in enumerate(inds):
        cdm[r] = EMD(pdds[i], pdds[j], metric=metric, **kwargs)
        if verbose:
            bar.update(r)

    return cdm


def emd(
        pdd: np.ndarray,
        pdd_: np.ndarray,
        metric: Optional[str] = 'chebyshev',
        return_transport: Optional[bool] = False,
        **kwargs):
    """Alias for amd.EMD()."""
    return EMD(pdd, pdd_, metric=metric, return_transport=return_transport, **kwargs)
