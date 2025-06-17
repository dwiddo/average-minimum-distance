"""Functions for comparing AMDs and PDDs of crystals."""

from typing import List, Optional, Union, Tuple
from functools import partial
from itertools import combinations

import numpy as np
import numba
from scipy.spatial.distance import cdist, pdist
from joblib import Parallel, delayed
import tqdm

from ._emd import network_simplex
from ._types import FloatArray

__all__ = ["EMD", "AMD_cdist", "AMD_pdist", "PDD_cdist", "PDD_pdist"]


def EMD(
    pdd: FloatArray,
    pdd_: FloatArray,
    metric: Optional[str] = "chebyshev",
    return_transport: Optional[bool] = False,
    **kwargs,
) -> Union[float, Tuple[float, FloatArray]]:
    r"""Calculate the Earth mover's distance (EMD) between two PDDs, aka
    the Wasserstein metric.

    Parameters
    ----------
    pdd : :class:`numpy.ndarray`
        PDD of a crystal.
    pdd\_ : :class:`numpy.ndarray`
        PDD of a crystal.
    metric : str or callable, default 'chebyshev'
        EMD between PDDs requires defining a distance between PDD rows.
        By default, Chebyshev (L-infinity) distance is chosen as with
        AMDs. Accepts any metric accepted by
        :func:`scipy.spatial.distance.cdist`.
    return_transport: bool, default False
        Instead return a tuple ``(emd, transport_plan)`` where
        transport_plan describes the optimal flow.

    Returns
    -------
    emd : float
        Earth mover's distance between two PDDs. If ``return_transport``
        is True, return a tuple (emd, transport_plan).

    Raises
    ------
    ValueError
        Thrown if ``pdd`` and ``pdd_`` do not have the same number of
        columns.
    """

    emd_dist, transport_plan = _EMD(
        pdd[:, 0], pdd_[:, 0], pdd[:, 1:], pdd_[:, 1:], metric=metric, **kwargs
    )
    if return_transport:
        return emd_dist, transport_plan
    return emd_dist


def _EMD(
    weights: FloatArray,
    weights_: FloatArray,
    dist: FloatArray,
    dist_: FloatArray,
    metric: Optional[str] = None,
    **kwargs,
) -> Tuple[float, FloatArray]:
    r"""Calculate the earth mover's distance (EMD) between two weighted
    distributions (collections of vectors).

    Parameters
    ----------
    dist : :class:`numpy.ndarray`
        ``(n, d)`` array of items in the first distribution.
    dist_ : :class:`numpy.ndarray`
        ``(m, d)`` array of items in the second distribution.
    weights : :class:`numpy.ndarray`
        Weights of items in ``dist``.
    weights\_ : :class:`numpy.ndarray`
        Weights of items in ``dist\_``.
    metric : str or callable, default 'chebyshev'
        Metric used as the base distance between items in ``dist`` and
        ``dist\_``. For a list of accepted metrics see
        :func:`scipy.spatial.distance.cdist`.

    Returns
    -------
    emd : float
        Earth mover's distance between two PDDs. If ``return_transport``
        is True, returns a tuple (emd, transport_plan).
    transport_plan : :class:`numpy.ndarray`
        Matrix of optimal flows between the two distributions.
    """

    dm = cdist(dist, dist_, metric=metric, **kwargs)
    return network_simplex(weights, weights_, dm)


def AMD_cdist(
    amds, amds_, metric: str = "chebyshev", low_memory: bool = False, **kwargs
) -> FloatArray:
    r"""Compare two sets of AMDs with each other, returning a distance
    matrix. This function is essentially
    :func:`scipy.spatial.distance.cdist` with the default metric
    ``chebyshev`` and a low memory option.

    Parameters
    ----------
    amds : ArrayLike
        A list/array of AMDs.
    amds\_ : ArrayLike
        A list/array of AMDs.
    metric : str or callable, default 'chebyshev'
        Usually AMDs are compared with the Chebyshev (L-infinitys)
        distance. Accepts any metric accepted by
        :func:`scipy.spatial.distance.cdist`.
    low_memory : bool, default False
        Use a slower but more memory efficient method for large
        collections of AMDs (metric 'chebyshev' only).
    **kwargs :
        Extra arguments for ``metric``, passed to
        :func:`scipy.spatial.distance.cdist`.

    Returns
    -------
    dm : :class:`numpy.ndarray`
        A distance matrix shape ``(len(amds), len(amds_))``. ``dm[ij]``
        is the distance (given by ``metric``) between ``amds[i]`` and
        ``amds[j]``.
    """

    amds = np.asarray(amds)

    if low_memory:
        if metric != "chebyshev":
            raise ValueError(
                "'low_memory' parameter of amd.AMD_cdist() only implemented "
                "with metric='chebyshev'"
            )
        dm = np.empty((len(amds), len(amds_)))
        for i, amd_vec in enumerate(amds):
            dm[i] = np.amax(np.abs(amds_ - amd_vec), axis=-1)
    else:
        dm = cdist(amds, amds_, metric=metric, **kwargs)
    return dm


def AMD_pdist(
    amds, metric: str = "chebyshev", low_memory: bool = False, **kwargs
) -> FloatArray:
    """Compare a set of AMDs pairwise, returning a condensed distance
    matrix. This function is essentially
    :func:`scipy.spatial.distance.pdist` with the default metric
    ``chebyshev`` and a low memory parameter.

    Parameters
    ----------
    amds : ArrayLike
        An list/array of AMDs.
    metric : str or callable, default 'chebyshev'
        Usually AMDs are compared with the Chebyshev (L-infinity)
        distance. Accepts any metric accepted by
        :func:`scipy.spatial.distance.pdist`.
    low_memory : bool, default False
        Use a slower but more memory efficient method for large
        collections of AMDs (metric 'chebyshev' only).
    **kwargs :
        Extra arguments for ``metric``, passed to
        :func:`scipy.spatial.distance.pdist`.

    Returns
    -------
    cdm : :class:`numpy.ndarray`
        Returns a condensed distance matrix. Collapses a square distance
        matrix into a vector, just keeping the upper half. See the
        function :func:`squareform <scipy.spatial.distance.squareform>`
        from SciPy to convert to a symmetric square distance matrix.
    """

    amds = np.asarray(amds)

    @numba.njit(cache=True, fastmath=True)
    def _pdist_lowmem(amds):
        m = amds.shape[0]
        cdm = np.empty((m * (m - 1)) // 2, dtype=np.float64)
        ind = 0
        for i in range(m):
            for j in range(i + 1, m):
                cdm[ind] = np.amax(np.abs(amds[i] - amds[j]))
        return cdm

    if low_memory:
        if metric != "chebyshev":
            raise ValueError(
                "'low_memory' parameter of amd.AMD_pdist() only implemented "
                "with metric='chebyshev'"
            )
        cdm = _pdist_lowmem(amds)
    else:
        cdm = pdist(amds, metric=metric, **kwargs)

    return cdm


def PDD_cdist(
    pdds: List[FloatArray],
    pdds_: List[FloatArray],
    metric: str = "chebyshev",
    backend: str = "multiprocessing",
    n_jobs: Optional[int] = None,
    verbose: bool = False,
    **kwargs,
) -> FloatArray:
    r"""Compare two sets of PDDs with each other, returning a distance
    matrix. Supports parallel processing via joblib. If using
    parallelisation, make sure to include an if __name__ == '__main__'
    guard around this function.

    Parameters
    ----------
    pdds : List[:class:`numpy.ndarray`]
        A list of PDDs.
    pdds\_ : List[:class:`numpy.ndarray`]
        A list of PDDs.
    metric : str or callable, default 'chebyshev'
        Usually PDD rows are compared with the Chebyshev/l-infinity
        distance. Accepts any metric accepted by
        :func:`scipy.spatial.distance.cdist`.
    backend : str, default 'multiprocessing'
        The parallelization backend implementation. For a list of
        supported backends, see the backend argument of
        :class:`joblib.Parallel`.
    n_jobs : int, default None
        Maximum number of concurrent jobs for parallel processing with
        ``joblib``. Set to -1 to use the maximum. Using parallel
        processing may be slower for small inputs.
    verbose : bool, default False
        Prints a progress bar. If using parallel processing
        (n_jobs > 1), the verbose argument of :class:`joblib.Parallel`
        is used, otherwise uses tqdm.
    **kwargs :
        Extra arguments for ``metric``, passed to
        :func:`scipy.spatial.distance.cdist`.

    Returns
    -------
    dm : :class:`numpy.ndarray`
        Returns a distance matrix shape ``(len(pdds), len(pdds_))``. The
        :math:`ij` th entry is the distance between ``pdds[i]`` and
        ``pdds_[j]`` given by Earth mover's distance.
    """

    kwargs.pop("return_transport", None)
    k = pdds[0].shape[-1] - 1
    _verbose = 3 if verbose else 0

    if n_jobs is not None and n_jobs not in (0, 1):
        # TODO: put results into preallocated empty array in place
        dm = Parallel(backend=backend, n_jobs=n_jobs, verbose=_verbose)(
            delayed(partial(EMD, metric=metric, **kwargs))(pdds[i], pdds_[j])
            for i in range(len(pdds))
            for j in range(len(pdds_))
        )
        dm = np.array(dm).reshape((len(pdds), len(pdds_)))

    else:
        n, m = len(pdds), len(pdds_)
        dm = np.empty((n, m))
        if verbose:
            desc = f"Comparing {len(pdds)}x{len(pdds_)} PDDs (k={k})"
            progress_bar = tqdm.tqdm(desc=desc, total=n * m)
            for i in range(n):
                for j in range(m):
                    dm[i, j] = EMD(pdds[i], pdds_[j], metric=metric, **kwargs)
                    progress_bar.update(1)
            progress_bar.close()
        else:
            for i in range(n):
                for j in range(m):
                    dm[i, j] = EMD(pdds[i], pdds_[j], metric=metric, **kwargs)

    return dm


def PDD_pdist(
    pdds: List[FloatArray],
    metric: str = "chebyshev",
    backend: str = "multiprocessing",
    n_jobs: Optional[int] = None,
    verbose: bool = False,
    **kwargs,
) -> FloatArray:
    """Compare a set of PDDs pairwise, returning a condensed distance
    matrix. Supports parallelisation via joblib. If using
    parallelisation, make sure to include a if __name__ == '__main__'
    guard around this function.

    Parameters
    ----------
    pdds : List[:class:`numpy.ndarray`]
        A list of PDDs.
    metric : str or callable, default 'chebyshev'
        Usually PDD rows are compared with the Chebyshev/l-infinity
        distance. Accepts any metric accepted by
        :func:`scipy.spatial.distance.cdist`.
    backend : str, default 'multiprocessing'
        The parallelization backend implementation. For a list of
        supported backends, see the backend argument of
        :class:`joblib.Parallel`.
    n_jobs : int, default None
        Maximum number of concurrent jobs for parallel processing with
        ``joblib``. Set to -1 to use the maximum. Using parallel
        processing may be slower for small inputs.
    verbose : bool, default False
        Prints a progress bar. If using parallel processing
        (n_jobs > 1), the verbose argument of :class:`joblib.Parallel`
        is used, otherwise uses tqdm.
    **kwargs :
        Extra arguments for ``metric``, passed to
        :func:`scipy.spatial.distance.cdist`.

    Returns
    -------
    cdm : :class:`numpy.ndarray`
        Returns a condensed distance matrix. Collapses a square distance
        matrix into a vector, just keeping the upper half. See the
        function :func:`squareform <scipy.spatial.distance.squareform>`
        from SciPy to convert to a symmetric square distance matrix.
    """

    kwargs.pop("return_transport", None)
    k = pdds[0].shape[-1] - 1
    _verbose = 3 if verbose else 0

    if n_jobs is not None and n_jobs > 1:
        # TODO: put results into preallocated empty array in place
        cdm = Parallel(backend=backend, n_jobs=n_jobs, verbose=_verbose)(
            delayed(partial(EMD, metric=metric, **kwargs))(pdds[i], pdds[j])
            for i, j in combinations(range(len(pdds)), 2)
        )
        cdm = np.array(cdm)

    else:
        m = len(pdds)
        cdm_len = (m * (m - 1)) // 2
        cdm = np.empty(cdm_len, dtype=np.float64)
        inds = ((i, j) for i in range(0, m - 1) for j in range(i + 1, m))
        if verbose:
            desc = f"Comparing {len(pdds)} PDDs pairwise (k={k})"
            progress_bar = tqdm.tqdm(desc=desc, total=cdm_len)
            for r, (i, j) in enumerate(inds):
                cdm[r] = EMD(pdds[i], pdds[j], metric=metric, **kwargs)
                progress_bar.update(1)
            progress_bar.close()
        else:
            for r, (i, j) in enumerate(inds):
                cdm[r] = EMD(pdds[i], pdds[j], metric=metric, **kwargs)

    return cdm
