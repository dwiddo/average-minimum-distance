"""Functions for comparing AMDs and PDDs of crystals.
"""

import inspect
from typing import List, Optional, Union, Tuple, Callable, Sequence
from functools import partial
from itertools import combinations
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.spatial.distance import cdist, pdist, squareform
from sklearn.neighbors import NearestNeighbors
from joblib import Parallel, delayed
import tqdm

from .io import CifReader, CSDReader
from .calculate import AMD, PDD
from ._emd import network_simplex
from .periodicset import PeriodicSet

FloatArray = npt.NDArray[np.floating]
IntArray = npt.NDArray[np.integer]

__all__ = [
    'compare',
    'EMD',
    'AMD_cdist',
    'AMD_pdist',
    'PDD_cdist',
    'PDD_pdist',
    'emd'
]

_SingleCompareInput = Union[PeriodicSet, str]
CompareInput = Union[_SingleCompareInput, List[_SingleCompareInput]]


def compare(
        crystals: CompareInput,
        crystals_: Optional[CompareInput] = None,
        by: str = 'AMD',
        k: int = 100,
        n_neighbors: Optional[int] = None,
        csd_refcodes: bool = False,
        verbose: bool = True,
        **kwargs
) -> pd.DataFrame:
    r"""Given one or two sets of crystals, compare by AMD or PDD and
    return a pandas DataFrame of the distance matrix.

    Given one or two paths to CIFs, periodic sets, CSD refcodes or lists
    thereof, compare by AMD or PDD and return a pandas DataFrame of the
    distance matrix. Default is to comapre by AMD with k = 100. Accepts
    any keyword arguments accepted by
    :class:`CifReader <.io.CifReader>`,
    :class:`CSDReader <.io.CSDReader>` and functions from
    :mod:`.compare`.

    Parameters
    ----------
    crystals : list of str or :class:`PeriodicSet <.periodicset.PeriodicSet>`
        A path, :class:`PeriodicSet <.periodicset.PeriodicSet>`, tuple
        or a list of those.
    crystals\_ : list of str or :class:`PeriodicSet <.periodicset.PeriodicSet>`, optional
        A path, :class:`PeriodicSet <.periodicset.PeriodicSet>`, tuple
        or a list of those.
    by : str, default 'AMD'
        Use AMD or PDD to compare crystals.
    k : int, default 100
        Parameter for AMD/PDD, the number of neighbor atoms to consider
        for each atom in a unit cell.
    n_neighbors : int, deafult None
        Find a number of nearest neighbors instead of a full distance
        matrix between crystals.
    csd_refcodes : bool, optional, csd-python-api only
        Interpret ``crystals`` and ``crystals_`` as CSD refcodes or
        lists thereof, rather than paths.
    verbose: bool, optional
        If True, prints a progress bar during reading, calculating and
        comparing items.
    **kwargs :
        Any keyword arguments accepted by the ``amd.CifReader``,
        ``amd.CSDReader``, ``amd.PDD`` and functions used to compare:
        ``reader``, ``remove_hydrogens``, ``disorder``,
        ``heaviest_component``, ``molecular_centres``,
        ``show_warnings``, (from class:`CifReader <.io.CifReader>`),
        ``refcode_families`` (from :class:`CSDReader <.io.CSDReader>`),
        ``collapse_tol`` (from :func:`PDD <.calculate.PDD>`),
        ``metric``, ``low_memory``
        (from :func:`AMD_pdist <.compare.AMD_pdist>`), ``metric``,
        ``backend``, ``n_jobs``, ``verbose``,
        (from :func:`PDD_pdist <.compare.PDD_pdist>`), ``algorithm``,
        ``leaf_size``, ``metric``, ``p``, ``metric_params``, ``n_jobs``
        (from :func:`_nearest_items <.compare._nearest_items>`).

    Returns
    -------
    df : :class:`pandas.DataFrame`
        DataFrame of the distance matrix for the given crystals compared
        by the chosen invariant.

    Raises
    ------
    ValueError
        If by is not 'AMD' or 'PDD', if either set given have no valid
        crystals to compare, or if crystals or crystals\_ are an invalid
        type.

    Examples
    --------
    Compare everything in a .cif (deafult, AMD with k=100)::

        df = amd.compare('data.cif')

    Compare everything in one cif with all crystals in all cifs in a
    directory (PDD, k=50)::

        df = amd.compare('data.cif', 'dir/to/cifs', by='PDD', k=50)

    **Examples (csd-python-api only)**

    Compare two crystals by CSD refcode (PDD, k=50)::

        df = amd.compare('DEBXIT01', 'DEBXIT02', csd_refcodes=True, by='PDD', k=50)

    Compare everything in a refcode family (AMD, k=100)::

        df = amd.compare('DEBXIT', csd_refcodes=True, families=True)
    """

    def _default_kwargs(func: Callable) -> dict:
        """Get the default keyword arguments from ``func``, if any
        arguments are in ``kwargs`` then replace with the value in
        ``kwargs`` instead of the default.
        """
        return {
            k: v.default for k, v in inspect.signature(func).parameters.items()
            if v.default is not inspect.Parameter.empty
        }

    def _unwrap_refcode_list(
            refcodes: List[str], **reader_kwargs
    ) -> List[PeriodicSet]:
        """Given string or list of strings, interpret as CSD refcodes
        and return a list of ``PeriodicSet`` objects.
        """
        if not all(isinstance(refcode, str) for refcode in refcodes):
            raise TypeError(
                f'amd.compare(csd_refcodes=True) expects a string or list of '
                'strings.'
            )
        return list(CSDReader(refcodes, **reader_kwargs))

    def _unwrap_pset_list(
            psets: List[Union[str, PeriodicSet]], **reader_kwargs
    ) -> List[PeriodicSet]:
        """Given a list of strings or ``PeriodicSet`` objects, interpret
        strings as paths and unwrap all items into one list of
        ``PeriodicSet``s.
        """
        ret = []
        for item in psets:
            if isinstance(item, PeriodicSet):
                ret.append(item)
            else:
                try:
                    path = Path(item)
                except TypeError:
                    raise ValueError(
                        'amd.compare() expects strings or amd.PeriodicSets, '
                        f'got {item.__class__.__name__}'
                    )
                ret.extend(CifReader(path, **reader_kwargs))
        return ret

    by = by.upper()
    if by not in ('AMD', 'PDD'):
        raise ValueError(
            "'by' parameter of amd.compare() must be 'AMD' or 'PDD' (passed "
            f"'{by}')"
        )

    # Sort out keyword arguments
    cifreader_kwargs = _default_kwargs(CifReader.__init__)
    csdreader_kwargs = _default_kwargs(CSDReader.__init__)
    csdreader_kwargs.pop('refcodes', None)
    pdd_kwargs = _default_kwargs(PDD)
    pdd_kwargs.pop('return_row_groups', None)
    compare_amds_kwargs = _default_kwargs(AMD_pdist)    
    compare_pdds_kwargs = _default_kwargs(PDD_pdist)
    nearest_items_kwargs = _default_kwargs(_nearest_items)
    nearest_items_kwargs.pop('XB', None)
    cifreader_kwargs['verbose'] = verbose
    csdreader_kwargs['verbose'] = verbose
    compare_pdds_kwargs['verbose'] = verbose

    for default_kwargs in (
        cifreader_kwargs, csdreader_kwargs, pdd_kwargs, compare_amds_kwargs,
        compare_pdds_kwargs, nearest_items_kwargs
    ):
        for kw in default_kwargs:
            if kw in kwargs:
                default_kwargs[kw] = kwargs[kw]

    # Get list of periodic sets from first input
    if not isinstance(crystals, list):
        crystals = [crystals]
    if csd_refcodes:
        crystals = _unwrap_refcode_list(crystals, **csdreader_kwargs)
    else:
        crystals = _unwrap_pset_list(crystals, **cifreader_kwargs)
    if not crystals:
        raise ValueError(
            'First argument passed to amd.compare() contains no valid '
            'crystals/periodic sets to compare.'
        )
    names = [s.name for s in crystals]
    if verbose:
        crystals = tqdm.tqdm(crystals, desc='Calculating', delay=1)

    # Get list of periodic sets from second input if given
    if crystals_ is None:
        names_ = names
    else:
        if not isinstance(crystals_, list):
            crystals_ = [crystals_]
        if csd_refcodes:
            crystals_ = _unwrap_refcode_list(crystals_, **csdreader_kwargs)
        else:
            crystals_ = _unwrap_pset_list(crystals_, **cifreader_kwargs)
        if not crystals_:
            raise ValueError(
                'Second argument passed to amd.compare() contains no '
                'valid crystals/periodic sets to compare.'
            )
        names_ = [s.name for s in crystals_]
        if verbose:
            crystals_ = tqdm.tqdm(crystals_, desc='Calculating', delay=1)

    if by == 'AMD':

        amds = np.empty((len(names), k), dtype=np.float64)
        for i, s in enumerate(crystals):
            amds[i] = AMD(s, k)

        if crystals_ is None:
            if n_neighbors is None:
                dm = squareform(AMD_pdist(amds, **compare_amds_kwargs))
                return pd.DataFrame(dm, index=names, columns=names_)
            else:
                nn_dm, inds = _nearest_items(
                    n_neighbors, amds, **nearest_items_kwargs
                )
                return _nearest_neighbors_dataframe(nn_dm, inds, names, names_)
        else:
            amds_ = np.empty((len(names_), k), dtype=np.float64)
            for i, s in enumerate(crystals_):
                amds_[i] = AMD(s, k)

            if n_neighbors is None:
                dm = AMD_cdist(amds, amds_, **compare_amds_kwargs)
                return pd.DataFrame(dm, index=names, columns=names_)
            else:
                nn_dm, inds = _nearest_items(
                    n_neighbors, amds, amds_, **nearest_items_kwargs
                )
                return _nearest_neighbors_dataframe(nn_dm, inds, names, names_)

    elif by == 'PDD':

        pdds = [PDD(s, k, **pdd_kwargs) for s in crystals]

        if crystals_ is None:
            dm = PDD_pdist(pdds, **compare_pdds_kwargs)
            if n_neighbors is None:
                dm = squareform(dm)
        else:
            pdds_ = [PDD(s, k, **pdd_kwargs) for s in crystals_]
            dm = PDD_cdist(pdds, pdds_, **compare_pdds_kwargs)

        if n_neighbors is None:
            return pd.DataFrame(dm, index=names, columns=names_)
        else:
            nn_dm, inds = _neighbors_from_distance_matrix(n_neighbors, dm)
            return _nearest_neighbors_dataframe(nn_dm, inds, names, names_)


def EMD(
        pdd: FloatArray,
        pdd_: FloatArray,
        metric: Optional[str] = 'chebyshev',
        return_transport: Optional[bool] = False,
        **kwargs
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
        By default, Chebyshev (L-infinity) distance is chosen like with
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

    dm = cdist(pdd[:, 1:], pdd_[:, 1:], metric=metric, **kwargs)
    emd_dist, transport_plan = network_simplex(pdd[:, 0], pdd_[:, 0], dm)

    if return_transport:
        return emd_dist, transport_plan
    return emd_dist


def AMD_cdist(
        amds,
        amds_,
        metric: str = 'chebyshev',
        low_memory: bool = False,
        **kwargs
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
    if low_memory:
        if metric != 'chebyshev':
            raise ValueError(
                "'low_memory' parameter of amd.AMD_cdist() only implemented "
                "with metric='chebyshev'."
            )
        dm = np.empty((len(amds), len(amds_)))
        for i, amd_vec in enumerate(amds):
            dm[i] = np.amax(np.abs(amds_ - amd_vec), axis=-1)
    else:
        dm = cdist(amds, amds_, metric=metric, **kwargs)
    return dm


def AMD_pdist(
        amds,
        metric: str = 'chebyshev',
        low_memory: bool = False,
        **kwargs
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
    if low_memory:
        m = len(amds)
        if metric != 'chebyshev':
            raise ValueError(
                "'low_memory' parameter of amd.AMD_pdist() only implemented "
                "with metric='chebyshev'."
            )
        cdm = np.empty((m * (m - 1)) // 2, dtype=np.float64)
        ind = 0
        for i in range(m):
            ind_ = ind + m - i - 1
            cdm[ind:ind_] = np.amax(np.abs(amds[i+1:] - amds[i]), axis=-1)
            ind = ind_
    else:
        cdm = pdist(amds, metric=metric, **kwargs)
    return cdm


def PDD_cdist(
        pdds: List[FloatArray],
        pdds_: List[FloatArray],
        metric: str = 'chebyshev',
        backend: str = 'multiprocessing',
        n_jobs: Optional[int] = None,
        verbose: bool = False,
        **kwargs
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

    kwargs.pop('return_transport', None)
    k = pdds[0].shape[-1] - 1
    _verbose = 3 if verbose else 0

    if n_jobs is not None and n_jobs not in (0, 1):
        # TODO: put results into preallocated empty array in place
        dm = Parallel(backend=backend, n_jobs=n_jobs, verbose=_verbose)(
            delayed(partial(EMD, metric=metric, **kwargs))(pdds[i], pdds_[j])
            for i in range(len(pdds)) for j in range(len(pdds_))
        )
        dm = np.array(dm).reshape((len(pdds), len(pdds_)))

    else:
        n, m = len(pdds), len(pdds_)
        dm = np.empty((n, m))
        if verbose:
            desc = f'Comparing {len(pdds)}x{len(pdds_)} PDDs (k={k})'
            progress_bar = tqdm.tqdm(desc=desc, total=n*m)
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
        metric: str = 'chebyshev',
        backend: str = 'multiprocessing',
        n_jobs: Optional[int] = None,
        verbose: bool = False,
        **kwargs
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

    kwargs.pop('return_transport', None)
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
            desc = f'Comparing {len(pdds)} PDDs pairwise (k={k})'
            progress_bar = tqdm.tqdm(desc=desc, total=cdm_len)
            for r, (i, j) in enumerate(inds):
                cdm[r] = EMD(pdds[i], pdds[j], metric=metric, **kwargs)
                progress_bar.update(1)
            progress_bar.close()
        else:
            for r, (i, j) in enumerate(inds):
                cdm[r] = EMD(pdds[i], pdds[j], metric=metric, **kwargs)

    return cdm


def emd(
        pdd: FloatArray, pdd_: FloatArray, **kwargs
) -> Union[float, Tuple[float, FloatArray]]:
    """Alias for :func:`EMD() <.compare.EMD>`."""
    return EMD(pdd, pdd_, **kwargs)


def _neighbors_from_distance_matrix(
        n: int, dm: FloatArray
) -> Tuple[FloatArray, IntArray]:
    """Given a distance matrix, find the n nearest neighbors of each
    item.

    Parameters
    ----------
    n : int
        Number of nearest neighbors to find for each item.
    dm : :class:`numpy.ndarray`
        2D distance matrix or 1D condensed distance matrix.

    Returns
    -------
    (nn_dm, inds) : tuple of :class:`numpy.ndarray` s
        ``nn_dm[i][j]`` is the distance from item :math:`i` to its
        :math:`j+1` st nearest neighbor, and ``inds[i][j]`` is the
        index of this neighbor (:math:`j+1` since index 0 is the first
        nearest neighbor).
    """

    inds = None
    if len(dm.shape) == 2:
        inds = np.array(
            [np.argpartition(row, n)[:n] for row in dm], dtype=np.int64
        )
    elif len(dm.shape) == 1:
        dm = squareform(dm)
        inds = []
        for i, row in enumerate(dm):
            inds_row = np.argpartition(row, n+1)[:n+1]
            inds_row = inds_row[inds_row != i][:n]
            inds.append(inds_row)
        inds = np.array(inds, dtype=np.int64)
    else:
        ValueError(
            'amd.neighbors_from_distance_matrix() accepts a distance matrix, '
            'either a 2D distance matrix or a condensed distance matrix as '
            'returned by scipy.spatial.distance.pdist().'
        )

    nn_dm = np.take_along_axis(dm, inds, axis=-1)
    sorted_inds = np.argsort(nn_dm, axis=-1)
    inds = np.take_along_axis(inds, sorted_inds, axis=-1)
    nn_dm = np.take_along_axis(nn_dm, sorted_inds, axis=-1)
    return nn_dm, inds


def _nearest_items(
        n_neighbors: int,
        XA: FloatArray,
        XB: Optional[FloatArray] = None,
        algorithm: str = 'kd_tree',
        leaf_size: int = 5,
        metric: str = 'chebyshev',
        n_jobs=None,
        **kwargs
) -> Tuple[FloatArray, IntArray]:
    """Find nearest neighbor distances and indices between all
    items/observations/rows in ``XA`` and ``XB``. If ``XB`` is None,
    find neighbors in ``XA`` for all items in ``XA``.
    """

    if XB is None:
        XB_ = XA
        _n_neighbors = n_neighbors + 1
    else:
        XB_ = XB
        _n_neighbors = n_neighbors

    dists, inds = NearestNeighbors(
        n_neighbors=_n_neighbors,
        algorithm=algorithm,
        leaf_size=leaf_size,
        metric=metric,
        n_jobs=n_jobs,
        **kwargs
    ).fit(XB_).kneighbors(XA)

    if XB is not None:
        return dists, inds

    final_shape = (dists.shape[0], n_neighbors)
    dists_ = np.empty(final_shape, dtype=np.float64)
    inds_ = np.empty(final_shape, dtype=np.int64)

    for i, (d_row, ind_row) in enumerate(zip(dists, inds)):
        i_ = 0
        for d, j in zip(d_row, ind_row):
            if i == j:
                continue
            dists_[i, i_] = d
            inds_[i, i_] = j
            i_ += 1
            if i_ == n_neighbors:
                break
    return dists_, inds_


def _nearest_neighbors_dataframe(nn_dm, inds, names, names_=None):
    """Make ``pandas.DataFrame`` from distances to and indices of
    nearest neighbors from one set to another (as returned by
    neighbors_from_distance_matrix() or _nearest_items()).
    DataFrame has columns ID 1, DIST1, ID 2, DIST 2..., and names as
    indices.
    """

    if names_ is None:
        names_ = names
    data = {}
    for i in range(nn_dm.shape[-1]):
        data['ID ' + str(i+1)] = [names_[j] for j in inds[:, i]]
        data['DIST ' + str(i+1)] = nn_dm[:, i]
    return pd.DataFrame(data, index=names)
