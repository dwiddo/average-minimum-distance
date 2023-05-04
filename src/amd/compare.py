"""Functions for comparing AMDs and PDDs of crystals.
"""

from typing import List, Optional, Union, Tuple
from functools import partial
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist, pdist, squareform
from joblib import Parallel, delayed
import tqdm

from .io import CifReader, CSDReader
from .calculate import AMD, PDD
from ._emd import network_simplex
from .periodicset import PeriodicSet
from .utils import neighbours_from_distance_matrix

__all__ = [
    'compare',
    'EMD',
    'AMD_cdist',
    'AMD_pdist',
    'PDD_cdist',
    'PDD_pdist',
    'emd'
]


def compare(
        crystals,
        crystals_=None,
        by: str = 'AMD',
        k: int = 100,
        nearest: Optional[int] = None,
        reader: str = 'gemmi',
        remove_hydrogens: bool = False,
        disorder: str = 'skip',
        heaviest_component: bool = False,
        molecular_centres: bool = False,
        csd_refcodes: bool = False,
        refcode_families: bool = False,
        show_warnings: bool = True,
        collapse_tol: float = 1e-4,
        metric: str = 'chebyshev',
        n_jobs: Optional[int] = None,
        backend: str = 'multiprocessing',
        verbose: bool = False,
        low_memory: bool = False,
        **kwargs
) -> pd.DataFrame:
    r"""Given one or two sets of crystals, compare by AMD or PDD and
    return a pandas DataFrame of the distance matrix.

    Given one or two paths to CIFs, periodic sets, CSD refcodes or lists
    thereof, compare by AMD or PDD and return a pandas DataFrame of the
    distance matrix. Default is to comapre by AMD with k = 100. Accepts
    most keyword arguments accepted by
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
        Parameter for AMD/PDD, the number of neighbour atoms to consider
        for each atom in a unit cell.
    nearest : int, deafult None
        Find a number of nearest neighbours instead of a full distance
        matrix between crystals.
    reader : str, optional
        The backend package used to parse the CIF. The default is
        :code:`gemmi`, :code:`pymatgen` and :code:`ase` are also
        accepted, as well as :code:`ccdc` if csd-python-api is
        installed. The ccdc reader should be able to read any format
        accepted by :class:`ccdc.io.EntryReader`, though only CIFs have
        been tested.
    remove_hydrogens : bool, optional
        Remove hydrogens from the crystals.
    disorder : str, optional
        Controls how disordered structures are handled. Default is
        ``skip`` which skips any crystal with disorder, since disorder
        conflicts with the periodic set model. To read disordered
        structures anyway, choose either :code:`ordered_sites` to remove
        atoms with disorder or :code:`all_sites` include all atoms
        regardless of disorder.
    heaviest_component : bool, optional, csd-python-api only
        Removes all but the heaviest molecule in
        the asymmeric unit, intended for removing solvents.
    molecular_centres : bool, default False, csd-python-api only
        Use the centres of molecules for comparison
        instead of centres of atoms.
    csd_refcodes : bool, optional, csd-python-api only
        Interpret ``crystals`` and ``crystals_`` as CSD refcodes or
        lists thereof, rather than paths.
    refcode_families : bool, optional, csd-python-api only
        Read all entries whose refcode starts with
        the given strings, or 'families' (e.g. giving 'DEBXIT' reads all
        entries with refcodes starting with DEBXIT).
    show_warnings : bool, optional
        Controls whether warnings that arise during reading are printed.
    collapse_tol: float, default 1e-4, ``by='PDD'`` only
        If two PDD rows have all elements closer
        than ``collapse_tol``, they are merged and weights are given to
        rows in proportion to the number of times they appeared.
    metric : str or callable, default 'chebyshev'
        The metric to compare AMDs/PDDs with. AMDs are compared directly
        with this metric. EMD is the metric used between PDDs, which
        requires giving a metric to use between PDD rows. Chebyshev
        (L-infinity) distance is the default. Accepts any metric
        accepted by :func:`scipy.spatial.distance.cdist`.
    n_jobs : int, default None, ``by='PDD'`` only
        Maximum number of concurrent jobs for
        parallel processing with :code:`joblib`. Set to -1 to use the
        maximum. Using parallel processing may be slower for small
        inputs.
    backend : str, default 'multiprocessing', ``by='PDD'`` only
        The parallelization backend implementation for PDD comparisons.
        For a list of supported backends, see the backend argument of
        :class:`joblib.Parallel`.
    verbose : bool, default False
        Prints a progress bar when reading crystals, calculating
        AMDs/PDDs and comparing PDDs. If using parallel processing
        (n_jobs > 1), the verbose argument of :class:`joblib.Parallel`
        is used, otherwise uses ``tqdm``.
    low_memory : bool, default False, ``by='AMD'`` only
        Use a slower but more memory efficient
        method for large collections of AMDs (metric 'chebyshev' only).

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

    by = by.upper()
    if by not in ('AMD', 'PDD'):
        raise ValueError(
            "'by' parameter of amd.compare() must be 'AMD' or 'PDD' (passed "
            f"'{by}')"
        )

    if heaviest_component or molecular_centres:
        reader = 'ccdc'

    if refcode_families:
        csd_refcodes = True

    reader_kwargs = {
        'reader': reader,
        'families': refcode_families,
        'remove_hydrogens': remove_hydrogens,
        'disorder': disorder,
        'heaviest_component': heaviest_component,
        'molecular_centres': molecular_centres,
        'show_warnings': show_warnings,
        'verbose': verbose,
    }

    compare_kwargs = {
        'metric': metric,
        'n_jobs': n_jobs,
        'backend': backend,
        'verbose': verbose,
        'low_memory': low_memory,
        **kwargs
    }

    # Get list(s) of periodic sets from first input
    if csd_refcodes:
        crystals = _unwrap_refcode_list(crystals, **reader_kwargs)
    else:
        crystals = _unwrap_pset_list(crystals, **reader_kwargs)

    if not crystals:
        raise ValueError(
            'First argument passed to amd.compare() contains no valid '
            'crystals/periodic sets'
        )
    names = [s.name for s in crystals]
    if verbose:
        container = tqdm.tqdm(crystals, desc='Calculating', delay=1)
    else:
        container = crystals

    # Get list(s) of periodic sets from second input if given
    if crystals_ is None:
        names_ = names
        container_ = None
    else:
        if csd_refcodes:
            crystals_ = _unwrap_refcode_list(crystals_, **reader_kwargs)
        else:
            crystals_ = _unwrap_pset_list(crystals_, **reader_kwargs)
        if not crystals_:
            raise ValueError(
                'Second argument passed to amd.compare() contains no '
                'valid crystals/periodic sets'
            )
        names_ = [s.name for s in crystals_]
        if verbose:
            container_ = tqdm.tqdm(crystals_, desc='Calculating', delay=1)
        else:
            container_ = crystals_

    if by == 'AMD':
        invs = [AMD(s, k) for s in container]
        if isinstance(container, tqdm.tqdm):
            container.close()
        compare_kwargs.pop('n_jobs', None)
        compare_kwargs.pop('backend', None)
        compare_kwargs.pop('verbose', None)

        if crystals_ is None:
            dm = AMD_pdist(invs, **compare_kwargs)
        else:
            invs_ = [AMD(s, k) for s in container_]
            dm = AMD_cdist(invs, invs_, **compare_kwargs)

    elif by == 'PDD':
        invs = [PDD(s, k, collapse_tol=collapse_tol) for s in container]
        compare_kwargs.pop('low_memory', None)

        if crystals_ is None:
            dm = PDD_pdist(invs, **compare_kwargs)
        else:
            invs_ = [PDD(s, k, collapse_tol=collapse_tol) for s in container_]
            dm = PDD_cdist(invs, invs_, **compare_kwargs)

    if nearest:
        nn_dm, inds = neighbours_from_distance_matrix(nearest, dm)
        data = {}
        for i in range(nearest):
            data['ID ' + str(i+1)] = [names_[j] for j in inds[:, i]]
            data['DIST ' + str(i+1)] = nn_dm[:, i]
        df = pd.DataFrame(data, index=names)
    else:
        if dm.ndim == 1:
            dm = squareform(dm)
        df = pd.DataFrame(dm, index=names, columns=names_)

    return df


def EMD(
        pdd: np.ndarray,
        pdd_: np.ndarray,
        metric: Optional[str] = 'chebyshev',
        return_transport: Optional[bool] = False,
        **kwargs
) -> Union[float, Tuple[float, np.ndarray]]:
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
) -> np.ndarray:
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
        Usually AMDs are compared with the Chebyshev (L-infinitys) distance.
        Accepts any metric accepted by :func:`scipy.spatial.distance.cdist`.
    low_memory : bool, default False
        Use a slower but more memory efficient method for large collections of
        AMDs (metric 'chebyshev' only).

    Returns
    -------
    dm : :class:`numpy.ndarray`
        A distance matrix shape ``(len(amds), len(amds_))``. ``dm[ij]`` is the
        distance (given by ``metric``) between ``amds[i]`` and ``amds[j]``.
    """

    amds, amds_ = np.asarray(amds), np.asarray(amds_)

    if len(amds.shape) == 1:
        amds = np.array([amds])
    if len(amds_.shape) == 1:
        amds_ = np.array([amds_])

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
        amds, metric: str = 'chebyshev', low_memory: bool = False, **kwargs
) -> np.ndarray:
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

    Returns
    -------
    cdm : :class:`numpy.ndarray`
        Returns a condensed distance matrix. Collapses a square distance
        matrix into a vector, just keeping the upper half. See the
        function :func:`squareform <scipy.spatial.distance.squareform>`
        from SciPy to convert to a symmetric square distance matrix.
    """

    amds = np.asarray(amds)

    if len(amds.shape) == 1:
        amds = np.array([amds])

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
        pdds: List[np.ndarray],
        pdds_: List[np.ndarray],
        metric: str = 'chebyshev',
        backend: str = 'multiprocessing',
        n_jobs: Optional[int] = None,
        verbose: bool = False,
        **kwargs
) -> np.ndarray:
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
        pdds: List[np.ndarray],
        metric: str = 'chebyshev',
        backend: str = 'multiprocessing',
        n_jobs: Optional[int] = None,
        verbose: bool = False,
        **kwargs
) -> np.ndarray:
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
        pdd: np.ndarray, pdd_: np.ndarray, **kwargs
) -> Union[float, Tuple[float, np.ndarray]]:
    """Alias for :func:`EMD() <.compare.EMD>`."""
    return EMD(pdd, pdd_, **kwargs)


def _unwrap_refcode_list(refcodes, **reader_kwargs):
    """Given string or list of strings, interpret as CSD refcodes and
    return a list of PeriodicSets.
    """

    reader_kwargs.pop('reader', None)
    if isinstance(refcodes, list):
        if not all(isinstance(refcode, str) for refcode in refcodes):
            raise TypeError(
                f'amd.compare(refcodes=True) expects a string or list of '
                'strings.'
            )
    elif not isinstance(refcodes, str):
        raise TypeError(
            f'amd.compare(refcodes=True) expects a string or list of '
            f'strings, got {refcodes.__class__.__name__}'
        )
    return list(CSDReader(refcodes, **reader_kwargs))


def _unwrap_pset_list(psets, **reader_kwargs):
    """Given a valid input for amd.compare(), return a list of
    PeriodicSets. Accepts paths, PeriodicSets, tuples or lists
    thereof."""

    def _extract_periodicsets(item, **reader_kwargs):
        """Given a path, PeriodicSet or tuple, return a list of the
        PeriodicSet(s)."""

        if isinstance(item, PeriodicSet):
            return [item]
        if isinstance(item, Tuple):
            return [PeriodicSet(item[0], item[1])]
        try:
            path = Path(item)
        except TypeError:
            raise ValueError(
                'amd.compare() expects a string, amd.PeriodicSet or tuple, '
                f'got {item.__class__.__name__}'
            )
        return list(CifReader(path, **reader_kwargs))

    reader_kwargs.pop('families', None)
    if isinstance(psets, list):
        return [s for i in psets
                for s in _extract_periodicsets(i, **reader_kwargs)]
    return _extract_periodicsets(psets, **reader_kwargs)
