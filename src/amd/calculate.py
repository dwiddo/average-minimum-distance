"""Calculation of isometry invariants from periodic sets."""

import warnings
import collections
from typing import Tuple, Union
import itertools

import numpy as np
import numpy.typing as npt
import numba
from scipy.spatial.distance import pdist, squareform

from ._types import FloatArray
from ._nearest_neighbors import nearest_neighbors, nearest_neighbors_minval
from .periodicset import PeriodicSet
from .utils import diameter
from .globals_ import MAX_DISORDER_CONFIGS


__all__ = [
    "PDD",
    "AMD",
    "ADA",
    "PDA",
    "PDD_to_AMD",
    "AMD_finite",
    "PDD_finite",
    "PDD_reconstructable",
    "AMD_estimate",
]


def PDD(
    pset: PeriodicSet,
    k: int,
    lexsort: bool = True,
    collapse: bool = True,
    collapse_tol: float = 1e-4,
    return_row_data: bool = False,
) -> Union[FloatArray, Tuple[FloatArray, list]]:
    """Return the pointwise distance distribution (PDD) of a periodic
    set (usually representing a crystal).

    The PDD is a geometry based descriptor independent of choice of
    motif and unit cell. It is a matrix with each row corresponding to a
    point in the motif, starting with a weight followed by distances to
    the k nearest neighbors of the point.

    Parameters
    ----------
    pset : :class:`amd.PeriodicSet <.periodicset.PeriodicSet>`
        A periodic set (crystal).
    k : int
        Number of neighbors considered for each point in a unit cell.
        The output has k + 1 columns with the first column containing
        weights.
    lexsort : bool, default True
        Lexicographically order rows.
    collapse: bool, default True
        Collapse duplicate rows (within ``collapse_tol`` in the
        Chebyshev metric).
    collapse_tol: float, default 1e-4
        If two rows are closer than ``collapse_tol`` in the Chebyshev
        metric, they are merged and weights are given to rows in
        proportion to their frequency.
    return_row_data: bool, default False
        Return a tuple ``(pdd, groups)`` where ``groups`` contains
        information about which rows in ``pdd`` correspond to which
        points. If ``pset.asym_unit`` is None, then ``groups[i]``
        contains indices of points in ``pset.motif`` corresponding to
        ``pdd[i]``. Otherwise, PDD rows correspond to points in the
        asymmetric unit, and ``groups[i]`` contains indices pointing to
        ``pset.asym_unit``.

    Returns
    -------
    pdd : :class:`numpy.ndarray`
        The PDD of ``pset``, a :class:`numpy.ndarray` with ``k+1``
        columns. If ``return_row_data`` is True, returns a tuple
        (:class:`numpy.ndarray`, list).

    Examples
    --------
    Make list of PDDs with ``k=100`` for crystals in data.cif::

        pdds = []
        for periodic_set in amd.CifReader('data.cif'):
            pdd = amd.PDD(periodic_set, 100)
            pdds.append(pdd)

    Make list of PDDs with ``k=10`` for crystals in these CSD refcode
    families (requires csd-python-api)::

        pdds = []
        for periodic_set in amd.CSDReader(['HXACAN', 'ACSALA'], families=True):
            pdds.append(amd.PDD(periodic_set, 10))

    Manually create a periodic set as a tuple (motif, cell)::

        # simple cubic lattice
        motif = np.array([[0,0,0]])
        cell = np.array([[1,0,0], [0,1,0], [0,0,1]])
        periodic_set = amd.PeriodicSet(motif, cell)
        cubic_pdd = amd.PDD(periodic_set, 100)
    """

    if not isinstance(pset, PeriodicSet):
        raise ValueError(
            f"Expected {PeriodicSet.__name__}, got {pset.__class__.__name__}"
        )

    weights, dists, groups = _PDD(
        pset, k, lexsort=lexsort, collapse=collapse, collapse_tol=collapse_tol
    )
    pdd = np.empty(shape=(len(dists), k + 1), dtype=np.float64)
    pdd[:, 0] = weights
    pdd[:, 1:] = dists
    if return_row_data:
        return pdd, groups
    return pdd


def _PDD(
    pset: PeriodicSet,
    k: int,
    lexsort: bool = True,
    collapse: bool = True,
    collapse_tol: float = 1e-4,
) -> Tuple[FloatArray, FloatArray, list[list[int]]]:
    """See PDD() for documentation. This core function always returns a
    tuple (weights, dists, groups), with weights and dists to be merged
    by PDD() and groups to be optionally returned.
    """

    asym_unit = pset.motif[pset.asym_unit]
    weights = pset.multiplicities / pset.motif.shape[0]

    # Disordered structures
    subs_disorder_info = {}  # i: [inds masked] where i is sub disordered

    if pset.disorder:
        # Gather which disorder assemblies must be considered
        _asym_mask = np.full((asym_unit.shape[0], ), fill_value=True)
        asm_sizes = {}
        for i, asm in enumerate(pset.disorder):

            # Ignore assmeblies with 1 group
            if len(asm.groups) < 2:
                continue

            # For substitutional disorder, mask all but one atom
            elif asm.is_substitutional:
                mask_inds = [asm.groups[j].indices[0] for j in range(1, len(asm.groups))]
                keep = asm.groups[0].indices[0]
                subs_disorder_info[keep] = mask_inds
                _asym_mask[mask_inds] = False

            else:
                asm_sizes[i] = len(asm.groups)

        asm_sizes_arr = np.array(list(asm_sizes.values()))
        if _array_product_exceeds(asm_sizes_arr, MAX_DISORDER_CONFIGS):
            warnings.warn(
                f"Disorder configs exceeds limit "
                f"amd.globals_.MAX_DISORDER_CONFIGS={MAX_DISORDER_CONFIGS}, "
                "defaulting to majority occupancy config"
            )
            configs = [[]]
            for asm in pset.disorder:
                i, _ = max(enumerate(asm.groups), key=lambda g: g[1].occupancy)
                configs[0].append(i)
        else:
            configs = itertools.product(*(range(t) for t in asm_sizes.values()))

        # Mask atoms already masked (substitututional disorder)
        _motif_mask = np.full((pset.motif.shape[0], ), fill_value=True)
        if not np.all(_asym_mask):
            for i in range(len(asym_unit)):
                if not _asym_mask[i]:
                    m_i = pset.asym_unit[i]
                    mul = pset.multiplicities[i]
                    _motif_mask[m_i : m_i + mul] = False

        # One PDD for each disorder configuration
        dists_list, inds_list = [], []
        for config_inds in configs:

            # Mask groups not selected
            asym_mask = _asym_mask.copy()
            motif_mask = _motif_mask.copy()

            # Mask groups not selected in this config
            for i, asm_ind in enumerate(asm_sizes.keys()):
                for j, grp in enumerate(pset.disorder[asm_ind].groups):
                    if j != config_inds[i]:
                        for t in grp.indices:
                            asym_mask[t] = False
                            m_i = pset.asym_unit[t]
                            mul = pset.multiplicities[t]
                            motif_mask[m_i : m_i + mul] = False

            dists = nearest_neighbors(
                pset.motif[motif_mask], pset.cell, asym_unit[asym_mask], k + 1
            )
            dists_list.append(dists[:, 1:])
            inds_list.append(np.where(asym_mask)[0])

        dists = np.vstack(dists_list)
        inds = list(np.concatenate(inds_list))
        weights = np.concatenate([weights[i] for i in inds_list])
        weights /= np.sum(weights)

    else:
        dists = nearest_neighbors(pset.motif, pset.cell, asym_unit, k + 1)
        dists = dists[:, 1:]
        inds = list(range(len(dists)))

    # Collapse rows within tolerance
    groups = None
    if collapse:
        weights, dists, group_labs = _merge_pdd_rows(weights, dists, collapse_tol)
        if dists.shape[0] != len(group_labs):
            groups = [[] for _ in range(weights.shape[0])]
            for old_ind, new_ind in enumerate(group_labs):
                groups[new_ind].append(int(inds[old_ind]))

    if groups is None:
        groups = [[int(i)] for i in inds]

    # Add back substitutionally disordered sites to group info
    if subs_disorder_info:
        for i, masked_inds in subs_disorder_info.items():
            for grp in groups:
                if i in grp:
                    grp.extend(masked_inds)

    if lexsort:
        lex_ordering = np.lexsort(dists.T[::-1])
        weights = weights[lex_ordering]
        dists = dists[lex_ordering]
        groups = [groups[i] for i in lex_ordering]

    return weights, dists, groups


def AMD(pset: PeriodicSet, k: int) -> FloatArray:
    """Return the average minimum distance (AMD) of a periodic set
    (usually representing a crystal).

    The AMD is the centroid or average of the PDD (pointwise distance
    distribution) and hence is also a independent of choice of motif and
    unit cell. It is a vector containing average distances from points
    to k neighbouring points.

    Parameters
    ----------
    pset : :class:`amd.PeriodicSet <.periodicset.PeriodicSet>`
        A periodic set (crystal).
    k : int
        Number of neighbors considered for each point in a unit cell.

    Returns
    -------
    :class:`numpy.ndarray`
        The AMD of ``pset``, a :class:`numpy.ndarray` shape ``(k, )``.

    Examples
    --------
    Make list of AMDs with k = 100 for crystals in data.cif::

        amds = []
        for periodic_set in amd.CifReader('data.cif'):
            amds.append(amd.AMD(periodic_set, 100))

    Make list of AMDs with k = 10 for crystals in these CSD refcode families::

        amds = []
        for periodic_set in amd.CSDReader(['HXACAN', 'ACSALA'], families=True):
            amds.append(amd.AMD(periodic_set, 10))

    Manually create a periodic set as a tuple (motif, cell)::

        # simple cubic lattice
        motif = np.array([[0,0,0]])
        cell = np.array([[1,0,0], [0,1,0], [0,0,1]])
        periodic_set = amd.PeriodicSet(motif, cell)
        cubic_amd = amd.AMD(periodic_set, 100)
    """
    weights, dists, _ = _PDD(pset, k, lexsort=False, collapse=False)
    return np.average(dists, weights=weights, axis=0)


@numba.njit(cache=True, fastmath=True)
def PDD_to_AMD(pdd: FloatArray) -> FloatArray:
    """Calculate an AMD from a PDD, faster than computing both from
    scratch.

    Parameters
    ----------
    pdd : :class:`numpy.ndarray`
        The PDD of a periodic set as given by :class:`PDD() <.PDD>`.
    Returns
    -------
    :class:`numpy.ndarray`
        The AMD of the periodic set, so that
        ``amd.PDD_to_AMD(amd.PDD(pset)) == amd.AMD(pset)``
    """

    amd_ = np.empty((pdd.shape[-1] - 1,), dtype=np.float64)
    for col in range(amd_.shape[0]):
        v = 0
        for row in range(pdd.shape[0]):
            v += pdd[row, 0] * pdd[row, col + 1]
        amd_[col] = v
    return amd_


def AMD_finite(motif: FloatArray) -> FloatArray:
    """Return the AMD of a finite m-point set up to k = m - 1.

    Parameters
    ----------
    motif : :class:`numpy.ndarray`
        Collection of points.

    Returns
    -------
    :class:`numpy.ndarray`
        The AMD of ``motif``, a vector shape ``(motif.shape[0] - 1, )``.

    Examples
    --------
    The (L-infinity) AMD distance between finite trapezium and kite
    point sets, which have the same list of inter-point distances::

        trapezium = np.array([[0,0],[1,1],[3,1],[4,0]])
        kite      = np.array([[0,0],[1,1],[1,-1],[4,0]])

        trap_amd = amd.AMD_finite(trapezium)
        kite_amd = amd.AMD_finite(kite)

        l_inf_dist = np.amax(np.abs(trap_amd - kite_amd))
    """

    dm = np.sort(squareform(pdist(motif)), axis=-1)[:, 1:]
    return np.average(dm, axis=0)


def PDD_finite(
    motif: FloatArray,
    lexsort: bool = True,
    collapse: bool = True,
    collapse_tol: float = 1e-4,
    return_row_data: bool = False,
) -> Union[FloatArray, Tuple[FloatArray, list]]:
    """Return the PDD of a finite m-point set up to k = m - 1.

    Parameters
    ----------
    motif : :class:`numpy.ndarray`
        Collection of points.
    lexsort : bool, default True
        Lexicographically order rows.
    collapse: bool, default True
        Collapse duplicate rows (within ``collapse_tol`` in the
        Chebyshev metric).
    collapse_tol: float, default 1e-4
        If two rows are closer than ``collapse_tol`` in the Chebyshev
        metric, they are merged and weights are given to rows in
        proportion to their frequency.
    return_row_data: bool, default False
        If True, return a tuple ``(pdd, groups)`` where ``groups[i]``
        contains indices of points in ``motif`` corresponding to
        ``pdd[i]``.

    Returns
    -------
    pdd : :class:`numpy.ndarray`
        The PDD of ``motif``, a :class:`numpy.ndarray` with ``k+1``
        columns. If ``return_row_data`` is True, returns a tuple
        (:class:`numpy.ndarray`, list).

    Examples
    --------
    The PDD distance between finite trapezium and kite point sets, which
    have the same list of inter-point distances::

        trapezium = np.array([[0,0],[1,1],[3,1],[4,0]])
        kite      = np.array([[0,0],[1,1],[1,-1],[4,0]])

        trap_pdd = amd.PDD_finite(trapezium)
        kite_pdd = amd.PDD_finite(kite)

        dist = amd.EMD(trap_pdd, kite_pdd)
    """

    m = motif.shape[0]
    dists = np.sort(squareform(pdist(motif)), axis=-1)[:, 1:]
    weights = np.full((m,), 1 / m)
    groups = [[i] for i in range(len(dists))]

    # TODO: use _merge_pdd_rows
    if collapse:
        overlapping = pdist(dists, metric="chebyshev") <= collapse_tol
        if overlapping.any():
            groups = _collapse_into_groups(overlapping)
            weights = np.array([np.sum(weights[group]) for group in groups])
            dists = np.array(
                [np.average(dists[group], axis=0) for group in groups], dtype=np.float64
            )

    pdd = np.empty(shape=(len(weights), m), dtype=np.float64)

    if lexsort:
        lex_ordering = np.lexsort(np.rot90(dists))
        pdd[:, 0] = weights[lex_ordering]
        pdd[:, 1:] = dists[lex_ordering]
        if return_row_data:
            groups = [groups[i] for i in lex_ordering]
    else:
        pdd[:, 0] = weights
        pdd[:, 1:] = dists

    if return_row_data:
        return pdd, groups
    return pdd


def PDD_reconstructable(pset: PeriodicSet, lexsort: bool = True) -> FloatArray:
    """Return the PDD of a periodic set with ``k`` (number of columns)
    large enough such that the periodic set can be reconstructed from
    the PDD with :func:`amd.reconstruct.reconstruct`. Does NOT return
    weights or collapse rows.

    Parameters
    ----------
    pset : :class:`amd.PeriodicSet <.periodicset.PeriodicSet>`
        A periodic set (crystal).
    lexsort : bool, default True
        Lexicographically order rows.

    Returns
    -------
    pdd : :class:`numpy.ndarray`
        The PDD of ``pset`` with enough columns to reconstruct ``pset``
        using :func:`amd.reconstruct.reconstruct`.
    """

    if not isinstance(pset, PeriodicSet):
        raise ValueError(
            f"Expected {PeriodicSet.__name__}, got {pset.__class__.__name__}"
        )

    if pset.ndim not in (2, 3):
        raise ValueError(
            "Reconstructing from PDD is only possible for 2 and 3 dimensions."
        )
    min_val = diameter(pset.cell) * 2
    pdd, _, _ = nearest_neighbors_minval(pset.motif, pset.cell, min_val)
    if lexsort:
        lex_ordering = np.lexsort(pdd.T[::-1])
        pdd = pdd[lex_ordering]
    return pdd


def AMD_estimate(pset: PeriodicSet, k: int) -> FloatArray:
    r"""Calculate an estimate of :class:`AMD <.AMD>` based on the
    :class:`PPC <.periodicset.PeriodicSet.PPC>` of ``pset``.

    Parameters
    ----------
    pset : :class:`amd.PeriodicSet <.periodicset.PeriodicSet>`
        A periodic set (crystal).

    Returns
    -------
    amd_est : :class:`numpy.ndarray`
        An array shape (k, ), where ``amd_est[i]``
        :math:`= \text{PPC} \sqrt[n]{k}` in n dimensions, whose ratio
        with AMD has been shown to converge to 1.
    """

    if not isinstance(pset, PeriodicSet):
        raise ValueError(
            f"Expected {PeriodicSet.__name__}, got {pset.__class__.__name__}"
        )
    arange = np.arange(1, k + 1, dtype=np.float64)
    return pset.PPC() * np.power(arange, 1.0 / pset.ndim)


def PDA(
    pset: PeriodicSet,
    k: int,
    lexsort: bool = True,
    collapse: bool = True,
    collapse_tol: float = 1e-4,
    return_row_data: bool = False,
) -> Union[FloatArray, Tuple[FloatArray, list]]:
    """Return the pointwise deviation from asymptotic distribution,
    essentially a normalisation of the pointwise distance distribution
    of ``pset``. The PDA records how much the distances in the PDD
    deviate from what is expected based on the asymptotic estimate.

    The PDD of ``pset`` is a geometry based descriptor independent of
    choice of motif and unit cell. Its asymptotic behaviour is well
    understood and depends on the point density of the periodic set.
    The PDA is the difference between the PDD and its asymptotic curve.

    Parameters
    ----------
    pset : :class:`amd.PeriodicSet <.periodicset.PeriodicSet>`
        A periodic set (crystal).
    k : int
        Number of neighbors considered for each point in a unit cell.
        The output has k + 1 columns with the first column containing
        weights.
    lexsort : bool, default True
        Lexicographically order rows.
    collapse: bool, default True
        Collapse duplicate rows (within ``collapse_tol`` in the
        Chebyshev metric).
    collapse_tol: float, default 1e-4
        If two rows are closer than ``collapse_tol`` in the Chebyshev
        metric, they are merged and weights are given to rows in
        proportion to their frequency.
    return_row_data: bool, default False
        Return a tuple ``(pda, groups)`` where ``groups`` contains
        information about which rows in ``pda`` correspond to which
        points. If ``pset.asym_unit`` is None, then ``groups[i]``
        contains indices of points in ``pset.motif`` corresponding to
        ``pda[i]``. Otherwise, PDA rows correspond to points in the
        asymmetric unit, and ``groups[i]`` contains indices pointing to
        ``pset.asym_unit``.

    Returns
    -------
    pda : :class:`numpy.ndarray`
        The PDA of ``pset``, a :class:`numpy.ndarray` with ``k+1``
        columns. If ``return_row_data`` is True, returns a tuple
        (:class:`numpy.ndarray`, list).
    """
    pdd, grps = PDD(
        pset,
        k,
        collapse=collapse,
        collapse_tol=collapse_tol,
        lexsort=lexsort,
        return_row_data=True,
    )
    pdd[:, 1:] -= AMD_estimate(pset, k)
    if return_row_data:
        return pdd, grps
    return pdd


def ADA(pset: PeriodicSet, k: int) -> FloatArray:
    """Return the average deviation from asymptotic, essentially a
    normalisation of the average minimum distance of ``pset``. The ADA
    records how much the distances in the AMD deviate from what is
    expected based on the asymptotic estimate.

    The AMD of ``pset`` is a geometry based descriptor independent of
    choice of motif and unit cell. Its asymptotic behaviour is well
    understood and depends on the point density of the periodic set.
    The ADA is the difference between the AMD and its asymptotic curve.

    Parameters
    ----------
    pset : :class:`amd.PeriodicSet <.periodicset.PeriodicSet>`
        A periodic set (crystal).
    k : int
        Number of neighbors considered for each point in a unit cell.

    Returns
    -------
    :class:`numpy.ndarray`
        The ADA of ``pset``, a :class:`numpy.ndarray` shape ``(k, )``.
    """
    return AMD(pset, k) - AMD_estimate(pset, k)


@numba.njit(cache=True, fastmath=True)
def _array_product_exceeds(values, limit):
    """Returns False if np.prod(values) > limit."""
    tot = 1
    for i in range(len(values)):
        tot *= values[i]
        if tot > limit:
            return True
    return False


@numba.njit(cache=True, fastmath=True)
def _merge_pdd_rows(weights, dists, collapse_tol):
    """Collpases weights & rows of a PDD, and return an array of group
    labels (new indices of old rows)."""

    n, k = dists.shape
    group_labels = np.empty((n,), dtype=np.int64)
    done = set()
    group = 0

    for i in range(n):
        if i in done:
            continue

        group_labels[i] = group

        for j in range(i + 1, n):
            if j in done:
                continue

            grouped = True
            for i_ in range(k):
                v = np.abs(dists[i, i_] - dists[j, i_])
                if v > collapse_tol:
                    grouped = False
                    break

            if grouped:
                group_labels[j] = group
                done.add(j)

        group += 1

    if group == n:
        return weights, dists, group_labels

    weights_ = np.zeros((group,), dtype=np.float64)
    dists_ = np.zeros((group, k), dtype=np.float64)
    group_counts = np.zeros((group,), dtype=np.int64)

    for i in range(n):
        row = group_labels[i]
        weights_[row] += weights[i]
        dists_[row] += dists[i]
        group_counts[row] += 1

    for i in range(group):
        dists_[i] /= group_counts[i]

    return weights_, dists_, group_labels


def _collapse_into_groups(overlapping: npt.NDArray[np.bool_]) -> list:
    """Return a list of groups of indices where all indices in the same
    group overlap. ``overlapping`` indicates for each pair of items in a
    set whether or not the items overlap, in the shape of a condensed
    distance matrix.
    """

    overlapping = squareform(overlapping)
    group_nums = {}
    group = 0
    for i, row in enumerate(overlapping):
        if i not in group_nums:
            group_nums[i] = group
            group += 1
            for j in np.argwhere(row).T[0]:
                if j not in group_nums:
                    group_nums[j] = group_nums[i]

    groups = collections.defaultdict(list)
    for row_ind, group_num in sorted(group_nums.items()):
        groups[group_num].append(row_ind)

    return list(groups.values())
