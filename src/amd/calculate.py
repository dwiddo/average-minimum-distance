"""Functions for calculating the average minimum distance (AMD) and
point-wise distance distribution (PDD) isometric invariants of
periodic crystals and finite sets.
"""

import collections
from typing import Tuple, Union

import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.special import factorial

from .periodicset import PeriodicSet
from ._nearest_neighbours import nearest_neighbours, nearest_neighbours_minval
from .utils import diameter

__all__ = [
    'AMD', 'PDD', 'PDD_to_AMD', 'AMD_finite', 'PDD_finite',
    'PDD_reconstructable', 'PPC', 'AMD_estimate'
]

PeriodicSetType = Union[PeriodicSet, Tuple[np.ndarray, np.ndarray]]


def AMD(periodic_set: PeriodicSetType, k: int) -> np.ndarray:
    """Return the average minimum distance (AMD) of a periodic set
    (crystal).
    
    The AMD of a periodic set is a geometry based descriptor independent
    of choice of motif and unit cell. It is a vector, the (weighted)
    average of the :func:`PDD <.calculate.PDD>` matrix, which has one
    row for each (unique) point in the unit cell containing distances to
    k nearest neighbours.

    Parameters
    ----------
    periodic_set :
        A periodic set represented by a
        :class:`amd.PeriodicSet <.periodicset.PeriodicSet>` or by a
        tuple of :class:`numpy.ndarray` s (motif, cell) with coordinates
        in Cartesian form and a square unit cell.
    k : int
        The number of neighbouring points (atoms) considered for each
        point in the unit cell.

    Returns
    -------
    :class:`numpy.ndarray`
        A :class:`numpy.ndarray` shape ``(k, )``, the AMD of
        ``periodic_set`` up to k.

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

    Manually pass a periodic set as a tuple (motif, cell)::

        # simple cubic lattice
        motif = np.array([[0,0,0]])
        cell = np.array([[1,0,0], [0,1,0], [0,0,1]])
        cubic_amd = amd.AMD((motif, cell), 100)
    """

    motif, cell, asymmetric_unit, weights = _get_structure(periodic_set)
    dists, _, _ = nearest_neighbours(motif, cell, asymmetric_unit, k + 1)
    return np.average(dists[:, 1:], axis=0, weights=weights)


def PDD(
        periodic_set: PeriodicSetType,
        k: int,
        lexsort: bool = True,
        collapse: bool = True,
        collapse_tol: float = 1e-4,
        return_row_groups: bool = False
) -> Union[np.ndarray, Tuple[np.ndarray, list]]:
    """Return the point-wise distance distribution (PDD) of a periodic
    set (crystal).
    
    The PDD of a periodic set is a geometry based descriptor independent
    of choice of motif and unit cell. It is a matrix where each row
    corresponds to a point in the motif, containing a weight followed by
    distances to the k nearest neighbours of the point.

    Parameters
    ----------
    periodic_set :
        A periodic set represented by a
        :class:`amd.PeriodicSet <.periodicset.PeriodicSet>` or by a
        tuple of :class:`numpy.ndarray` s (motif, cell) with Cartesian
        coordinates and a square matrix representing the unit cell.
    k : int
        The number of neighbours considered for each atom (point) in the
        unit cell. The returned matrix has k + 1 columns, the first
        column for weights of rows.
    lexsort : bool, default True
        Lexicographically order the rows.
    collapse: bool, default True
        Collapse repeated rows (within tolerance ``collapse_tol``).
    collapse_tol: float, default 1e-4
        If two rows have all elements closer than ``collapse_tol``, they
        are merged and weights are given to rows in proportion to the
        number of times they appeared.
    return_row_groups: bool, default False
        Return data about which PDD rows correspond to which points. If
        True, a tuple is returned ``(pdd, groups)`` where ``groups[i]``
        contains the indices of the point(s) corresponding to
        ``pdd[i]``. Note that these indices are for the asymmetric unit
        of the set, whose indices in ``periodic_set.motif`` are
        accessible through ``periodic_set.asymmetric_unit``.

    Returns
    -------
    pdd : :class:`numpy.ndarray`
        A :class:`numpy.ndarray` with k+1 columns, the PDD of
        ``periodic_set`` up to k. The first column contains the weights
        of rows. If ``return_row_groups`` is True, returns a tuple with
        types (:class:`numpy.ndarray`, list).

    Examples
    --------
    Make list of PDDs with ``k=100`` for crystals in data.cif::

        pdds = []
        for periodic_set in amd.CifReader('data.cif'):
            # do not lexicographically order rows
            pdds.append(amd.PDD(periodic_set, 100, lexsort=False))

    Make list of PDDs with ``k=10`` for crystals in these CSD refcode
    families::

        pdds = []
        for periodic_set in amd.CSDReader(['HXACAN', 'ACSALA'], families=True):
            # do not collapse rows
            pdds.append(amd.PDD(periodic_set, 10, collapse=False))

    Manually pass a periodic set as a tuple (motif, cell)::

        # simple cubic lattice
        motif = np.array([[0,0,0]])
        cell = np.array([[1,0,0], [0,1,0], [0,0,1]])
        cubic_amd = amd.PDD((motif, cell), 100)
    """

    motif, cell, asymmetric_unit, weights = _get_structure(periodic_set)
    dists, _, _ = nearest_neighbours(motif, cell, asymmetric_unit, k + 1)
    dists = dists[:, 1:]
    groups = [[i] for i in range(len(dists))]

    if collapse:
        overlapping = pdist(dists, metric='chebyshev') <= collapse_tol
        if overlapping.any():
            groups = _collapse_into_groups(overlapping)
            weights = np.array([np.sum(weights[group]) for group in groups])
            dists = np.array([
                np.average(dists[group], axis=0) for group in groups
            ], dtype=np.float64)

    pdd = np.empty(shape=(len(weights), k + 1), dtype=np.float64)

    if lexsort:
        lex_ordering = np.lexsort(np.rot90(dists))
        pdd[:, 0] = weights[lex_ordering]
        pdd[:, 1:] = dists[lex_ordering]
        if return_row_groups:
            groups = [groups[i] for i in lex_ordering]
    else:
        pdd[:, 0] = weights
        pdd[:, 1:] = dists

    if return_row_groups:
        return pdd, groups
    return pdd


def PDD_to_AMD(pdd: np.ndarray) -> np.ndarray:
    """Calculate an AMD from a PDD. Faster than computing both from
    scratch.

    Parameters
    ----------
    pdd : :class:`numpy.ndarray`
        The PDD of a periodic set.

    Returns
    -------
    :class:`numpy.ndarray`
        The AMD of the periodic set.
    """
    return np.average(pdd[:, 1:], weights=pdd[:, 0], axis=0)


def AMD_finite(motif: np.ndarray) -> np.ndarray:
    """Return the AMD of a finite m-point set up to k = m - 1.

    Parameters
    ----------
    motif : :class:`numpy.ndarray`
        Coordinates of a set of points.

    Returns
    -------
    :class:`numpy.ndarray`
        A vector shape (motif.shape[0] - 1, ), the AMD of ``motif``.

    Examples
    --------
    The (L-infinity) AMD distance between finite trapezium and kite
    point sets::

        trapezium = np.array([[0,0],[1,1],[3,1],[4,0]])
        kite      = np.array([[0,0],[1,1],[1,-1],[4,0]])

        trap_amd = amd.AMD_finite(trapezium)
        kite_amd = amd.AMD_finite(kite)

        l_inf_dist = np.amax(np.abs(trap_amd - kite_amd))
    """

    dm = np.sort(squareform(pdist(motif)), axis=-1)[:, 1:]
    return np.average(dm, axis=0)


def PDD_finite(
        motif: np.ndarray,
        lexsort: bool = True,
        collapse: bool = True,
        collapse_tol: float = 1e-4,
        return_row_groups: bool = False
) -> Union[np.ndarray, Tuple[np.ndarray, list]]:
    """Return the PDD of a finite m-point set up to k = m - 1.

    Parameters
    ----------
    motif : :class:`numpy.ndarray`
        Coordinates of a set of points.
    lexsort : bool, default True
        Whether or not to lexicographically order the rows.
    collapse: bool, default True
        Whether or not to collapse repeated rows (within tolerance
        ``collapse_tol``).
    collapse_tol: float, default 1e-4
        If two rows have all elements closer than ``collapse_tol``, they
        are merged and weights are given to rows in proportion to the
        number of times they appeared.
    return_row_groups: bool, default False
        Whether to return data about which PDD rows correspond to which
        points. If True, a tuple is returned ``(pdd, groups)`` where
        ``groups[i]`` contains the indices of the point(s) corresponding
        to ``pdd[i]``.

    Returns
    -------
    pdd : :class:`numpy.ndarray`
        A :class:`numpy.ndarray` with m columns (where m is the number
        of points), the PDD of ``motif``. The first column contains the
        weights of rows.

    Examples
    --------
    Find PDD distance between finite trapezium and kite point sets::

        trapezium = np.array([[0,0],[1,1],[3,1],[4,0]])
        kite      = np.array([[0,0],[1,1],[1,-1],[4,0]])

        trap_pdd = amd.PDD_finite(trapezium)
        kite_pdd = amd.PDD_finite(kite)

        dist = amd.EMD(trap_pdd, kite_pdd)
    """

    m = motif.shape[0]
    dists = np.sort(squareform(pdist(motif)), axis=-1)[:, 1:]
    weights = np.full((m, ), 1 / m)
    groups = [[i] for i in range(len(dists))]

    if collapse:
        overlapping = pdist(dists, metric='chebyshev') <= collapse_tol
        if overlapping.any():
            groups = _collapse_into_groups(overlapping)
            weights = np.array([np.sum(weights[group]) for group in groups])
            dists = np.array([
                np.average(dists[group], axis=0) for group in groups
            ], dtype=np.float64)

    pdd = np.empty(shape=(len(weights), m), dtype=np.float64)

    if lexsort:
        lex_ordering = np.lexsort(np.rot90(dists))
        pdd[:, 0] = weights[lex_ordering]
        pdd[:, 1:] = dists[lex_ordering]
        if return_row_groups:
            groups = [groups[i] for i in lex_ordering]
    else:
        pdd[:, 0] = weights
        pdd[:, 1:] = dists

    if return_row_groups:
        return pdd, groups
    return pdd


def PDD_reconstructable(
        periodic_set: PeriodicSetType, lexsort: bool = True
) -> np.ndarray:
    """Return the PDD of a periodic set with `k` (number of columns)
    large enough such that the periodic set can be reconstructed from
    the PDD.

    Parameters
    ----------
    periodic_set :
        A periodic set represented by a
        :class:`amd.PeriodicSet <.periodicset.PeriodicSet>` or by a
        tuple of :class:`numpy.ndarray` s (motif, cell) with coordinates
        in Cartesian form and a square unit cell.
    lexsort : bool, default True
        Whether or not to lexicographically order the rows.

    Returns
    -------
    pdd : :class:`numpy.ndarray`
        The PDD of ``periodic_set`` with enough columns to be
        reconstructable.
    """

    motif, cell, _, _ = _get_structure(periodic_set)
    dims = cell.shape[0]

    if dims not in (2, 3):
        raise ValueError(
            'Reconstructing from PDD is only possible for 2 and 3 dimensions.'
        )

    min_val = diameter(cell) * 2
    pdd, _, _ = nearest_neighbours_minval(motif, cell, min_val)

    if lexsort:
        lex_ordering = np.lexsort(np.rot90(pdd))
        pdd = pdd[lex_ordering]

    return pdd


def PPC(periodic_set: PeriodicSetType) -> float:
    r"""Return the point packing coefficient (PPC) of ``periodic_set``.

    The PPC is a constant of any periodic set determining the
    asymptotic behaviour of its AMD and PDD. As
    :math:`k \rightarrow \infty`, the ratio
    :math:`\text{AMD}_k / \sqrt[n]{k}` converges to the PPC, as does any
    row of its PDD.

    For a unit cell :math:`U` and :math:`m` motif points in :math:`n`
    dimensions,

    .. math::

        \text{PPC} = \sqrt[n]{\frac{\text{Vol}[U]}{m V_n}}

    where :math:`V_n` is the volume of a unit sphere in :math:`n`
    dimensions.

    Parameters
    ----------
    periodic_set :
        A periodic set represented by a
        :class:`amd.PeriodicSet <.periodicset.PeriodicSet>` or by a
        tuple of :class:`numpy.ndarray` s (motif, cell) with coordinates
        in Cartesian form.

    Returns
    -------
    ppc : float
        The PPC of ``periodic_set``.
    """

    motif, cell, _, _ = _get_structure(periodic_set)
    m, n = motif.shape
    t = int(n // 2)

    if n % 2 == 0:
        unit_sphere_vol = (np.pi ** t) / factorial(t)
    else:
        unit_sphere_vol = (2 * factorial(t) * (4 * np.pi) ** t) / factorial(n)

    return (np.linalg.det(cell) / (m * unit_sphere_vol)) ** (1.0 / n)


def AMD_estimate(periodic_set: PeriodicSetType, k: int) -> np.ndarray:
    r"""Calculate an estimate of AMD based on the PPC.

    Parameters
    ----------
    periodic_set :
        A periodic set represented by a
        :class:`amd.PeriodicSet <.periodicset.PeriodicSet>` or by a
        tuple of :class:`numpy.ndarray` s (motif, cell) with coordinates
        in Cartesian form.

    Returns
    -------
    amd_est : :class:`numpy.ndarray`
        An array shape (k, ), where ``amd_est[i]``
        :math:`= \text{PPC} \sqrt[n]{k}` in n dimensions.
    """

    motif, cell, _, _ = _get_structure(periodic_set)
    n = cell.shape[0]
    k_root = np.power(np.arange(1, k + 1, dtype=np.float64), 1.0 / n)
    return PPC((motif, cell)) * k_root


def _get_structure(periodic_set: PeriodicSetType) -> Tuple[np.ndarray, ...]:
    """Extract the motif and cell, and if present the asymmetric unit
    and scaled multiplicities, from a periodic set. ``periodic_set`` can
    be a :class:`amd.PeriodicSet <.periodicset.PeriodicSet>`, or a tuple
    of :class:`numpy.ndarray` s (motif, cell).
    """

    asymmetric_unit = None
    weights = None

    if isinstance(periodic_set, PeriodicSet):
        motif, cell = periodic_set.motif, periodic_set.cell
        asym_unit_inds = periodic_set.asymmetric_unit
        wyc_muls = periodic_set.wyckoff_multiplicities
        if asym_unit_inds is not None and wyc_muls is not None:
            asymmetric_unit = motif[asym_unit_inds]
            weights = wyc_muls / np.sum(wyc_muls)
    elif isinstance(periodic_set, (tuple, list)):
        motif, cell = periodic_set
    else:
        raise ValueError(
            'Expected amd.PeriodicSet or tuple, got '
            f'{periodic_set.__class__.__name__}'
        )

    if asymmetric_unit is None or weights is None:
        asymmetric_unit = motif
        weights = np.full((len(motif), ), 1 / len(motif), dtype=np.float64)

    return motif, cell, asymmetric_unit, weights


def _collapse_into_groups(overlapping: np.ndarray) -> list:
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
    groups = list(groups.values())

    return groups
