"""Functions for calculating the average minimum distance (AMD) and 
point-wise distance distribution (PDD) isometric invariants of 
periodic crystals and finite sets.
"""

from typing import Union, Tuple
import collections

import numpy as np
from scipy.spatial.distance import pdist, squareform

from . import utils
from ._nearest_neighbours import nearest_neighbours, nearest_neighbours_minval
from .periodicset import PeriodicSet

PeriodicSet_or_Tuple = Union[PeriodicSet, Tuple[np.ndarray, np.ndarray]]


def AMD(
        periodic_set: PeriodicSet_or_Tuple, 
        k: int
) -> np.ndarray:
    """The AMD of a periodic set (crystal) up to k.

    Parameters
    ----------
    periodic_set : :class:`.periodicset.PeriodicSet`  tuple of :class:`numpy.ndarray` s
        A periodic set represented by a :class:`.periodicset.PeriodicSet` or
        by a tuple (motif, cell) with coordinates in Cartesian form and a square unit cell.
    k : int
        Length of the AMD returned; the number of neighbours considered for each atom 
        in the unit cell to make the AMD.

    Returns
    -------
    numpy.ndarray
        A :class:`numpy.ndarray` shape (k, ), the AMD of ``periodic_set`` up to k.

    Examples
    --------
    Make list of AMDs with k = 100 for crystals in mycif.cif::

        amds = []
        for periodic_set in amd.CifReader('mycif.cif'):
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

    motif, cell, asymmetric_unit, multiplicities = _extract_motif_and_cell(periodic_set)
    pdd, _, _ = nearest_neighbours(motif, cell, k, asymmetric_unit=asymmetric_unit)
    return np.average(pdd, axis=0, weights=multiplicities)


def PDD(
        periodic_set: PeriodicSet_or_Tuple,
        k: int,
        lexsort: bool = True,
        collapse: bool = True,
        collapse_tol: float = 1e-4,
        return_row_groups: bool = False,
) -> np.ndarray:
    """The PDD of a periodic set (crystal) up to k.

    Parameters
    ----------
    periodic_set : :class:`.periodicset.PeriodicSet`  tuple of :class:`numpy.ndarray` s
        A periodic set represented by a :class:`.periodicset.PeriodicSet` or
        by a tuple (motif, cell) with coordinates in Cartesian form and a square unit cell.
    k : int
        The returned PDD has k+1 columns, an additional first column for row weights.
        k is the number of neighbours considered for each atom in the unit cell 
        to make the PDD.
    lexsort : bool, optional
        Lexicographically order the rows. Default True.
    collapse: bool, optional
        Collapse repeated rows (within the tolerance ``collapse_tol``). Default True.
    collapse_tol: float, optional
        If two rows have all elements closer than ``collapse_tol``, they are merged and
        weights are given to rows in proportion to the number of times they appeared.
        Default is 0.0001.
    return_row_groups: bool, optional
        Return data about which PDD rows correspond to which points.
        If True, a tuple is returned ``(pdd, groups)`` where ``groups[i]`` 
        contains the indices of the point(s) corresponding to ``pdd[i]``. 
        Note that these indices are for the asymmetric unit of the set, whose
        indices in ``periodic_set.motif`` are accessible through 
        ``periodic_set.asymmetric_unit``.

    Returns
    -------
    numpy.ndarray
        A :class:`numpy.ndarray` with k+1 columns, the PDD of ``periodic_set`` up to k.
        The first column contains the weights of rows. If ``return_row_groups`` is True, 
        returns a tuple (:class:`numpy.ndarray`, list).

    Examples
    --------
    Make list of PDDs with ``k=100`` for crystals in mycif.cif::

        pdds = []
        for periodic_set in amd.CifReader('mycif.cif'):
            # do not lexicographically order rows
            pdds.append(amd.PDD(periodic_set, 100, lexsort=False))

    Make list of PDDs with ``k=10`` for crystals in these CSD refcode families::

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

    motif, cell, asymmetric_unit, multiplicities = _extract_motif_and_cell(periodic_set)
    dists, _, _ = nearest_neighbours(motif, cell, k, asymmetric_unit=asymmetric_unit)
    groups = [[i] for i in range(len(dists))]

    if multiplicities is None:
        weights = np.full((motif.shape[0], ), 1 / motif.shape[0])
    else:
        weights = multiplicities / np.sum(multiplicities)

    if collapse:
        overlapping = pdist(dists, metric='chebyshev')
        overlapping = overlapping < collapse_tol
        if overlapping.any():
            groups = _collapse_into_groups(overlapping)
            weights = np.array([sum(weights[group]) for group in groups])
            dists = np.array([np.average(dists[group], axis=0) for group in groups])

    pdd = np.hstack((weights[:, None], dists))

    if lexsort:
        lex_ordering = np.lexsort(np.rot90(dists))
        groups = [groups[i] for i in lex_ordering]
        pdd = pdd[lex_ordering]

    if return_row_groups:
        return pdd, groups
    else:
        return pdd


def PDD_to_AMD(pdd: np.ndarray) -> np.ndarray:
    """Calculates an AMD from a PDD. Faster than computing both from scratch.

    Parameters
    ----------
    pdd : numpy.ndarray
        The PDD of a periodic set.

    Returns
    -------
    numpy.ndarray
        The AMD of the periodic set.
    """

    return np.average(pdd[:, 1:], weights=pdd[:, 0], axis=0)


def AMD_finite(motif: np.ndarray) -> np.ndarray:
    """The AMD of a finite m-point set up to k = m-1.

    Parameters
    ----------
    motif : numpy.ndarray
        Coordinates of a set of points.

    Returns
    -------
    numpy.ndarray
        A vector length m-1 (where m is the number of points), the AMD of ``motif``.

    Examples
    --------
    The AMD distance (L-infinity) between finite trapezium and kite point sets::

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
        return_row_groups: bool = False,
) -> np.ndarray:
    """The PDD of a finite m-point set up to k = m-1.

    Parameters
    ----------
    motif : numpy.ndarray
        Coordinates of a set of points.
    lexsort : bool, optional
        Whether or not to lexicographically order the rows. Default True.
    collapse: bool, optional
        Whether or not to collapse repeated rows (within the tolerance ``collapse_tol``). 
        Default True.
    collapse_tol: float
        If two rows have all elements closer than ``collapse_tol``, they are merged and
        weights are given to rows in proportion to the number of times they appeared.
        Default is 0.0001.
    return_row_groups: bool, optional
        Whether to return data about which PDD rows correspond to which points.
        If True, a tuple is returned ``(pdd, groups)`` where ``groups[i]`` 
        contains the indices of the point(s) corresponding to ``pdd[i]``.

    Returns
    -------
    numpy.ndarray
        A :class:`numpy.ndarray` with m columns (where m is the number of points), 
        the PDD of ``motif``. The first column contains the weights of rows.

    Examples
    --------
    Find PDD distance between finite trapezium and kite point sets::

        trapezium = np.array([[0,0],[1,1],[3,1],[4,0]])
        kite      = np.array([[0,0],[1,1],[1,-1],[4,0]])

        trap_pdd = amd.PDD_finite(trapezium)
        kite_pdd = amd.PDD_finite(kite)

        dist = amd.EMD(trap_pdd, kite_pdd)
    """

    dm = squareform(pdist(motif))
    m = motif.shape[0]
    dists = np.sort(dm, axis=-1)[:, 1:]
    weights = np.full((m, ), 1 / m)
    groups = [[i] for i in range(len(dists))]

    if collapse:
        overlapping = pdist(dists, metric='chebyshev')
        overlapping = overlapping < collapse_tol
        if overlapping.any():
            groups = _collapse_into_groups(overlapping)
            weights = np.array([sum(weights[group]) for group in groups])
            ordering = [group[0] for group in groups]
            dists = dists[ordering]

    pdd = np.hstack((weights[:, None], dists))

    if lexsort:
        lex_ordering = np.lexsort(np.rot90(dists))
        groups = [groups[i] for i in lex_ordering]
        pdd = pdd[lex_ordering]

    if return_row_groups:
        return pdd, groups
    else:
        return pdd


def PDD_reconstructable(
        periodic_set: PeriodicSet_or_Tuple,
        lexsort: bool = True
) -> np.ndarray:
    """The PDD of a periodic set with `k` (no of columns) large enough such that
    the periodic set can be reconstructed from the PDD.

    Parameters
    ----------
    periodic_set : :class:`.periodicset.PeriodicSet`  tuple of :class:`numpy.ndarray` s
        A periodic set represented by a :class:`.periodicset.PeriodicSet` or
        by a tuple (motif, cell) with coordinates in Cartesian form and a square unit cell.
    lexsort : bool, optional
        Whether or not to lexicographically order the rows. Default True.

    Returns
    -------
    numpy.ndarray
        An ndarray, the PDD of ``periodic_set`` with enough columns to be reconstructable.
    """

    motif, cell, _, _ = _extract_motif_and_cell(periodic_set)
    dims = cell.shape[0]

    if dims not in (2, 3):
        raise ValueError('Reconstructing from PDD only implemented for 2 and 3 dimensions')

    min_val = utils.diameter(cell) * 2
    pdd = nearest_neighbours_minval(motif, cell, min_val)

    if lexsort:
        pdd = pdd[np.lexsort(np.rot90(pdd))]

    return pdd


def PPC(periodic_set: PeriodicSet_or_Tuple) -> float:
    r"""The point packing coefficient (PPC) of ``periodic_set``.

    The PPC is a constant of any periodic set determining the
    asymptotic behaviour of its AMD and PDD as :math:`k \rightarrow \infty`.

    As :math:`k \rightarrow \infty`, the ratio :math:`\text{AMD}_k / \sqrt[n]{k}`
    approaches the PPC (as does any row of its PDD).

    For a unit cell :math:`U` and :math:`m` motif points in :math:`n` dimensions,

    .. math::

        \text{PPC} = \sqrt[n]{\frac{\text{Vol}[U]}{m V_n}}

    where :math:`V_n` is the volume of a unit sphere in :math:`n` dimensions.

    Parameters
    ----------
    periodic_set : :class:`.periodicset.PeriodicSet` or tuple of
        ndarrays (motif, cell) representing the periodic set in Cartesian form.

    Returns
    -------
    float
        The PPC of ``periodic_set``.
    """

    motif, cell, _, _ = _extract_motif_and_cell(periodic_set)
    m, n = motif.shape
    det = np.linalg.det(cell)
    t = (n - n % 2) / 2
    if n % 2 == 0:
        V = (np.pi ** t) / np.math.factorial(t)
    else:
        V = (2 * np.math.factorial(t) * (4 * np.pi) ** t) / np.math.factorial(n)

    return (det / (m * V)) ** (1./n)


def AMD_estimate(periodic_set: PeriodicSet_or_Tuple, k: int) -> np.ndarray:
    r"""Calculates an estimate of AMD based on the PPC, using the fact that

    .. math::

        \lim_{k\rightarrow\infty}\frac{\text{AMD}_k}{\sqrt[n]{k}} = \sqrt[n]{\frac{\text{Vol}[U]}{m V_n}}

    where :math:`U` is the unit cell, :math:`m` is the number of motif points and
    :math:`V_n` is the volume of a unit sphere in :math:`n`-dimensional space.
    """

    motif, cell, _, _ = _extract_motif_and_cell(periodic_set)
    n = motif.shape[1]
    c = PPC((motif, cell))
    return np.array([(x ** (1. / n)) * c for x in range(1, k + 1)])


def _extract_motif_and_cell(periodic_set: PeriodicSet_or_Tuple):
    """`periodic_set` is either a :class:`.periodicset.PeriodicSet`, or
    a tuple of ndarrays (motif, cell). If possible, extracts the asymmetric unit
    and wyckoff multiplicities and returns them, otherwise returns None.
    """

    asymmetric_unit, multiplicities = None, None

    if isinstance(periodic_set, PeriodicSet):
        motif, cell = periodic_set.motif, periodic_set.cell

        if 'asymmetric_unit' in periodic_set.tags and 'wyckoff_multiplicities' in periodic_set.tags:
            asymmetric_unit = periodic_set.asymmetric_unit
            multiplicities = periodic_set.wyckoff_multiplicities

    elif isinstance(periodic_set, np.ndarray):
        motif, cell = periodic_set, None
    else:
        motif, cell = periodic_set[0], periodic_set[1]

    return motif, cell, asymmetric_unit, multiplicities


def _collapse_into_groups(overlapping):
    """The vector `overlapping` indicates for each pair of items in a set whether 
    or not the items overlap, in the shape of a condensed distance matrix. Returns
    a list of groups of indices where all items in the same group overlap."""

    overlapping = squareform(overlapping)
    group_nums = {} # row_ind: group number
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
