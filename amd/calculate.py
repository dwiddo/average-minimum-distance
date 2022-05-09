"""Functions for calculating AMDs and PDDs (and SDDs) of periodic and finite sets.
"""

from typing import Union, Tuple
import collections

import numpy as np
import scipy.spatial
import scipy.special

from ._nearest_neighbours import nearest_neighbours, nearest_neighbours_minval, generate_concentric_cloud
from .periodicset import PeriodicSet
from .utils import diameter

PeriodicSet_or_Tuple = Union[PeriodicSet, Tuple[np.ndarray, np.ndarray]]


def AMD(periodic_set: PeriodicSet_or_Tuple, k: int) -> np.ndarray:
    """The AMD of a periodic set (crystal) up to `k`.

    Parameters
    ----------
    periodic_set : :class:`.periodicset.PeriodicSet` or tuple of ndarrays
        A periodic set represented by a :class:`.periodicset.PeriodicSet` or
        by a tuple (motif, cell) with coordinates in Cartesian form.
    k : int
        Length of AMD returned.

    Returns
    -------
    ndarray
        An ndarray of shape (k,), the AMD of ``periodic_set`` up to `k`.

    Examples
    --------
    Make list of AMDs with ``k=100`` for crystals in mycif.cif::

        amds = []
        for periodic_set in amd.CifReader('mycif.cif'):
            amds.append(amd.AMD(periodic_set, 100))
  
    Make list of AMDs with ``k=10`` for crystals in these CSD refcode families::

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
        collapse_tol: float = 1e-4
) -> np.ndarray:
    """The PDD of a periodic set (crystal) up to `k`.

    Parameters
    ----------
    periodic_set : :class:`.periodicset.PeriodicSet` or tuple of ndarrays
        A periodic set represented by a :class:`.periodicset.PeriodicSet` or
        by a tuple (motif, cell) with coordinates in Cartesian form.
    k : int
        Number of columns in the PDD (the returned matrix has an additional first
        column containing weights).
    lexsort : bool, optional
        Whether or not to lexicographically order the rows. Default True.
    collapse: bool, optional
        Whether or not to collapse identical rows (within a tolerance). Default True.
    collapse_tol: float
        If two rows have all entries closer than collapse_tol, they get collapsed.
        Default is 1e-4.

    Returns
    -------
    ndarray
        An ndarray with k+1 columns, the PDD of ``periodic_set`` up to `k`.

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

    if multiplicities is None:
        weights = np.full((motif.shape[0], ), 1 / motif.shape[0])
    else:
        weights = multiplicities / np.sum(multiplicities)

    if collapse:
        weights, dists = _collapse_rows(weights, dists, collapse_tol)

    pdd = np.hstack((weights[:, None], dists))

    if lexsort:
        pdd = pdd[np.lexsort(np.rot90(dists))]

    return pdd


def PDD_to_AMD(pdd: np.ndarray) -> np.ndarray:
    """Calculates AMD from a PDD. Faster than computing both from scratch.

    Parameters
    ----------
    pdd : np.ndarray
        The PDD of a periodic set.

    Returns
    -------
    ndarray
        The AMD of the periodic set.
    """

    return np.average(pdd[:, 1:], weights=pdd[:, 0], axis=0)


def AMD_finite(motif: np.ndarray) -> np.ndarray:
    """The AMD of a finite point set (up to k = `len(motif) - 1`).

    Parameters
    ----------
    motif : ndarray
        Cartesian coordinates of points in a set. Shape (n_points, dimensions)

    Returns
    -------
    ndarray
        An vector length len(motif) - 1, the AMD of ``motif``.

    Examples
    --------
    Find AMD distance between finite trapezium and kite point sets::

        trapezium = np.array([[0,0],[1,1],[3,1],[4,0]])
        kite      = np.array([[0,0],[1,1],[1,-1],[4,0]])

        trap_amd = amd.AMD_finite(trapezium)
        kite_amd = amd.AMD_finite(kite)

        dist = amd.AMD_pdist(trap_amd, kite_amd)
    """

    dm = np.sort(scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(motif)), axis=-1)[:, 1:]
    return np.average(dm, axis=0)


def PDD_finite(
        motif: np.ndarray,
        lexsort: bool = True,
        collapse: bool = True,
        collapse_tol: float = 1e-4
) -> np.ndarray:
    """The PDD of a finite point set (up to k = `len(motif) - 1`).

    Parameters
    ----------
    motif : ndarray
        Cartesian coordinates of points in a set. Shape (n_points, dimensions)
    lexsort : bool, optional
        Whether or not to lexicographically order the rows. Default True.
    collapse: bool, optional
        Whether or not to collapse identical rows (within a tolerance). Default True.
    collapse_tol: float
        If two rows have all entries closer than collapse_tol, they get collapsed.
        Default is 1e-4.

    Returns
    -------
    ndarray
        An ndarray with len(motif) columns, the PDD of ``motif``.

    Examples
    --------
    Find PDD distance between finite trapezium and kite point sets::

        trapezium = np.array([[0,0],[1,1],[3,1],[4,0]])
        kite      = np.array([[0,0],[1,1],[1,-1],[4,0]])

        trap_pdd = amd.PDD_finite(trapezium)
        kite_pdd = amd.PDD_finite(kite)

        dist = amd.emd(trap_pdd, kite_pdd)
    """

    dm = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(motif))
    m = motif.shape[0]
    dists = np.sort(dm, axis=-1)[:, 1:]
    weights = np.full((m, ), 1 / m)

    if collapse:
        weights, dists = _collapse_rows(weights, dists, collapse_tol)

    pdd = np.hstack((weights[:, None], dists))

    if lexsort:
        pdd = pdd[np.lexsort(np.rot90(dists))]

    return pdd


def PDD_reconstructable(
        periodic_set: PeriodicSet_or_Tuple,
        lexsort: bool = True
) -> np.ndarray:
    """The PDD of a periodic set with `k` (no of columns) large enough such that
    the periodic set can be reconstructed from the PDD.

    Parameters
    ----------
    periodic_set : :class:`.periodicset.PeriodicSet` or tuple of ndarrays
        A periodic set represented by a :class:`.periodicset.PeriodicSet` or
        by a tuple (motif, cell) with coordinates in Cartesian form.
    k : int
        Number of columns in the PDD, plus one for the first column of weights.
    order : int
        Order of the PDD, default 1. See papers for a description of higher-order PDDs.
    lexsort : bool, optional
        Whether or not to lexicographically order the rows. Default True.
    collapse: bool, optional
        Whether or not to collapse identical rows (within a tolerance). Default True.
    collapse_tol: float
        If two rows have all entries closer than collapse_tol, they get collapsed.
        Default is 1e-4.

    Returns
    -------
    ndarray
        An ndarray with k+1 columns, the PDD of ``periodic_set`` up to `k`.

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

    motif, cell, _, _ = _extract_motif_and_cell(periodic_set)
    dims = cell.shape[0]

    if dims not in (2, 3):
        raise ValueError('Reconstructing from PDD only implemented for 2 and 3 dimensions')

    min_val = diameter(cell) * 2
    pdd = nearest_neighbours_minval(motif, cell, min_val)

    if lexsort:
        pdd = pdd[np.lexsort(np.rot90(pdd))]

    return pdd


def PPC(periodic_set: PeriodicSet_or_Tuple) -> float:
    r"""The point packing coefficient (PPC) of ``periodic_set``.

    The PPC is a constant of any periodic set determining the
    asymptotic behaviour of its AMD or PDD as :math:`k \rightarrow \infty`.

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


def _collapse_rows(weights, dists, collapse_tol):
    """Given a vector `weights`, matrix `dists` and tolerance `collapse_tol`, collapse
    the identical rows of dists (if all entries in a row are within  `collapse_tol`)
    and collapse the same entires of `weights` (adding entries that merge).
    """

    diffs = np.abs(dists[:, None] - dists)
    overlapping = np.all(diffs <= collapse_tol, axis=-1)

    res = _group_overlapping_and_sum_weights(weights, overlapping)
    if res is not None:
        weights = res[0]
        dists = dists[res[1]]

    return weights, dists


def _group_overlapping_and_sum_weights(weights, overlapping):
    if np.triu(overlapping, 1).any():
        groups = {}
        group = 0
        for i, row in enumerate(overlapping):
            if i not in groups:
                groups[i] = group
                group += 1

            for j in np.argwhere(row).T[0]:
                groups[j] = groups[i]

        groupings = collections.defaultdict(list)
        for key, val in sorted(groups.items()):
            groupings[val].append(key)

        weights_ = []
        keep_inds = []
        for inds in groupings.values():
            keep_inds.append(inds[0])
            weights_.append(np.sum(weights[inds]))
        weights = np.array(weights_)

        return weights, keep_inds
