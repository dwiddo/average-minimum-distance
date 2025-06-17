"""Implements the :class:`PeriodicSet` class representing a periodic
set defined by a motif (collection of points) and unit cell (basis)
modelling a crystal with a point at the center of each atom.

:class:`PeriodicSet` s are yielded by the children of
:class:`amd.io._Reader <.io._Reader>` and are passed to
:func:`PDD() <.calculate.PDD>`, :func:`AMD() <.calculate.AMD>` etc to
calculate the invariants of crystals.
"""

from __future__ import annotations
from typing import Optional, List, NamedTuple
import math

import numpy as np
from scipy.special import factorial

from .utils import (
    cellpar_to_cell,
    cellpar_to_cell_2D,
    random_cell,
    cell_to_cellpar,
)
from ._types import FloatArray, UIntArray, AtomTypeArray
from .globals_ import ATOMIC_MASSES, ATOMIC_NUMBERS_TO_SYMS


__all__ = ["PeriodicSet", "DisorderGroup", "DisorderAssembly"]


class DisorderGroup(NamedTuple):
    indices: list[int]
    occupancy: float


class DisorderAssembly(NamedTuple):
    groups: list[DisorderGroup]
    is_substitutional: bool


class PeriodicSet:
    """A periodic set is a collection of points (motif) which
    periodically repeats according to a lattice (unit cell), which can
    model a crystal with a point in the centre of each atom.

    :class:`PeriodicSet` s are yielded by the children of
    :class:`amd.io._Reader <.io._Reader>` and are passed to
    :func:`PDD() <.calculate.PDD>`, :func:`AMD() <.calculate.AMD>` etc to
    calculate the invariants of crystals.

    Parameters
    ----------
    motif : :class:`numpy.ndarray`
        Collection of motif points in Cartesian (orthogonal)
        coordinates, shape ``(m, dims)``.
    cell : :class:`numpy.ndarray`
        Square array representing the unit cell in Cartesian
        (orthogonal) coordinates, shape ``(dims, dims)``. Use
        :func:`amd.cellpar_to_cell <.utils.cellpar_to_cell>` to convert
        a vector of 6 conventional cell parameters a,b,c,α,β,γ to a
        square matrix.
    name : str, optional
        Name of the periodic set.
    asym_unit : :class:`numpy.ndarray`, optional
        Indices of motif points constituting an asymmetric unit, so that
        ``motif[asym_unit]`` is an asymmetric unit. If given, the motif
        should be such that ``motif[asym_unit[i]:asym_unit[i+1]]`` are
        points corresponding to ``motif[asym_unit[i]``.
    multiplicities : :class:`numpy.ndarray`, optional
        Wyckoff multiplicities of points in the asymmetric unit (the
        number of unique sites generated under symmetries).
    types : list, optional
        Atomic numbers of atoms in the asymmetric unit.
    labels : list, optional
        Labels of atoms in the asymmetric unit.
    disorder : list, optional
        Contains disorder assemblies of the crystal.
    """

    def __init__(
        self,
        motif: FloatArray,
        cell: FloatArray,
        name: Optional[str] = None,
        asym_unit: Optional[UIntArray] = None,
        multiplicities: Optional[UIntArray] = None,
        types: Optional[AtomTypeArray] = None,
        labels: Optional[List[str]] = None,
        disorder: Optional[list[DisorderAssembly]] = None,
    ):
        self.motif = motif
        self.cell = cell
        self.name = name
        self._asym_unit = asym_unit
        self._multiplicities = multiplicities
        self.types = types
        self.labels = labels
        self.disorder = disorder

    @property
    def ndim(self) -> int:
        """Dimension of the ambient space."""
        return self.cell.shape[0]

    @property
    def m(self) -> int:
        """Number of motif points."""
        return self.motif.shape[0]

    @property
    def n_asym(self) -> int:
        """Number of points in the asymmetric unit."""
        if self.asym_unit is None:
            return self.m
        else:
            return self.asym_unit.shape[0]

    @property
    def asym_unit(self) -> np.ndarray:
        if self._asym_unit is None:
            return np.arange(self.m)
        else:
            return self._asym_unit

    @property
    def multiplicities(self) -> np.ndarray:
        if self._multiplicities is None:
            return np.ones((self.m,), dtype=np.uint64)
        else:
            return self._multiplicities

    @property
    def occupancies(self) -> np.ndarray:
        occs = np.ones((self.n_asym, ), dtype=np.float64)
        if self.disorder:
            for asm in self.disorder:
                for grp in asm.groups:
                    occs[grp.indices] = grp.occupancy
        return occs

    def __str__(self) -> str:
        m, n = self.motif.shape
        m_pl = "" if m == 1 else "s"
        n_pl = "" if n == 1 else "s"
        name_str = f"{self.name}: " if self.name is not None else ""
        return f"PeriodicSet({name_str}{m} point{m_pl} in {n} dimension{n_pl})"

    def __repr__(self) -> str:
        motif_str = str(self.motif).replace("\n ", "\n" + " " * 11)
        cell_str = str(self.cell).replace("\n ", "\n" + " " * 10)

        optional_attrs = []
        for attr in ("_asym_unit", "_multiplicities"):
            val = getattr(self, attr)
            if val is not None:
                st = str(val).replace("\n ", "\n" + " " * (len(attr) + 6))
                optional_attrs.append(f"{attr[1:]}={st}")

        for attr in ("types", "labels"):
            val = getattr(self, attr)
            if val is not None:
                st = str(val).replace("\n ", "\n" + " " * (len(attr) + 6))
                optional_attrs.append(f"{attr}={st}")

        optional_attrs_str = ",\n    " if optional_attrs else ""
        optional_attrs_str += ",\n    ".join(optional_attrs)

        return (
            f"PeriodicSet(name={self.name},\n"
            f"    motif={motif_str},\n"
            f"    cell={cell_str}{optional_attrs_str})"
        )

    def PPC(self):
        r"""Return the point packing coefficient (PPC) of the
        PeriodicSet.

        The PPC is a constant of any periodic set determining the
        asymptotic behaviour of its isometry invariants. As
        :math:`k \rightarrow \infty`, the ratio
        :math:`\text{AMD}_k / \sqrt[n]{k}` converges to the PPC, as does
        any row of its PDD.

        For a unit cell :math:`U` and :math:`m` motif points in
        :math:`n` dimensions,

        .. math::

            \text{PPC} = \sqrt[n]{\frac{\text{Vol}[U]}{m V_n}}

        where :math:`V_n` is the volume of a unit sphere in :math:`n`
        dimensions.

        Returns
        -------
        ppc : float
            The PPC of ``self``.
        """
        m, n = self.motif.shape
        t = int(n // 2)
        if n % 2 == 0:
            ball_vol = (np.pi ** t) / factorial(t)
        else:
            ball_vol = (2 * factorial(t) * (4 * np.pi) ** t) / factorial(n)
        return (np.abs(np.linalg.det(self.cell)) / (m * ball_vol)) ** (1.0 / n)

    def has_positional_disorder(self):
        if self.disorder is not None:
            return any(not asm.is_substitutional for asm in self.disorder)
        return False

    def has_substitutional_disorder(self):
        if self.disorder is not None:
            return any(asm.is_substitutional for asm in self.disorder)
        return False

    def density(self):
        """Return the physical density of the PeriodicSet representing a
        crystal in g/cm3.

        Returns
        -------
        density : float
            The physical density of ``self`` in g/cm3.
        """
        if self.types is None:
            raise ValueError(f"PeriodicSet(name={self.name}) has no types")
        cell_mass = sum(
            ATOMIC_MASSES[num] * tot_occ
            for num, tot_occ in zip(self.types, self.multiplicities * self.occupancies)
        )
        cell_vol = np.linalg.det(self.cell)
        return (cell_mass / cell_vol) * 1.66053907

    def formula(self):
        """Return the reduced formula of the PeriodicSet representing a
        crystal.

        Returns
        -------
        formula : str
            The reduced formula of ``self``.
        """
        counter = self._formula_dict()
        counter = {ATOMIC_NUMBERS_TO_SYMS[n]: count for n, count in counter.items()}
        return "".join(
            f"{lab}{num:.3g}" if num != 1 else lab for lab, num in counter.items()
        )

    def _formula_dict(self):
        """Return a dictionary of atomic types in the form
        {atomic_num: count}.
        """

        if self.types is None:
            raise ValueError(f"PeriodicSet(name={self.name}) has no types")

        counter = {}
        for num, mul, occ in zip(self.types, self.multiplicities, self.occupancies):
            if num not in counter:
                counter[num] = 0
            counter[num] += occ * mul

        if 0 in counter:
            del counter[0]

        gcd = math.gcd(*(int(v) for v in counter.values() if np.mod(v, 1.0) == 0.0))
        if gcd == 0:
            gcd = 1

        counter_ = {}
        if 6 in counter:
            counter_[6] = counter[6] / gcd
            del counter[6]
        if 1 in counter:
            counter_[1] = counter[1] / gcd
            del counter[1]

        for n, count in sorted(counter.items(), key=lambda x: x[0]):
            counter_[int(n)] = count / gcd

        return counter_

    def _equal_cell_and_motif(self, other: PeriodicSet, tol: float = 1e-8) -> bool:
        """Used for debugging. True if both 1) unit cells are (close to)
        identical, and 2) the motifs are the same shape, and
        every point in one motif has a (close to) identical point
        somewhere in the other, accounting for pbc.
        """

        if (
            self.cell.shape != other.cell.shape
            or self.motif.shape != other.motif.shape
            or not np.allclose(self.cell, other.cell)
        ):
            return False

        cell_inv = np.linalg.inv(self.cell)
        fm1 = np.mod(self.motif @ cell_inv, 1)
        fm2 = np.mod(other.motif @ cell_inv, 1)
        d1 = np.abs(fm2[:, None] - fm1)
        d2 = np.abs(d1 - 1)
        diffs = np.amax(np.minimum(d1, d2), axis=-1)

        if not np.all(
            (np.amin(diffs, axis=0) <= tol) | (np.amin(diffs, axis=-1) <= tol)
        ):
            return False
        return True

    @staticmethod
    def cubic(scale: float = 1.0, dims: int = 3) -> PeriodicSet:
        """Return a :class:`PeriodicSet` representing a cubic lattice."""
        return PeriodicSet(np.zeros((1, dims)), np.identity(dims) * scale)

    @staticmethod
    def hexagonal(scale: float = 1.0, dims: int = 3) -> PeriodicSet:
        """Return a :class:`PeriodicSet` representing a hexagonal
        lattice. Implemented for dimensions 2 and 3 only.
        """

        if dims == 3:
            cellpar = np.array([scale, scale, scale, 90.0, 90.0, 120.0])
            cell = cellpar_to_cell(cellpar)
        elif dims == 2:
            cell = cellpar_to_cell_2D(np.array([scale, scale, 60.0]))
        else:
            raise NotImplementedError(
                "amd.PeriodicSet.hexagonal() only implemented for dimensions "
                f"2 and 3, passed {dims}"
            )
        return PeriodicSet(np.zeros((1, dims)), cell)

    @staticmethod
    def random(
        n_points: int,
        length_bounds: tuple = (1.0, 2.0),
        angle_bounds: tuple = (60.0, 120.0),
        dims: int = 3,
    ) -> PeriodicSet:
        """Return a :class:`PeriodicSet` with a chosen number of
        randomly placed points in a random cell with edges between
        ``length_bounds`` and angles between ``angle_bounds``.
        Implemented for dimensions 2 and 3 only.
        """

        if dims not in (2, 3):
            raise NotImplementedError(
                "amd.PeriodicSet._random only implemented for dimensions 2 "
                f"and 3, passed {dims}"
            )

        cell = random_cell(
            length_bounds=length_bounds, angle_bounds=angle_bounds, dims=dims
        )
        frac_motif = np.random.uniform(size=(n_points, dims))
        return PeriodicSet(frac_motif @ cell, cell)

    def to_cif_str(self) -> str:
        """Return a CIF string representing the PeriodicSet. Does not
        write disorder data, atom labels or symmetries."""

        a, b, c, al, be, ga = cell_to_cellpar(self.cell)
        out = [
            f"data_{self.name}",
            "_symmetry_space_group_name_H-M   'P 1'",
            f"_cell_length_a   {a}",
            f"_cell_length_b   {b}",
            f"_cell_length_c   {c}",
            f"_cell_angle_alpha   {al}",
            f"_cell_angle_beta   {be}",
            f"_cell_angle_gamma   {ga}",
            "loop_",
            " _symmetry_equiv_pos_site_id",
            " _symmetry_equiv_pos_as_xyz",
            " 1  'x, y, z'",
            "loop_",
            " _atom_site_label",
            " _atom_site_type_symbol",
            " _atom_site_fract_x",
            " _atom_site_fract_y",
            " _atom_site_fract_z",
        ]

        frac_motif = self.motif @ np.linalg.inv(self.cell)
        ind = 0
        label_nums = {}
        for asym_ind, mul in enumerate(self.multiplicities):
            for _ in range(mul):

                sym = ATOMIC_NUMBERS_TO_SYMS[self.types[asym_ind]]
                if sym in label_nums:
                    label_nums[sym] += 1
                else:
                    label_nums[sym] = 1

                out.append(
                    "  ".join(
                        [
                            f"{sym}{label_nums[sym]}",
                            sym,
                            f"{frac_motif[ind, 0]:.4f}",
                            f"{frac_motif[ind, 1]:.4f}",
                            f"{frac_motif[ind, 2]:.4f}",
                        ]
                    )
                )
                ind += 1

        return "\n".join(out)

    def to_pymatgen_structure(self):

        if self.types is None:
            raise ValueError(
                "Cannot convert PeriodicSet to pymatgen Structure without "
                "atomic types."
            )

        from pymatgen.core import Structure

        species = [
            ATOMIC_NUMBERS_TO_SYMS[t]
            for m, t in zip(self.multiplicities, self.types)
            for _ in range(m)
        ]

        labels = None
        if self.labels is not None:
            labels = [
                f"{t}_{i+1}"
                for m, t in zip(self.multiplicities, self.labels)
                for i in range(m)
            ]

        return Structure(
            self.cell, species, self.motif, coords_are_cartesian=True, labels=labels
        )

    def majority_occupancy_config(self) -> PeriodicSet:

        if not self.disorder:
            return self

        asym_mask = np.full((self.n_asym, ), fill_value=True)
        motif_mask = np.full((self.m, ), fill_value=True)
        
        for asm in self.disorder:

            # find majority occupancy group
            majority_occ, majority_ind = 0, 0
            for i, grp in enumerate(asm.groups):
                if grp.occupancy > majority_occ:
                    majority_ind = i
                    majority_occ = grp.occupancy

            # mask all other groups
            for i, grp in enumerate(asm.groups):

                if i == majority_ind:
                    continue
                
                for j in grp.indices:
                    asym_mask[j] = False
                    m_i = self.asym_unit[j]
                    mul = self.multiplicities[j]
                    motif_mask[m_i : m_i + mul] = False

        muls = self.multiplicities[asym_mask]
        asym_unit = np.zeros_like(muls, dtype=np.int64)
        asym_unit[1:] = np.cumsum(muls)[:-1]
        masked_labels = [l for l, flag in zip(self.labels, asym_mask) if flag]

        return PeriodicSet(
            self.motif[motif_mask],
            self.cell,
            self.name,
            asym_unit,
            self.multiplicities[asym_mask],
            self.types[asym_mask],
            masked_labels
        )
    
    def compressed_dtypes(self) -> PeriodicSet:
        """Return a PeriodicSet using float32, uint32 and uint8 for
        atomic types."""
        
        if self._asym_unit is not None:
            asym_unit = np.array(self._asym_unit, dtype=np.uint32)
        else:
            asym_unit = None
        
        if self._multiplicities is not None:
            multiplicities = np.array(self._multiplicities, dtype=np.uint32)
        else:
            multiplicities = None
        
        if self.types is not None:
            types = np.array(self.types, dtype=np.uint8)
        else:
            types = None

        return PeriodicSet(
            np.array(self.motif, dtype=np.float32),
            np.array(self.cell, dtype=np.float32),
            name=self.name,
            asym_unit=asym_unit,
            multiplicities=multiplicities,
            types=types,
            labels=self.labels,
            disorder=self.disorder
        )
