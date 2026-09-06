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
from fractions import Fraction
import math

import numpy as np

from .utils import (
    cellpar_to_cell,
    cellpar_to_cell_2D,
    random_cell,
    cell_to_cellpar,
)
from ._types import FloatArray, UIntArray, AtomTypeArray
from .globals_ import ATOMIC_MASSES, ATOMIC_NUMBERS_TO_SYMS


__all__ = ["PeriodicSet", "DisorderGroup", "DisorderAssembly"]

# Rotation matrix entries are 0 or +-1, so anything smaller than this is
# a zero which floating point arithmetic has nudged off the mark.
_SYMOP_TOL = 1e-8


def _assembly_code(index: int) -> str:
    """Name disorder assembly `index` the way a CIF does: A, B, ... Z,
    AA, AB and so on."""
    code = ""
    index += 1
    while index > 0:
        index, remainder = divmod(index - 1, 26)
        code = chr(ord("A") + remainder) + code
    return code


class DisorderGroup(NamedTuple):
    """One mutually exclusive alternative configuration within a
    :class:`DisorderAssembly`. Atoms of the same group are simultaneously
    occupied; atoms of different groups in the same assembly are not.
    """

    indices: list[int]
    occupancy: float


class DisorderAssembly(NamedTuple):
    """A set of atoms whose disorder is coupled, comprised of two or more
    mutually exclusive :class:`DisorderGroup` s.

    ``source`` records how the assembly was determined: ``"cif"`` if it
    came from explicit ``_atom_site_disorder_assembly`` /
    ``_atom_site_disorder_group`` codes, otherwise the name of the
    heuristic which inferred it from occupancies and positions.
    """

    groups: list[DisorderGroup]
    is_substitutional: bool
    source: str = "cif"


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
    symop_rotations : :class:`numpy.ndarray`, optional
        Rotation parts of the symmetry operations of the crystal, shape
        ``(n_ops, dims, dims)``. These are the operations which were
        applied to ``motif[asym_unit]`` to build the motif, so they are
        expressed in the basis of ``cell``.
    symop_translations : :class:`numpy.ndarray`, optional
        Translation parts of the symmetry operations, shape
        ``(n_ops, dims)``, in fractional coordinates.
    """

    # Attributes added after the first release. They are absent from the
    # __dict__ of anything pickled by an earlier version, so __setstate__
    # fills them in rather than letting attribute access fail.
    _ADDED_ATTRS = ("symop_rotations", "symop_translations")

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
        symop_rotations: Optional[FloatArray] = None,
        symop_translations: Optional[FloatArray] = None,
    ):
        self.motif = motif
        self.cell = cell
        self.name = name
        self._asym_unit = asym_unit
        self._multiplicities = multiplicities
        self.types = types
        self.labels = labels
        self.disorder = disorder
        self.symop_rotations = symop_rotations
        self.symop_translations = symop_translations

    def __setstate__(self, state: dict) -> None:
        """Restore from a pickle, defaulting any attribute added since
        the pickle was written."""
        for attr in PeriodicSet._ADDED_ATTRS:
            state.setdefault(attr, None)
        self.__dict__.update(state)

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
        # One per asymmetric unit point, like types, labels and
        # occupancies - NOT one per motif point. The two agree only when
        # there is no asymmetric unit, and defaulting to self.m instead
        # silently misaligns every consumer of this the moment a set has
        # an asym_unit but no multiplicities of its own.
        if self._multiplicities is None:
            return np.ones((self.n_asym,), dtype=np.uint64)
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

        if self.has_symmetry():
            n_ops = len(self.symop_rotations)
            op_pl = "" if n_ops == 1 else "s"
            optional_attrs.append(f"symops={n_ops} operation{op_pl}")

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

        Notes
        -----
        For a disordered crystal :math:`m` is the *expected* number of
        atoms in a unit cell, the occupancies summed. That is the right
        count for the density, but it is not in general the constant the
        AMD of a disordered crystal converges to. A disordered crystal is
        a distribution over ordered ones, each of which has its own PPC,
        and the limit is the average of those.

        .. math::

            E[m^{1-1/n}] \le E[m]^{1-1/n}

        so this formula is exact when every configuration holds the same
        number of atoms - substitutional disorder, or alternatives of
        equal size - and an overestimate when that number varies, which
        is what a partially occupied site does. The exact constant is
        :attr:`PointwiseDistanceDistribution.PPC
        <.pointwise_distance_distribution.PointwiseDistanceDistribution.PPC>`,
        the average of the configurations' own PPCs.
        """
        _, n = self.motif.shape
        m = np.sum(self.occupancies * self.multiplicities)
        t = int(n // 2)
        if n % 2 == 0:
            ball_vol = (np.pi ** t) / math.factorial(t)
        else:
            ball_vol = (2 * math.factorial(t) * (4 * np.pi) ** t) / math.factorial(n)

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
        # abs: a left handed basis has a negative determinant, which is
        # not a negative volume.
        cell_vol = np.abs(np.linalg.det(self.cell))
        return (cell_mass / cell_vol) * 1.66053907

    def formula(self, sep="", latex=False) -> str:
        """Return the reduced formula of the PeriodicSet representing a
        crystal.

        Returns
        -------
        formula : str
            The reduced formula of ``self``.
        """
        counter = self._formula_dict()
        counter = {ATOMIC_NUMBERS_TO_SYMS[n]: count for n, count in counter.items()}
        if latex:
            pieces = []
            for lab, num in counter.items():
                if num == 1:
                    pieces.append(lab)
                elif num % 1 == 0 and num < 10:  # single digit
                    pieces.append(f"{lab}{int(num)}")
                else:
                    pieces.append(f"{lab}_{{{num:.3g}}}")
            return sep.join(pieces)

        else:
            return sep.join(
                f"{lab}{num:.3g}" if num != 1 else lab 
                for lab, num in counter.items()
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

        c_h = (6, 1)
        counter_ = {}
        for n in c_h:
            if n in counter:
                counter_[n] = counter[n] / gcd

        for n, count in sorted(counter.items(), key=lambda x: ATOMIC_NUMBERS_TO_SYMS[x[0]]):
            if n not in c_h:
                counter_[n] = count / gcd

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

    def has_symmetry(self) -> bool:
        """Whether the symmetry operations of the crystal are known."""
        return (
            self.symop_rotations is not None
            and self.symop_translations is not None
            and len(self.symop_rotations) > 0
        )

    def symops_xyz(self) -> List[str]:
        """Return the symmetry operations as xyz triplet strings, e.g.
        ``["x,y,z", "-x,y+1/2,-z+1/2"]``, the form a CIF's
        ``_symmetry_equiv_pos_as_xyz`` loop takes.

        Raises :class:`ValueError` if the operations are not known; check
        :meth:`has_symmetry` first.
        """

        if not self.has_symmetry():
            raise ValueError(
                "Cannot write the symmetry operations of a PeriodicSet "
                "which does not have any."
            )

        axes = "xyz"
        symops = []
        for rot, trans in zip(self.symop_rotations, self.symop_translations):
            terms = []
            for i in range(len(rot)):
                term = ""
                for j, coeff in enumerate(rot[i]):
                    if abs(coeff) < _SYMOP_TOL:
                        continue
                    sign = "-" if coeff < 0 else ("+" if term else "")
                    term += sign + axes[j]

                # Translations are always a multiple of 1/24 (gemmi's own
                # denominator), so bounding the denominator recovers the
                # exact fraction from the float rather than approximating
                # it - 1/3 comes back as 1/3, not 0.3333.
                frac = Fraction(float(trans[i])).limit_denominator(24)
                if frac:
                    magnitude = abs(frac)
                    text = (
                        str(magnitude.numerator)
                        if magnitude.denominator == 1
                        else f"{magnitude.numerator}/{magnitude.denominator}"
                    )
                    term += ("-" if frac < 0 else "+") + text

                terms.append(term.lstrip("+") or "0")
            symops.append(",".join(terms))

        return symops

    def site_labels(self) -> List[str]:
        """Return one label per point of the asymmetric unit.

        ``self.labels`` is used when it is set, which is the case for
        :class:`PeriodicSet` s read from a file. Otherwise labels are
        generated from the atomic types by numbering each element over
        the asymmetric unit (``Si1``, ``O1``, ``O2``, ...), which is what
        a CIF's own labels usually look like.
        """

        if self.labels is not None and len(self.labels) == self.n_asym:
            # Whitespace would split a label across two columns of the
            # CIF atom site loop. Real labels are single tokens, but
            # collapsing keeps generated labels and file text in step.
            return ["".join(str(lab).split()) for lab in self.labels]

        if self.types is None:
            raise ValueError(
                "Cannot label the points of a PeriodicSet without either "
                "labels or atomic types."
            )

        labels = []
        counts = {}
        for t in self.types:
            sym = ATOMIC_NUMBERS_TO_SYMS[t]
            counts[sym] = counts.get(sym, 0) + 1
            labels.append(f"{sym}{counts[sym]}")
        return labels

    def expanded_labels(self) -> List[str]:
        """Return one label per motif point, in the order of ``motif``.

        Point ``i`` of the asymmetric unit contributes
        ``multiplicities[i]`` points to the motif; its copies are
        labelled ``<site label>_1``, ``<site label>_2`` and so on, so
        that a label names both the site a point came from and which
        copy of that site the point is.

        This is what :meth:`to_cif_str` writes as ``_atom_site_label``,
        so a program displaying that CIF can map an atom it has picked
        back onto a row of ``motif`` - and from there, via
        ``multiplicities``, onto a row of the PDD.
        """

        labels = []
        for label, mul in zip(self.site_labels(), self.multiplicities):
            labels.extend(f"{label}_{copy + 1}" for copy in range(int(mul)))
        return labels

    def _disorder_codes(self) -> tuple[Optional[List[str]], Optional[List[str]]]:
        """Return the ``_atom_site_disorder_assembly`` and
        ``_atom_site_disorder_group`` code of each point of the
        asymmetric unit, or ``(None, None)`` if there is no disorder.

        Codes are generated from the position of each assembly and group
        rather than kept from the file, so they exist even for
        assemblies which were inferred rather than read
        (``DisorderAssembly.source != "cif"``) and so never had codes of
        their own. Points outside every assembly take ``"."``, the CIF
        way of writing "not applicable".
        """

        if not self.disorder:
            return None, None

        assemblies = ["."] * self.n_asym
        groups = ["."] * self.n_asym
        for asm_ind, asm in enumerate(self.disorder):
            code = _assembly_code(asm_ind)
            for grp_ind, grp in enumerate(asm.groups):
                for i in grp.indices:
                    assemblies[i] = code
                    groups[i] = str(grp_ind + 1)

        return assemblies, groups

    def to_cif_str(self) -> str:
        """Return a CIF string representing the PeriodicSet.

        If the symmetry operations of the crystal are known, only the
        asymmetric unit is listed, along with a
        ``_symmetry_equiv_pos_as_xyz`` loop for a reader to rebuild the
        rest of the motif from. Atom labels are then those of
        :meth:`site_labels`.

        Otherwise the cell is written in P1 with the motif listed in
        full, labelled with :meth:`expanded_labels` so that a label still
        identifies the asymmetric unit point its atom came from.

        Occupancies and disorder assembly/group codes are written when
        the crystal has any disorder.
        """

        if self.types is None:
            raise ValueError(
                "Cannot convert PeriodicSet to CIF without atomic types."
            )

        # A CIF names three cell lengths, three angles and an x, y and z
        # per atom; there is no form of it for any other dimension.
        if self.ndim != 3:
            raise ValueError(
                f"Cannot convert a PeriodicSet in {self.ndim} dimensions to CIF."
            )

        use_symmetry = self.has_symmetry()

        a, b, c, al, be, ga = cell_to_cellpar(self.cell)
        out = [f"data_{self.name}"]

        # With operations to write there is no need to name the space
        # group: a reader takes the operations over the name, and the
        # name is not known here anyway. Without them, P1 is exactly what
        # listing every motif point means.
        if not use_symmetry:
            out.append("_symmetry_space_group_name_H-M   'P 1'")

        out.extend(
            [
                f"_cell_length_a   {a}",
                f"_cell_length_b   {b}",
                f"_cell_length_c   {c}",
                f"_cell_angle_alpha   {al}",
                f"_cell_angle_beta   {be}",
                f"_cell_angle_gamma   {ga}",
                "loop_",
                " _symmetry_equiv_pos_site_id",
                " _symmetry_equiv_pos_as_xyz",
            ]
        )

        symops = self.symops_xyz() if use_symmetry else ["x,y,z"]
        for i, symop in enumerate(symops):
            out.append(f" {i + 1}  '{symop}'")

        occupancies = self.occupancies
        write_occupancies = not np.allclose(occupancies, 1.0)
        assemblies, groups = self._disorder_codes()

        tags = [
            " _atom_site_label",
            " _atom_site_type_symbol",
            " _atom_site_fract_x",
            " _atom_site_fract_y",
            " _atom_site_fract_z",
        ]
        if write_occupancies:
            tags.append(" _atom_site_occupancy")
        if assemblies is not None:
            tags.append(" _atom_site_disorder_assembly")
            tags.append(" _atom_site_disorder_group")
        out.append("loop_")
        out.extend(tags)

        # (label, asymmetric unit index, motif index) of every atom to be
        # listed. Under symmetry that is one atom per asymmetric unit
        # point; otherwise it is the whole motif.
        if use_symmetry:
            labels = self.site_labels()
            entries = [
                (labels[i], i, int(self.asym_unit[i])) for i in range(self.n_asym)
            ]
        else:
            labels = self.expanded_labels()
            entries = []
            motif_ind = 0
            for asym_ind, mul in enumerate(self.multiplicities):
                for _ in range(int(mul)):
                    entries.append((labels[motif_ind], asym_ind, motif_ind))
                    motif_ind += 1

        frac_motif = self.motif @ np.linalg.inv(self.cell)
        for label, asym_ind, motif_ind in entries:
            row = [
                label,
                ATOMIC_NUMBERS_TO_SYMS[self.types[asym_ind]],
                f"{frac_motif[motif_ind, 0]:.6f}",
                f"{frac_motif[motif_ind, 1]:.6f}",
                f"{frac_motif[motif_ind, 2]:.6f}",
            ]
            if write_occupancies:
                row.append(f"{occupancies[asym_ind]:.4f}")
            if assemblies is not None:
                row.append(assemblies[asym_ind])
                row.append(groups[asym_ind])
            out.append("  ".join(row))

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
        asym_unit = np.zeros_like(muls, dtype=np.uint64)
        asym_unit[1:] = np.cumsum(muls)[:-1]

        # types and labels are both optional, so a set which has neither
        # still has a majority configuration - mask them only if present.
        masked_labels = None
        if self.labels is not None:
            masked_labels = [
                lab for lab, flag in zip(self.labels, asym_mask) if flag
            ]
        masked_types = None
        if self.types is not None:
            masked_types = np.asarray(self.types)[asym_mask]

        # The operations still generate the points which survive the
        # mask, so they carry over; the disorder itself does not, having
        # just been collapsed to one configuration.
        return PeriodicSet(
            self.motif[motif_mask],
            self.cell,
            self.name,
            asym_unit,
            muls,
            masked_types,
            masked_labels,
            symop_rotations=self.symop_rotations,
            symop_translations=self.symop_translations,
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

        # The symmetry operations are left alone: there are at most a few
        # hundred tiny numbers, so narrowing them saves nothing worth the
        # precision, and they have to reproduce exact fractions.
        return PeriodicSet(
            np.array(self.motif, dtype=np.float32),
            np.array(self.cell, dtype=np.float32),
            name=self.name,
            asym_unit=asym_unit,
            multiplicities=multiplicities,
            types=types,
            labels=self.labels,
            disorder=self.disorder,
            symop_rotations=self.symop_rotations,
            symop_translations=self.symop_translations,
        )
