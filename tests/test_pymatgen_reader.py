"""Tests for reading pymatgen Structure objects.

pymatgen is an optional dependency, so most of these drive
:func:`amd.io.periodicset_from_pymatgen_structure` with stubs
implementing the parts of the Structure interface it uses. The tests at
the bottom of the file use pymatgen itself and are skipped when it is
not installed.
"""

import contextlib
import warnings

import numpy as np
import pytest

from amd.io import ParseError, periodicset_from_pymatgen_structure


@contextlib.contextmanager
def warnings_as_errors():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        yield


# --- stubs ------------------------------------------------------------


class FakeSpecies:
    """Stands in for pymatgen.core.Element / Species."""

    def __init__(self, symbol, Z):
        self.symbol = symbol
        self.Z = Z

    def __repr__(self):
        return self.symbol

    def __hash__(self):
        return hash((self.symbol, self.Z))

    def __eq__(self, other):
        return (
            isinstance(other, FakeSpecies)
            and (self.symbol, self.Z) == (other.symbol, other.Z)
        )


H = FakeSpecies("H", 1)
C = FakeSpecies("C", 6)
O = FakeSpecies("O", 8)
NA = FakeSpecies("Na", 11)
CO = FakeSpecies("Co", 27)
NI = FakeSpecies("Ni", 28)


class FakeComposition(dict):
    """Stands in for pymatgen.core.Composition, a mapping of species to
    occupancy."""

    def __str__(self):
        return "".join(f"{sp.symbol}{occ:g}" for sp, occ in self.items())


class FakeSite:
    """Stands in for pymatgen.core.PeriodicSite."""

    def __init__(self, species, frac_coords, label=None):
        if isinstance(species, FakeSpecies):
            species = FakeComposition({species: 1.0})
        else:
            species = FakeComposition(species)
        self.species = species
        self.frac_coords = np.array(frac_coords, dtype=np.float64)
        # PeriodicSite.label falls back to the species string when the
        # structure was built without labels.
        self._label = label

    @property
    def species_string(self):
        return str(self.species)

    @property
    def label(self):
        return self._label if self._label is not None else self.species_string


class FakeLattice:
    def __init__(self, matrix):
        self.matrix = np.array(matrix, dtype=np.float64)


class FakeStructureComposition:
    def __init__(self, reduced_formula):
        self.reduced_formula = reduced_formula


class FakeStructure:
    """Stands in for pymatgen.core.Structure."""

    def __init__(
        self,
        sites,
        matrix=None,
        properties=None,
        reduced_formula="Fake1",
    ):
        if matrix is None:
            matrix = np.identity(3) * 10.0
        self.sites = list(sites)
        self.lattice = FakeLattice(matrix)
        self.properties = {} if properties is None else dict(properties)
        self.composition = FakeStructureComposition(reduced_formula)

    def __iter__(self):
        return iter(self.sites)

    def __len__(self):
        return len(self.sites)

    @property
    def is_ordered(self):
        return all(
            len(s.species) == 1 and abs(sum(s.species.values()) - 1) < 1e-8
            for s in self.sites
        )


def frac(periodic_set):
    return periodic_set.motif @ np.linalg.inv(periodic_set.cell)


def group_indices(assembly):
    return [sorted(grp.indices) for grp in assembly.groups]


# --- the ordinary ordered structure -----------------------------------


def test_ordered_structure():
    structure = FakeStructure(
        [FakeSite(C, [0, 0, 0]), FakeSite(O, [0.5, 0.5, 0.5])]
    )
    with warnings_as_errors():
        ps = periodicset_from_pymatgen_structure(structure)

    assert ps.m == 2
    assert ps.n_asym == 2
    assert ps.types.tolist() == [6, 8]
    assert np.allclose(frac(ps), [[0, 0, 0], [0.5, 0.5, 0.5]])
    assert np.allclose(ps.cell, np.identity(3) * 10.0)
    assert ps.multiplicities.tolist() == [1, 1]
    assert ps.disorder is None
    assert np.allclose(ps.occupancies, 1.0)


def test_p1_symmetry_operations_are_recorded():
    structure = FakeStructure([FakeSite(C, [0, 0, 0])])
    ps = periodicset_from_pymatgen_structure(structure)

    # A Structure is a P1 listing, and the identity is genuinely the
    # operation the motif was built with.
    assert ps.has_symmetry()
    assert ps.symops_xyz() == ["x,y,z"]


def test_empty_structure_is_a_parse_error():
    with pytest.raises(ParseError, match="no coordinates"):
        periodicset_from_pymatgen_structure(FakeStructure([]))


# --- name -------------------------------------------------------------


def test_name_argument_wins():
    structure = FakeStructure(
        [FakeSite(C, [0, 0, 0])], properties={"material_id": "mp-1"}
    )
    ps = periodicset_from_pymatgen_structure(structure, name="chosen")
    assert ps.name == "chosen"


@pytest.mark.parametrize("key", ["name", "id", "material_id"])
def test_name_from_properties(key):
    structure = FakeStructure([FakeSite(C, [0, 0, 0])], properties={key: "mp-1"})
    ps = periodicset_from_pymatgen_structure(structure)
    assert ps.name == "mp-1"


def test_name_falls_back_to_reduced_formula():
    structure = FakeStructure([FakeSite(C, [0, 0, 0])], reduced_formula="C")
    ps = periodicset_from_pymatgen_structure(structure)
    assert ps.name == "C"


# --- labels -----------------------------------------------------------


def test_default_labels_are_not_kept():
    # A Structure built without labels sets site.label to the species
    # string, which is a formula and not a label; generating labels from
    # the types instead is more useful.
    structure = FakeStructure(
        [FakeSite(C, [0, 0, 0]), FakeSite(C, [0.5, 0, 0])]
    )
    ps = periodicset_from_pymatgen_structure(structure)
    assert ps.labels is None
    assert ps.site_labels() == ["C1", "C2"]


def test_real_labels_are_kept():
    structure = FakeStructure(
        [
            FakeSite(C, [0, 0, 0], label="C1A"),
            FakeSite(O, [0.5, 0, 0], label="O2B"),
        ]
    )
    ps = periodicset_from_pymatgen_structure(structure)
    assert ps.labels == ["C1A", "O2B"]


def test_unlabelled_sites_are_named_alongside_labelled_ones():
    structure = FakeStructure(
        [FakeSite(C, [0, 0, 0], label="C1A"), FakeSite(O, [0.5, 0, 0])]
    )
    ps = periodicset_from_pymatgen_structure(structure)
    assert ps.labels == ["C1A", "O1"]


def test_mixed_site_labels_name_their_species():
    structure = FakeStructure(
        [FakeSite({CO: 0.5, NI: 0.5}, [0, 0, 0], label="M1")]
    )
    ps = periodicset_from_pymatgen_structure(structure)
    assert ps.labels == ["M1_Co", "M1_Ni"]


# --- disorder ---------------------------------------------------------


def test_mixed_site_splits_into_one_point_per_species():
    structure = FakeStructure([FakeSite({CO: 0.5, NI: 0.5}, [0.25, 0.25, 0.25])])
    ps = periodicset_from_pymatgen_structure(structure)

    assert ps.m == 2
    assert ps.types.tolist() == [27, 28]
    assert np.allclose(frac(ps), [[0.25, 0.25, 0.25], [0.25, 0.25, 0.25]])

    assert len(ps.disorder) == 1
    assembly = ps.disorder[0]
    assert assembly.source == "cif"
    assert assembly.is_substitutional
    assert group_indices(assembly) == [[0], [1]]
    assert np.allclose([g.occupancy for g in assembly.groups], [0.5, 0.5])
    assert ps.has_substitutional_disorder()


def test_partially_occupied_site_becomes_an_assembly():
    structure = FakeStructure(
        [FakeSite(C, [0, 0, 0]), FakeSite({NA: 0.5}, [0.5, 0.5, 0.5])]
    )
    ps = periodicset_from_pymatgen_structure(structure)

    assert len(ps.disorder) == 1
    assembly = ps.disorder[0]
    assert assembly.source == "partial-occupancy"
    assert group_indices(assembly) == [[1]]
    assert assembly.groups[0].occupancy == pytest.approx(0.5)
    assert np.allclose(ps.occupancies, [1.0, 0.5])


def test_mixed_site_below_full_occupancy_keeps_its_vacancy():
    structure = FakeStructure([FakeSite({CO: 0.3, NI: 0.3}, [0, 0, 0])])
    ps = periodicset_from_pymatgen_structure(structure)

    assembly = ps.disorder[0]
    # The two are alternatives at one position and the remaining 0.4 is a
    # vacancy, so neither occupancy is scaled up.
    assert np.allclose([g.occupancy for g in assembly.groups], [0.3, 0.3])


def test_ordered_structure_has_no_disorder():
    structure = FakeStructure([FakeSite(C, [0, 0, 0])])
    assert periodicset_from_pymatgen_structure(structure).disorder is None


def test_skip_disorder():
    structure = FakeStructure([FakeSite({CO: 0.5, NI: 0.5}, [0, 0, 0])])
    with pytest.raises(ParseError, match="disorder"):
        periodicset_from_pymatgen_structure(structure, skip_disorder=True)


def test_skip_disorder_catches_a_lone_partial_occupancy():
    structure = FakeStructure([FakeSite({NA: 0.5}, [0, 0, 0])])
    with pytest.raises(ParseError, match="disorder"):
        periodicset_from_pymatgen_structure(structure, skip_disorder=True)


def test_skip_disorder_passes_an_ordered_structure():
    structure = FakeStructure([FakeSite(C, [0, 0, 0])])
    ps = periodicset_from_pymatgen_structure(structure, skip_disorder=True)
    assert ps.m == 1


# --- remove_hydrogens -------------------------------------------------


def test_remove_hydrogens():
    structure = FakeStructure(
        [
            FakeSite(C, [0, 0, 0]),
            FakeSite(H, [0.1, 0, 0]),
            FakeSite(O, [0.5, 0, 0]),
        ]
    )
    ps = periodicset_from_pymatgen_structure(structure, remove_hydrogens=True)
    assert ps.types.tolist() == [6, 8]


def test_remove_hydrogens_from_a_mixed_site():
    # Structure.remove_species cannot do this: the hydrogen is one
    # species of a site which also holds carbon.
    structure = FakeStructure([FakeSite({C: 0.5, H: 0.5}, [0, 0, 0])])
    ps = periodicset_from_pymatgen_structure(structure, remove_hydrogens=True)
    assert ps.types.tolist() == [6]


def test_remove_hydrogens_leaving_nothing_is_a_parse_error():
    structure = FakeStructure([FakeSite(H, [0, 0, 0])])
    with pytest.raises(ParseError, match="no coordinates"):
        periodicset_from_pymatgen_structure(structure, remove_hydrogens=True)


# --- missing_coords ---------------------------------------------------


def test_missing_coords_warn():
    structure = FakeStructure(
        [FakeSite(C, [0, 0, 0]), FakeSite(O, [np.nan, 0, 0])]
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ps = periodicset_from_pymatgen_structure(structure)

    assert any("missing coordinates" in str(w.message) for w in caught)
    assert ps.types.tolist() == [6]


def test_missing_coords_skip():
    structure = FakeStructure([FakeSite(C, [np.nan, 0, 0])])
    with pytest.raises(ParseError, match="missing coordinates"):
        periodicset_from_pymatgen_structure(structure, missing_coords="skip")


def test_missing_coords_silent():
    structure = FakeStructure(
        [FakeSite(C, [0, 0, 0]), FakeSite(O, [np.nan, 0, 0])]
    )
    with warnings_as_errors():
        ps = periodicset_from_pymatgen_structure(structure, missing_coords="ignore")
    assert ps.types.tolist() == [6]


# --- cell handling ----------------------------------------------------


def test_wrap_coords():
    structure = FakeStructure([FakeSite(C, [1.25, -0.5, 0])])
    ps = periodicset_from_pymatgen_structure(structure)
    assert np.allclose(frac(ps), [[0.25, 0.5, 0]])


def test_wrap_coords_off():
    structure = FakeStructure([FakeSite(C, [1.25, -0.5, 0])])
    ps = periodicset_from_pymatgen_structure(structure, wrap_coords=False)
    assert np.allclose(frac(ps), [[1.25, -0.5, 0]])


def test_reduce_cell_drops_the_symmetry_operations():
    # A very skewed cell so that the reduction actually changes it.
    matrix = [[1.0, 0.0, 0.0], [9.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    structure = FakeStructure([FakeSite(C, [0, 0, 0])], matrix=matrix)
    ps = periodicset_from_pymatgen_structure(structure, reduce_cell=True)

    assert not ps.has_symmetry()
    assert np.isclose(
        np.abs(np.linalg.det(ps.cell)), np.abs(np.linalg.det(np.array(matrix)))
    )


# --- pymatgen itself --------------------------------------------------


def test_real_structure_cell_and_motif():
    pytest.importorskip("pymatgen")
    from pymatgen.core import Lattice, Structure

    lattice = Lattice.cubic(4.0)
    structure = Structure(lattice, ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    ps = periodicset_from_pymatgen_structure(structure)

    assert np.allclose(ps.cell, lattice.matrix)
    assert np.allclose(frac(ps), [[0, 0, 0], [0.5, 0.5, 0.5]])
    assert ps.types.tolist() == [11, 17]
    assert ps.name == structure.composition.reduced_formula


def test_real_disordered_structure():
    pytest.importorskip("pymatgen")
    from pymatgen.core import Lattice, Structure

    structure = Structure(
        Lattice.cubic(3.5), [{"Co": 0.5, "Ni": 0.5}], [[0, 0, 0]]
    )
    ps = periodicset_from_pymatgen_structure(structure)

    assert ps.m == 2
    assert sorted(ps.types.tolist()) == [27, 28]
    assert len(ps.disorder) == 1
    assert ps.disorder[0].is_substitutional


def test_real_symprec_finds_the_asymmetric_unit():
    pytest.importorskip("pymatgen")
    from pymatgen.core import Lattice, Structure

    # Body centred cubic caesium: two sites, one orbit.
    structure = Structure(
        Lattice.cubic(6.0), ["Cs", "Cs"], [[0, 0, 0], [0.5, 0.5, 0.5]]
    )
    ps = periodicset_from_pymatgen_structure(structure, symprec=0.01)

    assert ps.has_symmetry()
    assert len(ps.symop_rotations) > 1
    assert ps.n_asym == 1
    assert ps.multiplicities.tolist() == [2]
    assert ps.m == 2


def test_real_symprec_none_leaves_the_structure_in_p1():
    pytest.importorskip("pymatgen")
    from pymatgen.core import Lattice, Structure

    structure = Structure(
        Lattice.cubic(6.0), ["Cs", "Cs"], [[0, 0, 0], [0.5, 0.5, 0.5]]
    )
    ps = periodicset_from_pymatgen_structure(structure)

    assert ps.n_asym == 2
    assert ps.multiplicities.tolist() == [1, 1]


def test_real_symprec_with_a_disordered_site():
    pytest.importorskip("pymatgen")
    from pymatgen.core import Lattice, Structure

    # Body centred, both sites the same Co/Ni mixture, so spglib sees one
    # orbit and each species becomes a point of the asymmetric unit.
    structure = Structure(
        Lattice.cubic(3.5),
        [{"Co": 0.5, "Ni": 0.5}, {"Co": 0.5, "Ni": 0.5}],
        [[0, 0, 0], [0.5, 0.5, 0.5]],
    )
    ps = periodicset_from_pymatgen_structure(structure, symprec=0.01)

    assert ps.n_asym == 2
    assert sorted(ps.types.tolist()) == [27, 28]
    assert ps.multiplicities.tolist() == [2, 2]
    assert ps.m == 4
    assert len(ps.disorder) == 1
    assert ps.disorder[0].is_substitutional


def test_real_symprec_preserves_the_geometry(data_dir):
    pytest.importorskip("pymatgen")
    from pymatgen.core import Structure

    import amd

    # This structure is exactly symmetric, so collapsing it onto an
    # asymmetric unit must leave its AMD untouched.
    structure = Structure.from_file(str(data_dir / "eq_sites_test.cif"))
    p1 = periodicset_from_pymatgen_structure(structure)
    symmetrised = periodicset_from_pymatgen_structure(structure, symprec=0.01)

    assert symmetrised.n_asym < p1.n_asym
    assert int(symmetrised.multiplicities.sum()) == p1.m
    assert np.allclose(amd.AMD(p1, 30), amd.AMD(symmetrised, 30))


def test_real_agrees_with_the_gemmi_reader(data_dir):
    pytest.importorskip("pymatgen")
    from pymatgen.core import Structure

    import amd

    path = str(data_dir / "eq_sites_test.cif")
    # pymatgen reads the first block of a multi-block CIF; take the
    # matching one from the gemmi reader.
    from_gemmi = list(amd.CifReader(path, show_warnings=False))[0]
    from_pymatgen = periodicset_from_pymatgen_structure(
        Structure.from_file(path)
    )

    assert from_pymatgen.m == from_gemmi.m
    assert np.allclose(amd.AMD(from_gemmi, 30), amd.AMD(from_pymatgen, 30))


def test_real_cubic_cif_matches_the_reference(data_dir, reference_data):
    pytest.importorskip("pymatgen")
    from pymatgen.core import Structure

    structure = Structure.from_file(str(data_dir / "cubic.cif"))
    ps = periodicset_from_pymatgen_structure(structure)
    # One entry per structure in the file; cubic.cif holds just one.
    expected = reference_data["cubic"][0]["PeriodicSet"]

    if not ps._equal_cell_and_motif(expected):
        pytest.fail(
            "periodicset_from_pymatgen_structure does not agree with the "
            "reference PeriodicSet for cubic.cif"
        )
