"""Tests for reading disorder from ccdc Crystal objects.

csd-python-api is not installable everywhere, so these drive
:func:`amd.io.periodicset_from_ccdc_crystal` with stubs implementing the
parts of the interface it uses, as documented at
https://downloads.ccdc.cam.ac.uk/documentation/API/modules/crystal_api.html
"""

import contextlib
import warnings

import numpy as np
import pytest

from amd.io import ParseError, periodicset_from_ccdc_crystal


@contextlib.contextmanager
def warnings_as_errors():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        yield


class FakeAtom:
    """Stands in for ccdc.molecule.Atom."""

    def __init__(
        self,
        label,
        coords,
        atomic_symbol="C",
        atomic_number=6,
        occupancy=1.0,
        atomic_weight=12.011,
    ):
        self.label = label
        self.fractional_coordinates = coords
        self.atomic_symbol = atomic_symbol
        self.atomic_number = atomic_number
        self.occupancy = occupancy
        self.atomic_weight = atomic_weight


class FakeMolecule:
    """Stands in for ccdc.molecule.Molecule."""

    def __init__(self, atoms):
        self.atoms = list(atoms)

    def remove_atoms(self, atoms):
        remove = {id(a) for a in atoms}
        self.atoms = [a for a in self.atoms if id(a) not in remove]

    @property
    def components(self):
        return [self]


class FakeGroup:
    """Stands in for ccdc.crystal.Crystal.Disorder.Group."""

    def __init__(self, id, atoms, occupancy=None):
        self.id = id
        self.atoms = list(atoms)
        self.occupancy = occupancy


class FakeAssembly:
    """Stands in for ccdc.crystal.Crystal.Disorder.Assembly."""

    def __init__(self, id, groups):
        self.id = id
        self.groups = tuple(groups)


class FakeDisorder:
    """Stands in for ccdc.crystal.Crystal.Disorder."""

    def __init__(self, assemblies, is_suppressed=False):
        self.assemblies = tuple(assemblies)
        self.is_suppressed = is_suppressed


class FakeCrystal:
    """Stands in for ccdc.crystal.Crystal."""

    def __init__(
        self,
        atoms,
        disorder=None,
        identifier="TEST01",
        symmetry_operators=("x,y,z",),
        cell_lengths=(10.0, 10.0, 10.0),
        cell_angles=(90.0, 90.0, 90.0),
    ):
        self._atoms = list(atoms)
        self.disorder = disorder
        self.identifier = identifier
        self.symmetry_operators = tuple(symmetry_operators)
        self.cell_lengths = tuple(cell_lengths)
        self.cell_angles = tuple(cell_angles)

    @property
    def has_disorder(self):
        return self.disorder is not None

    @property
    def asymmetric_unit_molecule(self):
        # ccdc rebuilds this on each access; the reader mutates it.
        return FakeMolecule(self._atoms)


def group_indices(assembly):
    """Sorted index sets of an assembly's groups, for comparison."""
    return sorted(sorted(grp.indices) for grp in assembly.groups)


# --------------------------------------------------------------------
# Occupancies
# --------------------------------------------------------------------


def test_missing_occupancy_does_not_raise():
    """Atom.occupancy is None when the structure did not record one; one
    unusable value must not take the whole structure down."""

    crystal = FakeCrystal(
        [
            FakeAtom("C1", (0.0, 0.0, 0.0), occupancy=None),
            FakeAtom("C2", (0.5, 0.0, 0.0), occupancy=1.0),
        ]
    )

    pset = periodicset_from_ccdc_crystal(crystal)

    assert len(pset.motif) == 2
    assert pset.disorder is None


def test_group_occupancy_fills_in_unrecorded_site_occupancy():
    """ccdc states the occupancy of an alternative on the group, which is
    the only source left when the sites themselves have none."""

    atoms = [
        FakeAtom("C1", (0.0, 0.0, 0.0), occupancy=None),
        FakeAtom("C2", (0.1, 0.0, 0.0), occupancy=None),
    ]
    disorder = FakeDisorder(
        [
            FakeAssembly(
                "A",
                [
                    FakeGroup(1, [atoms[0]], occupancy=0.7),
                    FakeGroup(2, [atoms[1]], occupancy=0.3),
                ],
            )
        ]
    )

    pset = periodicset_from_ccdc_crystal(FakeCrystal(atoms, disorder))

    assert pset.disorder is not None
    occs = {tuple(grp.indices): grp.occupancy for grp in pset.disorder[0].groups}
    assert occs == pytest.approx({(0,): 0.7, (1,): 0.3})


def test_recorded_site_occupancy_beats_group_occupancy():
    """A group value is a summary of its atoms' occupancies, not a
    correction to them."""

    atoms = [
        FakeAtom("C1", (0.0, 0.0, 0.0), occupancy=0.6),
        FakeAtom("C2", (0.1, 0.0, 0.0), occupancy=0.4),
    ]
    disorder = FakeDisorder(
        [
            FakeAssembly(
                "A",
                [
                    FakeGroup(1, [atoms[0]], occupancy=0.9),
                    FakeGroup(2, [atoms[1]], occupancy=0.1),
                ],
            )
        ]
    )

    pset = periodicset_from_ccdc_crystal(FakeCrystal(atoms, disorder))

    assert pset.disorder is not None
    occs = {tuple(grp.indices): grp.occupancy for grp in pset.disorder[0].groups}
    assert occs == pytest.approx({(0,): 0.6, (1,): 0.4})


def test_skip_disorder_catches_partial_occupancy():
    """A partial occupancy ccdc does not call disorder must still be
    skipped, as it is by the gemmi backend."""

    crystal = FakeCrystal(
        [
            FakeAtom("C1", (0.0, 0.0, 0.0), occupancy=0.5),
            FakeAtom("C2", (0.5, 0.0, 0.0), occupancy=1.0),
        ]
    )

    with pytest.raises(ParseError):
        periodicset_from_ccdc_crystal(crystal, skip_disorder=True)


# --------------------------------------------------------------------
# Assembly and group ids
# --------------------------------------------------------------------


@pytest.mark.parametrize("blank", [None, "", " "])
def test_assemblies_without_ids_are_not_fused(blank):
    """ccdc numbers groups within their assembly, so two assemblies
    which give nothing for an id must not have their group 1s treated as
    one set of alternatives."""

    atoms = [
        FakeAtom("C1", (0.0, 0.0, 0.0), occupancy=0.5),
        FakeAtom("C2", (0.1, 0.0, 0.0), occupancy=0.5),
        FakeAtom("C3", (0.0, 0.5, 0.0), occupancy=0.5),
        FakeAtom("C4", (0.1, 0.5, 0.0), occupancy=0.5),
    ]
    disorder = FakeDisorder(
        [
            FakeAssembly(blank, [FakeGroup(1, [atoms[0]]), FakeGroup(2, [atoms[1]])]),
            FakeAssembly(blank, [FakeGroup(1, [atoms[2]]), FakeGroup(2, [atoms[3]])]),
        ]
    )

    pset = periodicset_from_ccdc_crystal(FakeCrystal(atoms, disorder))

    assert pset.disorder is not None
    assert len(pset.disorder) == 2, "independent assemblies were fused"
    assert sorted(group_indices(a) for a in pset.disorder) == [
        [[0], [1]],
        [[2], [3]],
    ]
    assert all(asm.source == "cif" for asm in pset.disorder)


def test_distinct_assembly_ids_are_kept():
    atoms = [
        FakeAtom("C1", (0.0, 0.0, 0.0), occupancy=0.5),
        FakeAtom("C2", (0.1, 0.0, 0.0), occupancy=0.5),
        FakeAtom("C3", (0.0, 0.5, 0.0), occupancy=0.5),
        FakeAtom("C4", (0.1, 0.5, 0.0), occupancy=0.5),
    ]
    disorder = FakeDisorder(
        [
            FakeAssembly("A", [FakeGroup(1, [atoms[0]]), FakeGroup(2, [atoms[1]])]),
            FakeAssembly("B", [FakeGroup(1, [atoms[2]]), FakeGroup(2, [atoms[3]])]),
        ]
    )

    pset = periodicset_from_ccdc_crystal(FakeCrystal(atoms, disorder))

    assert pset.disorder is not None
    assert len(pset.disorder) == 2
    assert sorted(group_indices(a) for a in pset.disorder) == [
        [[0], [1]],
        [[2], [3]],
    ]


# --------------------------------------------------------------------
# Matching the disorder model onto the asymmetric unit
# --------------------------------------------------------------------


def test_codes_follow_labels_not_positions_after_removals():
    """atom.index is a position in the molecule the disorder model was
    built from, so the codes must survive hydrogens being removed."""

    hydrogens = [
        FakeAtom("H1", (0.7, 0.0, 0.0), atomic_symbol="H", atomic_number=1),
        FakeAtom("H2", (0.8, 0.0, 0.0), atomic_symbol="H", atomic_number=1),
    ]
    heavy = [
        FakeAtom("C1", (0.0, 0.0, 0.0), occupancy=0.5),
        FakeAtom("C2", (0.1, 0.0, 0.0), occupancy=0.5),
    ]
    disorder = FakeDisorder(
        [FakeAssembly("A", [FakeGroup(1, [heavy[0]]), FakeGroup(2, [heavy[1]])])]
    )
    crystal = FakeCrystal(hydrogens + heavy, disorder)

    pset = periodicset_from_ccdc_crystal(crystal, remove_hydrogens=True)

    assert pset.disorder is not None
    assert len(pset.disorder) == 1
    # C1 and C2 are sites 0 and 1 once the hydrogens are gone.
    assert group_indices(pset.disorder[0]) == [[0], [1]]


def test_duplicate_labels_fall_back_to_positions():
    """ccdc does not guarantee unique labels, so an ambiguous label must
    not silently drop a site out of its assembly."""

    atoms = [
        FakeAtom("C1", (0.0, 0.0, 0.0), occupancy=0.5),
        FakeAtom("C1", (0.1, 0.0, 0.0), occupancy=0.5),
    ]
    disorder = FakeDisorder(
        [FakeAssembly("A", [FakeGroup(1, [atoms[0]]), FakeGroup(2, [atoms[1]])])]
    )

    pset = periodicset_from_ccdc_crystal(FakeCrystal(atoms, disorder))

    assert pset.disorder is not None
    assert len(pset.disorder) == 1
    assert group_indices(pset.disorder[0]) == [[0], [1]]
    assert pset.disorder[0].source == "cif"


def test_unmatchable_disorder_model_warns_and_falls_back():
    """Nothing matching means the codes were lost, not that the sites are
    ordered."""

    atoms = [
        FakeAtom("C1", (0.0, 0.0, 0.0), occupancy=0.5),
        FakeAtom("C2", (0.1, 0.0, 0.0), occupancy=0.5),
    ]
    # A model whose atoms share no label with the asym unit and have no
    # coordinates either, so they cannot be restored into it.
    ghosts = [
        FakeAtom("Zz1", None, occupancy=0.5),
        FakeAtom("Zz2", None, occupancy=0.5),
    ]
    disorder = FakeDisorder(
        [FakeAssembly("A", [FakeGroup(1, [ghosts[0]]), FakeGroup(2, [ghosts[1]])])]
    )

    with pytest.warns(UserWarning, match="could not be matched"):
        pset = periodicset_from_ccdc_crystal(FakeCrystal(atoms, disorder))

    # Fell back to the occupancy heuristics rather than reporting none.
    assert pset.disorder is not None
    assert pset.disorder[0].source != "cif"


def test_group_absent_from_asymmetric_unit_warns():
    """One alternative of an assembly missing from the asymmetric unit,
    with nothing removed, means the sites never arrived."""

    present = FakeAtom("C1", (0.0, 0.0, 0.0), occupancy=0.6)
    # No coordinates, so it cannot be restored into the asymmetric unit.
    absent = FakeAtom("C2", None, occupancy=0.4)
    disorder = FakeDisorder(
        [FakeAssembly("A", [FakeGroup(1, [present]), FakeGroup(2, [absent])])]
    )

    with pytest.warns(UserWarning, match="no corresponding site"):
        periodicset_from_ccdc_crystal(FakeCrystal([present], disorder))


def test_no_warning_when_group_lost_to_removals():
    """A disorder group of hydrogens matching nothing under
    remove_hydrogens is expected, not a sign of a lost alternative."""

    heavy = FakeAtom("C1", (0.0, 0.0, 0.0), occupancy=0.6)
    hydrogens = [
        FakeAtom("H1", (0.2, 0.0, 0.0), atomic_symbol="H", atomic_number=1,
                 occupancy=0.6),
        FakeAtom("H2", (0.3, 0.0, 0.0), atomic_symbol="H", atomic_number=1,
                 occupancy=0.4),
    ]
    disorder = FakeDisorder(
        [
            FakeAssembly(
                "A",
                [FakeGroup(1, [heavy, hydrogens[0]]), FakeGroup(2, [hydrogens[1]])],
            )
        ]
    )
    crystal = FakeCrystal([heavy] + hydrogens, disorder)

    with warnings_as_errors():
        periodicset_from_ccdc_crystal(crystal, remove_hydrogens=True)


def test_suppressed_disorder_model_is_not_read_as_assemblies():
    """A suppressed model holds every minority atom in one assembly of
    two groups; reading its ids would couple unrelated disorder."""

    atoms = [
        FakeAtom("C1", (0.0, 0.0, 0.0), occupancy=0.5),
        FakeAtom("C2", (0.1, 0.0, 0.0), occupancy=0.5),
        FakeAtom("C3", (0.0, 0.5, 0.0), occupancy=0.5),
        FakeAtom("C4", (0.1, 0.5, 0.0), occupancy=0.5),
    ]
    disorder = FakeDisorder(
        [FakeAssembly("", [FakeGroup(1, atoms), FakeGroup(2, [])])],
        is_suppressed=True,
    )

    pset = periodicset_from_ccdc_crystal(FakeCrystal(atoms, disorder))

    assert pset.disorder is not None
    # Inferred from occupancies and positions, not from the model.
    assert all(asm.source != "cif" for asm in pset.disorder)


# --------------------------------------------------------------------
# Restoring the alternatives ccdc leaves out of the asymmetric unit
# --------------------------------------------------------------------


def test_minority_alternative_is_restored_to_the_asymmetric_unit():
    """ccdc's asymmetric unit describes one configuration of the crystal
    and so carries the majority alternative of a disordered site only.
    The others have to come back from the disorder model, or the motif
    loses them and every assembly is left with a single group."""

    major = FakeAtom("C1", (0.0, 0.0, 0.0), occupancy=0.6)
    minor = FakeAtom("C1A", (0.05, 0.0, 0.0), occupancy=0.4)
    disorder = FakeDisorder(
        [FakeAssembly("A", [FakeGroup(1, [major]), FakeGroup(2, [minor])])]
    )

    with warnings_as_errors():
        pset = periodicset_from_ccdc_crystal(FakeCrystal([major], disorder))

    assert len(pset.motif) == 2
    assert list(pset.labels) == ["C1", "C1A"]
    assert len(pset.disorder) == 1
    assert pset.disorder[0].source == "cif"
    assert group_indices(pset.disorder[0]) == [[0], [1]]


def test_suppressed_model_atoms_are_restored_though_its_codes_are_not():
    """Suppression makes a model's assembly and group ids meaningless,
    not its atoms. GAZJAW's Co/Li substitutions reach the motif only
    through them: without the minority atom a substituted site reads as
    the majority element against a vacancy."""

    co = FakeAtom(
        "Co1", (0.25, 0.25, 0.25), atomic_symbol="Co", atomic_number=27,
        occupancy=0.6,
    )
    # ccdc marks an atom outside the majority configuration with '?'.
    li = FakeAtom(
        "Li1?", (0.25, 0.25, 0.25), atomic_symbol="Li", atomic_number=3,
        occupancy=0.4,
    )
    disorder = FakeDisorder(
        [FakeAssembly("", [FakeGroup(1, [li]), FakeGroup(2, [])])],
        is_suppressed=True,
    )

    pset = periodicset_from_ccdc_crystal(FakeCrystal([co], disorder))

    assert list(pset.labels) == ["Co1", "Li1"]
    assert len(pset.disorder) == 1
    assembly = pset.disorder[0]
    assert assembly.is_substitutional
    # Inferred from the restored positions, not from the suppressed ids.
    assert assembly.source != "cif"
    assert sorted(g.occupancy for g in assembly.groups) == pytest.approx([0.4, 0.6])


def test_restored_site_of_the_same_element_in_the_same_place_is_a_duplicate():
    """Substitutional partners coincide, so position alone cannot mark a
    duplicate - but a site of the same element there is the same site
    under another name."""

    atom = FakeAtom("C1", (0.0, 0.0, 0.0), occupancy=0.5)
    twin = FakeAtom("Cx", (0.0, 0.0, 0.0), occupancy=0.5)
    disorder = FakeDisorder(
        [FakeAssembly("A", [FakeGroup(1, [atom]), FakeGroup(2, [twin])])]
    )

    pset = periodicset_from_ccdc_crystal(FakeCrystal([atom], disorder))

    assert len(pset.motif) == 1


def test_restored_hydrogens_are_dropped_under_remove_hydrogens():
    """An alternative which is a hydrogen is subject to the same removal
    as one read from the asymmetric unit."""

    heavy = FakeAtom("C1", (0.0, 0.0, 0.0), occupancy=0.6)
    h_major = FakeAtom(
        "H1", (0.05, 0.0, 0.0), atomic_symbol="H", atomic_number=1, occupancy=0.6
    )
    h_minor = FakeAtom(
        "H1A", (0.08, 0.0, 0.0), atomic_symbol="H", atomic_number=1, occupancy=0.4
    )
    disorder = FakeDisorder(
        [
            FakeAssembly(
                "A", [FakeGroup(1, [heavy, h_major]), FakeGroup(2, [h_minor])]
            )
        ]
    )
    crystal = FakeCrystal([heavy, h_major], disorder)

    pset = periodicset_from_ccdc_crystal(crystal, remove_hydrogens=True)

    assert list(pset.labels) == ["C1"]


def test_far_alternative_is_not_restored_under_heaviest_component():
    """Only one component survives heaviest_component, while the disorder
    model still describes the whole asymmetric unit. An alternative
    nowhere near a surviving site belongs to what was discarded."""

    kept = FakeAtom("C1", (0.0, 0.0, 0.0), occupancy=0.6)
    solvent_alt = FakeAtom("C9A", (0.5, 0.5, 0.5), occupancy=0.4)
    disorder = FakeDisorder(
        [FakeAssembly("A", [FakeGroup(1, [kept]), FakeGroup(2, [solvent_alt])])]
    )
    crystal = FakeCrystal([kept], disorder)

    pset = periodicset_from_ccdc_crystal(crystal, heaviest_component=True)

    assert list(pset.labels) == ["C1"]


def test_symmetry_copy_of_a_kept_site_is_not_restored():
    """ccdc's disorder model describes the whole disordered molecule,
    which can reach past one asymmetric unit, so some of its atoms are
    symmetry copies of sites already read. Restoring one puts a second
    motif point exactly on top of an existing one, i.e. a zero into the
    structure's PDD."""

    site = FakeAtom("C1", (0.1, 0.2, 0.3), occupancy=0.5)
    # The image of C1 under -x,-y,-z, under its own label.
    copy = FakeAtom("C1B", (-0.1, -0.2, -0.3), occupancy=0.5)
    disorder = FakeDisorder(
        [FakeAssembly("A", [FakeGroup(1, [site]), FakeGroup(2, [copy])])]
    )
    crystal = FakeCrystal(
        [site], disorder, symmetry_operators=("x,y,z", "-x,-y,-z")
    )

    with warnings_as_errors():
        pset = periodicset_from_ccdc_crystal(crystal)

    assert list(pset.labels) == ["C1"]
    # Both alternatives are still in the motif - as images of one site.
    assert len(pset.motif) == 2
    dists = np.linalg.norm(pset.motif[0] - pset.motif[1])
    assert dists > 1e-3


def test_disorder_across_a_symmetry_element_does_not_warn():
    """The alternatives being symmetry images of one another is one
    crystallographic site at partial occupancy, not an alternative
    missing from the motif."""

    a = FakeAtom("H1", (0.1, 0.0, 0.0), atomic_symbol="H", atomic_number=1,
                 occupancy=0.5)
    b = FakeAtom("H2", (-0.1, 0.0, 0.0), atomic_symbol="H", atomic_number=1,
                 occupancy=0.5)
    disorder = FakeDisorder(
        [FakeAssembly("A", [FakeGroup(1, [a]), FakeGroup(2, [b])])]
    )
    crystal = FakeCrystal([a], disorder, symmetry_operators=("x,y,z", "-x,-y,-z"))

    with warnings_as_errors():
        periodicset_from_ccdc_crystal(crystal)


def test_alternative_restored_when_a_symmetry_copy_is_of_another_element():
    """Coinciding with a symmetry image of a site of a *different*
    element is substitution, not duplication."""

    co = FakeAtom("Co1", (0.1, 0.2, 0.3), atomic_symbol="Co",
                  atomic_number=27, occupancy=0.6)
    li = FakeAtom("Li1", (0.1, 0.2, 0.3), atomic_symbol="Li",
                  atomic_number=3, occupancy=0.4)
    disorder = FakeDisorder(
        [FakeAssembly("", [FakeGroup(1, [li]), FakeGroup(2, [])])],
        is_suppressed=True,
    )
    crystal = FakeCrystal([co], disorder, symmetry_operators=("x,y,z", "-x,-y,-z"))

    pset = periodicset_from_ccdc_crystal(crystal)

    assert sorted(pset.labels) == ["Co1", "Li1"]
    assert pset.disorder[0].is_substitutional


def test_label_naming_a_different_element_does_not_match_that_site():
    """ccdc labels are not guaranteed unique across elements: CSD KAKMIX
    has a V atom labelled Co1 beside the Co atom of that name. Trusting
    the label puts a mixed metal site's codes on whichever it hit."""

    co = FakeAtom("Co1", (0.1, 0.1, 0.1), atomic_symbol="Co",
                  atomic_number=27, occupancy=0.6)
    v = FakeAtom("V1", (0.5, 0.5, 0.5), atomic_symbol="V",
                 atomic_number=23, occupancy=0.4)
    # The model names this vanadium "Co1" as well.
    ghost = FakeAtom("Co1", (0.5, 0.5, 0.5), atomic_symbol="V",
                     atomic_number=23, occupancy=0.4)
    disorder = FakeDisorder(
        [FakeAssembly("A", [FakeGroup(1, [co]), FakeGroup(2, [ghost])])]
    )

    pset = periodicset_from_ccdc_crystal(FakeCrystal([co, v], disorder))

    assert list(pset.labels) == ["Co1", "V1"]
    # The V atom was matched by position, not by its misleading label.
    assert len(pset.disorder) == 1
    assert group_indices(pset.disorder[0]) == [[0], [1]]


# --------------------------------------------------------------------
# Pairing alternatives by configuration (suppressed models)
# --------------------------------------------------------------------


def _suppressed_crystal(majority_atoms, minority_atoms, **kwargs):
    """A crystal whose asymmetric unit holds ``majority_atoms`` and whose
    suppressed disorder model names ``minority_atoms``, the shape ccdc
    gives a suppressed entry: one assembly of two groups, the first
    holding every minority atom, the second empty."""

    disorder = FakeDisorder(
        [FakeAssembly("", [FakeGroup(1, minority_atoms), FakeGroup(2, [])])],
        is_suppressed=True,
    )
    return FakeCrystal(list(majority_atoms), disorder, **kwargs)


def test_suppressed_half_half_disorder_pairs_kept_with_restored():
    """A suppressed 0.5:0.5 model gives every site the same occupancy, so
    nothing in the occupancies says which sites are alternatives of
    which. The configurations do: the asymmetric unit is one
    self-consistent configuration, and the restored atoms replace it."""

    maj = FakeAtom("F1", (0.10, 0.10, 0.10), atomic_symbol="F",
                   atomic_number=9, occupancy=0.5)
    mino = FakeAtom("F1A", (0.16, 0.10, 0.10), atomic_symbol="F",
                    atomic_number=9, occupancy=0.5)

    pset = periodicset_from_ccdc_crystal(_suppressed_crystal([maj], [mino]))

    assert len(pset.disorder) == 1
    assembly = pset.disorder[0]
    assert assembly.source == "majority-minority"
    assert len(assembly.groups) == 2
    assert group_indices(assembly) == [[0], [1]]
    assert all(g.occupancy == pytest.approx(0.5) for g in assembly.groups)


def test_distant_regions_of_suppressed_disorder_stay_separate_assemblies():
    """Regions of disorder further apart than a bond length are
    independent: fusing them halves the number of configurations."""

    maj1 = FakeAtom("F1", (0.10, 0.10, 0.10), atomic_symbol="F",
                    atomic_number=9, occupancy=0.5)
    min1 = FakeAtom("F1A", (0.16, 0.10, 0.10), atomic_symbol="F",
                    atomic_number=9, occupancy=0.5)
    maj2 = FakeAtom("F2", (0.60, 0.60, 0.60), atomic_symbol="F",
                    atomic_number=9, occupancy=0.5)
    min2 = FakeAtom("F2A", (0.66, 0.60, 0.60), atomic_symbol="F",
                    atomic_number=9, occupancy=0.5)

    pset = periodicset_from_ccdc_crystal(
        _suppressed_crystal([maj1, maj2], [min1, min2])
    )

    assert len(pset.disorder) == 2
    labels = list(pset.labels)
    pairs = sorted(
        sorted(labels[g.indices[0]] for g in asm.groups)
        for asm in pset.disorder
    )
    assert pairs == [["F1", "F1A"], ["F2", "F2A"]]


def test_fused_regions_salvaged_by_complementary_occupancies():
    """Two regions close enough to fuse spatially push the group
    occupancies past one; complementary majority/minority pairs within
    the region are still unambiguous."""

    maj1 = FakeAtom("C1", (0.10, 0.10, 0.10), occupancy=0.6)
    maj2 = FakeAtom("C2", (0.15, 0.10, 0.10), occupancy=0.7)
    min1 = FakeAtom("C1A", (0.10, 0.15, 0.10), occupancy=0.4)
    min2 = FakeAtom("C2A", (0.15, 0.15, 0.10), occupancy=0.3)

    pset = periodicset_from_ccdc_crystal(
        _suppressed_crystal([maj1, maj2], [min1, min2])
    )

    labels = list(pset.labels)
    pairings = {
        frozenset(
            (labels[g.indices[0]], round(g.occupancy, 2)) for g in asm.groups
        )
        for asm in pset.disorder
        if asm.source == "majority-minority"
    }
    assert frozenset({("C1", 0.6), ("C1A", 0.4)}) in pairings
    assert frozenset({("C2", 0.7), ("C2A", 0.3)}) in pairings


def test_three_way_suppressed_disorder_groups_minorities_by_occupancy():
    """A majority alternative and two minority ones of distinct
    occupancy are one assembly of three groups."""

    maj = FakeAtom("C1", (0.10, 0.10, 0.10), occupancy=0.70)
    min1 = FakeAtom("C1A", (0.16, 0.10, 0.10), occupancy=0.20)
    min2 = FakeAtom("C1B", (0.10, 0.16, 0.10), occupancy=0.10)

    pset = periodicset_from_ccdc_crystal(
        _suppressed_crystal([maj], [min1, min2])
    )

    assert len(pset.disorder) == 1
    assembly = pset.disorder[0]
    assert assembly.source == "majority-minority"
    assert sorted(round(g.occupancy, 2) for g in assembly.groups) == [
        0.10, 0.20, 0.70
    ]
