"""Tests for reading disorder assemblies and groups from CIFs.

An *assembly* is a set of atoms whose disorder is coupled; the *groups*
of an assembly are its mutually exclusive alternative configurations.
"""

import numpy as np
import pytest

import amd
from amd.io import (
    _disorder_assemblies,
    _group_indices_by_occupancy,
    _tuples_sum_to_one,
)


CELL = np.eye(3) * 10.0


def build_disorder(
    coords,
    occupancies,
    types=None,
    assemblies=None,
    groups=None,
    multiplicities=None,
    n_symmetries=1,
    site_sym_orders=None,
):
    """Call _disorder_assemblies with sensible defaults for a P1 cell."""

    coords = np.array(coords, dtype=np.float64)
    n = len(coords)
    if types is None:
        types = [6] * n
    if assemblies is None:
        assemblies = [None] * n
    if groups is None:
        groups = [None] * n
    if multiplicities is None:
        multiplicities = np.ones((n,), dtype=np.uint64)
    return _disorder_assemblies(
        coords,
        multiplicities,
        n_symmetries,
        CELL,
        types,
        assemblies,
        groups,
        np.array(occupancies, dtype=np.float64),
        site_sym_orders=site_sym_orders,
    )


def group_indices(assembly):
    """Sorted index sets of an assembly's groups, for comparison."""
    return sorted(sorted(grp.indices) for grp in assembly.groups)


# --------------------------------------------------------------------
# Explicit CIF disorder codes are authoritative
# --------------------------------------------------------------------


def test_groups_without_assembly_are_one_implicit_assembly():
    """_atom_site_disorder_assembly is only required when a structure has
    more than one assembly, so group codes alone must still be read."""

    disorder = build_disorder(
        [[0, 0, 0], [0.5, 0, 0], [0, 0.5, 0], [0.5, 0.5, 0]],
        [0.6, 0.6, 0.4, 0.4],
        groups=["1", "1", "2", "2"],
    )

    assert disorder is not None, "group codes with no assembly were discarded"
    assert len(disorder) == 1
    assert group_indices(disorder[0]) == [[0, 1], [2, 3]]
    assert disorder[0].source == "cif"


def test_explicit_groups_read_without_any_occupancies():
    """A CIF may give disorder groups and no _atom_site_occupancy loop at
    all, in which case every occupancy defaults to 1."""

    disorder = build_disorder(
        [[0, 0, 0], [0.5, 0, 0], [0, 0.5, 0], [0.5, 0.5, 0]],
        [1.0, 1.0, 1.0, 1.0],
        assemblies=["A"] * 4,
        groups=["1", "1", "2", "2"],
    )

    assert disorder is not None, "groups were dropped for want of occupancies"
    assert group_indices(disorder[0]) == [[0, 1], [2, 3]]
    assert [grp.occupancy for grp in disorder[0].groups] == [0.5, 0.5]


def test_fully_occupied_site_kept_inside_explicit_group():
    """A group can legitimately contain a site refined at occupancy 1."""

    disorder = build_disorder(
        [[0, 0, 0], [0.5, 0, 0], [0, 0.5, 0]],
        [1.0, 0.6, 0.4],
        assemblies=["A", "A", "A"],
        groups=["1", "1", "2"],
    )

    assert disorder is not None
    assert group_indices(disorder[0]) == [[0, 1], [2]]


def test_multiple_assemblies_stay_separate():
    disorder = build_disorder(
        [[0, 0, 0], [0.5, 0, 0], [0, 0.5, 0], [0.5, 0.5, 0]],
        [0.6, 0.4, 0.7, 0.3],
        assemblies=["A", "A", "B", "B"],
        groups=["1", "2", "1", "2"],
    )

    assert len(disorder) == 2
    assert sorted(group_indices(a) for a in disorder) == [
        [[0], [1]],
        [[2], [3]],
    ]


def test_quoted_codes_are_unquoted():
    disorder = build_disorder(
        [[0, 0, 0], [0.5, 0, 0]],
        [0.6, 0.4],
        assemblies=["'A'", "A"],
        groups=["'1'", "2"],
    )

    assert len(disorder) == 1, "quoted and unquoted codes were treated as different"
    assert group_indices(disorder[0]) == [[0], [1]]


def test_assembly_without_group_split_by_occupancy():
    """An assembly code with no group codes says the sites are coupled
    but not which alternative each belongs to."""

    disorder = build_disorder(
        [[0, 0, 0], [0.5, 0, 0]], [0.6, 0.4], assemblies=["A", "A"]
    )

    assert disorder is not None
    assert group_indices(disorder[0]) == [[0], [1]]
    assert disorder[0].source == "cif-assembly-only"


def test_inapplicable_group_excluded_from_heuristics():
    """"." means the atom takes no part in disorder. It must not be
    confused with "?", which means nothing was stated."""

    disorder = build_disorder(
        [[0, 0, 0], [0.5, 0, 0], [0, 0.5, 0]],
        [0.5, 0.5, 0.4],
        groups=["1", "2", "."],
    )

    assigned = {i for asm in disorder for grp in asm.groups for i in grp.indices}
    assert 2 not in assigned, "site explicitly marked '.' was assigned disorder"
    assert assigned == {0, 1}


def test_unknown_group_still_reaches_heuristics():
    disorder = build_disorder(
        [[0, 0, 0], [0.5, 0, 0], [0, 0.5, 0]],
        [0.5, 0.5, 0.4],
        groups=["1", "2", "?"],
    )

    assigned = {i for asm in disorder for grp in asm.groups for i in grp.indices}
    assert 2 in assigned, "site marked '?' was treated as ordered"


# --------------------------------------------------------------------
# Substitutional vs positional disorder
# --------------------------------------------------------------------


def test_different_elements_on_one_site_are_substitutional():
    disorder = build_disorder(
        [[0.25, 0.25, 0.25], [0.25, 0.25, 0.25]], [0.7, 0.3], types=[26, 25]
    )

    assert len(disorder) == 1
    assert disorder[0].is_substitutional
    assert [grp.occupancy for grp in disorder[0].groups] == [0.7, 0.3]


def test_same_element_displaced_is_positional():
    """Two sites of one element pulled apart by a real distance are
    alternative positions, not alternative identities."""

    disorder = build_disorder(
        [[0.25, 0.25, 0.25], [0.2505, 0.25, 0.25]], [0.5, 0.5], types=[6, 6]
    )

    assert len(disorder) == 1
    assert not disorder[0].is_substitutional
    assert group_indices(disorder[0]) == [[0], [1]]


def test_single_linkage_chain_is_not_one_cluster():
    """Sites strung out in a chain, each just inside the overlap
    tolerance of the next, are not all mutually overlapping and must not
    be merged into one substitutional site."""

    step = 0.029  # 0.29 Angstroms in a 10 Angstrom cell
    coords = [[i * step, 0, 0] for i in range(6)]
    disorder = build_disorder(coords, [0.2] * 6, types=[6] * 6)

    for asm in disorder or []:
        assert not asm.is_substitutional, "a chained cluster was called substitutional"


def test_overlapping_sites_over_occupied_are_not_an_assembly():
    """Sites sharing a position whose occupancies exceed one cannot be
    alternatives for that position."""

    disorder = build_disorder(
        [[0.25, 0.25, 0.25], [0.25, 0.25, 0.25]], [0.8, 0.7], types=[26, 25]
    )

    for asm in disorder or []:
        assert not asm.is_substitutional


# --------------------------------------------------------------------
# Special positions
# --------------------------------------------------------------------


def test_occupancy_accounting_for_special_position_is_not_disorder():
    """An atom on an inversion centre in P-1 is generated once rather
    than twice, and is commonly written with occupancy 0.5 despite being
    fully ordered."""

    disorder = build_disorder(
        [[0, 0, 0], [0.3, 0.3, 0.3]],
        [0.5, 1.0],
        types=[8, 6],
        multiplicities=np.array([1, 2], dtype=np.uint64),
        n_symmetries=2,
    )

    assert disorder is None


def test_half_occupancy_on_general_position_is_disorder():
    disorder = build_disorder(
        [[0.11, 0.13, 0.17], [0.3, 0.3, 0.3]],
        [0.5, 1.0],
        types=[8, 6],
        multiplicities=np.array([2, 2], dtype=np.uint64),
        n_symmetries=2,
    )

    assert disorder is not None
    assigned = {i for asm in disorder for grp in asm.groups for i in grp.indices}
    assert assigned == {0}


def test_stated_site_symmetry_order_takes_precedence():
    """When a file records site symmetry order separately, occupancy is a
    true site occupancy and must be believed rather than second guessed.
    """

    disorder = build_disorder(
        [[0, 0, 0], [0.3, 0.3, 0.3]],
        [0.5, 1.0],
        types=[8, 6],
        multiplicities=np.array([1, 2], dtype=np.uint64),
        n_symmetries=2,
        site_sym_orders=[2.0, 1.0],
    )

    assert disorder is not None, "a stated occupancy of 0.5 was explained away"
    assigned = {i for asm in disorder for grp in asm.groups for i in grp.indices}
    assert assigned == {0}


# --------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------


@pytest.mark.parametrize(
    "occupancies,expected",
    [
        ([0.4, 0.6], [{0, 1}]),
        ([0.5, 0.5], [{0, 1}]),
        ([0.2, 0.3, 0.5], [{0, 1, 2}]),
        ([0.25, 0.75, 0.4, 0.6], [{0, 1}, {2, 3}]),
        ([0.5], []),
        ([0.2, 0.3], []),
    ],
)
def test_tuples_sum_to_one(occupancies, expected):
    found = [set(c) for c in _tuples_sum_to_one(np.array(occupancies))]
    assert sorted(map(sorted, found)) == sorted(map(sorted, expected))


def test_tuples_sum_to_one_uses_each_index_once():
    found = _tuples_sum_to_one(np.array([0.3, 0.7, 0.1, 0.2, 0.9]))
    flat = [i for combination in found for i in combination]
    assert len(flat) == len(set(flat))


def test_group_indices_by_occupancy():
    occs = np.array([0.4, 0.6, 0.4, 0.55])
    grouped = _group_indices_by_occupancy(occs, [0, 1, 2, 3])
    assert sorted(map(sorted, grouped)) == [[0, 2], [1], [3]]


def test_group_indices_by_occupancy_keeps_every_site():
    """Regression: a transposed pdist call collapsed this to one site."""

    occs = np.array([0.5] * 15)
    grouped = _group_indices_by_occupancy(occs, list(range(15)))
    assert sorted(i for grp in grouped for i in grp) == list(range(15))


# --------------------------------------------------------------------
# End to end, through CifReader
# --------------------------------------------------------------------


CIF_TEMPLATE = """data_test
_cell_length_a 10.0
_cell_length_b 10.0
_cell_length_c 10.0
_cell_angle_alpha 90.0
_cell_angle_beta 90.0
_cell_angle_gamma 90.0
loop_
_symmetry_equiv_pos_as_xyz
 'x, y, z'
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_occupancy
_atom_site_disorder_assembly
_atom_site_disorder_group
{sites}
"""


def write_cif(tmp_path, sites):
    path = tmp_path / "disorder.cif"
    path.write_text(CIF_TEMPLATE.format(sites="\n".join(sites)))
    return str(path)


def test_cifreader_reads_groups_without_assembly(tmp_path):
    path = write_cif(
        tmp_path,
        [
            "C1 C 0.10 0.10 0.10 0.6 . 1",
            "C2 C 0.20 0.20 0.20 0.6 . 1",
            "C3 C 0.30 0.30 0.30 0.4 . 2",
            "C4 C 0.40 0.40 0.40 0.4 . 2",
        ],
    )
    pset = list(amd.CifReader(path, show_warnings=False))[0]

    assert pset.disorder is not None, "CifReader discarded group-only disorder"
    assert len(pset.disorder) == 1
    assert sorted(sorted(g.indices) for g in pset.disorder[0].groups) == [
        [0, 1],
        [2, 3],
    ]
    assert pset.has_positional_disorder()


def test_cifreader_substitutional_site(tmp_path):
    path = write_cif(
        tmp_path,
        [
            "Fe1 Fe 0.25 0.25 0.25 0.7 . .",
            "Mn1 Mn 0.25 0.25 0.25 0.3 . .",
            "O1 O 0.75 0.75 0.75 1.0 . .",
        ],
    )
    pset = list(amd.CifReader(path, show_warnings=False))[0]

    assert pset.has_substitutional_disorder()


def test_cifreader_ordered_structure_has_no_disorder(tmp_path):
    path = write_cif(
        tmp_path,
        [
            "C1 C 0.10 0.10 0.10 1.0 . .",
            "C2 C 0.60 0.60 0.60 1.0 . .",
        ],
    )
    pset = list(amd.CifReader(path, show_warnings=False))[0]

    assert pset.disorder is None


def test_cifreader_disordered_solvent_keeps_every_atom(data_dir):
    """Regression: only the first of the fifteen half-occupancy solvent
    atoms of this structure used to be recorded."""

    path = str(data_dir / "T2-alpha-solvent.cif")
    pset = list(amd.CifReader(path, show_warnings=False))[0]

    assert pset.disorder is not None
    assigned = {i for asm in pset.disorder for grp in asm.groups for i in grp.indices}
    partial = {i for i in range(pset.n_asym) if pset.occupancies[i] < 1}

    assert len(assigned) == 15
    assert assigned == partial
