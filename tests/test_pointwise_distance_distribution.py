"""Tests for the PDD object and its disorder handling.

A disordered crystal is a probability distribution over ordered ones, so
its PDD is a probability distribution over the PDDs of its
configurations. These check the properties that makes worth having:
above all that it is *continuous* in the occupancies, so that a crystal
refined at 0.99/0.01 sits next to the ordered crystal rather than half a
configuration away from it.
"""

import warnings

import numpy as np
import pytest

import amd
from amd.periodicset import PeriodicSet, DisorderAssembly, DisorderGroup
from amd.pointwise_distance_distribution import (
    PointwiseDistanceDistribution,
    WeightedVectors,
    centroid_lb,
    compare_structures,
    component_costs,
    emd,
)

CELL = np.eye(3) * 10.0
K = 6

# Asymmetric positions, so that the two alternatives below are genuinely
# different crystals whose PDDs do not coincide.
A = [0.0, 0.0, 0.0]
B = [0.31, 0.11, 0.23]
G1 = [0.62, 0.17, 0.27]
G2 = [0.44, 0.71, 0.53]


def build(coords, disorder=None, name="test"):
    """A P1 PeriodicSet, with the dtypes the readers produce."""
    motif = np.array(coords, dtype=np.float64) @ CELL
    n = len(motif)
    return PeriodicSet(
        motif,
        CELL,
        name=name,
        asym_unit=np.arange(n, dtype=np.uint64),
        multiplicities=np.ones(n, dtype=np.uint64),
        types=np.array([6] * n, dtype=np.uint8),
        labels=[f"C{i + 1}" for i in range(n)],
        disorder=disorder,
    )


def two_group(p1, name=None):
    """Sites A, B ordered; a positional assembly choosing G1 or G2."""
    return build(
        [A, B, G1, G2],
        disorder=[
            DisorderAssembly(
                groups=[DisorderGroup([2], p1), DisorderGroup([3], 1.0 - p1)],
                is_substitutional=False,
            )
        ],
        name=name or f"two_group_{p1}",
    )


# --------------------------------------------------------------------
# The ordered case is unchanged
# --------------------------------------------------------------------


def test_ordered_crystal_is_one_component():
    pdd = PointwiseDistanceDistribution(build([A, B, G1, G2]), K)
    assert pdd.is_ordered
    assert pdd.n_components == 1
    assert float(pdd.weights[0]) == pytest.approx(1.0)


def test_ordered_matches_the_matrix_implementation():
    pset = build([A, B, G1, G2])
    pdd = PointwiseDistanceDistribution(pset, K)
    assert np.allclose(pdd.to_matrix(), amd.PDD(pset, K))
    assert np.allclose(pdd.AMD(), amd.AMD(pset, K))


def test_barycenter_is_the_amd():
    pset = build([A, B, G1, G2])
    pdd = PointwiseDistanceDistribution(pset, K)
    assert np.allclose(pdd.barycenter, pdd.AMD())
    assert np.allclose(pdd.barycenter, amd.AMD(pset, K))


def test_collapsing_rows_does_not_move_the_amd():
    """merge_duplicates takes the weight-weighted centroid, so the
    barycenter - the AMD - is preserved exactly."""
    pset = build([A, B, G1, G2])
    loose = PointwiseDistanceDistribution(pset, K, collapse=False)
    tight = loose.merge_duplicates(1.0)
    assert tight.components[0].n < loose.components[0].n
    assert np.allclose(loose.AMD(), tight.AMD())


# --------------------------------------------------------------------
# Continuity in occupancy: the property the mixture exists for
# --------------------------------------------------------------------


def test_two_alternatives_give_two_components():
    pdd = PointwiseDistanceDistribution(two_group(0.5), K)
    assert pdd.n_components == 2
    assert np.allclose(sorted(pdd.weights), [0.5, 0.5])


def test_component_weights_follow_occupancy():
    pdd = PointwiseDistanceDistribution(two_group(0.75), K)
    assert np.allclose(sorted(pdd.weights), [0.25, 0.75])


def test_each_component_is_the_pdd_of_its_configuration():
    """The class invariant: a component is a PDD in its own right."""
    pdd = PointwiseDistanceDistribution(two_group(0.75), K)
    for coords in ([A, B, G1], [A, B, G2]):
        ordered = PointwiseDistanceDistribution(build(coords), K).components[0]
        assert any(c.equals(ordered) for c in pdd.components)


def test_emd_to_the_ordered_limit_vanishes_with_occupancy():
    """As the majority occupancy goes to 1 the disordered crystal must
    converge on the ordered one it is becoming."""
    reference = PointwiseDistanceDistribution(build([A, B, G1]), K)
    distances = [
        emd(PointwiseDistanceDistribution(two_group(p), K),
            reference, disorder="coupled")
        for p in (0.5, 0.9, 0.99)
    ]
    assert distances == sorted(distances, reverse=True)
    # linear in the minority occupancy, so continuous at p = 1
    assert distances[1] == pytest.approx(distances[0] * 0.2, rel=1e-6)
    assert distances[2] == pytest.approx(distances[0] * 0.02, rel=1e-6)


def test_matrix_implementation_does_not_converge():
    """The behaviour this replaces: stacking configurations without
    weighting them by probability leaves a 0.99/0.01 crystal exactly as
    far from the ordered one as a 0.5/0.5 crystal is."""
    reference = amd.PDD(build([A, B, G1]), K)
    far = amd.EMD(amd.PDD(two_group(0.5), K), reference)
    near = amd.EMD(amd.PDD(two_group(0.99), K), reference)
    assert near == pytest.approx(far)
    assert near > 0.1


# --------------------------------------------------------------------
# Vacancies
# --------------------------------------------------------------------


def test_partial_occupancy_against_nothing_is_enumerated():
    """A one-group assembly means a site disordered against a vacancy.
    Without an explicit vacancy alternative it would be treated as fully
    occupied."""
    pset = build(
        [A, B, G1, G2],
        disorder=[
            DisorderAssembly(
                groups=[DisorderGroup([3], 0.4)], is_substitutional=False
            )
        ],
    )
    pdd = PointwiseDistanceDistribution(pset, K)
    assert pdd.n_components == 2

    # q(c) = P(c) * m_c / E[m]: four atoms 40% of the time, three the rest
    expected = np.array([0.4 * 4, 0.6 * 3])
    assert np.allclose(sorted(pdd.weights), sorted(expected / expected.sum()))


def test_never_occupied_alternative_is_dropped():
    """Regression: an alternative with occupancy at or below OCCUPANCY_TOL
    was filtered out of the enumeration without its sites being dropped,
    so a site which is never there was treated as always there."""
    pset = build(
        [A, B, G1, G2],
        disorder=[
            DisorderAssembly(
                groups=[DisorderGroup([2], 1e-8), DisorderGroup([3], 1.0 - 1e-8)],
                is_substitutional=False,
            )
        ],
    )
    pdd = PointwiseDistanceDistribution(pset, K)
    assert pdd.equals(PointwiseDistanceDistribution(build([A, B, G2]), K))
    # and the same when the assembly still has two live alternatives
    pset = build(
        [A, B, G1, G2],
        disorder=[
            DisorderAssembly(
                groups=[
                    DisorderGroup([1], 0.5),
                    DisorderGroup([2], 0.0),
                    DisorderGroup([3], 0.5),
                ],
                is_substitutional=False,
            )
        ],
    )
    pdd = PointwiseDistanceDistribution(pset, K, collapse=False)
    assert pdd.n_components == 2
    assert all(c.n == 2 for c in pdd.components)


def test_full_occupancy_collapses_to_the_ordered_pdd():
    pset = build(
        [A, B, G1, G2],
        disorder=[
            DisorderAssembly(
                groups=[DisorderGroup([3], 1.0)], is_substitutional=False
            )
        ],
    )
    assert PointwiseDistanceDistribution(pset, K).equals(
        PointwiseDistanceDistribution(build([A, B, G1, G2]), K)
    )


# --------------------------------------------------------------------
# Substitutional disorder
# --------------------------------------------------------------------


def test_substitutional_sites_collapse_to_one_point():
    """Several elements refined at one position are one point to a PDD,
    which does not see atomic type."""
    pset = build(
        [A, B, G1, G2, G2],
        disorder=[
            DisorderAssembly(
                groups=[DisorderGroup([3], 0.6), DisorderGroup([4], 0.4)],
                is_substitutional=True,
            )
        ],
    )
    pdd = PointwiseDistanceDistribution(pset, K)
    assert pdd.is_ordered
    assert pdd.components[0].equals(
        PointwiseDistanceDistribution(build([A, B, G1, G2]), K).components[0]
    )
    # the folded-away site is still traceable from the row it joined
    assert any(4 in src for src in pdd.components[0].sources)


def test_substitutional_sites_short_of_full_occupancy_leave_a_vacancy():
    pset = build(
        [A, B, G1, G2, G2],
        disorder=[
            DisorderAssembly(
                groups=[DisorderGroup([3], 0.5), DisorderGroup([4], 0.3)],
                is_substitutional=True,
            )
        ],
    )
    pdd = PointwiseDistanceDistribution(pset, K)
    assert pdd.n_components == 2  # the site is there 80% of the time


# --------------------------------------------------------------------
# Canonical form
# --------------------------------------------------------------------


def test_identical_components_are_merged():
    """Two configurations related by a symmetry of the parent structure
    have identical PDDs. Merging them is exact - two Diracs at one point
    of the space of measures are one Dirac of the summed mass - and it is
    what makes equality well defined."""
    pset = build(
        [[0, 0, 0], [0.5, 0.5, 0.5], [0.25, 0.25, 0.25], [0.75, 0.75, 0.75]],
        disorder=[
            DisorderAssembly(
                groups=[DisorderGroup([2], 0.5), DisorderGroup([3], 0.5)],
                is_substitutional=False,
            )
        ],
    )
    pdd = PointwiseDistanceDistribution(pset, K)
    assert pdd.n_components == 1
    assert float(pdd.weights[0]) == pytest.approx(1.0)
    assert len(pdd.configurations[0]) == 2  # both are recorded


def test_merging_identical_components_leaves_distances_alone():
    pset = build(
        [[0, 0, 0], [0.5, 0.5, 0.5], [0.25, 0.25, 0.25], [0.75, 0.75, 0.75]],
        disorder=[
            DisorderAssembly(
                groups=[DisorderGroup([2], 0.5), DisorderGroup([3], 0.5)],
                is_substitutional=False,
            )
        ],
    )
    merged = PointwiseDistanceDistribution(pset, K)
    unmerged = PointwiseDistanceDistribution.from_components(
        [0.5, 0.5], [merged.components[0], merged.components[0]],
        canonicalise=False,
    )
    other = PointwiseDistanceDistribution(two_group(0.5), K)
    for mode in ("pooled", "coupled"):
        assert emd(merged, other, disorder=mode) == pytest.approx(
            emd(unmerged, other, disorder=mode)
        )


def test_from_components_round_trips():
    pdd = PointwiseDistanceDistribution(two_group(0.4), K)
    assert PointwiseDistanceDistribution.from_components(
        pdd.weights, pdd.components
    ).equals(pdd)


def test_equal_pdds_built_differently_compare_equal():
    a = PointwiseDistanceDistribution(two_group(0.3), K)
    b = PointwiseDistanceDistribution.from_components(
        a.weights[::-1], a.components[::-1]
    )
    assert a.equals(b)


# --------------------------------------------------------------------
# Comparison
# --------------------------------------------------------------------


def test_bound_cascade():
    """||AMD - AMD'|| <= pooled <= coupled, so the cheap ones prune the
    expensive one."""
    a = PointwiseDistanceDistribution(two_group(0.7), K)
    b = PointwiseDistanceDistribution(
        build([A, B, G1, [0.5, 0.5, 0.1]],
              disorder=[DisorderAssembly(
                  groups=[DisorderGroup([2], 0.6), DisorderGroup([3], 0.4)],
                  is_substitutional=False)]),
        K,
    )
    bound = centroid_lb(a, b)
    pooled = emd(a, b, disorder="pooled")
    coupled = emd(a, b, disorder="coupled")
    assert bound <= pooled + 1e-9
    assert pooled <= coupled + 1e-9


def test_coupled_is_a_metric_on_these():
    a = PointwiseDistanceDistribution(two_group(0.7), K)
    b = PointwiseDistanceDistribution(two_group(0.3), K)
    c = PointwiseDistanceDistribution(two_group(0.5), K)
    assert emd(a, a, disorder="coupled") == pytest.approx(0.0, abs=1e-12)
    assert emd(a, b, disorder="coupled") == pytest.approx(
        emd(b, a, disorder="coupled")
    )
    assert emd(a, b, disorder="coupled") <= (
        emd(a, c, disorder="coupled") + emd(c, b, disorder="coupled") + 1e-9
    )


def test_best_match_finds_the_shared_configuration():
    """Two crystals disordered over the same two alternatives share both
    configurations, so best-match is 0 while the metrics are not."""
    a = PointwiseDistanceDistribution(two_group(0.7), K)
    b = PointwiseDistanceDistribution(two_group(0.3), K)
    assert emd(a, b, disorder="best-match") == pytest.approx(0.0, abs=1e-12)
    assert emd(a, b, disorder="coupled") > 0.0


def test_disorder_mode_is_irrelevant_for_ordered_crystals():
    a = PointwiseDistanceDistribution(build([A, B, G1, G2]), K)
    b = PointwiseDistanceDistribution(build([A, B, G1]), K)
    values = {emd(a, b, disorder=m) for m in ("pooled", "coupled", "best-match")}
    assert len(values) == 1


def test_component_costs_shape():
    a = PointwiseDistanceDistribution(two_group(0.7), K)
    b = PointwiseDistanceDistribution(build([A, B, G1]), K)
    assert component_costs(a, b).shape == (2, 1)


def test_unknown_disorder_mode_raises():
    a = PointwiseDistanceDistribution(build([A, B, G1]), K)
    with pytest.raises(ValueError):
        emd(a, a, disorder="nonsense")


# --------------------------------------------------------------------
# Truncation
# --------------------------------------------------------------------


def test_truncation_warns_and_renormalises():
    """Independent assemblies multiply out; cutting the enumeration short
    costs at most the probability dropped, and says so."""
    pset = build(
        [A, B, G1, G2],
        disorder=[
            DisorderAssembly(
                groups=[DisorderGroup([i], 0.5), DisorderGroup([], 0.5)],
                is_substitutional=False,
            )
            for i in (1, 2, 3)
        ],
    )
    with pytest.warns(UserWarning, match="configurations"):
        pdd = PointwiseDistanceDistribution(pset, K, max_configs=3)
    assert pdd.n_components <= 3
    assert float(pdd.weights.sum()) == pytest.approx(1.0)


def test_configurations_come_out_most_probable_first():
    pset = build(
        [A, B, G1, G2],
        disorder=[
            DisorderAssembly(
                groups=[DisorderGroup([2], 0.7), DisorderGroup([3], 0.3)],
                is_substitutional=False,
            )
        ],
    )
    pdd = PointwiseDistanceDistribution(pset, K)
    assert list(pdd.weights) == sorted(pdd.weights, reverse=True)


# --------------------------------------------------------------------
# WeightedVectors
# --------------------------------------------------------------------


def test_merge_duplicates_sums_weights_and_keeps_sources():
    wv = WeightedVectors(
        [1.0, 1.0, 2.0],
        [[1.0, 2.0], [1.0, 2.0], [5.0, 6.0]],
        sources=[[0], [1], [2]],
    )
    merged = wv.merge_duplicates(1e-6)
    assert merged.n == 2
    assert merged.total_weight == pytest.approx(1.0)
    assert sorted(sorted(s) for s in merged.sources) == [[0, 1], [2]]


def test_equal_measures_compare_equal_whatever_the_row_order():
    """Rows equal to within rounding can come out of lexsorted() either
    way round when they share a leading distance, so equality must not
    depend on the order."""
    vectors = [[5.0, 6.0], [5.0, 7.0], [1.0, 2.0]]
    weights = [0.25, 0.25, 0.5]
    a = WeightedVectors(weights, vectors)
    order = [1, 0, 2]
    b = WeightedVectors(
        [weights[i] for i in order], [vectors[i] for i in order]
    )
    assert a.equals(b)
    # and a genuinely different measure still compares unequal
    assert not a.equals(WeightedVectors(weights, [[5.0, 6.0], [5.0, 8.0], [1.0, 2.0]]))


def test_equality_survives_rounding():
    """1e-12 was tight enough to call two routes to one crystal unequal."""
    a = WeightedVectors([0.5, 0.5], [[1.0, 2.0], [3.0, 4.0]])
    nudged = np.array([[1.0, 2.0], [3.0, 4.0]]) + 1e-11
    assert a.equals(WeightedVectors([0.5, 0.5], nudged))
    assert not a.equals(WeightedVectors([0.5, 0.5], nudged + 1e-6))


def test_lexsorted_is_a_canonical_order():
    vectors = [[3.0, 1.0], [1.0, 2.0], [2.0, 5.0]]
    weights = [0.2, 0.5, 0.3]
    order = [2, 0, 1]
    a = WeightedVectors(weights, vectors).lexsorted()
    b = WeightedVectors(
        [weights[i] for i in order], [vectors[i] for i in order]
    ).lexsorted()
    assert a.equals(b)


def test_to_matrix_round_trips():
    wv = WeightedVectors([0.25, 0.75], [[1.0, 2.0], [3.0, 4.0]])
    matrix = wv.to_matrix()
    assert np.allclose(matrix[:, 0], wv.weights)
    assert np.allclose(matrix[:, 1:], wv.vectors)


# --------------------------------------------------------------------
# compare_structures: the EMD plus an account of what did not line up
# --------------------------------------------------------------------


def typed(coords, types, labels, disorder=None, name="typed"):
    """Like build(), but with the atomic types and labels spelled out."""
    motif = np.array(coords, dtype=np.float64) @ CELL
    n = len(motif)
    return PeriodicSet(
        motif,
        CELL,
        name=name,
        asym_unit=np.arange(n, dtype=np.uint64),
        multiplicities=np.ones(n, dtype=np.uint64),
        types=np.array(types, dtype=np.uint64),
        labels=labels,
        disorder=disorder,
    )


def test_compare_structures_returns_a_float_without_describe():
    d = compare_structures(two_group(0.7), two_group(0.3), K)
    assert isinstance(d, float)


def test_compare_structures_agrees_with_pooled_emd():
    """The report needs uncollapsed rows so each names one site. Collapsing
    is exact for the EMD, so the distance must be unchanged by it."""
    a, b = two_group(0.7), two_group(0.3)
    assert compare_structures(a, b, K) == pytest.approx(
        emd(PointwiseDistanceDistribution(a, K),
            PointwiseDistanceDistribution(b, K), disorder="pooled"),
        abs=1e-12,
    )


def test_compare_structures_self_distance_is_zero_and_clean():
    a = two_group(0.7)
    d, report = compare_structures(a, a, K, describe=True)
    assert d == pytest.approx(0.0, abs=1e-12)
    assert report.startswith("EMD=0 ")
    assert "0/4!" in report  # none of the four matches flagged
    assert "->" not in report  # and a clean match is not written at all


def test_compare_structures_flags_types_the_pdd_cannot_see():
    """PDD is blind to atomic type, so two crystals can be identical and
    still differ chemically. The report is what says so."""
    a = typed([A, B, G1], [6, 6, 7], ["C1", "C2", "N1"], name="with_N")
    b = typed([A, B, G1], [6, 6, 15], ["C1", "C2", "P1"], name="with_P")

    d, report = compare_structures(a, b, 4, describe=True)
    assert d == pytest.approx(0.0, abs=1e-12)
    # The labels say which type went to which, so the match needs no
    # field of its own; C1 and C2 are clean and go unmentioned.
    assert report.endswith("N1->P1")
    assert "1/3!" in report


def test_compare_structures_flags_occupancy():
    full = build([A, B, G1])
    partial = build(
        [A, B, G1],
        disorder=[
            DisorderAssembly(groups=[DisorderGroup([2], 0.6)], is_substitutional=False)
        ],
    )
    _, report = compare_structures(partial, full, 4, describe=True)
    assert "o=0.6>1" in report


def test_compare_structures_sums_the_rows_of_a_disordered_site():
    """A site appears once per configuration it survives in, so its weight
    is the total over those rows: its expected presence in the ensemble."""
    a = two_group(0.7)
    b = build([A, B, G1], name="ordered")
    _, report = compare_structures(a, b, 4, describe=True)

    # Sites A and B are in both configurations, so two rows each, and the
    # two rows together carry the full 1/3 of an ordered site.
    assert "A=two_group_0.7 4s 6r 2c" in report
    # G1 is in one configuration only, weight 0.7 * 1/3.
    assert "w=0.233333>0.333333" in report


def test_compare_structures_reports_a_split_weight():
    a, b = two_group(0.5), build([A, B, G1], name="ordered")
    _, report = compare_structures(a, b, 4, describe=True)
    assert "C1->C1 50% | C1->C3* 50%" in report


def test_compare_structures_names_folded_substitutional_partners():
    sub = typed(
        [A, B, G1, G1],
        [6, 6, 7, 8],
        ["C1", "C2", "N1", "O1"],
        disorder=[
            DisorderAssembly(
                groups=[DisorderGroup([2], 0.6), DisorderGroup([3], 0.4)],
                is_substitutional=True,
            )
        ],
        name="sub",
    )
    ordered = typed([A, B, G1], [6, 6, 7], ["C1", "C2", "N1"], name="ordered")
    _, report = compare_structures(sub, ordered, 4, describe=True)
    assert "4s(3)" in report  # four sites, three left after folding
    assert "N1/O1->N1" in report  # both partners named, by the one match


def test_compare_structures_rejects_things_that_are_not_periodic_sets():
    a = two_group(0.7)
    with pytest.raises(TypeError, match="PeriodicSet"):
        compare_structures(PointwiseDistanceDistribution(a, K), a, K)

# --------------------------------------------------------------------
# PPC, PDA and ADA: the asymptotics
# --------------------------------------------------------------------


def partial(occ, name=None):
    """Sites A, B ordered; G1 present a fraction ``occ`` of the time, so
    the two configurations differ in how many atoms they hold."""
    return build(
        [A, B, G1],
        disorder=[
            DisorderAssembly(
                groups=[DisorderGroup([2], occ)], is_substitutional=False
            )
        ],
        name=name or f"partial_{occ}",
    )


def test_ordered_ppc_matches_the_periodic_set():
    pset = build([A, B, G1])
    pdd = PointwiseDistanceDistribution(pset, K)
    assert pdd.PPC == pytest.approx(pset.PPC(), rel=1e-12)
    assert pdd.AMD_estimate() == pytest.approx(amd.AMD_estimate(pset, K))
    assert pdd.ADA() == pytest.approx(amd.ADA(pset, K))


def test_ordered_pda_matches_the_matrix_version():
    """The PDA object flattens to the matrix the old function returns,
    give or take the order of the rows."""
    pset = build([A, B, G1])
    new = PointwiseDistanceDistribution(pset, K).PDA().to_matrix()
    old = amd.PDA(pset, K)
    assert np.allclose(np.sort(new, axis=0), np.sort(old, axis=0))


def test_pda_is_a_pdd_which_compares_like_any_other():
    a = PointwiseDistanceDistribution(two_group(0.7), K).PDA()
    b = PointwiseDistanceDistribution(two_group(0.3), K).PDA()
    assert isinstance(a, PointwiseDistanceDistribution)
    assert a.k == K and a.n_components > 1
    assert emd(a, a) == pytest.approx(0.0, abs=1e-12)
    assert emd(a, b) > 0.0


def test_pda_shifts_each_component_by_its_own_estimate():
    pdd = PointwiseDistanceDistribution(partial(0.5), K)
    pda = pdd.PDA()
    assert pdd.n_components == 2
    assert len(set(np.round(pdd.ppcs, 12))) == 2  # the vacancy changes the count
    for comp, shifted, est in zip(
        pdd.components, pda.components, pdd.component_estimates()
    ):
        assert np.allclose(shifted.vectors, comp.vectors - est)
        assert np.allclose(shifted.weights, comp.weights)


def test_ada_is_the_amd_of_the_pda():
    """The two routes the deviation can be taken by - subtract from every
    row, or subtract from the average - agree, because the average of the
    estimates subtracted is the estimate of the average."""
    for pset in (build([A, B, G1]), two_group(0.7), partial(0.4)):
        pdd = PointwiseDistanceDistribution(pset, K)
        assert pdd.ADA() == pytest.approx(pdd.PDA().AMD(), abs=1e-12)


def test_pda_of_a_pda_changes_nothing():
    """Its rows over the k-th root of k converge to zero, so its PPC is
    zero and there is nothing left to subtract."""
    pda = PointwiseDistanceDistribution(two_group(0.7), K).PDA()
    assert pda.PPC == pytest.approx(0.0, abs=1e-15)
    assert pda.PDA().equals(pda)


def test_ppc_of_a_disordered_set_is_the_average_of_its_configurations():
    """Each configuration is an ordered crystal converging to its own
    PPC, so the pooled AMD converges to the average of those, weighted
    the way their rows are."""
    pdd = PointwiseDistanceDistribution(partial(0.5), K)
    assert pdd.PPC == pytest.approx(float(pdd.weights @ pdd.ppcs))
    for comp, ppc in zip(pdd.components, pdd.ppcs):
        ordered = PointwiseDistanceDistribution.from_components(
            [1.0], [comp], ppcs=[ppc], ndim=3
        )
        assert ordered.PPC == pytest.approx(ppc)


def test_ppc_is_below_the_periodic_sets_when_the_atom_count_varies():
    """PeriodicSet.PPC uses the expected number of atoms, and by Jensen
    that overestimates the constant a partially occupied crystal really
    converges to."""
    pset = partial(0.5)
    pdd = PointwiseDistanceDistribution(pset, K)
    assert pdd.PPC < pset.PPC()
    assert pdd.PPC == pytest.approx(pset.PPC(), rel=0.05)  # and not by much


def test_ppc_matches_the_periodic_set_when_the_atom_count_does_not_vary():
    """Substitutional disorder and an assembly whose alternatives are the
    same size both leave the atom count alone, and there the expected
    count is the right one."""
    sub = build(
        [A, B, G1, G1],
        disorder=[
            DisorderAssembly(
                groups=[DisorderGroup([2], 0.6), DisorderGroup([3], 0.4)],
                is_substitutional=True,
            )
        ],
        name="sub",
    )
    for pset in (two_group(0.7), sub):
        pdd = PointwiseDistanceDistribution(pset, K)
        assert pdd.PPC == pytest.approx(pset.PPC(), rel=1e-12)
        assert np.allclose(pdd.ppcs, pdd.ppcs[0])


def test_ppc_converges():
    """The point of the constant: AMD_k / k^(1/n) tends to it."""
    pset = partial(0.5)
    ratios = []
    for k in (50, 400):
        pdd = PointwiseDistanceDistribution(pset, k)
        ratios.append(pdd.AMD()[-1] / pdd.AMD_estimate()[-1])
    assert abs(ratios[1] - 1.0) < abs(ratios[0] - 1.0)
    assert ratios[1] == pytest.approx(1.0, abs=0.05)


def test_asymptotics_need_a_periodic_set_to_come_from():
    """Built from bare components there is no cell and no atom count, so
    no PPC can be had; better to say so than to invent one."""
    pdd = PointwiseDistanceDistribution(build([A, B, G1]), K)
    bare = PointwiseDistanceDistribution.from_components(
        pdd.weights, pdd.components
    )
    assert bare.ppcs is None
    with pytest.raises(ValueError, match="no PPC"):
        bare.PPC
    with pytest.raises(ValueError, match="no PPC"):
        bare.PDA()


def test_merge_duplicates_keeps_the_asymptotics():
    pdd = PointwiseDistanceDistribution(two_group(0.7), K)
    merged = pdd.merge_duplicates()
    assert merged.PPC == pytest.approx(pdd.PPC)
    assert merged.ndim == pdd.ndim
