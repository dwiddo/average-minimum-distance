"""Tests for the gemmi backend of the CIF reader, built from small
in-memory CIFs rather than files on disk.
"""

import warnings

import numpy as np
import pytest

from amd.io import ParseError, periodicset_from_gemmi_block, _element_symbol


CELL_HEADER = """data_{name}
_cell_length_a 10
_cell_length_b 10
_cell_length_c 10
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
"""

P1 = "_symmetry_equiv_pos_as_xyz 'x,y,z'\n"

ONE_CARBON = """loop_
_atom_site_label
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
C1 0.1 0.2 0.3
"""


def read(body, name="test", **kwargs):
    gemmi = pytest.importorskip("gemmi")
    doc = gemmi.cif.read_string(CELL_HEADER.format(name=name) + body)
    return periodicset_from_gemmi_block(doc.sole_block(), **kwargs)


def frac(periodic_set):
    return periodic_set.motif @ np.linalg.inv(periodic_set.cell)


# --------------------------------------------------------------------
# Malformed symmetry data must be skipped, not crash the reader
# --------------------------------------------------------------------


@pytest.mark.parametrize("number", ["?", ".", "999", "abc"])
def test_unusable_space_group_number_defaults_to_p1(number):
    """The reader loop only catches ParseError, so anything else raised
    here would abort reading a whole file over one block."""

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ps = read(f"_space_group_IT_number {number}\n" + ONE_CARBON)

    assert ps.motif.shape[0] == 1
    assert any("defaulting to P1" in str(w.message) for w in caught)


def test_valid_space_group_number_is_used():
    ps = read("_space_group_IT_number 14\n" + ONE_CARBON)
    assert ps.motif.shape[0] == 4


# --------------------------------------------------------------------
# Tags are case-insensitive
# --------------------------------------------------------------------


def test_uppercase_tags_are_read():
    body = P1 + ONE_CARBON.replace("_atom_site_", "_ATOM_SITE_").replace(
        "fract_x", "FRACT_X"
    )
    ps = read(body)
    assert ps.motif.shape[0] == 1
    assert ps.labels == ["C1"]
    assert ps.types.tolist() == [6]


# --------------------------------------------------------------------
# wrap_coords
# --------------------------------------------------------------------


def test_wrap_coords_true_wraps():
    body = P1 + ONE_CARBON.replace("0.1 0.2 0.3", "-0.1 1.2 0.3")
    assert np.allclose(frac(read(body, wrap_coords=True)), [[0.9, 0.2, 0.3]])


def test_wrap_coords_false_leaves_asymmetric_unit_alone():
    body = P1 + ONE_CARBON.replace("0.1 0.2 0.3", "-0.1 1.2 0.3")
    assert np.allclose(frac(read(body, wrap_coords=False)), [[-0.1, 1.2, 0.3]])


def test_wrap_coords_false_leaves_symmetry_images_alone():
    body = "loop_\n_symmetry_equiv_pos_as_xyz\n'x,y,z'\n'-x,-y,-z'\n" + ONE_CARBON
    got = frac(read(body, wrap_coords=False))
    assert np.allclose(got, [[0.1, 0.2, 0.3], [-0.1, -0.2, -0.3]])


def test_wrap_coords_false_still_removes_duplicate_images():
    """Uniqueness of symmetry images is judged modulo 1 whether or not
    the points are wrapped."""
    body = "loop_\n_symmetry_equiv_pos_as_xyz\n'x,y,z'\n'-x,-y,-z'\n" + (
        ONE_CARBON.replace("0.1 0.2 0.3", "0 0 0")
    )
    assert read(body, wrap_coords=False).motif.shape[0] == 1


def test_wrap_coords_false_with_reduce_cell():
    body = P1 + ONE_CARBON.replace("0.1 0.2 0.3", "-0.1 1.2 0.3")
    ps = read(body, wrap_coords=False, reduce_cell=True)
    assert np.allclose(frac(ps), [[-0.1, 1.2, 0.3]])


# --------------------------------------------------------------------
# skip_disorder
# --------------------------------------------------------------------


def test_skip_disorder_sees_group_codes_at_full_occupancy():
    body = P1 + """loop_
_atom_site_label
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_occupancy
_atom_site_disorder_group
C1 0.1 0.2 0.3 1 1
C2 0.12 0.2 0.3 1 2
"""
    with pytest.raises(ParseError, match="disorder"):
        read(body, skip_disorder=True)


def test_skip_disorder_sees_assembly_codes():
    body = P1 + """loop_
_atom_site_label
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_disorder_assembly
C1 0.1 0.2 0.3 A
"""
    with pytest.raises(ParseError, match="disorder"):
        read(body, skip_disorder=True)


def test_skip_disorder_ignores_null_codes():
    body = P1 + """loop_
_atom_site_label
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_disorder_assembly
_atom_site_disorder_group
C1 0.1 0.2 0.3 . .
C2 0.5 0.2 0.3 ? ?
"""
    ps = read(body, skip_disorder=True)
    assert ps.motif.shape[0] == 2
    assert ps.disorder is None


# --------------------------------------------------------------------
# Element inference
# --------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,expected",
    [
        ("C1", "C"),
        ("HA", "H"),
        ("H1A", "H"),
        ("Ca1", "Ca"),
        ("CA", "Ca"),
        ("HG1", "Hg"),
        ("Cl-", "Cl"),
        ("O2-", "O"),
        ("D", "H"),
        ("D1", "H"),
        ("T1", "H"),
        ("?", ""),
        (".", ""),
        ("", ""),
        ("1", ""),
        ("Xx1", ""),
    ],
)
def test_element_symbol(text, expected):
    assert _element_symbol(text) == expected


def test_remove_hydrogens_by_label_with_one_letter_fallback():
    body = P1 + ONE_CARBON + "HA 0.2 0.2 0.3\nHB2 0.3 0.2 0.3\nCA 0.4 0.2 0.3\n"
    ps = read(body, remove_hydrogens=True)
    assert ps.labels == ["C1", "CA"]
    assert ps.types.tolist() == [6, 20]


def test_remove_hydrogens_removes_deuterium():
    body = P1 + """loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
C1 C 0.1 0.2 0.3
D1 D 0.2 0.2 0.3
H1 H 0.3 0.2 0.3
"""
    ps = read(body, remove_hydrogens=True)
    assert ps.labels == ["C1"]


def test_deuterium_has_hydrogen_type():
    body = P1 + ONE_CARBON + "D1 0.2 0.2 0.3\n"
    assert read(body).types.tolist() == [6, 1]
