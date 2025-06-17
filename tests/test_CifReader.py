import pytest
import amd
import warnings


@pytest.fixture(scope="module")
def data_cif_paths(data_dir):
    cif_names = ["cubic", "T2_experimental"]
    return {name: str(data_dir / f"{name}.cif") for name in cif_names}


@pytest.mark.parametrize("reader", ["gemmi", "ccdc"])
def test_CifReader(reader, data_cif_paths, reference_data):
    for name in data_cif_paths:
        try:
            read_in = list(amd.CifReader(data_cif_paths[name], reader=reader))
        except ImportError:
            pytest.skip(
                f"Skipping test_CifReader[{reader}] as {reader} failed to " "import."
            )

        references = reference_data[name]

        if (not len(references) == len(read_in)) or len(read_in) == 0:
            pytest.fail(
                f"There are {len(references)} references, but {len(read_in)}"
                f"structures were read from {name}."
            )

        for s, s_ in zip(read_in, references):
            if not s._equal_cell_and_motif(s_["PeriodicSet"]):
                pytest.fail(
                    f"Structure {name}:{s.name} read with CifReader disagrees "
                    "with reference."
                )


def test_CifReader_equiv_structs(data_dir):
    structs = list(amd.CifReader(data_dir / "eq_sites_test.cif", show_warnings=False))
    pdds = [amd.PDD(struct, 100) for struct in structs]

    if amd.EMD(pdds[0], pdds[1]) > 1e-10:  # change
        pytest.fail(
            "Asymmetric structure was read differently than identical "
            "expanded version."
        )


def test_heaviest_component(data_dir, ccdc_enabled):
    if not ccdc_enabled:
        pytest.skip("Skipping test_CSDReader as csd-python-api failed to import.")

    s = amd.CifReader(
        data_dir / "T2-alpha-solvent.cif",
        reader="ccdc",
        heaviest_component=True,
    ).read()

    if not s.asym_unit.shape[0] == 26:
        pytest.fail("Heaviest component test failed.")
