import pytest
import amd
import warnings


@pytest.fixture(scope='module')
def test_cif_paths(data_dir):
    cif_names = ['cubic', 'T2_experimental']
    return {name: str(data_dir / f'{name}.cif') for name in cif_names}


@pytest.mark.parametrize('reader', ['gemmi', 'pymatgen', 'ase', 'ccdc'])
def test_CifReader(reader, test_cif_paths, reference_data):
    for name in test_cif_paths:
        try:
            read_in = list(amd.CifReader(test_cif_paths[name], reader=reader))
        except ImportError:
            pytest.skip(
                f'Skipping test_CifReader[{reader}] as {reader} failed to '
                'import.'
            )
        references = reference_data[name]
        if (not len(references) == len(read_in)) or len(read_in) == 0:
            pytest.fail(
                f'There are {len(references)} references, but {len(read_in)}'
                f'structures were read from {name}.'
            )
        for s, s_ in zip(read_in, references):
            if not s == s_['PeriodicSet']:
                pytest.fail(
                    f'Structure {name}:{s.name} read with CifReader disagrees '
                    'with reference.'
                )


def test_periodicset_from_ase_atoms(test_cif_paths, reference_data):
    try:
        from ase.io import iread
    except ImportError:
        pytest.skip(
            'Skipping test_periodicset_from_ase_atoms as ase failed to import.'
        )

    for name in test_cif_paths:
        references = reference_data[name]
        # ignore ase warning: "crystal system x is not interpreted for space
        # group Spacegroup(1, setting=1). This may result in wrong setting!"
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            atoms = list(iread(test_cif_paths[name]))
        read_in = [amd.periodicset_from_ase_atoms(a) for a in atoms]
        if (not len(references) == len(read_in)) or len(read_in) == 0:
            pytest.fail(
                f'There are {len(references)} references, but {len(read_in)}'
                f'structures were read from {name}.'
            )

        for s, s_ in zip(read_in, references):
            if not s == s_['PeriodicSet']:
                n = s_['PeriodicSet'].name
                pytest.fail(
                    'Structure read with ase.io.iread disagrees with '
                    f'reference ({n}).'
                )


def test_CifReader_equiv_structs(data_dir):
    reader = amd.CifReader(data_dir / 'OJIGOG.cif', show_warnings=False)
    pdds = [amd.PDD(struct, 100) for struct in reader]
    if amd.emd(pdds[0], pdds[1]) > 0:
        pytest.fail(
            'Asymmetric structure was read differently than identical '
            'expanded version.'
        )


def test_equiv_sites(data_dir):
    reader = amd.CifReader(data_dir / 'BABMUQ.cif', show_warnings=False)
    pdds = [amd.PDD(s, 100) for s in reader]
    if amd.PDD_pdist(pdds):
        pytest.fail('Equivalent structures by symmetry differ by PDD.')


def test_heaviest_component(data_dir, ccdc_enabled):

    if not ccdc_enabled:
        pytest.skip(
            'Skipping test_CSDReader as csd-python-api failed to import.'
        )

    s = amd.CifReader(
        data_dir / 'T2-alpha-solvent.cif',
        reader='ccdc',
        disorder='all_sites',
        heaviest_component=True
    ).read()

    if not s.asymmetric_unit.shape[0] == 26:
        pytest.fail('Heaviest component test failed.')
