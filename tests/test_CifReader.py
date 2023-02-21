import pytest
import os
import amd
import warnings


@pytest.fixture(scope='module')
def cif_paths(root_dir):
    cif_names = [
        'cubic',
        'T2_experimental',
    ]
    return {name: os.path.join(root_dir, f'{name}.cif') for name in cif_names}

@pytest.mark.parametrize('reader', ['ase', 'pymatgen', 'gemmi', 'ccdc'])
def test_CifReader(reader, cif_paths, reference_data):
    for name in cif_paths:
        references = reference_data[name]
        try:
            read_in = list(amd.CifReader(cif_paths[name], reader=reader))
        except ImportError as _:
            pytest.skip(
                f'Skipping test_CifReader for {reader} as it failed to import.'
            )

        if (not len(references) == len(read_in)) or len(read_in) == 0:
            pytest.fail(
                f'There are {len(references)} references, but ' \
                f'{len(read_in)} structures were read.'
            )

        for s, s_ in zip(read_in, references):
            if not s == s_['PeriodicSet']:
                pytest.fail(
                    f'Structure {s.name} read with CifReader disagrees with ' \
                    'reference.'
                )

def test_periodicset_from_ase_atoms(cif_paths, reference_data):
    try:
        from ase.io import iread
    except ImportError as _:    
        pytest.fail('Failed to import ase.')

    for name in cif_paths:
        references = reference_data[name]
        # ignore ase warning: "crystal system x is not interpreted for space
        # group Spacegroup(1, setting=1). This may result in wrong setting!"
        with warnings.catch_warnings() as _:
            warnings.simplefilter('ignore') 
            atoms = list(iread(cif_paths[name]))
        read_in = [amd.periodicset_from_ase_atoms(a) for a in atoms]
        if (not len(references) == len(read_in)) or len(read_in) == 0:
            pytest.fail(
                f'There are {len(references)} references, but ' \
                f'{len(read_in)} structures were read.'
            )

        for s, s_ in zip(read_in, references):
            if not s == s_['PeriodicSet']:
                n = s_['PeriodicSet'].name
                pytest.fail(
                    'Structure read with ase.io.iread disagrees with ' \
                    f'reference ({n}).'
                )


@pytest.fixture(scope='module')
def asym_unit_test_cif_path(root_dir):
    return os.path.join(root_dir, 'OJIGOG.cif')

def test_CifReader_equiv_structs(asym_unit_test_cif_path):
    reader = amd.CifReader(asym_unit_test_cif_path, show_warnings=False)
    pdds = [amd.PDD(struct, 100) for struct in reader]
    if amd.emd(pdds[0], pdds[1]) > 0:
        pytest.fail(
            'Asymmetric structure was read differently than identical ' \
            'expanded version.'
        )


@pytest.fixture(scope='module')
def equiv_sites_cif_path(root_dir):
    return os.path.join(root_dir, 'BABMUQ.cif')

def test_equiv_sites(equiv_sites_cif_path):
    reader = amd.CifReader(equiv_sites_cif_path, show_warnings=False)
    pdds = [amd.PDD(s, 100) for s in reader]
    if amd.PDD_pdist(pdds):
        pytest.fail('Equivalent structures by symmetry differ by PDD.')


@pytest.fixture(scope='module')
def T2_alpha_cif_path(root_dir):
    return os.path.join(root_dir, 'T2-alpha-solvent.cif')


def test_heaviest_component(T2_alpha_cif_path, ccdc_enabled):

    if not ccdc_enabled:
        pytest.skip(
            'Skipping test_CSDReader as csd-python-api failed to import.'
        )

    s = amd.CifReader(
        T2_alpha_cif_path, 
        reader='ccdc',
        disorder='all_sites',
        heaviest_component=True
    ).read()

    if not s.asymmetric_unit.shape[0] == 26:
        pytest.fail('Heaviest component test failed.')
