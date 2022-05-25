import pytest
import os
import amd


@pytest.fixture(scope='module')
def cif_paths(root_dir):
    
    cif_names = [
        'cubic',
        'T2_experimental',
    ]
    
    return {name: os.path.join(root_dir, f'{name}.cif') for name in cif_names}


def test_CifReader(cif_paths, reference_data):
    
    for name in cif_paths:
        references = reference_data[name]
        read_in = list(amd.CifReader(cif_paths[name], show_warnings=True))

        if (not len(references) == len(read_in)) or len(read_in) == 0:
            pytest.fail(f'There are {len(references)} references, but {len(read_in)} structures were read.')
        
        for s, s_ in zip(read_in, references):
            if not s == s_:
                pytest.fail(f'Structure {s.name} read with CifReader disagrees with reference.')


@pytest.fixture(scope='module')
def asym_unit_test_cif_path(root_dir):
    return os.path.join(root_dir, 'OJIGOG.cif')

def test_CifReader_equiv_structs(asym_unit_test_cif_path):

    pdds = [amd.PDD(struct, 100) for struct in amd.CifReader(asym_unit_test_cif_path, show_warnings=False)]
    
    if amd.emd(pdds[0], pdds[1]) > 0:
        pytest.fail(f'Asymmetric structure was read differently than identical expanded version.')


@pytest.fixture(scope='module')
def equiv_sites_cif_path(root_dir):
    return os.path.join(root_dir, 'BABMUQ.cif')

def test_equiv_sites(equiv_sites_cif_path):
    
    pdds = [amd.PDD(s, 100) for s in amd.CifReader(equiv_sites_cif_path, show_warnings=False)]

    if amd.PDD_pdist(pdds):
        pytest.fail(f'Equivalent structures by symmetry differ by PDD.')


@pytest.fixture(scope='module')
def T2_alpha_cif_path(root_dir):
    return os.path.join(root_dir, 'T2-alpha-solvent.cif')

def test_heaviest_component(T2_alpha_cif_path):
    
    if not amd.io._CSD_PYTHON_API_ENABLED:
        pytest.skip(f'Skipping test_heaviest_component as csd-python-api is not installed.')
    
    s = amd.CifReader(T2_alpha_cif_path, 
                      reader='ccdc',
                      disorder='all_sites',
                      heaviest_component=True).read_one()

    if not s.asymmetric_unit.shape[0] == 26:
        pytest.fail(f'Heaviest component test failed.')
