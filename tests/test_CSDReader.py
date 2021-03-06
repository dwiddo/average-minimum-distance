import pytest
import amd


@pytest.fixture(scope='module')
def refcode_families():
    return ['DEBXIT', 'GLYCIN', 'HXACAN', 'ACSALA']


@pytest.mark.skipif(not amd.io._CSD_PYTHON_API_ENABLED,
    reason=f'Skipping test_CSDReader as csd-python-api is not installed.')
def test_CSDReader(reference_data, refcode_families):

    references = reference_data['CSD_families']
    read_in = list(amd.CSDReader(refcode_families, show_warnings=False, families=True))
    
    if (not len(references) == len(read_in)) or len(read_in) == 0:
        pytest.fail(f'There are {len(references)} references, but {len(read_in)} structures were read.')
    
    for s, s_ in zip(read_in, references):
        print(s, s_)
        if not s == s_:
            pytest.fail(f'Structure {s.name} is different from reference.')
