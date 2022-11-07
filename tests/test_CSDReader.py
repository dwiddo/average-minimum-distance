import pytest
import amd


@pytest.fixture(scope='module')
def refcode_families():
    return ['DEBXIT', 'GLYCIN', 'HXACAN', 'ACSALA']


def test_CSDReader(reference_data, refcode_families, ccdc_enabled):

    if not ccdc_enabled:
        pytest.skip('Skipping test_CSDReader as csd-python-api is not installed.')

    references = reference_data['CSD_families']
    read_in = list(amd.CSDReader(refcode_families, show_warnings=False, families=True))
    
    if (not len(references) == len(read_in)) or len(read_in) == 0:
        pytest.fail(f'There are {len(references)} references, but {len(read_in)} structures were read.')
    
    for s, s_ in zip(read_in, references):
        if not s == s_['PeriodicSet']:
            pytest.fail(f'Structure {s.name} is different from reference. {s}, {s_["PeriodicSet"]}')
