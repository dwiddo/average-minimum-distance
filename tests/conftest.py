import os
import pickle
import pytest


@pytest.fixture(scope='session', autouse=True)
def root_dir():
    return r'tests/data'      

@pytest.fixture(scope='session', autouse=True)
def ccdc_enabled():
    try:
        import ccdc
        return True
    except (ImportError, RuntimeError) as _:
        return False

@pytest.fixture(scope='session', autouse=True)
def reference_data(root_dir):

    filenames = {   # root_dir contains .pkl files with PeriodicSets
        'cubic':           'cubic',
        'T2_experimental': 'T2_experimental',
        'CSD_families':    'CSD_families',
    }

    refs = {}
    for name in filenames:
        path = os.path.join(root_dir, filenames[name] + '.pkl')
        with open(path, 'rb') as f:
            data = pickle.load(f)
            if not data:
                raise ValueError(f'Data not found in path {path}')
            refs[name] = data

    return refs
