import pytest
import os
import pickle


@pytest.fixture(scope='session', autouse=True)
def root_dir():
    return r'tests/data'


@pytest.fixture(scope='session', autouse=True)
def reference_data(root_dir):

    filenames = {   # .pkl files with PeriodicSets.
        'cubic':           'cubic',
        'T2_experimental': 'T2_experimental',
        'CSD_families':    'CSD_families',
    }

    refs = {}
    for name in filenames:
        with open(os.path.join(root_dir, filenames[name] + '.pkl'), 'rb') as f:
            refs[name] = pickle.load(f)

    return refs
