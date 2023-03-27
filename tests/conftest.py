import os
import pickle
import pytest
import pathlib


@pytest.fixture(scope='session', autouse=True)
def data_dir():
    return pathlib.Path(__file__).absolute().parent / 'data'


@pytest.fixture(scope='session', autouse=True)
def ccdc_enabled():
    try:
        import ccdc
        return True
    except Exception:
        return False


@pytest.fixture(scope='session', autouse=True)
def reference_data(data_dir):
    """Data fixture of amd.PeriodicSet objects to use in tests."""

    filenames = ['cubic', 'T2_experimental', 'CSD_families']
    refs = {}
    for name in filenames:
        path = str(data_dir / f'{name}.pkl')
        with open(path, 'rb') as f:
            data = pickle.load(f)
            if not data:
                raise ValueError(f'Data not found in path {path}')
            refs[name] = data
    return refs
