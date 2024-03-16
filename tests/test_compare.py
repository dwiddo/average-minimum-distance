import os
import numpy as np
import pytest
import amd


def test_PDD_pdist(data_dir, reference_data):
    for name in reference_data:
        if len(reference_data[name]) == 1:
            continue
        pdds = [s['PDD100'] for s in reference_data[name]]
        cdm = amd.PDD_pdist(pdds)
        loaded = np.load(str(data_dir / f'{name}_cdm.npz'))
        reference_cdm = loaded['cdm']
        if not np.amax(np.abs(cdm - reference_cdm)) < 1e-8:
            pytest.fail('PDD_pdist disagrees with reference.')
