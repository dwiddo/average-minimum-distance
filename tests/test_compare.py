import os
import numpy as np
import pytest
import amd


def test_PDD_pdist(root_dir, reference_data):
    for name in reference_data:
        if len(reference_data[name]) == 1:
            continue
        pdds = [s.pdd for s in reference_data[name]]
        cdm = amd.PDD_pdist(pdds)
        loaded = np.load(os.path.join(root_dir, rf'{name}_cdm.npz'))
        reference_cdm = loaded['cdm']
        
        if not np.amax(np.abs(cdm - reference_cdm)) < 1e-6:
            pytest.fail(f'PDD_pdist disagrees with reference.')
