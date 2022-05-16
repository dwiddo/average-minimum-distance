import os
import pytest
import amd
import numpy as np


def test_PDD_pdist(root_dir, reference_data):

    for name in reference_data:
        
        if 1 < len(reference_data[name]) < 1000:
            
            pdds = [s.pdd for s in reference_data[name]]
            cdm = amd.PDD_pdist(pdds)
            loaded = np.load(os.path.join(root_dir, rf'{name}_cdm.npz'))
            reference_cdm = loaded['cdm']
            
            if not np.amax(np.abs(cdm - reference_cdm)) < 1e-6:
                pytest.fail(f'PDD_pdist disagrees with reference.')
         
                
# def test_PDD_cdist(root_dir, reference_data):
    
#     t2_exp_pdds = [s.pdd for s in reference_data['T2_experimental']]
#     t2_sim_pdds = [s.pdd for s in reference_data['T2_simulated']]
    
#     cdm = amd.PDD_cdist(t2_exp_pdds, t2_sim_pdds)
    
#     loaded = np.load(os.path.join(root_dir, r'T2_exp_vs_sim_cdm.npz'))
#     reference_cdm = loaded['cdm']
    
#     if not np.amax(np.abs(cdm - reference_cdm)) < 1e-6:
#         pytest.fail(f'PDD_cdist disagrees with reference.')
