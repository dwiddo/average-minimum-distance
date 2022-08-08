import numpy as np
import pytest
import amd


def test_AMD(reference_data):
    for name in reference_data:
        for s in reference_data[name]:
            calc_amd = amd.AMD(s['PeriodicSet'], 100)
            if not np.allclose(calc_amd, s['AMD100']):
                pytest.fail(f'AMD of structure {s.name} disagrees with reference.\n' + \
                            'reference: ' + str(s.amd) + '\ncalculated: ' + str(calc_amd))


def test_PDD(reference_data):
    for name in reference_data:
        for s in reference_data[name]:
            calc_pdd = amd.PDD(s['PeriodicSet'], 100)
            if not np.allclose(calc_pdd, s['PDD100']):
                abs_diffs = np.abs(calc_pdd - s['PDD100'])
                print(abs_diffs, np.amax(abs_diffs))
                pytest.fail(f'PDD of structure {s.name} disagrees with reference.\n' + \
                            'reference: ' + str(s['PDD100']) + '\ncalculated: ' + str(calc_pdd) + \
                            'diffs' + str(np.sort(abs_diffs.flatten())[::-1][:10]))


def test_PDD_to_AMD(reference_data):
    for name in reference_data:
        for s in reference_data[name]:
            calc_pdd = amd.PDD(s['PeriodicSet'], 100)
            calc_amd = amd.AMD(s['PeriodicSet'], 100)
            amd_from_pdd = amd.PDD_to_AMD(calc_pdd)
            if not np.allclose(calc_amd, amd_from_pdd):
                pytest.fail(f'Directly calculated AMD of structure {s["PeriodicSet"].name} disagrees with ' + \
                            'AMD calculated from PDD.\n')


def test_AMD_finite():
    trapezium = np.array([[0,0],[1,1],[3,1],[4,0]])
    kite = np.array([[0,0],[1,1],[1,-1],[4,0]])
    trap_amd = amd.AMD_finite(trapezium)
    kite_amd = amd.AMD_finite(kite)
    dist = np.linalg.norm(trap_amd - kite_amd)
    if not abs(dist - 0.6180339887498952) < 1e-16:
        pytest.fail(f'AMDs of finite sets trapezium and kite are different than expected.\n')


def test_PDD_finite():
    trapezium = np.array([[0,0],[1,1],[3,1],[4,0]])
    kite = np.array([[0,0],[1,1],[1,-1],[4,0]])
    trap_pdd = amd.PDD_finite(trapezium)
    kite_pdd = amd.PDD_finite(kite)
    dist = amd.EMD(trap_pdd, kite_pdd)
    if not abs(dist - 0.874032) < 1e-8:
        pytest.fail(f'PDDs of finite sets trapezium and kite are different than expected.\n')
