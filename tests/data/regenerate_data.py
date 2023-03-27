"""Script for regenerating data found in the tests/data directory.
"""

import numpy as np
import amd
import pickle
import pathlib


parent = pathlib.Path(__file__).absolute().parent


def regenerate(name, generator):
    """Regenerate and overwrite ``PeriodicSet``s, AMDs and PDDs for a
    testing dataset from a ``generator`` of ``PeriodicSet``s.
    """

    data = []
    for s in generator:
        pdd = amd.PDD(s, 100)
        data.append(
            {
                'PeriodicSet': s,
                'AMD100': amd.PDD_to_AMD(pdd),
                'PDD100': pdd
            }
        )

    with open(str(parent / f'{name}.pkl'), 'wb') as f:
        pickle.dump(data, f)
    cdm = amd.PDD_pdist([d['PDD100'] for d in data])
    np.savez_compressed(str(parent / f'{name}_cdm.npz'), cdm=cdm)


def regenerate_from_cifs():
    """Regenerate and overwrite all test data using cifs in tests/data.
    Does not apply to test data extracted with ``csd-python-api``.
    """
    for name in ('cubic', 'T2_experimental'):
        path = (parent / f'{name}.cif')
        regenerate(name, amd.CifReader(path, show_warnings=False))


def regenerate_CSD_families():
    """Regenerate and overwrite all test data for 'CSD_families' using
    ``csd-python-api``.
    """
    csd_families = ['DEBXIT', 'GLYCIN', 'HXACAN', 'ACSALA']
    reader = amd.CSDReader(csd_families, families=True, show_warnings=False)
    regenerate('CSD_families', reader)


if __name__ == '__main__':
    pass
