"""Script for regenerating data found in the tests/data directory.
"""

import os
import numpy as np
import amd
import pickle


root = r'tests/data'


def regenerate(name, generator):
    """Regenerate and overwrite ``PeriodicSet``s, AMDs and PDDs for a
    testing dataset from a ``generator`` of ``PeriodicSet``s.
    """

    data = []
    for s in generator:
        pdd = amd.PDD(s, 100)
        data.append(
            {'PeriodicSet': s, 'AMD100': amd.PDD_to_AMD(pdd), 'PDD100': pdd}
        )

    with open(os.path.join(root, f'{name}.pkl'), 'wb') as f:
        pickle.dump(data, f)

    pdds = [d['PDD100'] for d in data]
    cdm = amd.PDD_pdist(pdds)
    np.savez_compressed(os.path.join(root, rf'{name}_cdm.npz'), cdm=cdm)


def regenerate_from_cifs():
    """Regenerate and overwrite all test data using cifs in tests/data.
    Does not apply to test data extracted with ``csd-python-api``.
    """
    for name in ('cubic', 'T2_experimental'):
        path = os.path.join(root, name)
        regenerate(name, amd.CifReader(path + '.cif', show_warnings=False))


def regenerate_CSD_families():
    """Regenerate and overwrite all test data for 'CSD_families' using
    ``csd-python-api``.
    """
    csd_families = ['DEBXIT', 'GLYCIN', 'HXACAN', 'ACSALA']
    generator = amd.CSDReader(csd_families, families=True, show_warnings=False)
    regenerate('CSD_families', generator)


if __name__ == '__main__':
    regenerate_CSD_families()
