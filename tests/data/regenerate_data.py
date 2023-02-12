import os
import numpy as np
import amd
import pickle


root = r'tests/data'

def regen(name, generator):

    data = []
    for s in generator:
        pdd = amd.PDD(s, 100)
        data.append({'PeriodicSet': s, 'AMD100': amd.PDD_to_AMD(pdd), 'PDD100': pdd})

    with open(os.path.join(root, f'{name}.pkl'), 'wb') as f:
        pickle.dump(data, f)

    pdds = [d['PDD100'] for d in data]
    cdm = amd.PDD_pdist(pdds)
    np.savez_compressed(os.path.join(root, rf'{name}_cdm.npz'), cdm=cdm)


def regenerate_from_cifs():

    filenames = ['cubic', 'T2_experimental']

    for name in filenames:
        path = os.path.join(root, name)
        regen(name, amd.CifReader(path + '.cif', show_warnings=False))


def regenerate_CSD_families():

    name = 'CSD_families'
    csd_families = ['DEBXIT', 'GLYCIN', 'HXACAN', 'ACSALA']
    generator = amd.CSDReader(csd_families, families=True, show_warnings=False)
    regen(name, generator)


if __name__ == '__main__':
    regenerate_CSD_families()