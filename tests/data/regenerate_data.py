import os
import amd
import pickle

datasets = []

root = r'tests/data'
filenames = {
    'cubic':           'cubic',
    'T2_experimental': 'T2_experimental',
}

for name in filenames:
    path = os.path.join(root, filenames[name])
    data = [{'PeriodicSet': s, 
             'AMD100': amd.AMD(s, 100),
             'PDD100': amd.PDD(s, 100)}
            for s in amd.CifReader(path + '.cif', show_warnings=False)]
    with open(path + '.pkl', 'wb') as f:
        pickle.dump(data, f)

csd_families = ['DEBXIT', 'GLYCIN', 'HXACAN', 'ACSALA']
data = [{'PeriodicSet': s, 
         'AMD100': amd.AMD(s, 100),
         'PDD100': amd.PDD(s, 100)}
        for s in amd.CSDReader(csd_families, families=True, show_warnings=False)]
with open(os.path.join(root, 'CSD_families.pkl'), 'wb') as f:
    pickle.dump(data, f)