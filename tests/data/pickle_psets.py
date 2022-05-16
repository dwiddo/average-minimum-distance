import os
import amd
import pickle

root_dir = r'average-minimum-distance\tests\data'

filenames = {
    'cubic':           r'cubic.hdf5',
    'T2_simulated':    r'T2_simulated.hdf5',
    'T2_experimental': r'T2_experimental.hdf5',
    'CSD_families':    r'CSD_families.hdf5',
}

paths = {name: os.path.join(root_dir, filenames[name]) for name in filenames}

refs = {name: list(amd.SetReader(paths[name])) for name in paths}

for name, psets in refs.items():
    with open(os.path.join(root_dir, name + '.pkl'), 'wb') as f:
        pickle.dump(psets, f)