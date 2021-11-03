### PeriodicSet io with h5py and .hdf5 format ###

import numpy as np
import h5py

from ..core.PeriodicSet import PeriodicSet

class SetWriter:
    """Write PeriodicSets to a .hdf5 file. Reading the .hdf5 is much faster than parsing a big .cif.
    """

    str_dtype = h5py.special_dtype(vlen=str)
    
    def __init__(self, filename):
        
        self.file = h5py.File(filename, 'w', track_order=True)
        
    def write(self, periodic_set, name=None):
        
        if not isinstance(periodic_set, PeriodicSet):
            raise ValueError(f'Object type {periodic_set.__class__.__name__} cannot be written with SetWriter')
        
        # need a name to store or you can't access items by key
        if periodic_set.name is None and name is None:
            raise ValueError('Periodic set must have a name to be written. Either set the name attribute of the PeriodicSet or pass a name to SetWriter.write()')
        
        if name is None:
            name = periodic_set.name
        
        # this group is the PeriodicSet
        group = self.file.create_group(name)
        
        # datasets in the group for motif and cell
        group.create_dataset('motif', data=periodic_set.motif)
        group.create_dataset('cell', data=periodic_set.cell)
        
        if periodic_set.tags:
            
            # a subgroup contains tags that are lists or ndarrays
            tags_group = group.create_group('tags')
            for tag in periodic_set.tags:
                
                data = periodic_set.tags[tag]
                
                # scalars (nums and strs) stored as attrs
                if np.isscalar(data):
                    tags_group.attrs[tag] = data
                
                elif isinstance(data, np.ndarray):
                    tags_group.create_dataset(tag, data=data)
                    
                elif isinstance(data, list):
                    
                    # lists of strings stored as special type for some reason
                    # if any string is in the tags
                    if any(isinstance(d, str) for d in data):
                        data = [str(d) for d in data]
                        tags_group.create_dataset(tag, data=data, 
                                                  dtype=SetWriter.str_dtype)
                        
                    # everything else castable to ndarray
                    else:
                        data = np.asarray(data)
                        tags_group.create_dataset(tag, data=np.array(data))
                else:
                    raise ValueError(f'Cannot store object {data} of type {type(data)} in hdf5')
    
    def iwrite(self, periodic_sets):
        for periodic_set in periodic_sets:
            self.write(periodic_set)
        
    def close(self):
        self.file.close()
        
    def __enter__(self):
        return self

    # handle exceptions?
    def __exit__(self, exc_type, exc_value, tb):
        self.file.close()

class SetReader:
    """Read PeriodicSets from a .hdf5 file written with SetWriter.
    Acts like a read-only dict that can be iterated over (preserves write order)."""
    
    def __init__(self, filename):
        
        self.file = h5py.File(filename, 'r', track_order=True)

    def _get_set(self, name):
        # take a name in the set and return the PeriodicSet
        
        group = self.file[name]
        periodic_set = PeriodicSet(group['motif'][:], group['cell'][:], name=name)
        
        if 'tags' in group:
            for tag in group['tags']:
                # tag: data e.g. 'types': ['C', 'H', 'O',...]
                data = group['tags'][tag][:]
                
                if any(isinstance(d, (bytes, bytearray)) for d in data):
                    periodic_set.tags[tag] = [d.decode() for d in data]
                else:
                    periodic_set.tags[tag] = data
        
        for attr in group.attrs:
            periodic_set.tags[tag] = group.attrs[attr]

        return periodic_set
    
    def close(self):
        self.file.close()
    
    def family(self, family):
        for name in self.keys():
            if name.startswith(family):
                yield self._get_set(name)
    
    def __getitem__(self, name):
        # index by name. Not found exc?
        return self._get_set(name)
    
    def __len__(self):
        return len(self.keys())
    
    def __iter__(self):
        # interface to loop over the SetReader
        # does not close the SetReader when done
        for name in self.keys():
            yield self._get_set(name)

    def __contains__(self, item):
        if item in self.keys():
            return True
        else:
            return False
        
    def keys(self):
        return self.file['/'].keys()
    
    def __enter__(self):
        return self

    # handle exceptions?
    def __exit__(self, exc_type, exc_value, tb):
        self.file.close()