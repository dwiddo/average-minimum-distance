from collections import namedtuple
import numpy as np

_PeriodicSet = namedtuple('_PeriodicSet', ['motif', 'cell', 'name'])

class PeriodicSet(_PeriodicSet):
    
    def __str__(self):
        return f"PeriodicSet({self.name}: {self.cell.shape} cell, {self.motif.shape} motif)"
    
    def __repr__(self):
        return self.__str__()
    
    # actually, this doesn't make complete sense. Used for debugging.
    def __eq__(self, other):
        return self.cell.shape == other.cell.shape and \
               self.motif.shape == other.motif.shape and \
               np.allclose(self.cell, other.cell) and \
               np.allclose(self.motif, other.motif)
    
    def __ne__(self, other):
        return not self.__eq__(other)
    
    def astype(self, dtype):
        return PeriodicSet(name=self.name, 
                           motif=self.motif.astype(dtype),
                           cell=self.cell.astype(dtype))