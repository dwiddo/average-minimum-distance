import numpy as np

class PeriodicSet:
    """
    Represents a periodic set. Has attributes motif, cell and name.
    PeriodicSet objects are returned by the Readers in amd.io.
    """
    def __init__(self, motif, cell, name=None, **kwargs):
        self.motif = motif
        self.cell  = cell
        self.name  = name
        self.tags = kwargs

    def __getattr__(self, attr):
        if attr in self.tags:
            return self.tags[attr]
        else:
            raise AttributeError(f"{self.__class__.__name__} object has no attribute or tag {attr}")

    def __str__(self):
        m, dims = self.motif.shape
        return f"PeriodicSet({self.name}: {m} motif points in {dims} dimensions)"
    
    def __repr__(self):
        return f"PeriodicSet(name: {self.name}, cell: {self.cell}, motif: {self.motif}, tags: {self.tags})"
    
    # used for debugging. Checks if the motif/cell agree point for point
    # (disregarding order), NOT up to isometry.
    # Also, this function is unfinished. It'll get confused by structures 
    # with overlapping sites --- but it's good enough for now.
    def __eq__(self, other):
        motif = self.motif
        motif_ = other.motif
        cell = self.cell
        cell_ = other.cell

        if not (cell.shape == cell_.shape and motif.shape == motif_.shape):
            return False

        if not np.allclose(cell, cell_):
            return False

        # this is the part that'd be confused by overlapping sites
        # just checks every motif point in either have a neighbour in the other
        diffs = np.amax(np.abs(motif[:, None] - motif), axis=-1)
        
        if not np.all((np.amin(diffs, axis=0) <= 1e-6) & (np.amin(diffs, axis=-1) <= 1e-6)):
            return False
        else:
            return True
    
    def __ne__(self, other):
        return not self.__eq__(other)
    
    def astype(self, dtype):
        return PeriodicSet(self.motif.astype(dtype), 
                           self.cell.astype(dtype),
                           name=self.name,
                           **self.tags)
