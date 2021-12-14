"""Implements the class :class:`PeriodicSet` representing a periodic set, 
defined by a motif and unit cell.

This is the object type yielded by the readers :class:`.io.CifReader` and 
:class:`.io.CSDReader`. The :class:`PeriodicSet` can be passed as the first argument 
to :func:`.calculate.AMD` or :func:`.calculate.PDD` to calculate its invariants.
They can be written to a file with :class:`.io.SetWriter` which can be read with 
:class:`.io.SetReader`.
"""

from typing import Optional
import numpy as np

class PeriodicSet:
    """Represents a periodic set (which mathematically represents a crystal). 
    
    Has attributes motif, cell and name (which can be :const:`None`). 
    :class:`PeriodicSet` objects are returned by the readers in the :mod:`.io` module.
    """

    def __init__(self, motif: np.ndarray, 
                       cell: np.ndarray, 
                       name: Optional[str] = None, 
                       **kwargs):
        
        self.motif = motif
        self.cell  = cell
        self.name  = name
        self.tags = kwargs

    def __getattr__(self, attr):
        if 'tags' not in self.__dict__:
            self.tags = {}
        # if attr not in self.__dict__:
        #     raise AttributeError(f"{self.__class__.__name__} object has no attribute or tag {attr}")
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
        
        shared_tags = set(self.tags.keys()).intersection(set(other.tags.keys()))
        for tag in shared_tags:
            if np.isscalar(self.tags[tag]):
                if self.tags[tag] != other.tags[tag]:
                    return False
            elif isinstance(self.tags[tag], np.ndarray):
                if not np.allclose(self.tags[tag], other.tags[tag]):
                    return False
            elif isinstance(self.tags[tag], list):
                for i, i_ in zip(self.tags[tag], other.tags[tag]):
                    if i != i_:
                        return False
                        
        return True
    
    def __ne__(self, other):
        return not self.__eq__(other)
    
    def astype(self, dtype):
        """Returns copy of the :class:`PeriodicSet` with ``.motif`` 
        and ``.cell`` casted to ``dtype``."""
        
        return PeriodicSet(self.motif.astype(dtype), 
                           self.cell.astype(dtype),
                           name=self.name,
                           **self.tags)

    def to_dict(self) -> dict:
        """Return ``dict`` with scalar data in the tags of the :class:`PeriodicSet`.
        
        Format of returned ``dict`` is for example::
        
            {
                'density': 1.231, 
                'family': 'CBMZPN', 
                ...
            }
        """
        
        data = {}
        
        for tag in self.tags:
            if np.isscalar(self.tags[tag]):
                data[tag] = self.tags[tag]

        return data