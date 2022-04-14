"""
average-minimum-distance: isometrically invariant crystal fingerprints
================================================
Documentation is available in the docstrings and
online at https://average-minimum-distance.readthedocs.io.
Using functions from this package does not require explicit import of
the relevant module.

Modules
-------
Using any of these subpackages requires an explicit import. For example,
``import scipy.cluster``.

::

 calculate   --- Calculate invariants (fingerprints) of periodic sets (crystals)
 compare     --- Compare fingerprints of crystals
 io          --- Read periodic sets from a file or the CSD
 periodicset --- Implements the PeriodicSet object
 pset_io     --- For writing/reading periodic sets + data with h5py
 reconstruct --- Implements function for recontructing a periodic set from PDD
 utils       --- General utility functions/classes (eg diameter, eta)
"""

from .io import *
from .pset_io import *
from .calculate import *
from .compare import *
from .utils import *
from .periodicset import PeriodicSet
from .reconstruct import reconstruct
