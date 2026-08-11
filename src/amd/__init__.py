"""
average-minimum-distance: geometry-based crystal descriptors
============================================================

Documentation is available in the docstrings and at
https://average-minimum-distance.readthedocs.io.

List of modules
***************

===================    ===================================================
Module                 Description
===================    ===================================================
:mod:`.calculate`      Calculate invariants (AMD, PDD, ADA, PDA)
:mod:`.compare`        Compare invariants
:mod:`.io`             Read crystals as PeriodicSets from files or the CSD
:mod:`.periodicset`    Implements the PeriodicSet object
:mod:`.reconstruct`    Implements the reconstruction PDD -> PeriodicSet
:mod:`.utils`          Utility functions
===================    ===================================================
"""

__version__ = "2.0.0a1"
__author__ = "Daniel Widdowson"
__maintainer__ = "Daniel Widdowson"
__email__ = "danwiddowson1@gmail.com"
__license__ = "CC-BY-NC-SA-4.0"
__copyright__ = "Copyright 2026, Daniel Widdowson"


from .calculate import *
from .compare import *
from .io import *
from .periodicset import *
from .utils import *
from .WeightedVectors import *

try:
    from ._private import *
except ModuleNotFoundError:
    pass
