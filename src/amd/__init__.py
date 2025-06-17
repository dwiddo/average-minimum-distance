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

__version__ = "1.6.0"
__author__ = "Daniel Widdowson"
__maintainer__ = "Daniel Widdowson"
__email__ = "D.E.Widdowson@liverpool.ac.uk"
__license__ = "CC-BY-NC-SA-4.0"
__copyright__ = "Copyright 2025, Daniel Widdowson"


from .calculate import *
from .compare import *
from .io import *
from .periodicset import *
from .utils import *

try:
    from ._private import *
except ModuleNotFoundError:
    pass
