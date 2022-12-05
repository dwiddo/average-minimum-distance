"""
average-minimum-distance: geometry-based crystal descriptors
============================================================

Documentation is available in the docstrings and online at
https://average-minimum-distance.readthedocs.io.

List of modules
***************

===================    ==============================================================
Module                 Description
===================    ==============================================================
:mod:`.calculate`      Calculate descriptors (invariants) of periodic sets (crystals)
:mod:`.compare`        Compare invariants of crystals
:mod:`.amdio`          Read crystals from a file or the CSD
:mod:`.periodicset`    Implements the PeriodicSet object
:mod:`.utils`          Utility functions
===================    ==============================================================
"""

__version__ = '1.3.5a2'
__author__ = 'Daniel Widdowson'
__maintainer__ = 'Daniel Widdowson'
__email__ = 'D.E.Widdowson@liverpool.ac.uk'
__license__ = "CC-BY-NC-SA-4.0"
__copyright__ = "Copyright 2022, Daniel Widdowson"


from .calculate import *
from .compare import *
from .amdio import *
from .periodicset import *
from .utils import *
