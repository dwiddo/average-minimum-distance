"""
average-minimum-distance: geometry-based crystal descriptors
============================================================

Documentation is available in the docstrings and at
https://average-minimum-distance.readthedocs.io.

List of modules
***************

===================    ===================================
Module                 Description
===================    ===================================
:mod:`.calculate`      Calculate descriptors (AMD, PDD)
:mod:`.compare`        Compare descriptors
:mod:`.io`             Read crystals from files or the CSD
:mod:`.periodicset`    Implements the PeriodicSet object
:mod:`.utils`          General utility functions
===================    ===================================
"""

__version__ = '1.5.2'
__author__ = 'Daniel Widdowson'
__maintainer__ = 'Daniel Widdowson'
__email__ = 'D.E.Widdowson@liverpool.ac.uk'
__license__ = "CC-BY-NC-SA-4.0"
__copyright__ = "Copyright 2023, Daniel Widdowson"


from .calculate import *
from .compare import *
from .io import *
from .periodicset import *
from .utils import *
