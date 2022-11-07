"""
average-minimum-distance: isometrically invariant crystal fingerprints
======================================================================

Documentation is available in the docstrings and online at
https://average-minimum-distance.readthedocs.io.

List of modules
***************

===================    ==============================================================
Module                 Description
===================    ==============================================================
:mod:`.calculate`      Calculate invariants (descriptors) of periodic sets (crystals)
:mod:`.compare`        Compare invariants of crystals
:mod:`.io`             Read periodic sets from a file or the CSD
:mod:`.periodicset`    Implements the PeriodicSet object
:mod:`.utils`          Utility functions
===================    ==============================================================
"""

__version__ = '1.3.3a3'
__author__ = 'Daniel Widdowson'
__maintainer__ = 'Daniel Widdowson'
__email__ = 'D.E.Widdowson@liverpool.ac.uk'
__license__ = "CC-BY-NC-SA-4.0"
__copyright__ = "Copyright 2022, Daniel Widdowson"


from .calculate import *
from .compare import *
from .io import *
from .periodicset import *
from .utils import *
