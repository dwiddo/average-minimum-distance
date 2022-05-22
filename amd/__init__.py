"""
average-minimum-distance: isometrically invariant crystal fingerprints
======================================================================

Documentation is available in the docstrings and
online at https://average-minimum-distance.readthedocs.io.
Using public functions from this amd does not require 
explicit import of the module.

List of modules
***************

===================    ===============================================================
Module                 Description
===================    ===============================================================
:mod:`.calculate`      Calculate invariants (fingerprints) of periodic sets (crystals)
:mod:`.compare`        Compare fingerprints of crystals
:mod:`.io`             Read periodic sets from a file or the CSD
:mod:`.periodicset`    Implements the PeriodicSet object
:mod:`.utils`          General utility functions/classes (eg diameter, eta)
===================    ===============================================================

"""

__version__ = '1.2.2'
__author__ = 'Daniel Widdowson'


from .calculate import *
from .compare import *
from .io import *
from .periodicset import PeriodicSet
from .pset_io import *
from .reconstruct import reconstruct
from .utils import *
