Using AMDs
==========

Calculating AMDs
----------------

The average minimum distance (AMD) of a crystal is given by ``amd.AMD()``. 
It accepts a crystal and an integer k, returning :math:`\text{AMD}_k` as a 1D NumPy array. 

If you have a .cif file, use :class:`amd.io.CifReader` to read the crystals 
(see :doc:`Reading_cifs`). If you have CSD refcodes and ``csd-python-api`` is installed, 
use :class:`amd.io.CSDReader` (see :doc:`Reading_from_the_CSD`).

::

    # get AMDs of crystals in a .cif
    crystals = list(amd.CifReader('file.cif'))
    amds = [amd.AMD(crystal, 100) for crystal in crystals]

    # get AMDs of crystals in DEBXIT family
    csd_reader = amd.CSDReader('DEBXIT', families=True)
    amds = [amd.AMD(crystal, 100) for crystal in csd_reader]

You can also give the coordinates of motif points and unit cell as a tuple of numpy 
arrays, in Cartesian form::

    # AMD (k=10) of 3D cubic lattice
    motif = np.array([[0,0,0]])
    cell = np.identity(3)
    cubic_lattice = (motif, cell)
    cubic_amd = amd.AMD(cubic_lattice, 10)

Each AMD returned by ``amd.AMD(crystal, k)`` is a vector length ``k``.

*Note:* The AMD of a crystal can be obtained from its PDD. If both are needed,
calculate the PDD and then use ``amd.calculate.PDD_to_AMD()``.

Comparing by AMD
----------------

AMDs are just vectors that can be compared with any metric, but the :mod:`amd.compare` module has functions to compare for you.

:func:`.compare.AMD_pdist` and :func:`.compare.AMD_cdist` are like scipy's functions 
``pdist`` and ``cdist``. ``pdist`` takes a set and compares all elements pairwise, 
whereas ``cdist`` takes two sets and compares elements in one with the other. 
``cdist`` returns a 2D distance matrix, but ``pdist`` returns a condensed distance matrix 
(see `scipy's pdist function <https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html>`_). 
The default metric for AMD comparisons is l-infinity, but it can be changed to any metric
accepted by scipy's pdist/cdist. ::

    # compare crystals in file1.cif with those in file2.cif by AMD, k=100
    amds1 = [amd.AMD(crystal, 100) for crystal in amd.CifReader('file1.cif')]
    amds2 = [amd.AMD(crystal, 100) for crystal in amd.CifReader('file2.cif')]
    distance_matrix = amd.AMD_cdist(amds1, amds2)

    # compare everything in file1.cif with each other (using l-inf)
    condensed_dm = amd.AMD_pdist(amds1)

Comparison options
******************

``amd.AMD_cdist`` and ``amd.AMD_pdist`` share the following optional arguments:

* ``metric`` chooses the metric used for comparison, see scipy's cdist/pdist for a list of accepted metrics.
* ``low_memory`` (default ``False``) uses an alternative slower algorithm that keeps memory use low for much larger inputs. Currently only ``metric='chebyshev'`` is accepted with ``low_memory``.