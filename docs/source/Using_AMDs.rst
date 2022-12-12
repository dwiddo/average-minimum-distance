Using AMDs
==========

Calculation
-----------

The *average minimum distance* (AMD) of a crystal is given by :func:`amd.AMD() <amd.calculate.AMD>`. 
It accepts a crystal and an integer k, returning :math:`\text{AMD}_k` as a 1D NumPy array. 

If you have a .cif file, use :class:`amd.CifReader <amd.io.CifReader>` to read the crystals 
(see :doc:`Reading_cifs`). If csd-python-api is installed, :class:`amd.CSDReader <amd.io.CSDReader>`
accepts CSD refcodes (see :doc:`Reading_from_the_CSD`).

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

The object returned by ``amd.AMD(crystal, k)`` is a vector with k elements.

*Note:* The AMD of a crystal can be calculated from its PDD with :func:`amd.PDD_to_AMD() <amd.calculate.PDD_to_AMD>`,
which is faster if both are needed.

Comparison
----------

AMDs are just vectors that can be compared with any metric, but the :mod:`amd.compare <amd.compare>`
module has functions to compare collections of AMDs for you.

:func:`amd.AMD_pdist() <amd.compare.AMD_pdist>` and :func:`amd.AMD_cdist() <amd.compare.AMD_cdist>` 
are like SciPy's functions ``pdist`` and ``cdist``. ``pdist`` takes a set and compares all elements pairwise, 
whereas ``cdist`` takes two sets and compares elements in one with the other. 
``cdist`` returns a 2D distance matrix, but ``pdist`` returns a condensed distance matrix 
(see `SciPy's pdist <https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html>`_). 
The default metric for AMD comparisons is L-infinity (aka Chebyshev), but it can be changed to any metric
accepted by SciPy's pdist/cdist. ::

    # compare crystals in file1.cif with those in file2.cif by AMD, k=100
    amds1 = [amd.AMD(crystal, 100) for crystal in amd.CifReader('file1.cif')]
    amds2 = [amd.AMD(crystal, 100) for crystal in amd.CifReader('file2.cif')]
    distance_matrix = amd.AMD_cdist(amds1, amds2)

    # compare everything in file1.cif with each other (using L-infinity)
    condensed_dm = amd.AMD_pdist(amds1)

Comparison options
******************

:func:`amd.AMD_pdist() <amd.compare.AMD_pdist>` and :func:`amd.AMD_cdist() <amd.compare.AMD_cdist>` share the following optional arguments:

* :code:`metric` (default ``chebyshev``) chooses the metric used for comparison, see `SciPy's cdist/pdist <https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html>`_ for a list of accepted metrics.
* :code:`low_memory` (default ``False``, requires ``metric='chebyshev'``) uses a slower algorithm with a smaller memory footprint, for larger inputs.