Using AMDs
==========

Calculating AMDs
----------------

The AMD (average minimum distance) of a crystal is given by ``amd.AMD()``. 
It accepts a crystal and an integer k, returning :math:`\text{AMD}_k` as a vector. 

If you have a .cif file, use :class:`amd.CifReader` to read the crystals. If
``csd-python-api`` is installed and you have CSD refcodes, use :class:`amd.CSDReader`.
You can also give the coordinates of motif points and unit cell as a tuple of numpy 
arrays, in Cartesian form. Examples::

    # get AMDs of crystals in a .cif
    crystals = list(amd.CifReader('file.cif'))
    amds = [amd.AMD(crystal, 100) for crystal in crystals]

    # get AMDs of crystals in DEBXIT family
    amds = [amd.AMD(crystal, 100) for crystal in amd.CSDReader('DEBXIT', families=True)]

    # AMD of 3D cubic lattice
    motif = np.array([[0,0,0]])
    cell  = np.identity(3)
    cubic_amd = amd.AMD((motif, cell), 100)

Each AMD returned by ``amd.AMD(c, k)`` is a vector length ``k``.

*Note:* If you want both the AMD and PDD of a crystal, first get the PDD and then use 
``amd.PDD_to_AMD()`` to get the AMD from it to save time.

Comparing by AMD
----------------

Any metric can be used to compare AMDs, but the :mod:`amd.compare` module has functions to compare for you.

Most useful are :func:`.compare.AMD_pdist` and :func:`.compare.AMD_cdist`, which mimic
the interface of scipy's functions ``pdist`` and ``cdist``. ``pdist`` takes one set and 
compares all elements pairwise, whereas ``cdist`` takes two sets and compares elements in
one with the other. ``cdist`` returns a 2D distance matrix, but ``pdist`` returns a 
condensed distance matrix (see `scipy's pdist function <https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html>`_). 
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
* ``k`` will truncate the passed AMDs to length ``k`` before comaparing (so ``k`` must not be larger than the passed AMDs). Useful if comparing for several ``k`` values.
* ``low_memory`` (default ``False``) uses an alternative slower algorithm that keeps memory use low for much larger input sizes. Currently only ``metric='chebyshev'`` is accepted with ``low_memory``.