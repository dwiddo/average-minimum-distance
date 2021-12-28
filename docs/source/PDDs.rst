Using PDDs
==========

Calculating PDDs
----------------

The PDD (pointwise distance distribution) of a crystal is given by ``amd.PDD()``.
It accepts a crystal and an integer k, returning the :math:`\text{PDD}_k` as a matrix. 

If you have a .cif file, use :class:`amd.CifReader` to read the crystals. If
``csd-python-api`` is installed and you have CSD refcodes, use :class:`amd.CSDReader`.
You can also give the coordinates of motif points and unit cell as a tuple of numpy 
arrays, in Cartesian form. Examples::

    # get PDDs of crystals in a .cif
    crystals = list(amd.CifReader('file.cif'))
    pdds = [amd.PDD(crystal, 100) for crystal in crystals]

    # get PDDs of crystals in DEBXIT family
    pdds = [amd.PDD(crystal, 100) for crystal in amd.CSDReader('DEBXIT', families=True)]

    # PDD of 3D cubic lattice
    motif = np.array([[0,0,0]])
    cell  = np.identity(3)
    cubic_pdd = amd.PDD((motif, cell), 100)

Each PDD returned by ``amd.PDD(c, k)`` is a matrix with ``k`` columns. 

Calculation options
*******************

``amd.PDD`` accepts a few optional arguments (not relevant to ``amd.AMD``)::

    amd.PDD(periodic_set, k, order=True, collapse=True, collapse_tol=1e-4)

``order`` lexicograpgically orders the rows of the PDD, and ``collapse`` merges rows
if all the elements are within ``collapse_tol``. The technical definition of PDD 
requires doing both in order for PDD to satisfy invariance, but sometimes it's useful to 
disable the behaviours, particularly so that the PDD rows are in the same order as the 
passed motif points. Without ordering and collapsing, isometric inputs could give different 
PDDs; but the Earth mover's distance between the PDDs would still be 0.

Comparing by PDD
----------------

The `Earth mover's distance <https://en.wikipedia.org/wiki/Earth_mover%27s_distance>`_ is
an appropriate metric to compare PDDs, and the :mod:`amd.compare` module has functions
for these comparisons.

Most useful are :func:`.compare.PDD_pdist` and :func:`.compare.PDD_cdist`, which mimic
the interface of scipy's functions ``pdist`` and ``cdist``. ``pdist`` takes one set and 
compares all elements pairwise, whereas ``cdist`` takes two sets and compares elements in
one with the other. ``cdist`` returns a 2D distance matrix, but ``pdist`` returns a 
condensed distance matrix (see `scipy's pdist function <https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html>`_). 
The default metric for AMD comparisons is l-infinity, but it can be changed to any metric
accepted by scipy's pdist/cdist. ::

    # compare crystals in file1.cif with those in file2.cif by PDD, k=100
    pdds1 = [amd.PDD(crystal, 100) for crystal in amd.CifReader('file1.cif')]
    pdds2 = [amd.PDD(crystal, 100) for crystal in amd.CifReader('file2.cif')]
    distance_matrix = amd.PDD_cdist(pdds1, pdds2)

    # compare everything in file1.cif with each other
    condensed_dm = amd.PDD_pdist(pdds1)

You can compare one PDD with another with :func:`.compare.emd`::

    # compare DEBXIT01 and DEBXIT02 by PDD, k=100
    pdds = [amd.PDD(crystal, 100) for crystal in amd.CSDReader(['DEBXIT01', 'DEBXIT02'])]
    distance = amd.emd(pdds[0], pdds[1])

:func:`.compare.emd`, :func:`.compare.PDD_pdist` and :func:`.compare.PDD_cdist` all accept 
an optional argument ``metric``, which can be anything accepted by `scipy's pdist/cdist functions <https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html>`_. 
The metric used to compare PDD matrices is always Earth mover's distance, but this still requires another metric
between the rows of PDDs (so there's a different Earth mover's distance for each choice of metric).


Comparison options
******************

``amd.PDD_cdist`` and ``amd.PDD_pdist`` share the following optional arguments:

* ``metric`` chooses the metric used for comparison of PDD rows, as explained above. See scipy's cdist/pdist for a list of accepted metrics.
* ``k`` will truncate the passed PDDs to length ``k`` before comaparing (so ``k`` must not be larger than the passed PDDs). Useful if comparing for several ``k`` values.
* ``verbose`` (default ``False``) prints an ETA to the terminal. 