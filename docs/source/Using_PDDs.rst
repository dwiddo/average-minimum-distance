Using PDDs
==========

Calculation
-----------

The pointwise distance distribution (PDD) of a crystal is given by ``amd.PDD()``.
It accepts a crystal and an integer k, returning the :math:`\text{PDD}_k` as a 2D
NumPy array with k+1 columns, the weights of each row being in the first column.

If you have a .cif file, use :class:`amd.io.CifReader` to read the crystals 
(see :doc:`Reading_cifs`). If you have CSD refcodes and ``csd-python-api`` is installed, 
use :class:`amd.io.CSDReader` (see :doc:`Reading_from_the_CSD`).

::

    # get PDDs of crystals in a .cif
    reader = amd.CifReader('file.cif')
    pdds = [amd.PDD(crystal, 100) for crystal in reader]

    # get PDDs of DEBXIT01 and DEBXIT02 from the CSD 
    crystals = list(amd.CSDReader(['DEBXIT01', 'DEBXIT02']))
    pdds = [amd.PDD(crystal, 50) for crystal in crystals]

You can also give the coordinates of motif points and unit cell as a tuple of numpy 
arrays, in Cartesian form::

    # PDD (k=10) of 3D cubic lattice
    motif = np.array([[0,0,0]])
    cell = np.identity(3)
    cubic_lattice = (motif, cell)
    cubic_pdd = amd.PDD(cubic_lattice, 10)

The object returned by ``amd.PDD`` is a NumPy array with ``k+1`` columns. 

Calculation options
*******************

``amd.PDD`` accepts a few optional arguments (not relevant to ``amd.AMD``)::

    amd.PDD(periodic_set, k, lexsort=True, collapse=True, collapse_tol=1e-4, return_row_groups=False)

``lexsort`` lexicograpgically orders the rows of the PDD, and ``collapse`` merges rows
if all elements of rows are within ``collapse_tol``. The definition of PDD requires both 
in order to satisfy invariance (that two isometric sets have equal PDDs). However,
earth mover's distance does not depend on the order of rows, so ``lexsort=False`` makes no 
difference to comparisons. Setting ``collapse=False`` will not change the earth mover's 
distances either, but comparisons can take longer if rows aren't collapsed.

Sometimes it's useful to know which rows of the PDD came from which motif points in the input.
Setting ``return_row_groups=True`` makes the function return a tuple ``(pdd, groups)``, where 
``groups[i]`` contains the indices of the point(s) corresponding to ``pdd[i]``. Note that these 
indices are for the asymmetric unit of the set, whose indices in ``periodic_set.motif`` are 
accessible through ``periodic_set.asymmetric_unit``.

Comparison
----------

The `Earth mover's distance <https://en.wikipedia.org/wiki/Earth_mover%27s_distance>`_ is
the appropriate metric to compare PDDs. The :mod:`amd.compare` module contains functions 
for these comparisons.

:func:`.compare.PDD_pdist` and :func:`.compare.PDD_cdist` are like scipy's functions 
``pdist`` and ``cdist``. ``pdist`` takes one set and compares all elements pairwise, 
whereas ``cdist`` takes two sets and compares elements in one with the other. 
``cdist`` returns a 2D distance matrix, but ``pdist`` returns a condensed distance matrix 
(see `scipy's pdist function <https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html>`_). 

::

    # compare crystals in file1.cif with those in file2.cif by PDD, k=100
    pdds1 = [amd.PDD(crystal, 100) for crystal in amd.CifReader('file1.cif')]
    pdds2 = [amd.PDD(crystal, 100) for crystal in amd.CifReader('file2.cif')]
    distance_matrix = amd.PDD_cdist(pdds1, pdds2)

    # compare everything in file1.cif with each other
    condensed_dm = amd.PDD_pdist(pdds1)

You can compare one PDD with another with :func:`.compare.EMD`::

    # compare DEBXIT01 and DEBXIT02 by PDD, k=100
    pdds = [amd.PDD(crystal, 100) for crystal in amd.CSDReader(['DEBXIT01', 'DEBXIT02'])]
    distance = amd.EMD(pdds[0], pdds[1])

:func:`.compare.EMD`, :func:`.compare.PDD_pdist` and :func:`.compare.PDD_cdist` all accept 
an optional argument ``metric``, which can be anything accepted by `scipy's pdist/cdist functions <https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html>`_. 
The metric used to compare PDD matrices is always Earth mover's distance, but this still requires another metric
between the rows of PDDs (so there's a different Earth mover's distance for each choice of metric).


Comparison options and multiprocessing
**************************************

``amd.PDD_cdist`` and ``amd.PDD_pdist`` share the following optional arguments:

* ``metric`` (default ``chebyshev``) chooses the metric used to compare PDD rows, as explained above. See scipy's cdist/pdist for a list of accepted metrics.
* ``n_jobs`` (new in 1.2.3, default ``None``) is the number of cores to use for multiprocessing (passed to ``joblib.Parallel``). Pass -1 to use the maximum.
* ``verbose`` (changed in 1.2.3, default 0) controls the verbosity level, increasing with larger numbers. This is passed to ``joblib.Parallel``, see their documentation for details.
