Using PDDs
==========

Calculation
-----------

The *pointwise distance distribution* (PDD) of a crystal is given by :func:`amd.PDD() <amd.calculate.PDD>`.
It accepts a crystal and an integer k, returning the :math:`\text{PDD}_k` as a 2D
NumPy array with k + 1 columns, the weights of each row being in the first column.

If you have a .cif file, use :class:`amd.CifReader <amd.io.CifReader>` to read the crystals 
(see :doc:`Reading_cifs`). If csd-python-api is installed, :class:`amd.CSDReader <amd.io.CSDReader>`
accepts CSD refcodes (see :doc:`Reading_from_the_CSD`).

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

The object returned by :func:`amd.PDD() <amd.calculate.PDD>` is a NumPy array with k + 1 columns. 

Calculation options
*******************

:func:`amd.PDD() <amd.calculate.PDD>` accepts a few optional arguments (not relevant to :func:`amd.AMD() <amd.calculate.AMD>`)::

    amd.PDD(
        periodic_set,              # input crystal (a PeriodicSet or tuple)
        k,                         # k value (number of neighbours)
        lexsort=True,              # lexicograpgically sort rows
        collapse=True,             # collpase identical rows (within tolerance)
        collapse_tol=1e-4,         # tolerance for collapsing rows
        return_row_groups=False    # return info about which rows collapsed
    )

``lexsort`` lexicograpgically orders the rows of the PDD, and ``collapse`` merges rows
if all elements of rows are within ``collapse_tol``. The definition of PDD requires both 
in order to technically satisfy invariance (that two isometric sets have equal PDDs). But,
Earth mover's distance does not depend on row order, so ``lexsort=False`` makes no 
difference to comparisons. Setting ``collapse=False`` will also not change Earth mover's 
distances, but comparisons may take longer.

Sometimes it's useful to know which rows of the PDD came from which motif points in the input.
Setting ``return_row_groups=True`` makes the function return a tuple ``(pdd, groups)``, where 
``groups[i]`` contains the indices of the point(s) corresponding to ``pdd[i]``. Note that these 
indices are for the asymmetric unit, whose indices in ``periodic_set.motif`` are 
accessible through ``periodic_set.asymmetric_unit`` if it exists.

Comparison
----------

The `Earth mover's distance <https://en.wikipedia.org/wiki/Earth_mover%27s_distance>`_ is
the appropriate metric to compare PDDs. The :mod:`amd.compare <amd.compare>` module contains functions 
for these comparisons.

:func:`amd.PDD_pdist() <amd.compare.PDD_pdist>` and :func:`amd.PDD_cdist() <amd.compare.PDD_cdist>` are like SciPy's functions 
``pdist`` and ``cdist``. ``pdist`` takes one set and compares all elements pairwise, 
whereas ``cdist`` takes two sets and compares elements in one with the other. 
``cdist`` returns a 2D distance matrix, but ``pdist`` returns a condensed distance matrix 
(see `SciPy's pdist function <https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html>`_). 

::

    # compare crystals in file1.cif with those in file2.cif by PDD, k=100
    pdds1 = [amd.PDD(crystal, 100) for crystal in amd.CifReader('file1.cif')]
    pdds2 = [amd.PDD(crystal, 100) for crystal in amd.CifReader('file2.cif')]
    distance_matrix = amd.PDD_cdist(pdds1, pdds2)

    # compare everything in file1.cif with each other
    condensed_dm = amd.PDD_pdist(pdds1)

You can compare one PDD with another with :func:`amd.EMD() <amd.compare.EMD>`::

    # compare DEBXIT01 and DEBXIT02 by PDD, k=100
    pdds = [amd.PDD(crystal, 100) for crystal in amd.CSDReader(['DEBXIT01', 'DEBXIT02'])]
    distance = amd.EMD(pdds[0], pdds[1])

:func:`amd.EMD() <amd.compare.EMD>`, :func:`amd.PDD_pdist() <amd.compare.PDD_pdist>` and :func:`amd.PDD_cdist() <amd.compare.PDD_cdist>` all accept 
an optional argument ``metric``, which can be anything accepted by `SciPy's pdist/cdist functions <https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html>`_. 
The metric used to compare PDDs is always Earth mover's distance, but this still requires another metric
between the rows of PDDs (so technically there's a different Earth mover's distance for each choice of metric).


Comparison options and multiprocessing
**************************************

:func:`amd.PDD_pdist() <amd.compare.PDD_pdist>` and :func:`amd.PDD_cdist() <amd.compare.PDD_cdist>` share the following optional arguments:

* :code:`metric` (default ``chebyshev``) chooses the metric used to compare PDD rows. See `SciPy's cdist/pdist <https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html>`_ for a list of accepted metrics.
* :code:`n_jobs` (new in 1.2.3, default ``None``) is the number of cores to use for multiprocessing (passed to :class:`joblib.Parallel`). Pass -1 to use the maximum.
* :code:`backend` (default ``multiprocessing``) is the parallelization backend implementation for PDD comparisons.
* :code:`verbose` (requires ``by='PDD'``, default ``False``) controls the verbosity level. With parallel processing the verbose argument of :class:`joblib.Parallel` is used, otherwise ``tqdm`` is used.
