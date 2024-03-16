Getting Started
===============

Use pip to install average-minimum-distance::

    pip install average-minimum-distance

Then import average-minimum-distance with ``import amd``.

Compare crystals in .CIF files with :func:`amd.compare() <amd.compare.compare>`
-------------------------------------------------------------------------------

:func:`amd.compare() <amd.compare.compare>` extracts crystals from CIF files
and compares them by AMD or PDD. Passing one path to a CIF will compare all
crystals in the CIF with each other, passing two will compare all in one
against all in the other. For example, to compare crystals by AMD with k = 100::

    >>> import amd
    >>> distance_matrix = amd.compare('crystals.cif', by='AMD', k=100)
    >>> print(distance_matrix)

               crystal_1  crystal_2  crystal_3  crystal_4
    crystal_1   0.000000   1.319907   0.975221   2.023880
    crystal_2   1.319907   0.000000   0.520115   0.703973
    crystal_3   0.975221   0.520115   0.000000   1.072211
    crystal_4   2.023880   0.703973   1.072211   0.000000

The distance matrix returned is a :class:`pandas.DataFrame`.

If csd-python-api is installed, :func:`amd.compare() <amd.compare.compare>`
can also accept lists of CSD refcodes (pass ``refcode_families=True``) or other
formats (pass ``reader='ccdc'``).

Read CIFs, calculate descriptors and compare separately
-------------------------------------------------------

:func:`amd.compare() <amd.compare.compare>` reads from CIFs,
calculates the descriptors and compares them, but these steps can be done
separately if needed. The code above using
:func:`amd.compare() <amd.compare.compare>` is equivalent to the following::

    import pandas as pd
    from scipy.spatial.distance import squareform

    crystals = list(amd.CifReader('crystals.cif'))          # read crystals
    amds = [amd.AMD(crystal, 100) for crystal in crystals]  # calculate AMDs, k=100
    cdm = amd.AMD_pdist(amds)                               # compare AMDs pairwise
    dm = squareform(cdm)                                    # condensed -> square distance matrix
    names = [crystal.name for crystal in crystals]
    distance_matrix = pd.DataFrame(dm, index=names, columns=names)

:class:`amd.CifReader <amd.io.CifReader>` reads a CIF and yields the crystals
represented as :class:`amd.PeriodicSet <amd.periodicset.PeriodicSet>` objects.
A :class:`amd.PeriodicSet <amd.periodicset.PeriodicSet>` can be passed to
:func:`amd.AMD() <amd.calculate.AMD>` to calculate its AMD (similar to 
:func:`amd.PDD() <amd.calculate.PDD>`). Then
:func:`amd.AMD_pdist() <amd.compare.AMD_pdist>` is used to compare the
AMDs pairwise, returning a condensed distance matrix which is converted to a
symmetric square distance matrix with :func:`scipy.spatial.distance.squareform`.

The equivalent function for comparing PDDs is :func:`amd.PDD_pdist() <amd.compare.PDD_pdist>`.
There is also :func:`amd.AMD_cdist() <amd.compare.AMD_cdist>` and
:func:`amd.PDD_cdist() <amd.compare.PDD_cdist>`, which take
two collections of AMDs/PDDs and compares everything in one set with the other
returning a 2D distance matrix.

Optional parameters of :func:`amd.compare() <amd.compare.compare>`
------------------------------------------------------------------

:func:`amd.compare() <amd.compare.compare>` accepts nearly all optional parameters
included in reading CIFs, computing descriptors or comparing, listed below.

Comparison options
^^^^^^^^^^^^^^^^^^

The parameters ``n_jobs`` and ``verbose`` below only apply to PDD comparisons, and ``low_memory`` only applies to AMD comparisons.

* :code:`by` (default ``AMD``) chooses whether to compare by AMD or PDD invariants.
* :code:`k` (default ``100``) is the parameter passed to AMD or PDD, the number of nearest neighbor atoms to consider.
* :code:`n_neighbors` (default ``None``) if given, only finds a number of nearest neighbors for each item rather than a full distance matrix.
* :code:`metric` (default ``chebyshev``) chooses the metric used to compare AMDs or PDD rows (the metric used for PDDs is always Earth Mover's distance, which requires a chosen metric to compare PDD rows). See `SciPy's cdist/pdist <https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html#scipy-spatial-distance-pdist>`_ for a list of accepted metrics.
* :code:`n_jobs` (requires ``by='PDD'``, default ``None``) is the number of cores to use for multiprocessing comparisons (passed to :class:`joblib.Parallel`). Pass -1 to use the maximum.
* :code:`backend` (requires ``by='PDD'``, default ``multiprocessing``) is the parallelization backend implementation for PDD comparisons.
* :code:`verbose` (requires ``by='PDD'``, default ``False``) controls the verbosity level. With parallel processing the verbose argument of :class:`joblib.Parallel` is used, otherwise ``tqdm`` is used.
* :code:`low_memory` (requires ``by='AMD'`` and ``metric='chebyshev'``, default ``False``) uses a slower algorithm with a smaller memory footprint, better for large input sizes.
* :code:`**kwargs` are additional keyword arguments passed to `SciPy's cdist/pdist <https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html#scipy-spatial-distance-pdist>`_ or `scikit-learn's NearestNeighbors <https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html>`_.

Reading options
^^^^^^^^^^^^^^^

Parameters of :class:`amd.CifReader <amd.io.CifReader>` or :class:`amd.CSDReader <amd.io.CSDReader>`.

* :code:`reader` (default ``gemmi``) controls the backend package used to parse the file. Accepts ``gemmi``, ``pycodcif``, ``pymatgen``, ``ase`` and ``ccdc`` (if installed). The ccdc reader can read any format accepted by :class:`ccdc.io.EntryReader`.
* :code:`remove_hydrogens` (default ``False``) removes Hydrogen atoms from the structure.
* :code:`disorder` (default ``skip``) controls how disordered structures are handled. The default skips (ignores) any crystal with disorder, since disorder conflicts with the model of a periodic set. Alternatively, :code:`ordered_sites` removes sites with disorder and :code:`all_sites` includes all sites regardless of disorder.
* :code:`show_warnings` (default ``True``) chooses whether to print warnings during reading, e.g. from disordered structures or crystals with missing data.
* :code:`verbose` (default ``False``) displays progress bar if True.
* :code:`heaviest_component` (``csd-python-api`` only, default ``False``) removes all but the heaviest connected molecule in the asymmetric unit, intended for removing solvents.
* :code:`molecular_centres` (``csd-python-api`` only, default ``False``) makes periodic sets using molecular centers of mass rather than atomic centers.
* :code:`csd_refcodes` (``csd-python-api`` only, default ``False``) interprets all inputs as CSD refcodes.
* :code:`families` (``csd-python-api`` only, default ``False``) interprets the list of strings given as CSD refcode families and reads all crystals in those families.

PDD options
^^^^^^^^^^^

Parameters of :func:`amd.PDD() <amd.calculate.PDD>`. :func:`amd.AMD() <amd.calculate.AMD>` does not accept any optional parameters.

* :code:`collapse` (default ``True``) chooses whether to collpase rows of PDDs which are similar enough (elementwise).
* :code:`collapse_tol` (default 0.0001) is the tolerance for considering PDD rows as the same and collapsing them into one. The merged row is the average of those collapsed. 
