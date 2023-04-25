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
against all in the other. For example, to compare crystals in 'crystals.cif' by
AMD with k = 100::

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

:func:`amd.compare() <amd.compare.compare>` reads crystals from CIFs,
calculates descriptors and compares them. These steps can be done separately,
e.g. to analyse the descriptors themselves. The line of code above using
:func:`amd.compare() <amd.compare.compare>` is equivalent to the following::

    import pandas as pd
    from scipy.spatial.distance import squareform

    crystals = list(amd.CifReader('crystals.cif'))          # read crystals
    amds = [amd.AMD(crystal, 100) for crystal in crystals]  # calculate AMDs
    dm = squareform(amd.AMD_pdist(amds))                    # compare AMDs pairwise
    names = [crystal.name for crystal in crystals]
    distance_matrix = pd.DataFrame(dm, index=names, columns=names)

:class:`amd.CifReader <amd.io.CifReader>` reads a CIF and yields the crystals
represented as :class:`amd.PeriodicSet <amd.periodicset.PeriodicSet>` objects.
A :class:`amd.PeriodicSet <amd.periodicset.PeriodicSet>` can be passed to
:func:`amd.AMD() <amd.calculate.AMD>` to calculate its AMD (similarly pass to 
:func:`amd.PDD() <amd.calculate.PDD>` for the PDD).
Here, :func:`amd.AMD_pdist() <amd.compare.AMD_pdist>` is used to compare the AMDs pairwise, returning a condensed distance matrix (see
:func:`scipy.spatial.distance.squareform`, which converts it to a symmetric 2D distance matrix). There is
an equivalent function for comparing PDDs, :func:`amd.PDD_pdist() <amd.compare.PDD_pdist>`. There are also two cdist functions, which take
two collections of descriptors and compares everything in one set with the other returning a 2D distance matrix.

Optional parameters of :func:`amd.compare() <amd.compare.compare>`
------------------------------------------------------------------

:func:`amd.compare() <amd.compare.compare>` reads crystals, computes their
invariants and compares them in one function for convinience. It accepts
most of the optional parameters from these steps, all are listed below.

Reading options
^^^^^^^^^^^^^^^

Parameters of :class:`amd.CifReader <amd.io.CifReader>` or :class:`amd.CSDReader <amd.io.CSDReader>`.

* :code:`reader` (default ``gemmi``) controls the backend package used to parse the file. Accepts ``gemmi``, ``pycodcif``, ``pymatgen``, ``ase`` and ``ccdc`` (if installed). The ccdc reader can read formats accepted by :class:`ccdc.io.EntryReader`.
* :code:`remove_hydrogens` (default ``False``) removes Hydrogen atoms from the structure.
* :code:`disorder` (default ``skip``) controls how disordered structures are handled. The default skips any crystal with disorder, since disorder conflicts with the model of a periodic set. Alternatively, :code:`ordered_sites` removes atoms with disorder and :code:`all_sites` includes all atoms regardless of disorder.
* :code:`show_warnings` (default ``True``) chooses whether to print warnings during reading, e.g. from disordered structures or crystals with missing data.
* :code:`verbose` (default ``False``) prints a progress bar showing the number of items read so far.
* :code:`heaviest_component` (``csd-python-api`` only, default ``False``) removes all but the heaviest connected molecule in the asymmetric unit, intended for removing solvents.
* :code:`molecular_centres` (``csd-python-api`` only, default ``False``) uses molecular centres of mass instead of atoms as the motif of the periodic set.
* :code:`csd_refcodes` (``csd-python-api`` only, default ``False``) interprets the string(s) given as CSD refcodes.
* :code:`families` (``csd-python-api`` only, default ``False``) interprets the list of strings given as CSD refcode families and reads all crystals in those families.

PDD options
^^^^^^^^^^^

Parameters of :func:`amd.PDD() <amd.calculate.PDD>`. :func:`amd.AMD() <amd.calculate.AMD>` does not accept any optional parameters.

* :code:`collapse` (default ``True``) chooses whether to collpase rows of PDDs which are similar enough (elementwise).
* :code:`collapse_tol` (default 0.0001) is the tolerance for collapsing PDD rows into one. The merged row is the average of those collapsed. 

Comparison options
^^^^^^^^^^^^^^^^^^

The parameters ``n_jobs`` and ``verbose`` below only apply to PDD comparisons, and ``low_memory`` only applies to AMD comparisons.

* :code:`metric` (default ``chebyshev``) chooses the metric used to compare AMDs or PDD rows (the metric used for PDDs is always Earth Mover's distance, which requires a chosen 'base' metric to compare rows). See `SciPy's cdist/pdist <https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html#scipy-spatial-distance-pdist>`_ for a list of accepted metrics.
* :code:`n_jobs` (requires ``by='PDD'``, default ``None``) is the number of cores to use for multiprocessing (passed to :class:`joblib.Parallel`). Pass -1 to use the maximum.
* :code:`backend` (requires ``by='PDD'``, default ``multiprocessing``) is the parallelization backend implementation for PDD comparisons.
* :code:`verbose` (requires ``by='PDD'``, default ``False``) controls the verbosity level. With parallel processing the verbose argument of :class:`joblib.Parallel` is used, otherwise ``tqdm`` is used.
* :code:`low_memory` (requires ``by='AMD'`` and ``metric='chebyshev'``, default ``False``) uses a slower algorithm with a smaller memory footprint, better for large input sizes.