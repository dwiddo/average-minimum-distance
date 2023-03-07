Getting Started
===============

Comparing crystals
------------------

:func:`amd.compare() <amd.compare.compare>` extracts crystals from one or more CIFs
and compares them by AMD or PDD. For example, to compare all crystals in a
cif by AMD with k = 100::

    import amd
    df = amd.compare('crystals.cif', by='AMD', k=100)

To compare by PDD, use ``by='PDD'``. A distance matrix is returned as a pandas DataFrame. 
:func:`amd.compare() <amd.compare.compare>` can also take two paths to compare all
crystals in one file with those in the other.

If csd-python-api is installed, :func:`amd.compare() <amd.compare.compare>` can also accept lists of
CSD refcodes, or other formats.

Read, calculate descriptors and compare separately 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:func:`amd.compare() <amd.compare.compare>` reads crystals, calculates AMD or PDD, and compares them. It is
sometimes useful to do these steps separately, e.g. to save the descriptors to a file. The code above using
:func:`amd.compare() <amd.compare.compare>` is equivalent to the following::

    import amd
    import pandas as pd
    from scipy.spatial.distance import squareform

    crystals = list(amd.CifReader('crystals.cif'))          # read crystals
    amds = [amd.AMD(crystal, 100) for crystal in crystals]  # calculate AMDs
    dm = squareform(amd.AMD_pdist(amds))                    # compare AMDs pairwise
    names = [crystal.name for crystal in crystals]
    df = pd.DataFrame(dm, index=names, columns=names)

Here, :func:`amd.AMD_pdist() <amd.compare.AMD_pdist>` is used to compare the AMDs pairwise, returning a condensed distance matrix (see
:func:`scipy.spatial.distance.squareform`, which converts it to a symmetric 2D distance matrix). There is
an equivalent function for comparing PDDs, :func:`amd.PDD_pdist() <amd.compare.PDD_pdist>`. There are also two cdist functions, which take
two collections of descriptors and compares everything in one set with the other returning a 2D distance matrix.

Write crystals or their descriptors to a file
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``pickle`` is an easy way to store crystals or their descriptors. 
::

    import amd
    import pickle

    crystals = list(amd.CifReader('crystals.cif'))

    with open('crystals.pkl', 'wb') as f: # write
        pickle.dump(crystals, f)

    with open('crystals.pkl', 'rb') as f: # read
        crystals = pickle.load(f)

List of optional parameters
---------------------------

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
* :code:`heaviest_component` (``reader='ccdc'`` only, default ``False``) removes all but the heaviest connected molecule in the asymmetric unit, intended for removing solvents.
* :code:`molecular_centres` (``reader='ccdc'`` only, default ``False``) uses molecular centres of mass instead of atoms as the motif of the periodic set.
* :code:`families` (``reader='ccdc'`` only, default ``False``) interprets the list of strings given as CSD refcode families and reads all crystals in those families. Only applies if the inputs are CSD refcodes.
* :code:`verbose` (default ``False``) prints a progress bar showing the number of items read so far.

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