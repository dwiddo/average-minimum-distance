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

An option for large collections of PDDs is ``h5py``, which can read and write the descriptors lazily
(only one item is in memory at a time).
::

    import h5py

    # write lazily
    with h5py.File('PDDs.hdf5', 'w', track_order=True) as file:
        for crystal in amd.CifReader('crystals.cif'):
            file.create_dataset(crystal.name, data=amd.PDD(crystal, 100))

    # read lazily
    with h5py.File('PDDs.hdf5', 'r', track_order=True) as file:
        for name in file['/'].keys():
            pdd = file[name][:]
            ...

List of optional parameters
---------------------------

:func:`amd.compare() <amd.compare.compare>` reads crystals, computes their
invariants and compares them in one function for convinience. It accepts
most of optional parameters from any of these steps, all are listed below.

Reader options
^^^^^^^^^^^^^^

Parameters of :class:`amd.CifReader <amd.io.CifReader>` or :class:`amd.CSDReader <amd.io.CSDReader>`.

* :code:`reader` (default ``ase``) controls the backend package used to parse the file. Accepts ``ase``, ``pycodcif``, ``pymatgen``, ``gemmi`` and ``ccdc`` (if these packages are installed). The ccdc reader can read formats accepted by :class:`ccdc.io.EntryReader`.
* :code:`remove_hydrogens` (default ``False``) removes Hydrogen atoms from the structure.
* :code:`disorder` (default ``skip``) controls how disordered structures are handled. The default skips any crystal with disorder, since disorder conflicts somewhat with the periodic set model. Alternatively, :code:`ordered_sites` removes atoms with disorder and :code:`all_sites` includes all atoms regardless.
* :code:`show_warnings` (default ``True``) chooses whether to print warnings during reading, e.g. from disordered structures or crystals with missing data.
* :code:`heaviest_component` (default ``False``, CSD Python API only) removes all but the heaviest molecule in the asymmetric unit, intended for removing solvents.
* :code:`molecular_centres` (default ``False``, CSD Python API only) uses centres of molecules instead of atoms as the motif of the periodic set.
* :code:`families` (default ``False``, CSD Python API only) interprets the list of strings given as CSD refcode families and reads all crystals in those families.

PDD options
^^^^^^^^^^^

Parameters of :func:`amd.PDD() <amd.calculate.PDD>`. :func:`amd.AMD() <amd.calculate.AMD>` does not accept any optional parameters.

* :code:`collapse` (default ``True``) chooses whether to collpase rows of PDDs which are similar enough (elementwise).
* :code:`collapse_tol` (default 0.0001) is the tolerance for collapsing PDD rows into one. The merged row is the average of those collapsed. 

Comparison options
^^^^^^^^^^^^^^^^^^

The first parameter ``metric`` below is available to :func:`amd.PDD_pdist() <amd.compare.PDD_pdist>`, 
:func:`amd.PDD_cdist() <amd.compare.PDD_cdist>`, :func:`amd.AMD_pdist() <amd.compare.AMD_pdist>` and
:func:`amd.AMD_cdist() <amd.compare.AMD_cdist>`. ``n_jobs`` and ``verbose`` only apply to PDD comparisons and
``low_memory`` only applies to AMD comparisons.

* :code:`metric` (default ``chebyshev``) chooses the metric used to compare AMDs or PDD rows. See SciPy's cdist/pdist for a list of accepted metrics.
* :code:`n_jobs` (requires ``by='PDD'``, default ``None``) is the number of cores to use for multiprocessing (passed to :class:`joblib.Parallel`). Pass -1 to use the maximum.
* :code:`verbose` (requires ``by='PDD'``, default 0) controls the verbosity level, increasing with larger numbers. This is passed to :class:`joblib.Parallel`, see its documentation for details.
* :code:`low_memory` (requires ``by='AMD'`` and ``metric='chebyshev'``, default ``False``) uses a slower algorithm with a smaller memory footprint, better for large input sizes.