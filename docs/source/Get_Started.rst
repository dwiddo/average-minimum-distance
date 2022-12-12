Get Started
===========

Comparing crystals
------------------

Compare crystals in one line
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:func:`amd.compare() <amd.compare.compare>` is one function for reading crystals
and comparing them by their AMD or PDD. For example, to compare all crystals in a
cif by PDD with k = 100::

    import amd
    df = amd.compare('crystals.cif', by='PDD', k=100)

A pandas DataFrame is returned, a table of the distance matrix with names in rows 
and columns. The function can also take a second path to a cif, and compare all
crystals in one with all in the other. To compare by AMD, use ``by='AMD'``.

If csd-python-api is installed, the compare function can also accept lists of
CSD refcodes, or paths to other file formats.

Read, calculate descriptors and compare separately 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:func:`amd.compare() <amd.compare.compare>` above does three things: reads crystals, calculates the AMD/PDD, and compares them. It is
often useful to do these steps separately, e.g. to save the descriptors to a file (see section below). The code above using
:func:`amd.compare() <amd.compare.compare>` is equivalent to::

    import amd
    import pandas as pd
    from scipy.spatial.distance import squareform

    crystals = list(amd.CifReader('crystals.cif'))          # read crystals
    pdds = [amd.PDD(crystal, 100) for crystal in crystals]  # calculate PDDs
    dm = squareform(amd.PDD_pdist(pdds))                    # compare PDDs
    names = [crystal.name for crystal in crystals]
    df = pd.DataFrame(dm, index=names, columns=names)

Here, :func:`amd.PDD_pdist() <amd.compare.PDD_pdist>` is used to compare the PDDs pairwise, returning a condensed distance matrix (see
:func:`scipy.spatial.distance.squareform`, which is used to convert to a symmetric 2D distance matrix). There is
an equivalent function for comparing AMDs, :func:`amd.AMD_pdist() <amd.compare.AMD_pdist>`. There are also two cdist functions, which take
two collections of descriptors and compares everything in one set with the other returning a 2D distance matrix,
used when :func:`amd.compare() <amd.compare.compare>` is given two paths.

Write crystals or their descriptors to file
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The pickle module is an easy way to store crystals or their descriptors. 
::

    import pickle

    with open('PDDs.pkl', 'wb') as f:
        pickle.dump(pdds, f)    # write

    with open('PDDs.pkl', 'rb') as f:
        pdds = pickle.load(f)   # read

Another example with h5py, which lazily writes the descriptors (only one crystal/PDD is in memory at a time). Preferable 
for large collections, though can only be used to store descriptors, not the objects given by the readers.
::

    import h5py

    # write
    with h5py.File('PDDs.hdf5', 'w', track_order=True) as file:
        for crystal in amd.CifReader('crystals.cif'):
            file.create_dataset(crystal.name, data=amd.PDD(crystal, 100))

    # read. modify as needed to read lazily
    with h5py.File('PDDs.hdf5', 'r', track_order=True) as file:
        pdds = [file[name][:] for name in file['/'].keys()]

Full list of optional parameters
--------------------------------

The compare function reads crystals, computes their invariants and compares them in one function for
convinience. It accepts most optional parameters of the crystal readers, invariant calculators and
functions which compare the invariants. All are listed below.

Reader options
^^^^^^^^^^^^^^

Parameters of :class:`amd.CifReader <amd.io.CifReader>` or :class:`amd.CSDReader <amd.io.CSDReader>`.

* :code:`reader` (default ``ase``) controls the backend package used to parse the file. To use csd-python-api change to ``ccdc``. The ccdc reader should be able to read any format accepted by :class:`ccdc.io.EntryReader`, though only cifs have been tested.
* :code:`remove_hydrogens` (default ``False``) removes Hydrogen atoms from the structure.
* :code:`disorder` (default ``skip``) controls how disordered structures are handled. The default skips any crystal with disorder, since disorder conflicts with the periodic set model. Alternatively, :code:`ordered_sites` removes sites with disorder and :code:`all_sites` includes all sites regardless.
* :code:`heaviest_component` (default ``False``, csd-python-api only) removes all but the heaviest molecule in the asymmetric unit, intended for removing solvents.
* :code:`molecular_centres` (default ``False``, csd-python-api only) uses the centres of molecules for comparisons instead of atoms (as in `our paper comparing across landscapes <https://pubs.acs.org/doi/10.1021/jacs.2c02653>`_).
* :code:`show_warnings` (default ``True``) chooses whether to print warnings during reading, e.g. from disordered structures or crystals with missing data.
* :code:`families` (default ``False``, csd-python-api only) chooses whether to read refcodes or refcode families from the CSD.

PDD options
^^^^^^^^^^^

Parameters of :func:`amd.PDD() <amd.calculate.PDD>`. :func:`amd.AMD() <amd.calculate.AMD>` does not accept any optional parameters.

* :code:`collapse` (default ``True``) chooses whether to collpase rows of PDDs which are similar enough (elementwise).
* :code:`collapse_tol` (default 0.0001) is the tolerance for collapsing PDD rows into one. The merged row is the average of those collapsed. 

Comparison options
^^^^^^^^^^^^^^^^^^

The first parameter ``metric`` below is available to :func:`amd.PDD_pdist() <amd.compare.PDD_pdist>`, :func:`amd.PDD_cdist() <amd.compare.PDD_cdist>`,
:func:`amd.AMD_pdist() <amd.compare.AMD_pdist>` and :func:`amd.AMD_cdist() <amd.compare.AMD_cdist>`. The rest are only for the
PDD related functions.

* :code:`metric` (default ``chebyshev``) chooses the metric used to compare AMDs or PDD rows. See SciPy's cdist/pdist for a list of accepted metrics.
* :code:`n_jobs` (new in 1.2.3, default ``None``) is the number of cores to use for multiprocessing (passed to :class:`joblib.Parallel`). Pass -1 to use the maximum.
* :code:`verbose` (changed in 1.2.3, default 0) controls the verbosity level, increasing with larger numbers. This is passed to :class:`joblib.Parallel`, see its documentation for details.
* :code:`low_memory` (default ``False``, requires ``by='AMD'`` and ``metric='chebyshev'``) uses a slower algorithm with a smaller memory footprint, for larger inputs.