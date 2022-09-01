Quick reference
===============

amd.compare() is one function for reading crystals and comparing them by their AMD or PDD. 
For example, to compare all crystals in a cif by PDD with k = 100::

    import amd
    df = amd.compare('crystals.cif', by='PDD', k=100)

A pandas DataFrame is returned, a table of the distance matrix with names with rows and
columns indexed by name. The function can also take a second path to a cif, and compare
all crystals in one with all in the other. To compare by AMD, just change ``by='AMD'``.

If ``csd-python-api`` is installed, the compare function can also accept CSD refcodes instead of cif files.

Compare options
---------------

The compare function reads crystals, computes their invariants and compares them in one function for
convinience. It accepts many keyword arguments for reading, calculating and comparing, all listed in one place below
for convinience.


Reader options
**************

* :code:`reader` (default ``ase``) controls the backend package used to parse the file. To use csd-python-api change to :code:`ccdc`. The ccdc reader should be able to read any format accepted by :class:`ccdc.io.EntryReader`, though only .cifs have been tested.
* :code:`remove_hydrogens` (default ``False``) removes Hydrogen atoms from the structure.
* :code:`disorder` (default ``skip``) controls how disordered structures are handled. The default skips any crystal with disorder, since disorder conflicts with the periodic set model. Alternatively, :code:`ordered_sites` removes sites with disorder and :code:`all_sites` includes all sites regardless.
* :code:`heaviest_component` (default ``False``, ``csd-python-api`` only) removes all but the heaviest molecule in the asymmetric unit, intended for removing solvents.
* :code:`show_warnings` (default ``True``) chooses whether to print warnings during reading, e.g. from disordered structures or crystals with missing data.
* :code:`families` (default ``False``, ``csd-python-api`` only) chooses whether to read refcodes or refcode families.

PDD options
***********

* :code:`collapse` (default ``True``) chooses whether to collpase rows of PDDs which are similar enough (elementwise).
* :code:`collapse_tol` (default ``1e-4``) is the tolerance for collapsing PDD rows into one. The merged row is the average of those collapsed. 

Comparison options
******************

* :code:`metric` (default ``chebyshev``) chooses the metric used to compare AMDs or PDD rows. See SciPy's cdist/pdist for a list of accepted metrics.
* :code:`n_jobs` (new in 1.2.3, default ``None``) is the number of cores to use for multiprocessing (passed to ``joblib.Parallel``). Pass -1 to use the maximum.
* :code:`verbose` (changed in 1.2.3, default 0) controls the verbosity level, increasing with larger numbers. This is passed to ``joblib.Parallel``, see their documentation for details.
* :code:`low_memory` (default ``False``, by='AMD' only) uses an alternative slower algorithm that keeps memory use low for much larger inputs. Only ``metric='chebyshev'`` is accepted with ``low_memory``.