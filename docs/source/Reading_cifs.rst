Reading cifs
============

If you have a .cif file, use :class:`amd.CifReader <amd.io.CifReader>` to extract the crystals::

    # list of crystals in a .cif
    crystals = list(amd.CifReader('file.cif'))

    # also accepts a directory, reading all cifs inside
    crystals = list(amd.CifReader('path/to/folder'))

    # loop over the reader and get AMDs (k=100) of crystals
    amds = [amd.AMD(c, 100) for c in amd.CifReader('file.cif')]

    # .read() returns one crystal if there is one, otherwise a list
    crystal = amd.CifReader('single_crystal.cif').read()

The :class:`CifReader <amd.io.CifReader>` yields :class:`PeriodicSet <amd.periodicset.PeriodicSet>` objects representing the crystals, 
which can be passed to :func:`amd.AMD() <amd.calculate.AMD>` or :func:`amd.PDD() <amd.calculate.PDD>` to calculate their invariants. 

*CSD Python API only*: :class:`CifReader <amd.io.CifReader>` can accept other file formats when passed ``reader='ccdc'``.

Reading options
---------------

:class:`CifReader <amd.io.CifReader>` accepts the following parameters (many shared with :class:`amd.CSDReader <amd.io.CSDReader>`)::

    amd.CifReader(
        'file.cif',                # path to file/folder
        reader='gemmi',            # backend cif parser
        remove_hydrogens=False,    # remove Hydrogens
        skip_disorder=False,       # skip disordered strucures
        missing_coords='warn',     # warn or skip if missing coordinates
        eq_site_tol=1e-3,          # tolerance for equivalent atomic positions
        heaviest_component=False,  # keep only heaviest molecule (CSD Python API only)
        show_warnings=True,        # silence warnings
        verbose=False,             # print progress bar
        molecular_centres=False    # use molecular centres as the motif (CSD Python API only)
    )

* :code:`reader` (default ``'gemmi'``) controls the backend package used to parse the file. Accepts ``'gemmi'`` and ``'ccdc'`` (if installed). The ccdc reader can read formats accepted by :class:`ccdc.io.EntryReader`.
* :code:`remove_hydrogens` (default ``False``) removes Hydrogen atoms from the structure.
* :code:`skip_disorder` (default ``False``) does not read structures with disorder.
* :code:`missing_coords` (default ``'warn'``) warns on missing coordinates, or skips structures with missing coordinates if set to ``'skip'``.
* :code:`eq_site_tol` (default ``1e-3``) sets the threshold below which atoms are conisdered to be in the same position.
* :code:`show_warnings` (default ``True``) chooses whether to print warnings during reading, e.g. from disordered structures or crystals with missing data.
* :code:`verbose` (default ``False``) prints a progress bar showing the number of items read so far.
* :code:`heaviest_component` (``csd-python-api`` only, default ``False``) removes all but the heaviest connected molecule in the asymmetric unit, intended for removing solvents.
* :code:`molecular_centres` (``csd-python-api`` only, default ``False``) uses molecular centres of mass instead of atoms as the motif of the periodic set.

See the references :class:`amd.io.CifReader` or :class:`amd.periodicset.PeriodicSet` for more.