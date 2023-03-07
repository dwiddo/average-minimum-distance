Reading cifs
============

If you have a .cif file, use :class:`amd.CifReader <amd.io.CifReader>` to extract the crystals::

    # list of crystals in a .cif
    crystals = list(amd.CifReader('file.cif'))

    # also accepts a directory, reading all cifs inside
    crystals = list(amd.CifReader('path/to/folder'))

    # loop over the reader and get AMDs (k=100) of crystals
    amds = [amd.AMD(c, 100) for c in amd.CifReader('file.cif')]

The :class:`CifReader <amd.io.CifReader>` yields :class:`PeriodicSet <amd.periodicset.PeriodicSet>` objects representing the crystals, 
which can be passed to :func:`amd.AMD() <amd.calculate.AMD>` or :func:`amd.PDD() <amd.calculate.PDD>` to calculate their invariants. 
The :class:`PeriodicSet <amd.periodicset.PeriodicSet>` has attributes ``name``, ``motif``, ``cell``, ``types`` (atomic numbers), 
as well as ``asymmetric_unit`` and ``wyckoff_multiplicities`` for use in AMD/PDD calculations.

*CSD Python API only*: :class:`CifReader <amd.io.CifReader>` can accept other file formats when passed ``reader='ccdc'``.

Reading options
---------------

:class:`CifReader <amd.io.CifReader>` accepts the following parameters (many shared with :class:`amd.CSDReader <amd.io.CSDReader>`)::

    amd.CifReader(
        'file.cif',                # path to file/folder
        reader='ase',              # backend cif parser
        remove_hydrogens=False,    # remove Hydrogens
        disorder='skip',           # how to handle disorder
        heaviest_component=False,  # keep only heaviest molecule (CSD Python API only)
        molecular_centres=False,   # take molecular centres as the motif (CSD Python API only)
        show_warnings=True,        # silence warnings
        verbose=False              # print number of items processed
    )

* :code:`reader` (default ``gemmi``) controls the backend package used to parse the file. Accepts ``gemmi``, ``pycodcif``, ``pymatgen``, ``ase`` and ``ccdc`` (if these packages are installed). The ccdc reader can read formats accepted by :class:`ccdc.io.EntryReader`.
* :code:`remove_hydrogens` (default ``False``) removes Hydrogen atoms from the structure.
* :code:`disorder` (default ``skip``) controls how disordered structures are handled. The default skips any crystal with disorder, since disorder conflicts somewhat with the periodic set model. Alternatively, :code:`ordered_sites` removes atoms with disorder and :code:`all_sites` includes all atoms regardless.
* :code:`heaviest_component` (``reader='ccdc'`` only, default ``False``) removes all but the heaviest connected molecule in the asymmetric unit, intended for removing solvents.
* :code:`molecular_centres` (``reader='ccdc'`` only, default ``False``) uses molecular centres of mass instead of atoms as the motif of the periodic set.
* :code:`show_warnings` (default ``True``) chooses whether to print warnings during reading, e.g. from disordered structures or crystals with missing data.
* :code:`verbose` (default ``False``) prints a progress bar showing the number of items read so far.

See the references :class:`amd.io.CifReader` or :class:`amd.periodicset.PeriodicSet` for more.