Reading cifs
============

If you have a .cif file, use :class:`amd.CifReader <amd.amdio.CifReader>` to extract the crystals::

    # create list of crystals in a .cif
    crystals = list(amd.CifReader('file.cif'))

    # Can also accept path to a directory, reading all files inside
    crystals = list(amd.CifReader('path/to/folder'))

    # loop over the reader and get AMDs (k=100) of crystals
    amds = []
    for p_set in amd.CifReader('file.cif'):
        amds.append(amd.AMD(p_set, 100))

The :class:`CifReader <amd.amdio.CifReader>` returns :class:`PeriodicSet <amd.periodicset.PeriodicSet>` objects representing the crystals, 
which can be passed to :func:`amd.AMD() <amd.calculate.AMD>` or :func:`amd.PDD() <amd.calculate.PDD>` to calculate their invariants. 
The :class:`PeriodicSet <amd.periodicset.PeriodicSet>` has attributes ``.name``, ``.motif``, ``.cell``, ``.types`` (atomic numbers), 
as well as ``.asymmetric_unit`` and ``.wyckoff_multiplicities`` for use in calculations.

csd-python-api only: :class:`CifReader <amd.amdio.CifReader>` can accept other file formats when passed ``reader='ccdc'``.

Reading options
---------------

The :class:`CifReader <amd.amdio.CifReader>` accepts the following parameters (many shared by :class:`amd.CSDReader <amd.amdio.CSDReader>`)::

    amd.CifReader(
        'file.cif',                  # path to file/folder
        reader='ase',                # backend cif parser
        remove_hydrogens=False,      # remove Hydrogens
        disorder='skip',             # handling disorder
        heaviest_component=False,    # only keep heaviest component in asym unit
        molecular_centres=False,     # read centres of molecules
        show_warnings=True           # silence warnings
    )

* :code:`reader` (default ``ase``) controls the backend package used to parse the file. The default is :code:`ase`; to use csd-python-api change to :code:`ccdc`. The ccdc reader should be able to read any format accepted by :class:`ccdc.io.EntryReader`, though only .cifs have been tested.
* :code:`remove_hydrogens` (default ``False``) removes Hydrogen atoms from the structure.
* :code:`disorder` (default ``skip``) controls how disordered structures are handled. The default skips any crystal with disorder, since disorder conflicts with the periodic set model. Alternatively, :code:`ordered_sites` removes sites with disorder and :code:`all_sites` includes all sites regardless.
* :code:`heaviest_component` (default ``False``, csd-python-api only) removes all but the heaviest molecule in the asymmetric unit, intended for removing solvents.
* :code:`molecular_centres` (default ``False``, csd-python-api only) uses centres of molecules instead of atoms as the motif of the periodic set.
* :code:`show_warnings` (default ``True``) chooses whether to print warnings during reading, e.g. from disordered structures or crystals with missing data.

See the references :class:`amd.amdio.CifReader` or :class:`amd.periodicset.PeriodicSet` for more.