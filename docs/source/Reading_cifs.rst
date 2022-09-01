Reading cifs
============

If you have a .cif file, use :class:`amd.CifReader` to extract the crystals::

    # create list of crystals in a .cif
    crystals = list(amd.CifReader('file.cif'))

    # Can also accept path to a directory, reading all files inside
    crystals = list(amd.CifReader('path/to/folder'))

    # loop over the reader and get AMDs (k=100) of crystals
    amds = []
    for p_set in amd.CifReader('file.cif'):
        amds.append(amd.AMD(p_set, 100))

The :class:`CifReader` returns :class:`.periodicset.PeriodicSet` objects representing the crystals, 
which can be passed to :func:`amd.AMD` or :func:`amd.PDD` to calculate their invariants. 
The :class:`PeriodicSet` has attributes ``.name``, ``.motif``, ``.cell``, ``.types`` (atomic numbers), 
as well as ``.asymmetric_unit`` and ``.wyckoff_multiplicities`` for use in calculations. When the path
is to a folder, it will also have an attribute ``.filename``.

:class:`CifReader` can accept other file formats, if you have csd-python-api installed. This should work
if you pass ``reader='ccdc'`` to the reader, though formats other than .cif have not been tested.

Reading options
---------------

:class:`amd.io.CifReader` accepts the following parameters (many shared by :class:`.io.CSDReader`)::

    amd.CifReader(
        'file.cif',                  # path to file or folder
        reader='ase',                # backend cif parser
        remove_hydrogens=False,      # remove Hydrogens
        disorder='skip',             # handling disorder
        heaviest_component=False,    # just keep the heaviest component in asym unit
        show_warnings=True           # silence warnings
    )

* :code:`reader` controls the backend package used to parse the file. The default is :code:`ase`; to use csd-python-api change to :code:`ccdc`. The ccdc reader should be able to read any format accepted by :class:`ccdc.io.EntryReader`, though only .cifs have been tested.
* :code:`remove_hydrogens` removes Hydrogen atoms from the structure.
* :code:`disorder` controls how disordered structures are handled. The default is to ``skip`` any crystal with disorder, since disorder conflicts with the periodic set model. To read disordered structures anyway, choose either :code:`ordered_sites` to remove sites with disorder or :code:`all_sites` include all sites regardless.
* :code:`heaviest_component` csd-python-api only. Removes all but the heaviest molecule in the asymmetric unit, intended for removing solvents.
* :code:`show_warnings` will not print warnings during reading if False, e.g. from disordered structures or crystals with missing data.

See the references :class:`.io.CifReader` or :class:`.periodicset.PeriodicSet` for more.