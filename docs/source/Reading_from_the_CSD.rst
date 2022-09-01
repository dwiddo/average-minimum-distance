Reading from the CSD
====================

If csd-python-api is installed, ``amd`` can use it to read crystals directly from the CSD. 

:class:`amd.io.CSDReader` accepts a list of CSD refcode(s) and yields the crystals. 
If ``None`` or ``'CSD'`` are passed instead of refcodes, it reads the whole CSD::

    # Put crystals with these refcodes in a list
    refcodes = ['DEBXIT01', 'DEBXIT05', 'HXACAN01']
    structures = list(amd.CSDReader(refcodes))
        
    # Read refcode families (any whose refcode starts with strings in the list)
    refcodes = ['ACSALA', 'HXACAN']
    structures = list(amd.CSDReader(refcodes, families=True))

    # Create a generic reader, read crystals 'on demand' with CSDReader.entry()
    reader = amd.CSDReader()
    debxit01 = reader.entry('DEBXIT01')
    
    # looping over this generic reader will yield all CSD entries
    for periodic_set in reader:
        ...

    # Make list of AMDs for crystals in these families
    refcodes = ['ACSALA', 'HXACAN']
    amds = []
    for structure in amd.CSDReader(refcodes, families=True):
        amds.append(amd.AMD(structure, 100))

The :class:`CSDReader` returns :class:`.periodicset.PeriodicSet` objects representing the crystals,
which can be passed to :func:`amd.AMD` or :func:`amd.PDD` to calculate their invariants. 
The :class:`PeriodicSet` has attributes ``.name``, ``.motif``, ``.cell``, ``.types`` (atomic numbers), 
as well as ``.asymmetric_unit`` and ``.wyckoff_multiplicities`` for use in calculations.

Reading options
---------------

The :code:`amd.io.CSDReader` accepts the following parameters (many shared by :class:`.io.CifReader`)::

    amd.CSDReader(
        refcodes=None,              # list of refcodes (or families) or 'CSD' 
        families=False,             # interpret refcodes as families (include if refcode starts with)
        remove_hydrogens=False,     # remove Hydrogens
        disorder='skip',            # handling disorder
        heaviest_component=False,   # just keep the heaviest component in asym unit
        molecular_centres=False,    # read centres of molecules
        show_warnings=True          # silence warnings
    )

* As described above, :code:`families` chooses whether to read refcodes or refcode families.
* :code:`remove_hydrogens` removes Hydrogen atoms from the structure.
* :code:`disorder` controls how disordered structures are handled. The default is to ``skip`` any crystal with disorder, since disorder conflicts with the periodic set model. To read disordered structures anyway, choose either :code:`ordered_sites` to remove sites with disorder or :code:`all_sites` include all sites regardless.
* :code:`heaviest_component` takes the heaviest connected molecule in the motif, intended for removing solvents.
* :code:`molecular_centres` extracts the centres of molecules in the unit cell and stores in the attribute ``molecular_centres``.
* :code:`show_warnings` will not print warnings during reading if False, e.g. from disordered structures or crystals with missing data.

See the references :class:`amd.io.CSDReader` or :class:`amd.periodicset.PeriodicSet` for more.