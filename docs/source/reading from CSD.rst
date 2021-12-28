Reading from the CSD
====================

If csd-python-api is installed, amd can use it to read crystals
directly from your local version of the CSD. 

:class:`amd.CSDReader` accepts refcode(s) and yields the chosen crystals. 
If ``None`` or ``'CSD'`` are passed instead of refcode(s), it reads the whole CSD. 
If the optional parameter ``families`` is ``True``, the refcode(s) given are 
interpreted as refcode families: any entry whose ID starts with the given string is included.

Here are some common patterns for the CSDReader::

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
    for periodic_set in amd.CSDReader(refcodes, families=True):
        amds.append(amd.AMD(periodic_set, 100))

The :class:`CSDReader` yields :class:`PeriodicSet` objects, which can be passed to :func:`amd.AMD` 
or :func:`amd.PDD`. The :class:`PeriodicSet` has attributes ``.name``, 
``.motif``, ``.cell``, ``.types`` (atomic types), and information about the asymmetric unit 
to help with AMD and PDD calculations.

See the references :class:`amd.io.CSDReader` or :class:`amd.periodicset.PeriodicSet` for more.

Optional parameters
-------------------

The :code:`CSDReader` accepts the following parameters (many shared by :class:`.io.CifReader`)::

    amd.CSDReader(refcodes=None,
                  families=False,
                  remove_hydrogens=False,
                  disorder='skip',
                  heaviest_component=False,
                  extract_data=None,
                  include_if=None)

* As described above, :code:`families` chooses whether to read refcodes or refcode families.
* :code:`remove_hydrogens` removes Hydrogen atoms from the structure.
* :code:`disorder` controls how disordered structures are handled. The default is to ``skip`` any crystal with disorder, since disorder conflicts with the periodic set model. To read disordered structures anyway, choose either :code:`ordered_sites` to remove sites with disorder or :code:`all_sites` include all sites regardless.
* :code:`heaviest_component` takes the heaviest connected molecule in the motif, intended for removing solvents. Only available when ``reader='ccdc'``.
* :code:`extract_data` is used to extract more data from crystals.
* :code:`include_if` can remove unwanted structures.
