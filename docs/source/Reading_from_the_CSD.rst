Reading from the CSD
====================

If csd-python-api is installed, ``amd`` can use it to read crystals directly from the CSD. 
:class:`amd.CSDReader <amd.io.CSDReader>` accepts a list of CSD refcode(s) and yields the crystals. 
If ``None`` or ``'CSD'`` are passed instead of refcodes, it reads the whole CSD::

    # Put crystals with these refcodes in a list
    refcodes = ['DEBXIT01', 'DEBXIT05', 'HXACAN01']
    structures = list(amd.CSDReader(refcodes))
        
    # Read refcode families (any whose refcode starts with strings in the list)
    refcodes = ['ACSALA', 'HXACAN']
    structures = list(amd.CSDReader(refcodes, families=True))

    # Giving the reader nothing reads from the whole CSD.
    for periodic_set in amd.CSDReader():
        ...

:class:`CSDReader <amd.io.CSDReader>` returns :class:`PeriodicSet <amd.periodicset.PeriodicSet>` objects representing the crystals, 
which can be passed to :func:`amd.AMD() <amd.calculate.AMD>` or :func:`amd.PDD() <amd.calculate.PDD>` to calculate their invariants. 
The :class:`PeriodicSet <amd.periodicset.PeriodicSet>` has attributes ``name``, ``motif``, ``cell``, ``types`` (atomic numbers), 
as well as ``asymmetric_unit`` and ``wyckoff_multiplicities`` for use in AMD/PDD calculations.

Reading options
---------------

:class:`CSDReader <amd.io.CSDReader>` accepts the following parameters (many shared by :class:`CifReader <amd.io.CifReader>`)::

    amd.CSDReader(
        refcodes=None,              # list of refcodes (or families) or 'CSD' 
        families=False,             # interpret refcodes as families
        remove_hydrogens=False,     # remove Hydrogens
        disorder='skip',            # how to handle disorder
        heaviest_component=False,   # keep only heaviest molecule
        molecular_centres=False,    # take molecular centres as the motif
        show_warnings=True          # silence warnings
    )

* :code:`families` (default ``False``) will interpret the list of strings given as refcode families, i.e. all crystals with refcodes starting with any in the list are read.
* :code:`remove_hydrogens` (default ``False``) removes Hydrogen atoms from the structure.
* :code:`disorder` (default ``skip``) controls how disordered structures are handled. The default is to ``skip`` any crystal with disorder, since disorder conflicts with the periodic set model. To read disordered structures anyway, choose either :code:`ordered_sites` to remove sites with disorder or :code:`all_sites` include all sites regardless.
* :code:`heaviest_component` (default ``False``) removes all but the heaviest molecule in the asymmetric unit, intended for removing solvents.
* :code:`molecular_centres` (default ``False``) uses centres of molecules instead of atoms as the motif of the periodic set.
* :code:`show_warnings` (default ``True``) chooses whether to print warnings during reading, e.g. from disordered structures or crystals with missing data.

See the references :class:`amd.io.CSDReader` or :class:`amd.periodicset.PeriodicSet` for more.