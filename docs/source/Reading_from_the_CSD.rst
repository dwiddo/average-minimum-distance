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

The :class:`CifReader <amd.io.CifReader>` yields :class:`PeriodicSet <amd.periodicset.PeriodicSet>` objects representing the crystals, 
which can be passed to :func:`amd.AMD() <amd.calculate.AMD>` or :func:`amd.PDD() <amd.calculate.PDD>` to calculate their invariants. 

Reading options
---------------

:class:`CSDReader <amd.io.CSDReader>` accepts the following parameters (many shared by :class:`CifReader <amd.io.CifReader>`)::

    amd.CSDReader(
        refcodes=None,              # list of refcodes (or families) or 'CSD' 
        families=False,             # interpret refcodes as families
        remove_hydrogens=False,     # remove Hydrogens
        skip_disorder=False,        # skip disordered strucures
        missing_coords='warn',      # warn or skip if missing coordinates
        eq_site_tol=1e-3,           # tolerance for equivalent atomic positions
        show_warnings=True,         # silence warnings
        verbose=False,              # print number of items processed
        heaviest_component=False,   # keep only heaviest molecule
        molecular_centres=False     # take molecular centres as the motif
    )

* :code:`families` (default ``False``) will interpret the list of strings given as refcode families, i.e. all crystals with refcodes starting with any in the list are read.
* :code:`remove_hydrogens` (default ``False``) removes Hydrogen atoms from the structure.
* :code:`skip_disorder` (default ``False``) does not read structures with disorder.
* :code:`missing_coords` (default ``'warn'``) warns on missing coordinates, or skips structures with missing coordinates if set to ``'skip'``.
* :code:`eq_site_tol` (default ``1e-3``) sets the threshold below which atoms are conisdered to be in the same position.
* :code:`show_warnings` (default ``True``) chooses whether to print warnings during reading, e.g. from disordered structures or crystals with missing data.
* :code:`verbose` (default ``False``) prints a progress bar showing the number of items read so far.
* :code:`heaviest_component` (default ``False``) removes all but the heaviest connected molecule in the asymmetric unit, intended for removing solvents.
* :code:`molecular_centres` (default ``False``) uses molecular centres of mass instead of atoms as the motif of the periodic set.

See the references :class:`amd.io.CSDReader` or :class:`amd.periodicset.PeriodicSet` for more.