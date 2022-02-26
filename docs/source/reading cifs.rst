Reading .CIFs
=============

If you have a .CIF file, use :class:`amd.CifReader` to extract the crystals. Here are 
some common patterns when reading .CIFs::

    # loop over the reader and get AMDs (k=100) of crystals
    amds = []
    for p_set in amd.CifReader('file.cif'):
        amds.append(amd.AMD(p_set, 100))

    # create list of crystals in a .cif
    crystals = list(amd.CifReader('file.cif'))

    # Use folder=True to read all cifs in a folder
    crystals = list(amd.CifReader(folder_path, folder=True))

    # if you need the file names as well
    import os
    crystals = [amd.CifReader(os.path.join(folder_path, file)).read_one(), file
                for file in os.listdir(folder_path)]

The :class:`CifReader` yields :class:`PeriodicSet` objects, which can be passed to :func:`amd.AMD` 
or :func:`amd.PDD`. The :class:`PeriodicSet` has attributes ``.name``, 
``.motif``, ``.cell``, ``.types`` (atomic types), and information about the asymmetric unit 
to help with AMD and PDD calculations.

See the references :class:`amd.io.CifReader` or :class:`amd.periodicset.PeriodicSet` for more.

Reading options
---------------

:class:`CifReader` accepts the following parameters (many shared by :class:`.io.CSDReader`)::

    amd.CifReader('file.cif',                  # path to file/folder
                  reader='ase',                # backend cif parser
                  remove_hydrogens=False,      # remove H/D
                  disorder='skip',             # handling disorder
                  heaviest_component=False,    # remove 
                  extract_data=None,           # extract more data
                  include_if=None,             # ignore some crystals
                  folder=False)                # if path is to a folder

* :code:`reader` controls the backend package used to parse the file. The default is :code:`ase`; to use csd-python-api pass :code:`ccdc`. The ccdc reader can read any format accepted by `ccdc's EntryReader <https://downloads.ccdc.cam.ac.uk/documentation/API/modules/io_api.html#ccdc.io.EntryReader>`_, though only .CIFs have been tested.
* :code:`remove_hydrogens` removes Hydrogen atoms from the structure.
* :code:`disorder` controls how disordered structures are handled. The default is to ``skip`` any crystal with disorder, since disorder conflicts with the periodic set model. To read disordered structures anyway, choose either :code:`ordered_sites` to remove sites with disorder or :code:`all_sites` include all sites regardless.
* :code:`heaviest_component` takes the heaviest connected molecule in the motif, intended for removing solvents. Only available when ``reader='ccdc'``.
* :code:`extract_data` is used to extract more data from crystals, as in the example below.
* :code:`include_if` can remove unwanted structures, as in the example below.
* :code:`folder` will read all crystals from files in a folder.

An example only reading entries with a ``.polymorph`` label and extracting additional data::

    import amd
    from amd import ccdc_utils as cu

    def has_polymorph_label(entry):
        if entry.polymorph is None:
            return False
        return True

    include_if = [has_polymorph_label]

    # data to extract. can be any functions accepting an entry, but ccdc_utils has some useful ones
    extract = {
        'density':    cu.density,
        'formula':    cu.formula,
        'cell_str':   cu.cell_str_as_html,
        'spacegroup': cu.spacegroup
    }

    reader = amd.CifReader('file.cif', reader='ccdc', extract_data=extract, include_if=include_if)
    ...