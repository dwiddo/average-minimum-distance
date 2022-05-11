Write crystals and fingerprints to a file
=========================================

For large sets of crystals, parsing .CIF files can be slow. The readers in
:mod:`.io` yield :class:`.periodicset.PeriodicSet` objects which represent 
the crystals, and the  :class:`.io.SetWriter` and :class:`.io.SetReader` are 
for writing and reading these crystals to a compressed hdf5 file. ::

    # write the crystals and their AMDs to a file using the crystal's .tags
    # amd.SetReader and amd.SetWriter can both be used in context managers
    with amd.SetWriter('crystals_and_AMD100.hdf5') as writer:
        for crystal in amd.CifReader('file.cif'):
            crystal.tags['AMD100'] = amd.AMD(crystal, 100)
            writer.write(crystal)

    # read back the crystals and their AMDs
    crystals = list(amd.SetReader('crystals_and_AMD100.hdf5'))
    amds = [crystal.tags['AMD100'] for crystal in crystals]