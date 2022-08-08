Miscellaneous
=============

Fingerprints of finite point sets
----------------------------------
AMDs and PDDs also work for finite point sets. :func:`.calculate.AMD_finite` and
:func:`.calculate.PDD_finite` accept a NumPy array containing the points and return the AMD/PDD. 
Unlike ``amd.AMD`` and ``amd.PDD`` no integer ``k`` is passed; instead the distances to all
neighbours are found (number of columns = no of points - 1).

::

    # Compare AMDs of trapezium and kite shaped finite point sets
    trapezium = np.array([[0,0],[1,1],[3,1],[4,0]])
    kite      = np.array([[0,0],[1,1],[1,-1],[4,0]])

    trap_amd = amd.AMD_finite(trapezium)
    kite_amd = amd.AMD_finite(kite)

    amd_dist = np.amax(np.abs(trap_amd - kite_amd))

Reconstruction of a periodic set from its PDD
---------------------------------------------
It is possible to reconstruct a periodic set up to isometry from its PDD if the periodic set 
satisfies certain conditions (a 'general position') and the PDD has enough columns. This is 
implemented via the functions :func:`amd.PDD_reconstructable`, which returns the PDD 
of a periodic set with enough columns, and :func:`amd.reconstruct` which returns 
the motif given the PDD and unit cell. Reconstruction is not optimised and very slow for 
most real crystals, it's also not well tested and may not work for some crystals.