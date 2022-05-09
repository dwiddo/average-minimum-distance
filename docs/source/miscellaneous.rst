Miscellaneous
=============

Fingerprints of finite point sets
----------------------------------
AMDs and PDDs also work for finite point sets. The functions :func:`.calculate.finite_AMD` and
:func:`.calculate.finite_PDD` accept just a numpy array containing the points, returning the 
fingerprint of the finite point set. Unlike ``amd.AMD`` and ``amd.PDD`` no integer ``k`` is passed;
instead the distances to all neighbours are found (number of columns = no of points - 1).

Inverse design
--------------
It is possible to reconstruct a periodic set up to isometry from its PDD if the periodic set 
satisfies certain conditions (a 'general position') and the PDD has enough columns. This is 
implemented via the  functions :func:`.calculate.PDD_reconstructable`, which returns the PDD 
of a periodic set with enough columns, and :func:`.reconstruct.reconstruct` which returns 
the motif given the PDD and unit cell.