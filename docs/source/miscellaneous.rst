Miscellaneous
==================

Minimum spanning trees
----------------------
Previously, minimum spanning trees have been used to visualise crystal landscapes
with AMD or PDD, particularly with `tmap <https://tmap.gdb.tools/>`_. The 
:mod:`.compare` module includes two functions to produce MSTs,
:func:`.compare.AMD_mst` and :func:`.compare.PDD_mst`. They each accept a list of
AMDs/PDDs and return an edge list ``(i, j, d)`` where the crystals at indices ``i`` and ``j``
are connected by an edge with weight ``d``. 

The mst functions take the same optional arguments as other compare functions, including
``low_memory`` for ``AMD_mst`` and ``filter`` for ``PDD_mst``.

AMDs and PDDs of finite point sets
----------------------------------
AMDs and PDDs also work for finite point sets. The functions :func:`.calculate.finite_AMD` and
:func:`.calculate.finite_PDD` accept just a numpy array containing the points, returning the 
fingerprint of the finite point set. Unlike ``amd.AMD`` and ``amd.PDD`` no integer ``k`` is passed;
instead the distances to all neighbours are found (number of columns = no of points - 1).

Higher order PDDs
-----------------
Not yet implemented.

Inverse design
--------------
Not yet implemented. 