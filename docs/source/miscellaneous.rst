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
In addition to the AMD and regular PDD, there is a sequence of 'higher order' PDD invariants 
accessible by passing an int to the optional parameter ``order`` in ``amd.PDD()`` or ``amd.finite_PDD()``. 
Each invariant in the sequence contains more information and grows in complexity.

To summarise, each row of PDD^h pertains to an h-tuple of motif points, hence PDD^h contains m choose h rows
before collapsing. Apart from weights, the first element of each row is the finite PDD (order 1) of the 
h-tuple of points (just the distance between them if h = 2). The rest of a row is lexicographically ordered
h-tuples of distances from the 3 points to other points in the set. 

Note that before 1.1.7 the ``order`` parameter controlled the sorting of rows, that is now ``lexsort``. 
The default is ``order=1`` which is the normal PDD (a matrix shape ``(m, k+1)`` with weights in the first 
column). For any integer > 1, ``amd.PDD()`` returns a tuple ``(weights, dist, pdd)`` where ``weights`` 
are the usual weights (number of row appearances / total rows), ``dist`` contains the finite PDDs (or
distances) and ``pdd`` contains the h-tuples (as described above).

Inverse design
--------------
Not yet implemented. 