Description of AMD/PDD
======================

AMD and PDD are geometry-based descriptors of crystals. For a crystal :math:`S`,
:math:`\text{PDD}(S)_{ik}` is the distance between an atom corresponding to row
:math:`i` and its :math:`k`-th nearest neighbour in the infinite crystal.
So the PDD is a matrix of distances between atoms, each row corresponding to an
atom and having a weight as not all atoms are considered equal. The
AMD is the average of the PDD, so :math:`\text{AMD}(S)_{k}` is the average
distance to the :math:`k`-th nearest neighbour over atoms in the unit cell. Both
take a parameter :math:`k`, and all entries up to the :math:`k`-th are given.

The full construction of PDD is as follows: 1) List distances to the nearest :math:`k`
neighbours in order for each atom in a unit cell; 2) Collect into rows of a matrix
and order the rows lexicographically; 3) If any rows are the same, merge into one
and assign weights to each row proportional to their frequency (in implementation,
these weights are in the first column of the PDD).

A much more detailed description can be found in our papers:

- *Average minimum distances of periodic point sets - foundational invariants for mapping periodic crystals*. MATCH Communications in Mathematical and in Computer Chemistry, 87(3):529-559 (2022). `<https://doi.org/10.46793/match.87-3.529W>`_
- *Resolving the data ambiguity for periodic crystals*. Advances in Neural Information Processing Systems (NeurIPS 2022), v.35. <https://openreview.net/forum?id=4wrB7Mo9_OQ>.

Comparing by AMD/PDD
********************

AMDs are just vectors which can be compared with any metric, as long as k (length of the AMD) is the same. 
The default metric used in this package is L-infinity (aka Chebyshev), 
since it does not so much accumulate differences in distances across many neighbours. 
PDDs are matrices with weighted rows; the appropriate metric to compare them is the `Earth mover's distance <https://en.wikipedia.org/wiki/Earth_mover%27s_distance>`_ (aka Wasserstein metric), 
which itself needs a metric to compare two PDD rows (without their weights), where L-infinity is again the default.
