Description of AMD/PDD
======================

The AMD of a crystal is an infinite sequence calculated from inter-atomic distances in the crystal. 
In contrast, the PDD is a matrix which can have arbitrarily many columns. 
In practice, both are calculated up to some chosen number k of entries/columns.

The kth AMD value of a periodic set is the average distance to the kth nearest neighbour over atoms in a unit cell. 
That is, to find the AMD for a periodic set up to k, list (in order) distances to the nearest k neighbours (in the infinite crystal) 
for every atom in a unit cell take the average, giving a vector length k.

The PDD is related to AMD but contains more information as it avoids the averaging step. 
Like AMD, list distances to the nearest k neighbours in order for each atom in a unit cell. 
Collect these lists into one matrix with a row for each atom. Then order the rows of the matrix lexicographically. 
If any rows are not unique, keep only one and give each a weight proportional to how many copies there are. 
The result is the kth PDD of the periodic set. In practice, the weights are kept in the first column of the matrix.

A much more detailed description can be found in the papers on AMD and PDD:

- Average minimum distances of periodic point sets - foundational invariants for mapping periodic crystals. MATCH Communications in Mathematical and in Computer Chemistry, 87(3):529-559 (2022). `<https://doi.org/10.46793/match.87-3.529W>`_
- Pointwise distance distributions of periodic point sets. arXiv preprint arXiv:2108.04798 (2021). `<https://arxiv.org/abs/2108.04798>`_



Comparing by AMD/PDD
********************

AMDs are just vectors which can be compared with any metric, as long as k (length of the AMD) is the same. 
The default metric used in this package is L-infinity (aka Chebyshev), 
since it does not so much accumulate differences in distances across many neighbours. 
PDDs are matrices with weighted rows; the appropriate metric to compare them is the *Earth mover's distance* (aka Wasserstein metric), 
which itself needs a metric to compare two PDD rows (without their weights), where L-infinity is again our default.