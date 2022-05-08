# average-minimum-distance: isometrically invariant crystal fingerprints

[![PyPI](https://img.shields.io/pypi/v/average-minimum-distance.svg)](https://pypi.org/project/average-minimum-distance/)
[![Status](https://img.shields.io/pypi/status/average-minimum-distance)](https://pypi.org/project/average-minimum-distance/)
[![Read the Docs](https://img.shields.io/readthedocs/average-minimum-distance)](https://average-minimum-distance.readthedocs.io)
[![MATCH Paper](https://img.shields.io/badge/DOI-10.46793%2Fmatch.87--3.529W-blue)](https://doi.org/10.46793/match.87-3.529W)
[![CC-0 license](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-blue.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

Implements fingerprints (*isometry invariants*) of crystal structures based on geometry: average minimum distances (AMD) and point-wise distance distributions (PDD). Includes .cif reading tools.

- **Papers:** https://doi.org/10.46793/match.87-3.529W or on arXiv at https://arxiv.org/abs/2009.02488
- **PyPI project:** https://pypi.org/project/average-minimum-distance/
- **Documentation:** https://average-minimum-distance.readthedocs.io
- **Source code:** https://github.com/dwiddo/average-minimum-distance

If you use our code in your work, please cite us. The bib reference is at the bottom of this page; [click here jump to it](#citeus).

## What's amd?

A crystal is an arrangement of atoms which periodically repeats in all directions according to some lattice. Crystals are usually given by listing parameters defining the lattice (or unit cell) and the atomic coordinates, e.g. in a .CIF file, but this representation is ambiguous because different .CIF files can define the same crystal. This package implements new *isometric invariants* based on inter-point distances which are guaranteed to take the same value for all (isometrically) equivalent representations of a crystal, called average minimum distances (AMDs) and point-wise distance distributions (PDDs). They are invariants that are also continuous, meaning crystals which are similar have similar AMDs and PDDs.

### Description of the invariants

A *periodic set* models a crystal with a point in the centre of each atom. The AMD of a structure is an infinite increasing sequence of real numbers calculated from inter-point distances in the periodic set. In contrast, the PDD is a matrix which can have arbitrarily many columns. In practice, both are calculated up to some chosen number k.

The k-th AMD value of a periodic set is the average distance to the k-th nearest neighbours of points in a unit cell. That is, to find the AMD of a periodic set up to k, first list distances to the nearest k neighbours in the infinite crystal for every point in a unit cell, then take the average over points in the unit cell.

The PDD is related to AMD but contains more information as it avoids the averaging step. Again start by and listing distances to the k nearest neighbours in order for each point in a unit cell, and collect all these lists into one matrix with a row for each point. Then order the matrix rows as in a dictionary (lexicographically), by comparing elements of two rows in order until one is smaller and putting that row first. Finally, if any rows are duplicated keep only one but give each a weight proportional to its original frequency, appending the weight to the start of each row to make the first column the weights. The result is the k-th PDD of the periodic set.

For a formal description, see our papers listed above. Detailed documentation for this package is [available on readthedocs](https://average-minimum-distance.readthedocs.io/en/latest/).

## Getting started

Use pip to install average-minimum-distance:

```shell
pip install average-minimum-distance
```

Then import average-minimum-distance with ```import amd```.

The central functions of this package are ```amd.AMD()``` and ```amd.PDD()```, which take a crystal and a positive integer k, returning the crystal's AMD/PDD up to k. An AMD is a 1D numpy array, whereas PDDs are 2D arrays. The AMDs or PDDs can then be passed to functions to compare them.

### Reading crystals

This example reads a .CIF with ```amd.CifReader``` and computes the AMDs (k=100):

```py
import amd

# read all structures in a .cif and put their amds (k=100) in a list
reader = amd.CifReader('path/to/file.cif')
amds = [amd.AMD(crystal, 100) for crystal in reader]
```

*Note: CifReader accepts optional arguments, e.g. for removing hydrogen and handling disorder. See the documentation for details.*

A crystal can also be read from the CSD using ```amd.CSDReader``` (if csd-python-api is installed), or created manually.

### Comparing AMDs or PDDs

The package includes functions for comparing sets of AMDs or PDDs.

They behave like scipy's function ```scipy.distance.spatial.pdist```,
which takes a set of points and compares them pairwise, returning a *condensed distance matrix*, a 1D vector containing the distances. It contains the upper half of the 2D distance matrix in one list, since for pairwise comparisons the matrix is symmetric. The function ```amd.AMD_pdist``` similarly takes a list of AMDs and compares them pairwise, returning the condensed distance matrix:

```py
cdm = amd.AMD_pdist(amds)
```

The default metric for comparison is ```chebyshev``` (l-infinity), though it can be changed to anything accepted by scipy's ```pdist```, e.g. ```euclidean```.

It is preferable to store the condensed matrix, though if you want the symmetric 2D distance matrix, use scipy's ```squareform```:

```py
from scipy.distance.spatial import squareform
dm = squareform(cdm)
# now dm[i][j] is the AMD distance between amds[i] and amds[j].
```

The function ```amd.AMD_pdist``` has an equivalent for PDDs, ```amd.PDD_pdist```. There are also the equivalents of ```scipy.distance.spatial.cdist```, ```amd.AMD_cdist``` and ```amd.PDD_cdist```, which take two sets and compares one vs the other, returning a 2D distance matrix.

## Example: PDD-based dendrogram of crystals in a .CIF

This example reads crystals from a .CIF, compares them by PDD and plots a single linkage dendrogram:

```py
import amd
import matplotlib.pyplot as plt
from scipy.cluster import hierarchy

crystals = list(amd.CifReader('crystals.cif'))
names = [crystal.name for crystal in crystals]
pdds = [amd.PDD(crystal, 100) for crystal in crystals]
cdm = amd.PDD_pdist(pdds)
Z = hierarchy.linkage(cdm, 'single')
dn = hierarchy.dendrogram(Z, labels=names)
plt.show()
```

## Example: Finding n nearest neighbours in one set from another

This example shows how to read two sets of crystals from .CIFs ```set1.cif``` and ```set2.cif``` and find the 10 nearest PDD-neighbours in set 2 for every crystal in set 1.

```py
import numpy as np
import amd

n = 10
k = 100

set1 = list(amd.CifReader('set1.cif'))
set2 = list(amd.CifReader('set2.cif'))

set1_pdds = [amd.PDD(s, k) for s in set1]
set2_pdds = [amd.PDD(s, k) for s in set2]

dm = amd.PDD_cdist(set1_pdds, set2_pdds)

# the following uses np.argpartiton (like argsort but not for the whole list)
# and np.take_along_axis to find nearest neighbours of each item given the
# distance matrix.
# nn_dists[i][j] = distance from set1[i] to its (j+1)st nearest neighbour in set2 
# nn_inds[i][j] = index of set1[i]'s (j+1)st nearest neighbour in set2
# it's (j+1)st as index 0 refers to the first nearest neighbour

nn_inds = np.array([np.argpartition(row, n)[:n] for row in dm])
nn_dists = np.take_along_axis(dm, nn_inds, axis=-1)
sorted_inds = np.argsort(nn_dists, axis=-1)
nn_inds = np.take_along_axis(nn_inds, sorted_inds, axis=-1)
nn_dists = np.take_along_axis(nn_dists, sorted_inds, axis=-1)

# now to print the names of these nearest neighbours and their distances:
set1_names = [s.name for s in set1]
set2_names = [s.name for s in set2]

for i in range(len(set1)):
    print('neighbours of', set1_names[i])
    for j in range(n):
        jth_nn_index = nn_inds[i][j]
        print('neighbour', j+1, set2_names[jth_nn_index], 'dist:', nn_dists[i][j])
```

## Cite us <a name="citeus"></a>

The arXiv paper for this package is [here](arxiv.org/abs/2009.02488). Use the following bib reference to cite us:

```bibtex
@article{amd2022,
  title = {Average Minimum Distances of periodic point sets - foundational invariants for mapping all periodic crystals},
  author = {Daniel Widdowson and Marco M Mosca and Angeles Pulido and Vitaliy Kurlin and Andrew I Cooper},
  journal = {MATCH Communications in Mathematical and in Computer Chemistry},
  doi = {10.46793/match.87-3.529W},
  volume = {87},
  number = {3},
  pages = {529-559},
  year = {2022}
}
```
