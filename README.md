# average-minimum-distance: isometrically invariant crystal fingerprints

[![PyPI](https://img.shields.io/pypi/v/average-minimum-distance.svg)](https://pypi.org/project/average-minimum-distance/)
[![Status](https://img.shields.io/pypi/status/average-minimum-distance)](https://pypi.org/project/average-minimum-distance/)
[![Format](https://img.shields.io/pypi/format/average-minimum-distance)](https://pypi.org/project/average-minimum-distance/)
[![MATCH Paper](https://img.shields.io/badge/DOI-10.46793%2Fmatch.87--3.529W-blue)](https://doi.org/10.46793/match.87-3.529W)
[![CC-0 license](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-blue.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

For calculation and comparison of AMD/PDD isometric invariants of periodic sets. Includes .cif reading functionality.

- **Paper:** https://doi.org/10.46793/match.87-3.529W
- **Documentation:** https://average-minimum-distance.readthedocs.io/en/latest/
- **Source code:** https://github.com/dwiddo/average-minimum-distance
- 
If you use our code in your work, please cite our paper at [arxiv.org/abs/2009.02488](https://arxiv.org/abs/2009.02488). The bib reference is at the bottom of this page; [click here jump to it](#citeus).

## What's amd?

A crystal is an arrangement of atoms which periodically repeats according to some lattice. The atoms and lattice defining a crystal are typically recorded in a .CIF file, but this representation is ambiguous, i.e. different .CIF files can define the same crystal. This package implements new *isometric invariants* called AMD (average minimum distance) and PDD (point-wise distance distribution) based on inter-point distances, which are guaranteed to take the same value for all equivalent representations of a crystal. They do this in a continuous way; crystals which are similar have similar AMDs and PDDs.

For a technical description of AMD, [see our paper on arXiv](https://arxiv.org/abs/2009.02488). Detailed documentation of this package is [available on readthedocs](https://average-minimum-distance.readthedocs.io/en/latest/).

Use pip to install average-minimum-distance:

```shell
pip install average-minimum-distance
```

Then import average-minimum-distance with ```import amd```.

## Getting started

The central functions of this package are ```amd.AMD()``` and ```amd.PDD()```, which take a crystal and a positive integer k, returning the crystal's AMD/PDD up to k. An AMD is a 1D numpy array, whereas PDDs are 2D arrays. The AMDs or PDDs can then be passed to functions to compare them.

### Reading crystals

The following example reads a .CIF with ```amd.CifReader``` and computes the AMDs (k=100):

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
which takes a set of points and compares them pairwise, returning a *condensed distance matrix*, a 1D vector containing the distances. This vector is the upper half of the 2D distance matrix in one list, since for pairwise comparisons the matrix is symmetric. The function ```amd.AMD_pdist``` similarly takes a list of AMDs and compares them pairwise, returning the condensed distance matrix:

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

## Example: PDD-based dendrogram of a family of crystals

This example reads crystals in the DEBXIT family, compares them by PDD and plots a single linkage dendrogram:

```py
import amd
import matplotlib.pyplot as plt
from scipy.cluster import hierarchy

crystals = list(amd.CSDReader('DEBXIT', families=True))
names = [crystal.name for crystal in crystals]
pdds = [amd.PDD(crystal, 100) for crystal in crystals]
cdm = amd.PDD_pdist(pdds)
Z = hierarchy.linkage(cdm, 'single')
dn = hierarchy.dendrogram(Z, labels=names)
plt.show()
```

## Example: Finding n nearest neighbours in one set from another

Here is an example showing how to read two sets of crystals from .CIFs ```set1.cif``` and ```set2.cif``` and find the 10 nearest PDD-neighbours in set 2 for every crystal in set 1. This can be done with the handy function ```amd.neighbours_from_distance_matrix```, which also accepts condensed distance matrices.

```py
import amd

n = 10
k = 100

set1 = list(amd.CifReader('set1.cif'))
set2 = list(amd.CifReader('set2.cif'))

set1_pdds = [amd.PDD(s, k) for s in set1]
set2_pdds = [amd.PDD(s, k) for s in set2]

dm = amd.PDD_cdist(set1_pdds, set2_pdds)

# amd.neighbours_from_distance_matrix calculates nearest neighbours for you
# nn_dists[i][j] = distance from set1[i] to its (j+1)st nearest neighbour in set2 
# nn_inds[i][j] = index of set1[i]'s (j+1)st nearest neighbour in set2
# it's (j+1)st as index 0 refers to the first nearest neighbour
nn_dists, nn_inds = amd.neighbours_from_distance_matrix(n, dm)

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

@article{widdowson2022average,
  title={Average Minimum Distances of periodic point sets - fundamental invariants for mapping all periodic crystals},
  author={Daniel Widdowson and Marco M Mosca and Angeles Pulido and Vitaliy Kurlin and Andrew I Cooper},
  journal={MATCH Communications in Mathematical and in Computer Chemistry},
  doi={10.46793/match.87-3.529W},
  volume={87},
  pages={529-559},
  year={2022}
}
