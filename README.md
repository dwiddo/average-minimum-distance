# average-minimum-distance: isometrically invariant crystal fingerprints

[![PyPI](https://img.shields.io/pypi/v/average-minimum-distance.svg)](https://pypi.org/project/average-minimum-distance/)
[![Status](https://img.shields.io/pypi/status/average-minimum-distance)](https://pypi.org/project/average-minimum-distance/)
[![Read the Docs](https://img.shields.io/readthedocs/average-minimum-distance)](https://average-minimum-distance.readthedocs.io)
[![Build Status](https://app.travis-ci.com/dwiddo/average-minimum-distance.svg?branch=master)](https://app.travis-ci.com/github/dwiddo/average-minimum-distance)
[![MATCH Paper](https://img.shields.io/badge/DOI-10.46793%2Fmatch.87--3.529W-blue)](https://doi.org/10.46793/match.87-3.529W)
[![CC-0 license](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-blue.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

Implements fingerprints (*isometry invariants*) of crystal structures based on geometry: average minimum distances (AMD) and pointwise distance distributions (PDD). Includes .cif reading tools.

- **PyPI project:** <https://pypi.org/project/average-minimum-distance>
- **Documentation:** <https://average-minimum-distance.readthedocs.io>
- **Source code:** <https://github.com/dwiddo/average-minimum-distance>
- **References** ([jump to bib references](#citeus)):
  - *Average minimum distances of periodic point sets - foundational invariants for mapping periodic crystals*. MATCH Communications in Mathematical and in Computer Chemistry, 87(3):529-559 (2022). <https://doi.org/10.46793/match.87-3.529W>
  - *Pointwise distance distributions of periodic point sets*. arXiv preprint arXiv:2108.04798 (2021). <https://arxiv.org/abs/2108.04798>
  - *Analogy Powered by Prediction and Structural Invariants: Computationally Led Discovery of a Mesoporous Hydrogen-Bonded Organic Cage Crystal*. Journal of the American Chemical Society, 144(22):9893–9901 (2022). <https://doi.org/10.1021/jacs.2c02653>

## What's amd?

The typical representation of a crystal as a motif and cell is ambiguous, as there are many ways to define the same crystal. This package implements *isometric invariants*, called average minimum distances (AMD) and pointwise distance distributions (PDD), which always take the same value for any two (isometrically) identical input crystals. They do this in a continuous way, so similar crystals have similar invariants.

### Description of AMD and PDD

The pointwise distance distribution (PDD) of a crystal records the environment of each atom in a unit cell. It does this by listing distances to their neighbouring atoms in order, closest first. When PDDs are compared we try to find an optimal matching of these environments, so a small distance between PDDs means the lists of distances to neighbours of atoms in one crystal line up with the other. This is done in a way that is independent of the choice of unit cell.

The average minimum distance (AMD) averages the PDD over over atoms in the unit cell to make a single vector. In theory this loses some information about the crystal, but in practice two crystals which distinguished by their PDDs are also by their AMDs. As AMDs are vectors, comparing them is much faster than comparing PDDs.

### A more formal description

The average minimum distance (AMD) of a crystal is an infinite increasing sequence of numbers calculated from inter-point distances in the crystal. In contrast, the point-wise distance distribution (PDD) is a matrix which can have arbitrarily many columns. In practice, both are calculated up to some chosen number k.

The kth AMD value of a periodic set is the average distance to the kth nearest neighbour over points in a unit cell. That is, to find the AMD for a periodic set up to k, list (in order) distances to the nearest k neighbours (in the infinite periodic set) for every point in a unit cell and average over points in the cell.

The PDD is related to AMD but contains more information as it avoids the averaging step. Like the AMD, start by and listing distances to the k nearest neighbours in order for each point in a unit cell. Collect these lists into one matrix with a row for each point. Then order the rows of the matrix as in a dictionary (lexicographically). If any rows are not unique, keep only one but give each a weight proportional to how many copies there are, appending the weight to the start of each row so the first column contains weights. The result is the kth PDD of the periodic set.

### Comparing with AMD or PDD

AMDs are just vectors which can be compared with any metric, as long as k (length of the AMD) is the same. The default metric used in the package is L-infinity (aka Chebyshev), since it does not so much accumulate differences in distances across many neighbours. PDDs are matrices with weighted rows; the appropriate metric to compare them is the *Earth mover's distance* (aka Wasserstein metric), which itself needs a metric to compare two PDD rows (without weights), where L-infinity is again the default.

For a formal description, see our papers listed above. Detailed documentation for this package is [available on readthedocs](https://average-minimum-distance.readthedocs.io/en/latest/).

## Getting started

Use pip to install average-minimum-distance:

```shell
pip install average-minimum-distance
```

Then import average-minimum-distance with ```import amd```.

The central functions of this package are ```amd.AMD()``` and ```amd.PDD()```, which take a crystal and a positive integer k, returning the crystal's AMD/PDD up to k. An AMD is a 1D NumPy array, whereas PDDs are 2D arrays. The AMDs or PDDs can then be passed to functions to compare them.

### Reading crystals from a file

This example reads a .cif with ```amd.CifReader``` and computes the AMDs (k=100):

```py
import amd

# read all structures in a .cif and put their amds (k=100) in a list
reader = amd.CifReader('path/to/file.cif')
amds = [amd.AMD(crystal, 100) for crystal in reader]
```

*Note: CifReader accepts optional arguments, e.g. for removing hydrogen and handling disorder. See the documentation for details.*

If csd-python-api is installed, CifReader can accept file formats other than cif, and crystals can be read directly from your local CSD database with ```amd.CSDReader```.

### Choosing a value of k

The second argument (k) of ```amd.AMD``` and ```amd.PDD``` controls the number of nearest neighbour atoms considered for each atom in the unit cell. So choosing a small k (e.g. k=5) only takes into account close neighbours of atoms, often in the same molecule, so there is usually a small distance between crystals with the same or similar unit molecules when k is small. A larger k (e.g. k=100) is more 'strict', meaning to get a small distance between crystals, the environments of atoms in one must line up with those in the other up to a larger radius. It is not recommended to choose very large values, usually over 1000, as the invariants start to converge and comparisons depend only on density.

### Comparing AMDs or PDDs

The package includes functions for comparing sets of AMDs or PDDs.

They behave like SciPy's function ```scipy.distance.spatial.pdist```,
which takes a set of points and compares them pairwise, returning a *condensed distance matrix*, a 1D vector containing the distances. It contains the upper half of the 2D distance matrix in one list, since for pairwise comparisons the matrix is symmetric. The function ```amd.AMD_pdist``` similarly takes a list of AMDs and compares them pairwise, returning the condensed distance matrix:

```py
cdm = amd.AMD_pdist(amds)
```

The default metric for comparison is ```chebyshev``` (L-infinity), though it can be changed to anything accepted by SciPy's ```pdist```, e.g. ```euclidean```.

It is preferable to store the condensed matrix, though if you want the symmetric 2D distance matrix, use SciPy's ```squareform```:

```py
from scipy.distance.spatial import squareform
dm = squareform(cdm)
# now dm[i][j] is the AMD distance between amds[i] and amds[j].
```

The function ```amd.AMD_pdist``` has an equivalent for PDDs, ```amd.PDD_pdist```. There are also the equivalents of ```scipy.distance.spatial.cdist```, ```amd.AMD_cdist``` and ```amd.PDD_cdist```, which take two sets and compares one vs the other, returning a 2D distance matrix.

## Example: PDD-based dendrogram of crystals in a .cif

This example reads crystals from a .cif, compares them by PDD and plots a single linkage dendrogram:

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

This example shows how to read two sets of crystals from .cifs ```set1.cif``` and ```set2.cif``` and find the 10 nearest PDD-neighbours in set 2 for every crystal in set 1.

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

Use the following bib reference to cite us.

*Average minimum distances of periodic point sets - foundational invariants for mapping periodic crystals*. MATCH Communications in Mathematical and in Computer Chemistry, 87(3), 529-559 (2022). <https://doi.org/10.46793/match.87-3.529W>.

```bibtex
@article{amd2022,
  title = {Average Minimum Distances of periodic point sets - foundational invariants for mapping periodic crystals},
  author = {Widdowson, Daniel and Mosca, Marco M and Pulido, Angeles and Kurlin, Vitaliy and Cooper, Andrew I},
  journal = {MATCH Communications in Mathematical and in Computer Chemistry},
  doi = {10.46793/match.87-3.529W},
  volume = {87},
  number = {3},
  pages = {529-559},
  year = {2022}
}
```

```bibtex
@misc{pdd2022,
  doi = {10.48550/ARXIV.2108.04798},
  url = {https://arxiv.org/abs/2108.04798},
  author = {Widdowson, Daniel and Kurlin, Vitaliy},
  title = {Pointwise distance distributions of periodic point sets},
  publisher = {arXiv},
  year = {2021},
}
```
