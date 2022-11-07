# average-minimum-distance: geometry based crystal descriptors

[![PyPI](https://img.shields.io/pypi/v/average-minimum-distance.svg)](https://pypi.org/project/average-minimum-distance/)
[![Status](https://img.shields.io/pypi/status/average-minimum-distance)](https://pypi.org/project/average-minimum-distance/)
[![Read the Docs](https://img.shields.io/readthedocs/average-minimum-distance)](https://average-minimum-distance.readthedocs.io)
[![Build Status](https://scrutinizer-ci.com/g/dwiddo/average-minimum-distance/badges/build.png?b=master)](https://scrutinizer-ci.com/g/dwiddo/average-minimum-distance/)
[![CC-0 license](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-blue.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

Implements descriptors of crystal structures based on geometry (*isometry invariants*): average minimum distances (AMD) and pointwise distance distributions (PDD).

- **PyPI project:** <https://pypi.org/project/average-minimum-distance>
- **Documentation:** <https://average-minimum-distance.readthedocs.io>
- **Source code:** <https://github.com/dwiddo/average-minimum-distance>
- **References** ([bib references at the bottom of this page](#citeus)):
  - *Average minimum distances of periodic point sets - foundational invariants for mapping periodic crystals*. MATCH Communications in Mathematical and in Computer Chemistry, 87(3):529-559 (2022). <https://doi.org/10.46793/match.87-3.529W>
  - *Resolving the data ambiguity for periodic crystals*. Advances in Neural Information Processing Systems (Proceedings of NeurIPS 2022), to appear. <https://arxiv.org/abs/2108.04798>

## What's amd?

The typical representation of a crystal as a motif and unit cell is ambiguous, as there are many ways to define the same crystal. This package implements new *isometric invariants*: average minimum distances (AMD) and pointwise distance distributions (PDD), which take the same value for any two crystals related by an *isometry* (rotation, translation, or reflection). They do this in a continuous way, so similar crystals have a small distance between their invariants.

### Brief description of AMD and PDD

The pointwise distance distribution (PDD) records the environment of each atom in a unit cell by listing the distances from each atom to neighbouring atoms in order, with some extra steps to ensure it is independent of choice of unit cell and motif. A PDD is a collection of lists, each with an attached weight. Two PDDs are compared by finding an optimal matching between the two sets of lists while respecting the weights ([Earth Mover's distance](https://doi.org/10.46793/match.87-3.529W)), and when the crystals are geometrically identical there is always a perfect matching resulting in a distance of zero.

The average minimum distance (AMD) averages the PDD over atoms in a unit cell to make a vector, which keeps the important properties of the PDD. Since AMDs are just vectors, comparing by AMD is much faster than PDD, but an AMD contains less information in theory.

Both AMD and PDD have a parameter k, the number of nearest neighbours to consider for each atom, which is the length of the AMD vector or the number of columns in the PDD (plus an extra column for weights of rows).

## Getting started

Use pip to install average-minimum-distance:

```shell
pip install average-minimum-distance
```

Then import average-minimum-distance with ```import amd```.

```amd.compare()``` compares sets of crystals by AMD or PDD in one line, e.g. by PDD with k = 100:

```py
import amd
df = amd.compare('crystals.cif', by='PDD', k=100)
```

A [pandas DataFrame](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html) is returned of the distance matrix with names of crystals in rows and columns. It can also take two paths and compare crystals in one file with the other, for example

```py
df = amd.compare('crystals_1.cif', 'crystals_2.cif' by='AMD', k=100)
```

Either first or second argument can be lists of cif paths (or file objects) which are combined in the final distance matrix.

```amd.compare()``` does three things: reads crystals, calculates their AMD/PDD, and compares them. These steps can be done separately for more flexibility (e.g, saving the  descriptors to a file), explained below. ```amd.compare()``` accepts most of the optional parameters from any of these steps, see the documentation for details.

If `csd-python-api` is installed, the compare function can also accept one or more CSD refcodes or other file formats instead of cifs (pass ```reader='ccdc'```).

### Choosing a value of k

The parameter k of the invariants is the number of neighbouring atoms considered for each atom in the unit cell. Two crystals with the same unit molecule will have a small AMD/PDD distance for small enough k. A larger k will mean the environments of atoms in one crystal must line up with those in the other up to a larger radius to have a small AMD/PDD distance.

### Reading crystals from a file, calculating the AMDs and PDDs

This code reads a .cif with ```amd.CifReader``` and computes the AMDs (k = 100):

```py
import amd
reader = amd.CifReader('path/to/file.cif')
amds = [amd.AMD(crystal, 100) for crystal in reader]  # calc AMDs
```

*Note: CifReader accepts optional arguments, e.g. for removing hydrogen and handling disorder. See the documentation for details.*

To calculate PDDs, just replace ```amd.AMD``` with ```amd.PDD```.

If `csd-python-api` is installed, crystals can be read directly from your local copy of the CSD with ```amd.CSDReader```, which accepts a list of refcodes. CifReader can accept file formats other than cif by passing ```reader='ccdc'```.

### Comparing by AMD or PDD

```amd.AMD_pdist``` and ```amd.PDD_pdist``` take a list of invariants and compares them pairwise, returning a *condensed distance matrix* like SciPy's ```pdist``` function.

```py
# read and calculate AMDs and PDDs (k=100)
crystals = list(amd.CifReader('path/to/file.cif'))
amds = [amd.AMD(crystal, 100) for crystal in reader]
pdds = [amd.PDD(crystal, 100) for crystal in reader]

amd_cdm = amd.AMD_pdist(amds) # compare a list of AMDs pairwise
pdd_cdm = amd.PDD_pdist(pdds) # compare a list of PDDs pairwise

# Use SciPy's squareform for a symmetric 2D distance matrix
from scipy.distance.spatial import squareform
amd_dm = squareform(amd_cdm)
```

*Note: if you want both AMDs and PDDs like above, it's faster to compute the PDDs first and use `amd.PDD_to_AMD()` rather than computing both from scratch.*

The default metric for comparison is ```chebyshev``` (L-infinity), though it can be changed to anything accepted by SciPy's ```pdist```, e.g. ```euclidean```.

If you have two sets of crystals and want to compare all crystals in one to the other, use ```amd.AMD_cdist``` or ```amd.PDD_cdist```.

```py
set1 = amd.CifReader('set1.cif')
set2 = amd.CifReader('set2.cif')
amds1 = [amd.AMD(crystal, 100) for crystal in set1]
amds2 = [amd.AMD(crystal, 100) for crystal in set2]

# dm[i][j] = distance(amds1[i], amds2[j])
dm = amd.AMD_cdist(amds)
```

## Example: PDD-based dendrogram

This example compares some crystals in a cif by PDD (k = 100) and plots a single linkage dendrogram:

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

This example finds the 10 nearest PDD-neighbours in set 2 for every crystal in set 1.

```py
import numpy as np
import amd

n = 10
df = amd.compare('set1.cif', 'set2.cif', k=100)
dm = df.values

# Uses np.argpartiton (partial argsort) and np.take_along_axis to find 
# nearest neighbours of each item in set1. Works for any distance matrix.
nn_inds = np.array([np.argpartition(row, n)[:n] for row in dm])
nn_dists = np.take_along_axis(dm, nn_inds, axis=-1)
sorted_inds = np.argsort(nn_dists, axis=-1)
nn_inds = np.take_along_axis(nn_inds, sorted_inds, axis=-1)
nn_dists = np.take_along_axis(nn_dists, sorted_inds, axis=-1)

for i in range(len(set1)):
    print('neighbours of', df.index[i])
    for j in range(n):
        print('neighbour', j+1, df.columns[nn_inds[i][j]], 'dist:', nn_dists[i][j])
```

## Cite us <a name="citeus"></a>

Use the following bib references to cite AMD or PDD.

*Average minimum distances of periodic point sets - foundational invariants for mapping periodic crystals*. MATCH Communications in Mathematical and in Computer Chemistry, 87(3), 529-559 (2022). <https://doi.org/10.46793/match.87-3.529W>.

```bibtex
@article{widdowson2022average,
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

*Resolving the data ambiguity for periodic crystals*. Advances in Neural Information Processing Systems (NeurIPS 2022), v.35, to appear. <https://arxiv.org/abs/2108.04798>.

```bibtex
@misc{widdowson2022resolving,
  title = {Resolving the data ambiguity for periodic crystals},
  author = {Widdowson, Daniel and Kurlin, Vitaliy},
  journal = {Advances in Neural Information Processing Systems (Proceedings of NeurIPS 2022)},
  volume = {35},
  year = {2022},
  eprint = {arXiv:2108.04798},
}
```
