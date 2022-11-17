# average-minimum-distance: geometry based crystal descriptors

[![PyPI](https://img.shields.io/pypi/v/average-minimum-distance.svg)](https://pypi.org/project/average-minimum-distance/)
[![Status](https://img.shields.io/pypi/status/average-minimum-distance)](https://pypi.org/project/average-minimum-distance/)
[![Read the Docs](https://img.shields.io/readthedocs/average-minimum-distance)](https://average-minimum-distance.readthedocs.io)
[![Build Status](https://scrutinizer-ci.com/g/dwiddo/average-minimum-distance/badges/build.png?b=master)](https://scrutinizer-ci.com/g/dwiddo/average-minimum-distance/)
[![CC-0 license](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-blue.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

Descriptors of crystal structures based on geometry (*isometry invariants*).

- **PyPI project:** <https://pypi.org/project/average-minimum-distance>
- **Documentation:** <https://average-minimum-distance.readthedocs.io>
- **Source code:** <https://github.com/dwiddo/average-minimum-distance>
- **References** ([bib references at the bottom of this page](#citeus)):
  - *Average minimum distances of periodic point sets - foundational invariants for mapping periodic crystals*. MATCH Communications in Mathematical and in Computer Chemistry, 87(3):529-559 (2022). <https://doi.org/10.46793/match.87-3.529W>
  - *Resolving the data ambiguity for periodic crystals*. Advances in Neural Information Processing Systems (Proceedings of NeurIPS 2022), to appear. <https://arxiv.org/abs/2108.04798>

## What's amd?

The typical representation of a crystal as a motif and unit cell is ambiguous, as there are many choices of cell and motif that define the same crystal. This package implements crystal descriptors that are *isometry invariants*, meaning they are the same for any two crystals which are equivalent by an *isometry* (rotation, translation and reflection). They do this in a continuous way; if two crystals have similar geometry there is a small distance between their invariants.

### Description of AMD and PDD invariants

The *pointwise distance distribution* (PDD) of a crystal records the environment of each atom in the unit cell by listing distances to neighbouring atoms, with some extra steps to ensure it is independent of choice of unit cell and motif. A PDD is a weighted collection of lists of distances, and two PDDs are compared by finding an optimal matching between them ([Earth Mover's distance](https://en.wikipedia.org/wiki/Earth_mover%27s_distance)). When the crystals are geometrically identical there is always a perfect matching and The Earth Mover's distance is 0.

Taking a weighted average over the lists in a PDD gives the *average minimum distance* (AMD), a vector. In theory the averaging means the AMD contains less information, but they are significantly faster to compare.

Both AMD and PDD require a parameter k, the number of nearest neighbours to consider for each atom. Then an AMD will be a vector length k, whereas a PDD matrix will have k+1 columns (the first column contains weights).

## Getting started

Use pip to install average-minimum-distance:

```shell
pip install average-minimum-distance
```

Then import average-minimum-distance with ```import amd```.

```amd.compare()``` compares a set of crystals from a .cif by AMD or PDD, for example

```py
import amd
df = amd.compare('crystals.cif', by='PDD', k=100)
```

The distance matrix is returned as a [pandas DataFrame](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html). It can also take two paths and compare crystals in one file with the other:

```py
df = amd.compare('crystals_1.cif', 'crystals_2.cif' by='AMD', k=100)
```

Either argument can a list of paths to .cif files, which are combined in the final distance matrix.

```amd.compare()``` does three things: reads crystals, calculates their AMD/PDD, and compares them. These steps can be done separately for more flexibility (e.g, saving the descriptors to a file), explained below. ```amd.compare()``` accepts most of the optional parameters from any of these steps, see the documentation for details.

If `csd-python-api` is installed, the compare function can also accept one or more CSD refcodes or other file formats instead of cifs (pass ```reader='ccdc'```).

### Choosing a value of k

The parameter k is the number of neighbouring atoms considered for each atom in the unit cell. Two crystals with the same unit molecule will have a small AMD/PDD distance for small enough k. A larger k will mean the environments of atoms in one crystal must line up with those in the other up to a larger radius to have a small AMD/PDD distance.

### Reading crystals from a file, calculating the AMDs and PDDs

This code reads a .cif with ```amd.CifReader``` and computes the AMDs (k = 100):

```py
import amd
reader = amd.CifReader('path/to/file.cif')
amds = [amd.AMD(crystal, 100) for crystal in reader]
```

*Note: CifReader accepts optional arguments, e.g. for removing Hydrogens and handling disorder. See the documentation for details.*

To calculate PDDs instead, just replace ```amd.AMD``` with ```amd.PDD```.

*csd-python-api only:* crystals can be read directly from your local copy of the CSD with ```amd.CSDReader```, which accepts a list of refcodes. CifReader can accept file formats other than .cif by passing ```reader='ccdc'```.

### Comparing by AMD or PDD

```amd.AMD_pdist``` and ```amd.PDD_pdist``` take a list of AMDs/PDDs respectively and compares them pairwise, returning a *condensed distance matrix* like SciPy's ```pdist``` function.

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

# dm[i][j] = AMD distance(amds1[i], amds2[j])
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

This example finds the 10 nearest PDD-neighbours in set 2 for every crystal in set 1. As of 1.3.4, the `compare` function accepts a parameter `nearest` which implements this behaviour.

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
