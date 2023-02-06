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

The representation of a crystal as a motif and unit cell is ambiguous, as many choices of cell and motif define the same crystal. This package implements crystal descriptors that are *isometry invariants*, meaning they are same for any two crystals which are equivalent by an *isometry* (rotation, translation and reflection). The descriptors can be compared quickly to give a distance which is 0 for geometrically identical crystals, and close to 0 for similar crystals.

### How the descriptors work

The *pointwise distance distribution* (PDD) of a crystal is a collection of atomic environments described by distances to neighbouring atoms. Two PDDs are compared by finding an optimal matching between them ([Earth Mover's distance](https://en.wikipedia.org/wiki/Earth_mover%27s_distance)).

The average of a PDD is a vector called the *average minimum distance* (AMD), which are significantly faster to compare but still contain most of the geometric information. Both AMD and PDD take a parameter k, the number of neighbouring atoms to consider for each atom in the unit cell.

## Getting started

Use pip to install average-minimum-distance:

```shell
pip install average-minimum-distance
```

Then import average-minimum-distance with ```import amd```.

The function ```amd.compare()``` compares crystals in cif files by AMD or PDD, for example

```py
import amd
# compare all items in one cif by PDD, k = 100
df = amd.compare('file.cif', by='PDD', k=100)
# compare all in one with all in the other by AMD, k = 100
df = amd.compare('file1.cif', 'file2.cif', by='AMD', k=100)
```

The distance matrix is returned as a [pandas DataFrame](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html). The first argument can also be a folder or list of cifs.

```amd.compare()``` does three things: reads crystals, calculates their AMDs or PDDs, and compares them. These steps can be done separately for more flexibility (e.g, saving the descriptors to a file), explained below. ```amd.compare()``` accepts most of the optional parameters from any of these steps, see the documentation for details. 

*CSD Python API only:* ```amd.compare()``` also accepts one or more CSD refcodes or other file formats instead of cifs (pass ```reader='ccdc'```).

### Choosing a value of k

The parameter k is the number of neighbouring atoms considered for each atom in the unit cell. Two crystals with the same unit molecule will have a small AMD/PDD distance for small enough k. A larger k will mean the environments of atoms in one crystal must line up with those in the other up to a larger radius to have a small AMD/PDD distance.

### Reading crystals, calculating AMDs & PDDs

This code reads a cif file and computes the list of AMDs (k = 100):

```py
import amd
reader = amd.CifReader('file.cif')
amds = [amd.AMD(crystal, 100) for crystal in reader]
# # To calculate the PDDs, replace 'AMD' with 'PDD':
# pdds = [amd.PDD(crystal, 100) for crystal in reader]
```

CifReader accepts optional arguments, e.g. for removing Hydrogens and handling disorder. See the documentation for details.

*CSD Python API only:* crystals can be read directly from your local copy of the CSD with ```amd.CSDReader```, which accepts a list of refcodes. CifReader can accept file formats other than .cif by passing ```reader='ccdc'```.

### Comparing by AMD or PDD

```amd.AMD_pdist()``` and ```amd.PDD_pdist()``` take a list of AMDs/PDDs respectively and compares them pairwise, returning a *condensed distance matrix* like SciPy's ```pdist``` function.

```py
import amd

# read and calculate AMDs and PDDs (k = 100)
crystals = list(amd.CifReader('path/to/file.cif'))
amds = [amd.AMD(crystal, 100) for crystal in reader]
pdds = [amd.PDD(crystal, 100) for crystal in reader]

amd_cdm = amd.AMD_pdist(amds) # compare AMDs pairwise
pdd_cdm = amd.PDD_pdist(pdds) # compare PDDs pairwise

# Use squareform for a symmetric 2D distance matrix
from scipy.distance.spatial import squareform
amd_dm = squareform(amd_cdm)
```

*Note: AMDs can be computed from PDDs, so if both are required, compute PDD first and use `amd.PDD_to_AMD()`.*

The default metric for comparison is ```chebyshev``` (L-infinity), though it can be changed to anything accepted by SciPy's ```pdist```, e.g. ```euclidean```.

If you have two sets of crystals and want to compare all crystals in one to the other, use ```amd.AMD_cdist``` or ```amd.PDD_cdist```.

```py
import amd
amds1 = [amd.AMD(c, 100) for c in amd.CifReader('set1.cif')]
amds2 = [amd.AMD(c, 100) for c in amd.CifReader('set2.cif')]
# dm[i][j] = AMD distance between amds1[i] & amds2[j]
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

*Resolving the data ambiguity for periodic crystals*. Advances in Neural Information Processing Systems (NeurIPS 2022), v.35. <https://openreview.net/forum?id=4wrB7Mo9_OQ>.

```bibtex
@inproceedings{widdowson2022resolving,
  title = {Resolving the data ambiguity for periodic crystals},
  author = {Widdowson, Daniel and Kurlin, Vitaliy},
  booktitle = {Advances in Neural Information Processing Systems},
  year = {2022},
  url = {https://openreview.net/forum?id=4wrB7Mo9_OQ}
}
```
