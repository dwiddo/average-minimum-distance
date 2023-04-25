# average-minimum-distance: geometry based crystal descriptors

[![PyPI](https://img.shields.io/pypi/v/average-minimum-distance)](https://pypi.org/project/average-minimum-distance/)
[![Status](https://img.shields.io/pypi/status/average-minimum-distance)](https://pypi.org/project/average-minimum-distance/)
[![Build Status](https://scrutinizer-ci.com/g/dwiddo/average-minimum-distance/badges/build.png?b=master)](https://scrutinizer-ci.com/g/dwiddo/average-minimum-distance/)
[![Read the Docs](https://img.shields.io/readthedocs/average-minimum-distance)](https://average-minimum-distance.readthedocs.io)
[![CC-0 license](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-blue)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

- **PyPI project:** <https://pypi.org/project/average-minimum-distance>
- **Documentation:** <https://average-minimum-distance.readthedocs.io>
- **Source code:** <https://github.com/dwiddo/average-minimum-distance>
- **References** ([bib references at the bottom of this page](#citeus)):
  - *Average minimum distances of periodic point sets - foundational invariants for mapping periodic crystals*. MATCH Communications in Mathematical and in Computer Chemistry, 87(3):529-559 (2022). <https://doi.org/10.46793/match.87-3.529W>
  - *Resolving the data ambiguity for periodic crystals*. Advances in Neural Information Processing Systems (NeurIPS 2022), v.35. <https://openreview.net/forum?id=4wrB7Mo9_OQ>.

## What's amd?

The typical representation of a crystal as a motif and unit cell is ambiguous, because many choices of cell and motif define the same crystal. This package implements crystal descriptors designed to be *isometry invariants*, meaning they are always the same for any two crystals which are geometrically equivalent, independent of the unit cell and motif. The descriptors can be compared to give a distance which is 0 for identical crystals, and close to 0 for similar crystals (a *continuous metric*).

The *pointwise distance distribution* (PDD) is a descriptor that records the environment of each atom in the unit cell by listing distances to neighbouring atoms. Two PDDs are compared using an optimal matching algorithm ([Earth Mover's distance](https://en.wikipedia.org/wiki/Earth_mover%27s_distance)). Taking the average of a PDD gives a vector called the *average minimum distance* (AMD), which are simpler and faster to compare (by several orders of magnitude) but can still identify crystals with similar geometry. Both have a parameter k, the number of neighbouring atoms considered for each atom in the unit cell.

## Getting started

Use pip to install average-minimum-distance:

```shell
pip install average-minimum-distance
```

Then import average-minimum-distance with ```import amd```.

```amd.compare()``` compares crystals in cif files by AMD or PDD descriptors:

```py
import amd

# compare items in file.cif pairwise by AMD, k=100
dm = amd.compare('file.cif', by='AMD', k=100)
# compare items in file1.cif vs file2.cif by PDD, k=100
dm = amd.compare('file1.cif', 'file2.cif', by='PDD', k=100)
```

The distance matrix returned is a [pandas DataFrame](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html). ```amd.compare()``` can also accept paths to folders or lists of paths.

```amd.compare()``` reads crystals from CIFs, calculates their descriptors and compares them, but these steps can be done separately if needed (see below). ```amd.compare()``` accepts several optional parameters, see [the documentation for a full list](https://average-minimum-distance.readthedocs.io/en/latest/Getting_Started.html#list-of-optional-parameters).

*CSD Python API only:* ```amd.compare()``` interfaces with ``csd-python-api``. It can accept one or more CSD refcodes if passed ``refcode_families=True`` or other file formats instead of cifs if passed ``reader='ccdc'``.

#### Choosing a value of k

The parameter k is the number of neighbouring atoms considered for each atom in a unit cell. Two crystals with the same unit molecule will have a small AMD/PDD distance for small enough k (e.g. k = 5), and a larger k means the geometry must be similar up to a larger radius for the distance to be small. The default for ```amd.compare()``` is k = 100, but if this is significantly less than the number of atoms in the unit molecule, it may be better to choose a larger value. It is usually not useful to choose k too large (many times larger than the number of atoms in a unit cell).

### Reading crystals, calculating AMDs/PDDs

This code reads a cif file and computes the list of AMDs (k = 100):

```py
import amd

reader = amd.CifReader('file.cif')
amds = [amd.AMD(crystal, 100) for crystal in reader]
# # To calculate the PDDs:
# pdds = [amd.PDD(crystal, 100) for crystal in reader]
```

CifReader accepts some optional parameters, e.g. for removing Hydrogen and handling disorder, [see here for a full list](https://average-minimum-distance.readthedocs.io/en/latest/Reading_cifs.html).

*CSD Python API only:* CSD entries can be accessed via the CSD Python API with ```amd.CSDReader```, [see the documentation for details](https://average-minimum-distance.readthedocs.io/en/latest/Reading_from_the_CSD.html). CifReader can accept file formats other than .cif by passing ```reader='ccdc'```.

### Comparing by AMD or PDD

To compare all crystals in one collection with each other, use ```amd.AMD_pdist()``` or ```amd.PDD_pdist()```, which accept a list of AMDs/PDDs and return a *condensed distance matrix* like SciPy's ```pdist()```. Here's a full example of reading crystals from a .cif, calculating the descriptors and comparing them:

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

*Note: AMDs can be quickly computed from PDDs with `amd.PDD_to_AMD()`.*

The default metric for comparison is ```chebyshev``` (L-infinity), though it can be changed to anything accepted by SciPy's ```pdist```, e.g. ```euclidean```.

To compare crystals in one set with those in another set, use ```amd.AMD_cdist``` or ```amd.PDD_cdist```:

```py
import amd

amds1 = [amd.AMD(c, 100) for c in amd.CifReader('set1.cif')]
amds2 = [amd.AMD(c, 100) for c in amd.CifReader('set2.cif')]
# dm[i][j] = AMD distance between amds1[i] & amds2[j]
dm = amd.AMD_cdist(amds)
```

## Example: AMD-based dendrogram

This example compares some crystals in a cif by AMD (k = 100) and plots a single linkage dendrogram:

```py
import amd
import matplotlib.pyplot as plt
from scipy.cluster import hierarchy

crystals = list(amd.CifReader('crystals.cif'))
names = [crystal.name for crystal in crystals]
amds = [amd.AMD(crystal, 100) for crystal in crystals]
cdm = amd.AMD_pdist(amds)
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
