# amd: distance-based isometry invariants

For calculation and comparison of [AMD](<https://arxiv.org/abs/2009.02488>)/PDD isometric invariants. Includes functions for extracting periodic set representations of crystal structures from .cif files.

## Requirements

- numpy and scipy
- ase or ccdc to read in .cif files (ase recommended).

Use pip to install average-minimum-distance and ase (if required):

```shell
pip install average-minimum-distance ase
```

average-minimum-distance is imported with ```import amd```.

## Guide

See the iPython notebook in the tests folder for examples with real cif files.

### Reading .cifs

amd includes functionality to read .cif files and extract their motif and cell in Cartesian form. To do so requires either ase or ccdc. ase is the default and recommended, as it can be easily pip installed. ccdc is not recommended, but in some specific cases it provides useful options. Using ccdc requires a valid license.

All readers return ```PeriodicSet``` objects, which have attributes ```name```, ```motif``` and ```cell```. ```PeriodicSet```s are intended for easy use with the AMD/PDD calculators.

The following creates a ```CifReader``` object which can be iterated over to get all (valid) structures in a .cif:

```py
import amd
reader = amd.CifReader('path/to/file.cif')
```

This can be used in a loop, comprehension or converted to a list:

```py
for periodic_set in reader:
    print(periodic_set.motif.shape[0]) # prints number of motif points
```

By default, the reader will skip structures that cannot be read and print a warning.

If your .cif contains just one structure, use:

```py
periodic_set = list(amd.CifReader('path/to/one_structure.cif'))[0]
```

If you have a folder with many .cifs each with one structure:

```py
import os
folder = 'path/to/cif_folder'
periodic_sets = [list(amd.CifReader(os.path.join(folder, filename)))[0] 
                 for filename os.listdir(folder)]
```

```CifReader``` has several optional arguments:

```py
reader = amd.CifReader(filename,
                       reader='ase',
                       remove_hydrogens=False,
                       allow_disorder=False,
                       dtype=np.float64,
                       heaviest_component=False)
```

Most useful (and stable) is ```remove_hydrogens```. The rest are usually not needed and should be changed from the defaults with some caution.

```reader``` ('ase' or 'ccdc') is the backend package used to read the .cif. ase is recommended and can be easily pip installed. Choosing ccdc allows setting ```heaviest_component``` to True, this is used to remove solvents by removing all but the heaviest connected component in the asymmetric unit. **For some .cifs this can produce unintended results.**

```allow_disorder``` contols handling of disordered structures. By default they are skipped by the reader and a warning is printed. Disordered structures don't make sense under the periodic set model; there is no good way to interpret them. Setting this to True will ignore disorder, including every atomic site regardless. **Note: when disorder information is missing on a site, the reader assumes there is no disorder there.**

```dtype``` is the numpy datatype of the motif and cell returned by the reader. The default ```np.float64``` should be fine for most cases. If the size of the data is limiting it may help to set ```dtype=np.float32```.

### Calculating AMDs and PDDs

The functions ```amd.amd``` and ```amd.pdd``` are for AMD and PDD calculations respectively. They have 2 required arguments:

- either a ```PeriodicSet``` given by a reader or a tuple ```(motif, cell)``` of numpy arrays,
- an integer ```k``` > 0.

The following creates a list of AMDs (with ```k=100```) for structures in a .cif:

```py
from amd import CifReader, amd
amds = [amd(periodic_set, 100) for periodic_set in amd.CifReader('path/to/file.cif')]
```

The functions also accept a tuple ```(motif, cell)``` to allow quick tests without a .cif, for example this calculates PDD (k=100) for a simple cubic lattice:

```py
import numpy as np
from amd import pdd
motif = np.array([[0,0,0]]) # one point at the origin
cell = np.identity(3)       # unit cell = identity
cubic_pdd = pdd((motif, cell), 100)
```

PDDs are returned as a concatenated matrix with weights in the first column.

Remember that amd and pdd always expect Cartesian forms of a motif and cell, with all points inside the cell. If you have unit cell parameters or fractional coodinates, then use ```amd.cellpar_to_cell``` to convert a,b,c,α,β,γ to a 3x3 Cartesian cell, then ```motif = np.matmul(frac_motif, cell)``` to get the motif in Cartesian form before passing to ```amd``` or ```pdd```.

### Comparing AMDs and PDDs

AMDs are usually compared with l-infinity/Chebyshev distance. ```amd.compare``` is used for comparing sets of (or single) AMDs. It has two required arguments, ```reference``` and ```comparison```, which both may either be single vectors or  matrices with the vectors to compare in rows. The number of columns in ```reference``` and ```comparison``` must agree. ```amd.compare``` also optionally accepts any metric supported by ```scipy.spatial.distance.cdist```, default chebyshev. For input shapes (m,k) and (n,k), a distance matrix with shape ```(m, n)``` is returned where the (i,j)-th entry is the distance between ```reference[i]``` and ```comparison[j]```. For example, this compares everything in one .cif to everything in another with AMD100:

```py
import amd

k = 100
set_1_amds = [amd.amd(s, k) for s in amd.CifReader('set_1.cif')]
set_2_amds = [amd.amd(s, k) for s in amd.CifReader('set_2.cif')]

distance_matrix = amd.compare(set_1_amds, set_2_amds)
```

To compare a collection pairwise, just pass it in twice:

```py
amds = [amd.amd(s, 100) for s in amd.CifReader('structures.cif')]
distance_matrix = amd.compare(amds, amds)
```

To compare two PDDs, use ```amd.emd```. This gives the Earth mover's distance between crystals in two seperate .cifs:

```py
pdd_1 = amd.pdd(list(amd.CifReader('crystal_1.cif'))[0])
pdd_2 = amd.pdd(list(amd.CifReader('crystal_2.cif'))[0])
dist = amd.emd(pdd_1, pdd_2)
```

A simple function for comparing many PDDs is not yet implimented, but is easy enough with a loop over two lists of PDDs.
