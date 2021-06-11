# amd

For the calculation and comparison of AMD/PDD isometric invariants. Includes functions for extracting periodic set representations of crystal structures from .cif files.

## Requirements

- numpy and scipy
- ase or ccdc to read in .cif files (ase recommended).

## Guide

### Reading .cifs

```amd``` includes functionality to read .cif files and extract their motif and cell in Cartesian form. To do so requires either  ```ase``` or ```ccdc```. ```ase``` is the default and recommended, as it can be easily pip installed. ```ccdc``` is not recommended, but in some specific cases it provides useful options. Using ```ccdc``` requires a valid license.

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

periodic_sets = list(reader)
```

If your .cif contains just one structure, use the syntax:

```py
periodic_set = list(amd.CifReader('path/to/one_structure.cif'))[0]
```

If you have a folder with many .cifs each with one structure, you can use this list comprehension:

```py
import os
folder = 'path/to/cif_folder'
periodic_sets = [list(amd.CifReader(os.path.join(folder, filename)))[0] for filename os.listdir(folder)]
```

The ```CifReader``` has several optional arguments. Most useful is ```remove_hydrogens``` which defaults to False. To remove Hydrogen atoms when reading, use

```py
reader = amd.CifReader('path/to/file.cif', remove_hydrogens=True)
```

The rest of the arguments of ```CifReader``` are usually not of use and should be changed from the defaults with caution.

```dtype```  controls the floating point precision to which motif/cell data is stored, which may be of use for many .cifs (default ```np.float64```; should only change to ```np.float32```). 

If ```allow_disorder``` is False (default), when any disorder is detected in a structure it will be skipped and a warning. Disordered structures don't make sense under the periodic set model; there is no good way to interpret them. Setting this to True will ignore disorder, including every atomic site regardless. 

The argument ```reader``` (default ```'ase'```) chooses the backend package used to read the .cif. ```ase``` is recommended and can be easily pip installed. Choosing ```'ccdc'``` provides one extra optional argument for the ```CifReader```, ```heaviest_component``` (default False). This is used to remove solvents, when True it will remove all but the heaviest connected component in the asymmetric unit. For many .cifs this can produce unintended results.

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

Remember that amd and pdd always expect Cartesian forms of a motif and cell, with all points inside the cell. If you have unit cell parameters or fractional coodinates, then use ```amd.cellpar_to_cell``` to convert a,b,c,alpha,beta,gamma to a 3x3 Cartesian cell, then ```motif = np.matmul(frac_motif, cell)``` to get the motif in Cartesian form before passing to ```amd``` or ```pdd```.

### Comparing AMDs and PDDs

So far, AMDs are simply compared with l-infinity/chebyshev distance. To compare AMDs, it is recommended you use ```amd.compare```. It has two required arguments, ```reference``` and ```comparison```, which both may either by a single AMD, a list of AMDs, or a numpy array with AMDs in rows. In any case the length of all AMDs must be equal. For m reference AMDs and n comparison AMDs, ```amd.compare``` returns a distance matrix with shape ```(m, n)```, where the (i,j)-th entry is the AMD distance between reference i and comparison j. For example, this code compares all structures in one .cif to all structures in another by AMD100:

```py
import amd

k = 100
set_1_amds = [amd.amd(s, k) for s in amd.CifReader('set_1.cif')]
set_2_amds = [amd.amd(s, k) for s in amd.CifReader('set_2.cif')]

distance_matrix = amd.compare(set_1_amds, set_2_amds)
```

To compare a collection pairwise, just pass it in twice, e.g.:

```py
amds = [amd.amd(s, 100) for s in amd.CifReader('structures.cif')]
distance_matrix = amd.compare(amds, amds)
```
