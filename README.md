# amd: distance-based isometry invariants

For calculation and comparison of [AMD](<https://arxiv.org/abs/2009.02488>) and PDD isometric invariants. Includes functions for extracting periodic set representations of crystal structures from .cif files.

## Contents

1. [Requirements](#Requirements)
2. [Reading .cifs](#Readingcifs)
3. [From CSD refcode(s)](#Refcode)
4. [Calculating AMDs and PDDs](#Calculating)
5. [Comparing AMDs and PDDs](#Comparing)
6. [Read and write PeriodicSets](#PeriodicSetsIO)

## Requirements <a name="Requirements"></a>

- numpy, scipy (>=1.6.1)
- ase or ccdc to read in .cif files (ase recommended).

Use pip to install average-minimum-distance and ase (if required):

```shell
pip install average-minimum-distance ase
```

average-minimum-distance is imported with ```import amd```.

## Reading .cifs <a name="Readingcifs"></a>

The core use of this package is AMD and PDD calculations. They take a periodic set described by a unit cell and motif. While these can be passed manually, if you have a .cif file then ```amd.CifReader``` can read it and extract the cell and motif which can be easily passed to the invariant calculators. To read .cifs either ase or ccdc is required, ase is default and recommended. Using ccdc requires a valid license but allows you to use the optional parameter ```heaviest_component``` which attempts to remove solvents from your structure.

All readers return ```PeriodicSet``` objects, which have attributes ```motif```, ```cell``` and ```name```. They also have a dictionary ```.tags``` which can store additional information, e.g. invariants, atomic types or density.

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
                       heaviest_component=False,
                       types=False)
```

Most useful is ```remove_hydrogens```. ```types``` (may be removed/changed in future) will also read the atomic types; if the reader yielded a ```set``` then ```set.tags['types']``` is a list of atomic symbols as strings (in order with the motif). The rest are usually not needed and should be changed from the defaults with some caution.

```reader``` (one of 'ase' or 'ccdc') is the backend package used to parse the .cif. ase is recommended and available with pip. Choosing ccdc allows setting ```heaviest_component``` to True, this is used to remove solvents by removing all but the heaviest connected component in the asymmetric unit. **For some .cifs this can produce unintended results.**

```disorder``` (one of 'skip', 'ordered_sites', 'all_sites') contols handling of disordered structures. The default is to skip structures with disorder and print a warning. Disordered structures don't make sense under the periodic set model. Other arguments will not skip: 'ordered_sites' will remove atoms with disorder, and 'all_sites' will include all atoms. **Note: when disorder is missing on a site, the reader assumes there is no disorder.**

```dtype``` is the numpy datatype of the motif and cell returned by the reader.

## From CSD refcode(s) <a name="Refcode"></a>

If you have ccdc installed, you can also get the periodic set form of structures from their CSD refcodes. ```amd.CSDReader``` has a similar interface to the cif reader, except you pass a list of refcodes:

```py
reader = amd.CSDReader(['DEBXIT', 'DEBXIT01'])               # get exactly DEBXIT and DEBXIT01
reader = amd.CSDReader(['DEBXIT', 'ACSALA'], families=True)  # get families (refcodes starting with these)
```

The CSD reader can take the same optional arguments as the cif reader (except ```reader```).

## Calculating AMDs and PDDs <a name="Calculating"></a>

The functions ```amd.amd``` and ```amd.pdd``` are for AMD and PDD calculations respectively. They have 2 required arguments:

- either a ```PeriodicSet``` given by a reader or a tuple ```(motif, cell)``` of numpy arrays,
- an integer ```k``` > 0.

The following creates a list of AMDs (with ```k=500```) for structures in a .cif:

```py
from amd import CifReader, amd
amds = [amd(periodic_set, 500) for periodic_set in amd.CifReader('path/to/file.cif')]
```

The functions also accept a tuple ```(motif, cell)``` to allow quick tests without a .cif, for example this calculates PDD (k=500) for a simple cubic lattice:

```py
import numpy as np
from amd import pdd
motif = np.array([[0,0,0]]) # one point at the origin
cell = np.identity(3)       # unit cell = identity
cubic_pdd = pdd((motif, cell), 500)
```

PDDs are returned as a concatenated matrix with weights in the first column.

Remember that amd and pdd always expect Cartesian forms of a motif and cell, with all points inside the cell. If you have unit cell parameters or fractional coodinates, then use ```amd.cellpar_to_cell``` to convert a,b,c,α,β,γ to a 3x3 Cartesian cell, then ```motif = np.matmul(frac_motif, cell)``` to get the motif in Cartesian form before passing to ```amd``` or ```pdd```.

## Comparing AMDs and PDDs <a name="Comparing"></a>

**Note: the comparison API is most likely to be subject to changes.**

AMDs are vectors which can be compared with any valid metric, usually l-infinity/Chebyshev distance. PDDs are compared with earth mover's distance (aka the Wasserstein metric).

### Comparison functions

The comparison part of this package has several functions with several options. Most useful might be ```amd.amd_pdist``` and ```amd.pdd_pdist```, which each take a collection of AMDs/PDDs and does a pairwise comparison, returning a [condensed distance matrix/vector](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.squareform.html). In contrast, ```amd.amd_cdist``` and ```amd.pdd_cdist``` are for comparison of one set vs another and return 2D distance matrices. The more advanced function ```amd.filter``` takes PDDs and only compares on each set's ```n``` nearest AMD-neighbours, comprimising the speed of AMDs and the accuracy of PDDs. There's also ```amd.amd_mst``` and ```amd.pdd_mst``` which calculates a minimum spanning tree.

### Comparison options

All comparison functions share several optional arguments: ```k=None```, ```metric='chebyshev'```, and ```ord=None```. ```k``` can be any int less than or equal to the number of columns/length of the passed invariants, or a range of k values. If ```k``` is a range, comparisons are done over a range of k making a vector of distances (for each comparison) which is collapsed to a single value using a norm specified by ```ord``` (default Euclidean; for all ```ord``` options see [numpy.linalg.norm](https://numpy.org/doc/stable/reference/generated/numpy.linalg.norm.html)).

The ```metric``` parameter decides how invariants are acutally compared. The options for ```metric``` are the same as those for [scipy.spatial.distance.cdist](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html), default is Chebyshev/l-infinity. With AMDs, this refers directly to the metric used to comapre AMD vectors. For two PDDs, the rows of the PDDs are compared by this metric, creating a distance matrix on PDD rows which is passed to the earth mover's distance to get a single value.

Functions that compare using PDDs have an optional ```verbose``` parameter which prints an ETA to the terminal.

To compare a collection pairwise by PDD:

```py
import amd

pdds = [amd.pdd(s, 100) for s in amd.CifReader('structures.cif')]
condensed_distance_matrix = amd.pdd_pdist(pdds)
```

To compare one set vs another with AMDs:

```py
set_1_amds = [amd.amd(s, 100) for s in amd.CifReader('set_1.cif')]
set_2_amds = [amd.amd(s, 100) for s in amd.CifReader('set_2.cif')]

distance_matrix = amd.amd_cdist(set_1_amds, set_2_amds)
```

## Read and write PeriodicSets <a name="PeriodicSetsIO"></a>

You can write and read PeriodicSet objects with .hdf5 files (requires ```h5py```), along with their invariants, atomic types or other data. This is much faster than parsing a .cif for the sets. When writing a ```set```, the name, motif and cell are stored (a name is required to write). Example:

```py
import amd

# write
with amd.SetWriter('path/to/sets_file.h5py') as writer:

    for p_set in periodic_sets:
        writer.write(p_set)

    # above loop is equivalent to
    writer.iwrite(periodic_sets)

# read
with amd.SetReader('path/to/sets_file.h5py') as reader:
    for p_set in reader: # yields PeriodicSets
        print(p_set.name) # print names of all sets in file
    
    print(reader['set_name'].motif) # reader supports key indexing
```

If creating the reader or writer manually instead of using a context manager, remember to close them with ```.close()```. The reader and writer preserve order, and also try to store the data in a set's ```tags``` dictionary (supports at least scalars e.g. float, int, str, as well as numerical arrays and lists of strings). For example, to write AMDs (k=100) of periodic sets:

```py
import amd

for p_set in periodic_sets:
    p_set.tags['amd'] = amd.amd(p_set, 100) 

with amd.SetWriter('path/to/sets_file.h5py') as writer:
    writer.iwrite(periodic_sets)

with amd.SetReader('path/to/sets_file.h5py') as reader:
    for p_set in reader:
        print(p_set.tags['amd']) 
```
