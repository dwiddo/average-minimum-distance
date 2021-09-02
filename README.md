# average-minimum-distance: distance-based isometry invariants

If you use our code in your work, please cite our paper at [arxiv.org/abs/2009.02488](https://arxiv.org/abs/2009.02488). The bib reference is at the bottom of this page; [click here jump to it](#citeus).

This package is for calculation and comparison of new [AMD](https://arxiv.org/abs/2009.02488) and PDD isometric invariants of crystal structures. Also includes a .cif reader, functions to compare AMDs/PDDs and integration with the ccdc package (if installed).

## Contents

1. [What is amd?](#Whatisamd)
2. [Requirements](#Requirements)
3. [Getting started](#GettingStarted)
4. [Reading .CIF files](#Readingcifs)
5. [Calculating AMDs and PDDs](#Calculating)
6. [Comparing AMDs and PDDs](#Comparing)
7. [PeriodicSets and storing crystals or invariants](#PeriodicSetsIO)
8. [CSD refcodes and the ccdc package](#Refcode)

## What is amd? <a name="Whatisamd"></a>

The structure of solid crystalline materials is usually described with a 'motif' of atomic positions inside a 'unit cell' which periodically repeats, data typically stored in a .CIF file. This representation is ambiguous---one crystal can be represented in many different ways. This package implements new *isometric invariants* called AMD (average minimum distance) and PDD (point-wise distance distribution) based on inter-point distances, which are guaranteed to take the same value for all equivalent representations of a crystal. They do this in a continuous way; crystals which are similar have similar AMDs and PDDs.

The AMD of a crystal is a vector, while PDDs are matrices. The length of an AMD and number of columns in a PDD can be any chosen integer. AMD vectors can be compared with any metric, but typically we use the l-infinity (aka Chebyshev) metric; PDDs are compared with the earth mover's distance (aka Wasserstein) metric.

For a technical description of AMD, [see our paper on arXiv](https://arxiv.org/abs/2009.02488).

## Requirements <a name="Requirements"></a>

- numpy, scipy, numba, h5py and ase.
- Optionally, ccdc for reading from the CSD (requires a license).

## Getting started <a name="GettingStarted"></a>

Use pip to install average-minimum-distance:

```shell
pip install average-minimum-distance
```

average-minimum-distance is imported with ```import amd```.

The central functions of this package are ```amd.AMD()``` and ```amd.PDD()```, which take a 'crystal' and a positive integer k, returning the crystal's AMD/PDD up to k as a vector/matrix (not a single value). There are a few ways to get a 'crystal' which can be passed to these functions:

- If you have a .CIF file, ```amd.CifReader``` will extract the crystal(s) from it,
- If you have CSD refcodes, ```amd.CSDReader``` can read the crystals from the CSD (requires ccdc),
- A crystal can be manually created as a tuple of numpy arrays (motif, cell).

Here's a simple example of each case:

```py
import amd

k = 100

# read all structures in a .cif and put their amds in a list
reader = amd.CifReader('path/to/file.cif')
amds = [amd.AMD(crystal, k) for crystal in reader]

# grab crystals from a list of CSD refcodes; requires ccdc.
reader = amd.CSDReader(['DEBXIT01', 'SEMFAU'])
amds = [amd.AMD(crystal, k) for crystal in reader]

# manually create a simple cubic lattice
import numpy as np
motif = np.array([[0,0,0]])                 # must be a 2D array
cell = np.array([[1,0,0],[0,1,0],[0,0,1]])  # must be a square array
crystal = (motif, cell)
cubic_amd = amd.AMD(crystal, k)
```

A list of AMDs as created above can be compared pairwise with the function ```amd.AMD_pdist```, which returns a condensed distance matrix as does ```scipy.spatial.distance.pdist```. The PDD invariants share a similar interface.

## Reading .CIF files <a name="Readingcifs"></a>

Crystallographic Information Files (.CIF) are a common way of storing crystal structure data. ```amd.CifReader``` can read a .CIF file and extract the crystal(s) (as ```amd.PeriodicSet``` objects, described in detail below) which can be passed to ```amd.AMD``` or ```amd.PDD```. The reader is an iterator, so it can be used in a loop or you can get a list of the crystals by passing it to ```list()```. **Note that this reader will skip structures that can't be read and print a warning**, so it's possible for the reader to yield fewer items than are in the .CIF or no items at all.

```py
import amd

# read all structures in a .cif and put their amds in a list (in order)
reader = amd.CifReader('path/to/file.cif')
amds = [amd.AMD(crystal, 100) for crystal in reader]
```

If your file contains just one structure, this will extract it:

```py
crystal = list(amd.CifReader('path/to/file.cif'))[0]
```

If you have a folder with many .CIFs each with one crystal, you can use:

```py
import os
folder = 'path/to/cif_folder'
crystals = [list(amd.CifReader(os.path.join(folder, filename)))[0] 
            for filename os.listdir(folder)]
```

```amd.CifReader``` can take the optional argument ```remove_hydrogens``` (default False), when True this removes all Hydrogen atoms in the structure.

Another argument ```disorder``` (one of 'skip', 'ordered_sites', 'all_sites') controls handling of disordered structures. The default is to skip structures with disorder and print a warning, because disorder conflicts somewhat with the mathematical model we use for crystals. Other arguments will not skip: 'ordered_sites' will remove atoms with disorder, and 'all_sites' will include all atoms. **Note: if disorder info is missing, the reader assumes there is none.**

All readers return ```PeriodicSet``` objects, whose most important attributes are ```motif```, ```cell``` and ```name```. They also have a dictionary `.tags`, which can contain additional data. From version 1.1.3, atomic types are automatically read as well as indices of the asymmetric unit and Wyckoff multiplicities (these are used in AMD/PDD calculations). More on ```PeriodicSet```s (and storing them) are in a section below.

## Calculating AMDs and PDDs <a name="Calculating"></a>

The functions ```amd.AMD()``` and ```amd.PDD()``` are for AMD and PDD calculations respectively. They have 2 required arguments, a 'crystal' and positive integer ```k```. The crystal can come from a ```CifReader``` (for .CIF files), a ```CSDReader``` (for CSD refcodes) or created manually.

```amd.AMD()``` returns a numpy vector of length `k`, whereas `amd.PDD()` returns a numpy array with shape `(m, k+1)` where `m` is the number of points in the asymmetric unit of the crystal. The first column of the array contains the weights for each row, hence there are `k+1` columns.

### Converting cell parameters and fractional coordinates

The functions ```amd.AMD()``` and ```amd.PDD()``` expect Cartesian forms of a motif and cell, with all points inside the cell. If using `amd.CifReader` or `amd.CSDReader` this is taken care of for you. But, if you only have 6 unit cell parameters and/or fractional coordinates, then use ```amd.cellpar_to_cell()``` to convert a,b,c,α,β,γ to the proper cell representation (passing the 6 values in that order), then ```motif = np.matmul(frac_motif, cell)``` to get the motif in Cartesian form. The Cartesian cell and motif can then be passed to `amd.AMD()` or `amd.PDD()` in a tuple, i.e. `amd.AMD((motif, cell), 100)`.

## Comparing AMDs and PDDs <a name="Comparing"></a>

**Note: the comparison API is new and is most likely to be subject to changes. These functions are not yet generally suitable for large collections, e.g. >30,000 items.**

AMDs are vectors which can be compared with any valid metric, usually l-infinity/Chebyshev distance. PDDs are compared with earth mover's distance (aka the Wasserstein metric). This package has several functions which can be given lists of AMDs or PDDs to do comparisons on the collection for you.

### Pairwise comparison on a set

`amd.AMD_pdist` and `amd.PDD_pdist` can handle comparing a set pairwise (compare everything vs everything else in one set) by AMD and PDD respectively. They work similarly to `scipy.spatial.distance.pdist`, taking a list of AMDs/PDDs and returning a [condensed distance matrix](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.squareform.html) (use `scipy.spatial.distance.squareform` to convert to a standard square distance matrix). Example:

```py
import amd

reader = amd.CifReader('crystals.cif')
amds = [amd.AMD(crystal, 100) for crystal in reader]
condensed_dm = amd.AMD_pdist(amds)
```

### Comparing one set against another

`amd.AMD_cdist` and `amd.PDD_cdist` can be used when you have two collections of crystals you want to compare against each other (e.g. for finding the nearest neighbours of a small subset of a collection). Like the pdist functions above, they work like `scipy.spatial.distance.cdist`, taking two collections of AMDs/PDDs and returning a 2D distance matrix. 

```py
import amd

ref_reader = amd.CifReader('reference_crystals.cif')
reference_pdds = [amd.PDD(crystal, 100) for crystal in ref_reader]

com_reader = amd.CifReader('comparison_crystals.cif')
comparison_pdds = [amd.PDD(crystal, 100) for crystal in com_reader]

dm = amd.PDD_cdist(reference_pdds, comparison_pdds)
```

### Minimum spanning trees

Since we have often made minimum spanning trees from these comparisons, functions are included that can construct them for you. `amd.AMD_mst` and `amd.PDD_mst` take one collection of AMDs/PDDs (like pdist) and return a list of tuples `(i,j,w)` representing edges where `i,j` are indices pointing to the crystals in the passed list (nodes) and w is the weight of the edge.

### Get the n nearest neighbours to each crystal

It is often useful to find the n nearest neighbours to every crystal in a set (or from one set to another). The convenient function `amd.neighbours_from_distance_matrix(n, dm)` takes an integer `n` and distance matrix `dm`, finding the `n` nearest neighbours (in order). The distance matrix input can generally be any distance matrix, condensed or not.

The function returns two arrays `(nn_dm, inds)`, with `nn_dm` containing distances and `inds` being indices pointing to which crystals are the nearest neighbours. So, `nn_dm[i]` is a vector length `n` of the smallest distances from the `i`th crystal (in the original list) to the others, and `inds[i]` contains the indices of the corresponding nearest neighbours so they can be found in the original list of crystals.

```py
import amd

crystals = list(amd.CifReader('crystals.cif'))
amds = [amd.AMD(crystal, 100) for crystal in crystals]

condensed_dm = amd.AMD_pdist(amds)
nn_dm, inds = amd.neighbours_from_distance_matrix(3, condensed_dm)

>>> nn_dm[0], inds[0]
([0.001,0.004,0.010] , [1,4,7])
# so crystals[0]'s 3 nearest neighbours are:
#   - crystals[1] with distance 0.001
#   - crystals[4] with distance 0.004
#   - crystals[7] with distance 0.010
```

### 'Filtering' by AMD before comparing by PDD

PDD comparisons should be 'more accurate' than AMD, at the cost of taking longer. The function `amd.filter(n, pdds)` works as a compromise: given an int `n` and PDDs, it first compares everything by AMD and then compares only the `n` nearest AMD neighbours of each crystal by PDD. For suitable `n`, this gives the better PDD results while taking far less time than a pairwise PDD comparison, especially for large collections. This function returns two matrices in the same format as `amd.neighbours_from_distance_matrix` above, one matrix for distances and one for indices of the nearest neighbours. One or optionally two lists of PDDs can be given to `filter`, giving pdist (pairwise) or cdist (one set vs another) behaviour respectively.

### Comparison options

The comparison functions (includes all functions above except `neighbours_from_distance_matrix`) share several optional arguments:

- `k=None` can be any integer not larger than the `k` used to create the AMDs/PDDs, or a range of such values. The comparison then happens for that value, or range, of k. When this is a range, the parameter `ord` (below) controls how the sequence of distances over k are collapsed into a single value. Default `None` compares with the entire invariant i.e. the largest `k`.

- `metric='chebyshev'` controls how invariants will be compared. Accepted values are the same as those for [scipy.spatial.distance.cdist](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html), but is Chebyshev (aka l-infinity) by default.  With AMDs, this refers directly to the metric used to compare two AMD vectors. The metric used between PDDs is always the Earth mover's distance (EMD), but the input to EMD is actually a matrix of distances between the rows of the two PDDs. So for PDD comparisons, `metric` refers to the metric used for distances between rows of the PDDs before computing EMD.

- `ord=None` is only used when `k` is a range. Then, every comparison happens for all `k` in the range, making a sequence of distance matrices. To keep output a consistent shape, we take a norm `ord` across the values of `k` so we get one value per comparison instead of a sequence. `ord` can be any value accepted by [numpy.linalg.norm](https://numpy.org/doc/stable/reference/generated/numpy.linalg.norm.html), default `None` is Euclidean.

- Only for functions involving PDD comparisons, `verbose=False` can be set to `True` which will print an ETA to terminal.

- Specifically `amd.PDD_mst` has the optional argument `amd_filter_cutoff=None`, when set to an integer this applies the behaviour of `amd.filter` instead of making a full PDD distance matrix before creating the minimum spanning tree.

- The function `amd.filter()` can accept either one or two lists of PDDs, behaving like `PDD_pdist` in the former case and `PDD_cdist` in the latter. The first list is a required argument (after the int `n`), the second list is the optional keyword argument `pdds_=None`.

## PeriodicSets and storing crystals with their invariants <a name="PeriodicSetsIO"></a>

`amd.CifReader` and `amd.CSDReader` both yield `PeriodicSet` objects, which are what can be given to `amd.AMD` or `amd.PDD`. Several attributes come with a `PeriodicSet` generated from a reader: `name`, `motif`, `cell`, `types`, `asymmetric_unit`, and `wyckoff_multiplicities`. `motif` and `cell` are required for computing any invariant; the last two aren't required but make things faster. `types` stores the atomic types of each motif point in order.

You can write and read `PeriodicSet`s to .hdf5 files, along with other data. Reading an .hdf5 is much faster than parsing a .CIF to recover the crystals. All the attributes of the set are stored; in addition you can attach other data to be written e.g. the density, or AMD/PDD, by setting them in the sets `.tags` dictionary as in this example:

```py
import amd

periodic_sets = list(amd.CifReader('crystals.cif'))

# attach AMD to each PeriodicSet
for periodic_set in periodic_sets:
    periodic_set.tags['AMD100'] = amd.AMD(periodic_set, 100)

# write PeriodicSets and their AMDs
with amd.SetWriter('periodic_sets.h5py') as writer:

    for periodic_set in periodic_sets:
        writer.write(periodic_set)

    # above loop is equivalent to
    writer.iwrite(periodic_sets)


with amd.SetReader('periodic_sets.h5py') as reader:

    for periodic_set in reader:
        print(periodic_set.name) # print names of all sets in file

    # AMDs were stored in the .tags dict
    amds = [periodic_set.tags['AMD100'] for periodic_set in reader]

    print(reader['set_name'].motif) # reader supports name indexing
```

Note that you cannot save arbitrary objects in `.tags` to .hdf5, currently the only allowed types are scalars (numbers and strings), numpy arrays, lists of strings or lists castable to numpy arrays.

The reader and writer preserve order, i.e. a `SetReader` will read back sets in the same order they were written. If creating the reader or writer manually instead of using a context manager, remember to close them with `.close()`.

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

## CSD refcodes and the ccdc package <a name="Refcode"></a>

```amd.CSDReader``` is used to read in crystals from their CSD refcodes using the `ccdc` package. It has a similar interface to `CifReader`, except you pass a list of refcodes instead of a file path:

```py
reader = amd.CSDReader(['DEBXIT', 'DEBXIT01'])               # get exactly DEBXIT and DEBXIT01
reader = amd.CSDReader(['DEBXIT', 'ACSALA'], families=True)  # get any with refcodes starting with these
```

The `CSDReader` takes the same optional keyword arguments as `CifReader`, `remove_hydrogens` and `disorder`. It allows for one additional argument that's rarely needed: when `heaviest_component` is set to `True`, the reader will only keep the heaviest component of the asymmetric unit. This is intended to remove solvents from the structure. The `CifReader` can have this enabled too, if you set `reader='ccdc'` as well (which is not generally recommended).

## Cite us <a name="citeus"></a>

The arXiv paper for this package is [here](arxiv.org/abs/2009.02488). Use the following bib reference to cite us:

@article{widdowson2020average,  
  title={Average Minimum Distances of periodic point sets},  
  author={Daniel Widdowson and Marco Mosca and Angeles Pulido and
          Vitaliy Kurlin and Andrew Cooper},  
  journal={arXiv:2009.02488},  
  year={2020}  
}
