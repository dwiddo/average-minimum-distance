# Changelog

All notable changes to this project will be documented in this file.

## [1.1.6] - 03/11/2021

### Added

- Optional parameter low_memory added to AMD comparison functions. Uses a slower but more memory efficient method, useful for larger collections of AMDs.

- The SetReader now has .to_dict() which returns a dict containing the scalar (strings and numbers) data attached to PeriodicSets. These could either be attrs of the h5py group containing the PeriodicSet, or values stored in the .tags dict of the PeriodicSet. The format of the returned dict is easily passable to pandas.DataFrame().

- To compliment the addition of .to_dict(), the CifReader and CSDReader now have ways to extract additional data from ccdc Entries (later ase CifBlocks). The parameter extract_data can be passed as a dictionary with data (column) titles as keys and callable functions as values, which take the Entry object and return the data. E.g. if the custom function get_density() takes the entry and returns the density, use extract_data={'Density': get_density}, then the PeriodicSets will have the tag 'Density' with a value given by get_density(entry). This approach is flexible to any function, but some common ones are included in amd.io.ccdc_utils.

- CifReader and CSDReader accept include_if parameter, which can be None or a list of callables. This can be used to discard structures with specific properties. If include_if is a list, the Entry/CifBlock is passed to each callable and is skipped if any do not return True.

### Changed

### Removed

- amd.compare() has been removed, as have the old lower case versions of functions amd.amd(), amd.amd_pdist(), etc.

- types=True removed as optional parameter to Readers, types are now always read.

- The option to pass a range of k values to comparison funcitons, along with the optional parameter ord, have been removed. Comparing over a range of k can be useful, but often its not if the range of distances aren't stored. No computational savings were really made in the function as opposed to a naive approach.

## [1.1.5] - 02/11/2021

### Added

- SetReader has additional convinient functions .keys() and .family().

- CSDReader has function .entry(refcode) which returns a PeriodicSet given a ccdc refocde. This can be any refcode, not just those given when the reader is initialised.

### Changed

- EMD has updated internals which is 2-3x faster.

## [1.1.4] - 13/09/2021

### Added

- Major changes were made to some internals of the CIF and CSD readers, as well as AMD/PDD computation functions. Now, a PeriodicSet read from one of these readers will contain the tags 'asymmetric_unit' and 'wyckoff_multiplicities'. Since equivalent sites have the same environment, it's only necessary to compute nearest neighbour distances for atoms in the asymmetric unit, and the PDD weights can be computed from the multiplicities. For crystals with an asymmetric unit, this improves AMD/PDD computation speed, PDD size and EMD speed.

### Changed

- **Functions amd.amd() and amd.pdd() are deprecated, replaced with amd.AMD() and amd.PDD(). Similarly, the 'amd' and 'pdd' in the names of comparison functions have been capitalized and old versions deprecated**. It was a bad idea to have a function named amd in a module of the same name. Other less used functions pdd_to_amd() and ppc() are now PDD_to_AMD() and PPC().

- The earth mover's distance algorithm no longer incorrectly divides by 1e+6. All earth mover's distances from this update will be 1e+6 times larger than before.

- ase and h5py are no longer an optional dependencies but required. numba was also added to dependencies as it was missing.

- The optional keyword argument types is no longer used for the Readers; types are read in regardless.

## [1.1.1] - 18/07/2021

### Added

- PeriodicSet object now impliments getattr to search tags for the attribute, e.g. if atomic types are in the set's tags under the name 'types' it can now be accesed by set.types.

- neighbours_from_distance_matrix(n, dm) takes an integer and a distance matrix and returns nearest neighbours.

### Changed

- Readers no longer 'normalise' the output by rounding and lexicographically sorting. Sorting the motif means the list of types must be sorted as well, so it's now left in the order it appears in the .cif (sans symops).

- Fixed bug reading .cifs whose symmetric operators don't start with the identity.

## [1.1.0] - 01/07/2021

### Added

- Several functions for comparison:
  - amd_pdist and pdd_pdist compare one collection pairwise by AMD/PDD respectively
  - amd_cdist and pdd_cdist compare one collection vs another by AMD/PDD respectively
  - filter takes PDDs. It gets the n closest items to every item by AMDs, then calculates PDD distances for these comparisons only. This does n x len(input) PDD comparisons instead of len(input) choose 2 for pdist, or len(input_1) * len(input_2) for cdist.
  - amd_mst and pdd_mst take a collection of AMDs/PDDs respectively and return edges of a minimum spanning tree.

  All of these functions support a range of metrics, any integer k (less than the input's max k) or range of k. The PDD functions include a verbose argument which prints an ETA to the terminal.

- The SetWriter and SetReader objects write and read PeriodicSets as compressed .hdf5 files (requires h5py). Optionally accepts any keyword arguments, stored alongside the sets (intended to store AMDs and PDDs).

- CifReader and CSDReader no longer take the optional boolean parameter 'allow_disorder', but instead the parameter 'disorder' which can be any out of {'skip', 'ordered_sites', 'all_sites'}. allow_disorder=False is now disorder='skip' (default: skip disordered structures), allow_disorder=True is now disorder='all_sites' (take all sites ignoring any disorder) and newly added disorder='ordered_sites' reads in the set but removes disordered sites.

- PeriodicSets now have an attribute .tags, a dict storing any additional information. This data is saved with the PeriodicSet if written with the SetWriter.

- CifReader and CSDReader include a parameter types (default False) which includes a list of atomic types of atoms in the .tags dict (with key 'types') of the PeriodicSets yielded. Access with set.tags['types'].

- Requirements now includes version requirements (numpy>=1.20.1, scipy>=1.6.1).

### Changed

- CifReader and CSDReader no longer take an optional boolean parameter 'allow_disorder', but instead a string 'disorder' out of {'skip', 'ordered_sites', 'all_sites'}. allow_disorder=False is now disorder='skip' (default: skip disordered structures), allow_disorder=True is now disorder='all_sites' (take all sites ignoring any disorder) and newly added disorder='ordered_sites' reads in the set but removes disordered sites.

- The ase reader backend has been cleaned up. It no longer handles ase Atoms or Spacegroup objects, which were throwing errors/warnings in some cases. Instead it deals with the tags directly.

- With the ccdc reader, if crystal.molecule.all_atoms_have_sites is False the structure is no longer automatically skipped but a warning is still printed.

- PeriodicSet is no longer a wrapped namedtuple. A tuple (motif, cell) is still accepted by the calculate functions, which dispatch to another function handling each case.

### Removed

- compare() has been removed and replaced with amd_cdist().
