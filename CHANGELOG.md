# Changelog

All notable changes to this project will be documented in this file.

## [1.1.0] - 18/06/2021

### Added

- Several functions for comparison:
  - amd_pdist and pdd_pdist compare one collection pairwise by AMD/PDD respectively
  - amd_cdist and pdd_cdist compare one collection vs another by AMD/PDD respectively
  - filter takes PDDs. It gets the n closest items to every item by AMDs, then calculates PDD distances for these comparisons only. This does n x len(input) PDD comparisons instead of len(input) choose 2 (for pdist) or len(input) ** 2 (for cdist).
  - amd_mst and pdd_mst take a collection of AMDs/PDDs respectively and return edges of a minimum spanning tree.

  All of these functions support a range of metrics, any integer k (less than the input's max k) or range of k. The PDD functions include a verbose argument which prints an ETA to the terminal.

- The SetWriter and SetReader objects write and read PeriodicSets as compressed .hdf5 files (requires h5py). Optionally accepts any keyword arguments, stored alongside the sets (intended to store AMDs and PDDs).

- CifReader and CSDReader no longer take an optional boolean parameter 'allow_disorder', but instead a string 'disorder' out of {'skip', 'ordered_sites', 'all_sites'}. allow_disorder=False is now disorder='skip' (default: skip disordered structures), allow_disorder=True is now disorder='all_sites' (take all sites ignoring any disorder) and newly added disorder='ordered_sites' reads in the set but removes disordered sites.

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
