# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog][],
and this project adheres to [Semantic Versioning][].

[keep a changelog]: https://keepachangelog.com/en/1.0.0/
[semantic versioning]: https://semver.org/spec/v2.0.0.html

## Unreleased

### Added
- `pycea.tl.parsimony` and `pycea.tl.fitch_count` for small-parsimony scoring and the FitchCount transition-count algorithm (migrated from Cassiopeia)
- `pycea.pl.ancestral_linkage` for plotting the pairwise linkage matrix as a clustered heatmap (#58)
- Added fast `"sum"` method to `pycea.tl.ancestral_states` (#54, #56)
- Added additional node ploting customization - `outline_width` and option to directly specify color with hex code (#53)
- `pycea.tl.ancestral_linkage` for computing relatedness of cells in different categories, with options for pairwise or target-category linkage permutation testing, and normalization. (#52, #55, #58)
- Added `angle_range` parameter for plotting polar trees with `pycea.pl.tree` and `pycea.pl.branches` (#51)

### Changed
- Added support for `tdata.alignment != 'leaves'` to relevant functions in `pycea.pl` and `pycea.tl` (#50)
- Vectorized continuous color computation in `pycea.pl.tree` and `pycea.pl.branches` for improved node plotting performance (#48)

### Fixed
- Fixed `IndexError` retrieving `var_names`/layer data under `anndata>=0.13` by indexing the column directly instead of via the deprecated `obs_vector` (#58)
- `pycea.get.palette` now correctly collects unique categories across all columns of array data, not just the first column (#57)
- Fixed bug in LCA distance computation when for non-ultrametric tree (#49)
- Legend placement now works with tight and constrained layouts (#45)

## [0.2.0] - 2025-11-14

### Added
- Added `pycea.tl.expansion_test` for computing expansion p-values to detect clades under selection

- `pycea.tl.partition_test` to test for statistically significant differences between leaf partitions. (#40)
- `pycea.tl.expansion_test` for computing expansion p-values to detect expanding clades. (#38)

### Changed

- Replaced `tdata.obs_keys()` with `tdata.obs.keys()` to conform with anndata API changes. (#41)
- `pycea.tl.fitness` no longer returns a multi-indexed DataFrame when `tdata` contains a single tree. (#38)

### Fixed

- Fixed node plotting when `isinstance(nodes,str)`. (#39)

## [0.1.0] - 2025-09-19

### Added

- `pycea.get` module for data retrieval (#32)
- Added `pycea.tl.n_extant` and `pycea.pl.n_extant` for calculating and plotting the number of extant lineages over time (#33)
- Added `pycea.tl.fitness` for estimating fitness of nodes in a tree (#35)

### Changed

- Only require `tree` parameter to be specified when trees in `tdata` actually overlap (#37)

### Fixed

- Sorting now preserves edge metadata (#31)

## [0.0.1]

### Added

-   Basic tool, preprocessing and plotting functions
