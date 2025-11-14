# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog][],
and this project adheres to [Semantic Versioning][].

[keep a changelog]: https://keepachangelog.com/en/1.0.0/
[semantic versioning]: https://semver.org/spec/v2.0.0.html

## Unreleased

### Added

### Changed

### Fixed

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
