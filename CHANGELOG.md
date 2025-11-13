# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog][],
and this project adheres to [Semantic Versioning][].

[keep a changelog]: https://keepachangelog.com/en/1.0.0/
[semantic versioning]: https://semver.org/spec/v2.0.0.html

## [Unrealeased]

### Added
- Added `pycea.tl.expansion_test` for computing expansion p-values to detect clades under selection

### Changed

### Fixed

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
