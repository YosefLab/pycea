# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog][],
and this project adheres to [Semantic Versioning][].

[keep a changelog]: https://keepachangelog.com/en/1.0.0/
[semantic versioning]: https://semver.org/spec/v2.0.0.html

## Unreleased

### Added
- `pycea.tl.ancestral_linkage` gained a `min_size` parameter; in pairwise mode, categories with fewer than `min_size` cells are excluded from the linkage matrix
- `pycea.tl.ancestral_linkage` now supports `aggregate='max'` with `metric='lca'` on non-ultrametric trees via an exact subtree walk-up (previously raised a `ValueError`)
- `pycea.pl.ancestral_linkage` for plotting the pairwise linkage matrix as a clustered heatmap, with options to symmetrize (`mean`/`max`/`min`) and normalize to `permuted_value` or `z_score`
- `pycea.pl.ancestral_linkage` gained a `cluster_mode` parameter (`'similarity'`/`'dissimilarity'`/`None`) controlling whether values are negated before clustering; `None` infers it from the recorded `metric` (`'dissimilarity'` for `metric='path'`, `'similarity'` otherwise)

### Changed
- `pycea.tl.ancestral_linkage` closest-target linkage (`min`/`path` and `max`/`lca`) is now computed with a subtree walk-up rather than per-category Dijkstra, making it substantially faster (single bottom-up pass regardless of category count)
- `pycea.tl.ancestral_linkage` now always runs at least one permutation, so `permuted_value` is available and `normalize=True` works even when `test=None`; `z_score`/`p_value` are still only computed with `test='permutation'`
- `pycea.tl.ancestral_linkage` now warns and lists categories with fewer than 10 cells (whose linkage is noisy), suggesting `min_size` to exclude them
- `pycea.tl.ancestral_linkage` default `permutation_mode` is now `'non_target'`
- `pycea.tl.ancestral_linkage` `permutation_mode='non_target'` permutations are vectorized with `numpy.bincount` (identical results, ~3–9× faster), making it faster than `'all'` again
- `pycea.tl.ancestral_linkage` defaults changed: `metric='lca'` and `normalize=True`
- `pycea.tl.ancestral_linkage` `symmetrize` now uses `False` to disable symmetrization instead of `None`, and defaults to `'mean'`
- `pycea.pl.ancestral_linkage` `normalize` and `symmetrize` default to `None`, inheriting the values used by `pycea.tl.ancestral_linkage` (recorded in `tdata.uns`) so the plot matches the stored matrix
- `pycea.tl.clades` now resets `tdata.uns["clade_colors"]` when number of clades differs from number of colors (#45)
- `pycea.pl.branches` and `pycea.pl.tree` now default `depth_key` to `None`, resolving to `tdata.uns["default_depth"]` if present, otherwise `"depth"`

### Fixed
- `pycea.get.palette` now correctly collects unique categories across all columns of array data, not just the first column
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
