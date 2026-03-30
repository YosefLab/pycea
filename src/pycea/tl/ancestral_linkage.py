from __future__ import annotations

import multiprocessing as mp
import sys
import warnings
from collections import defaultdict
from collections.abc import Callable, Sequence
from typing import Literal, overload

import networkx as nx
import numpy as np
import pandas as pd
import scipy as sp
import treedata as td
from tqdm import tqdm

from pycea.utils import _check_tree_overlap, check_tree_has_key, get_leaves, get_trees

from ._aggregators import _get_aggregator
from ._metrics import _TreeMetric
from ._utils import _set_random_state

# ── internal helpers ──────────────────────────────────────────────────────────


def _dijkstra_min_scores(
    tree: nx.DiGraph,
    source_leaves: list,
    target_cats: list,
    cat_to_leaves_in_tree: dict,
    depth_key: str,
    metric: str,
) -> dict:
    """Per-leaf 'closest target' score for each category via multi-source Dijkstra.

    Uses ``|depth[u] - depth[v]|`` as edge weights to compute path distances.
    For ``metric='lca'``, converts path distance to LCA depth via the identity:

        lca(i, j) = (depth[i] + depth[j] - path_dist(i, j)) / 2

    Self-distances are excluded for within-category scoring.
    """
    G = nx.Graph(tree)

    def _weight(u, v, _):
        return abs(G.nodes[u][depth_key] - G.nodes[v][depth_key])

    scores: dict = {leaf: {} for leaf in source_leaves}

    for cat in target_cats:
        target_leaves = cat_to_leaves_in_tree.get(cat, [])
        if not target_leaves:
            continue

        if metric == "lca":
            dists, paths = nx.multi_source_dijkstra(G, target_leaves, weight=_weight)
            for leaf in source_leaves:
                if leaf in dists:
                    nearest = paths[leaf][0]  # path goes source → ... → leaf
                    scores[leaf][cat] = (G.nodes[leaf][depth_key] + G.nodes[nearest][depth_key] - dists[leaf]) / 2
        else:
            dists = nx.multi_source_dijkstra_path_length(G, target_leaves, weight=_weight)
            for leaf in source_leaves:
                if leaf in dists:
                    scores[leaf][cat] = dists[leaf]

    return scores


def _all_pairs_scores(
    tdata: td.TreeData,
    trees: dict,
    source_leaves_by_tree: dict,
    target_cats: list,
    cat_to_leaves_by_tree: dict,
    depth_key: str,
    metric: str,
    agg_fn: Callable,
) -> dict:
    """Per-leaf aggregated distance to each target category via all-pairs distance matrix."""
    from pycea.tl.tree_distance import tree_distance as _td

    tree_keys = list(trees.keys())
    result = _td(tdata, depth_key=depth_key, metric=metric, tree=tree_keys, copy=True)
    precomputed = result.toarray() if sp.sparse.issparse(result) else result

    scores: dict = {}
    obs_names = tdata.obs_names
    for tree_key, t_leaves in source_leaves_by_tree.items():
        for leaf in t_leaves:
            if leaf not in obs_names:
                continue
            src_idx = obs_names.get_loc(leaf)
            leaf_scores: dict = {}
            for cat in target_cats:
                tgt_leaves = cat_to_leaves_by_tree[tree_key].get(cat, [])
                tgt_indices = [obs_names.get_loc(l) for l in tgt_leaves if l in obs_names]
                if not tgt_indices:
                    continue
                row = precomputed[src_idx, tgt_indices]
                leaf_scores[cat] = float(agg_fn(row))
            scores[leaf] = leaf_scores

    return scores


def _compute_scores(
    tdata: td.TreeData,
    trees: dict,
    leaf_to_cat: dict,
    target_cats: list,
    aggregate: str | Callable,
    metric: str,
    depth_key: str,
) -> dict:
    """Route to the appropriate per-leaf scoring method and return leaf → {cat → score}."""
    # Build per-tree leaf / category maps
    source_leaves_by_tree: dict = {}
    cat_to_leaves_by_tree: dict = {}
    for tree_key, t in trees.items():
        t_leaves = [l for l in get_leaves(t) if l in leaf_to_cat]
        source_leaves_by_tree[tree_key] = t_leaves
        cat_to_leaves_by_tree[tree_key] = defaultdict(list)
        for l in t_leaves:
            cat_to_leaves_by_tree[tree_key][leaf_to_cat[l]].append(l)

    # Choose strategy: Dijkstra handles the natural "closest" direction for each metric.
    # NOTE: the lca+max Dijkstra shortcut is only correct for ultrametric trees (all
    # leaves at equal depth).  On non-ultrametric trees, the nearest-path target is not
    # always the deepest-LCA target, so Dijkstra can underestimate linkage.
    # TODO: add a fast non-ultrametric path for lca+max.
    is_named = isinstance(aggregate, str)
    use_dijkstra = is_named and ((aggregate == "min" and metric == "path") or (aggregate == "max" and metric == "lca"))

    if use_dijkstra:
        if metric == "lca":
            for tree_key, t in trees.items():
                t_leaves = source_leaves_by_tree[tree_key]
                depths = [t.nodes[l][depth_key] for l in t_leaves]
                if depths and not np.allclose(depths, depths[0]):
                    raise ValueError(
                        f"Tree '{tree_key}' is not ultrametric (leaves have unequal depths). "
                        "aggregate='max' with metric='lca' requires an ultrametric tree. "
                        "Use aggregate='mean' or metric='path' for non-ultrametric trees."
                    )
        all_scores: dict = {}
        for tree_key, t in trees.items():
            tree_scores = _dijkstra_min_scores(
                t,
                source_leaves_by_tree[tree_key],
                target_cats,
                cat_to_leaves_by_tree[tree_key],
                depth_key,
                metric,
            )
            all_scores.update(tree_scores)
        return all_scores

    # Fallback: all-pairs distance matrix (mean, max-path, or custom callable)
    agg_fn = _get_aggregator(aggregate) if is_named else aggregate  # type: ignore[arg-type]
    return _all_pairs_scores(
        tdata,
        trees,
        source_leaves_by_tree,
        target_cats,
        cat_to_leaves_by_tree,
        depth_key,
        metric,
        agg_fn,
    )


def _scores_to_linkage_matrix(
    all_scores: dict,
    all_cats: list,
    cat_to_leaves: dict,
) -> pd.DataFrame:
    """Aggregate per-leaf scores to a (source category × target category) DataFrame."""
    matrix: dict = {}
    for src_cat in all_cats:
        row: dict = {}
        src_leaves = [l for l in cat_to_leaves.get(src_cat, []) if l in all_scores]
        for tgt_cat in all_cats:
            values = [all_scores[l][tgt_cat] for l in src_leaves if tgt_cat in all_scores.get(l, {})]
            row[tgt_cat] = float(np.mean(values)) if values else np.nan
        matrix[src_cat] = row
    return pd.DataFrame(matrix, dtype=float).T  # index=src, columns=tgt


def _symmetrize_matrix(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    """Symmetrize a square DataFrame in-place."""
    arr = df.values.astype(float)
    arr_T = arr.T
    if mode == "mean":
        sym = (arr + arr_T) / 2
    elif mode == "max":
        sym = np.maximum(arr, arr_T)
    elif mode == "min":
        sym = np.minimum(arr, arr_T)
    else:
        raise ValueError(f"symmetrize must be 'mean', 'max', 'min', or None; got '{mode}'.")
    return pd.DataFrame(sym, index=df.index, columns=df.columns)


# ── fork-based parallel permutation workers ───────────────────────────────────
# These are module-level functions so they are picklable (required even with
# fork-based pools, since tasks are dispatched via a queue).  Heavy shared data
# (tdata, trees, …) is placed in the module globals below immediately before the
# pool is created; fork copies the parent's address space to child processes
# via copy-on-write, so no serialisation overhead is incurred for that data.

_PERM_PAIRWISE_DATA: dict = {}
_PERM_SINGLE_DATA: dict = {}
_PERM_NON_TARGET_PAIRWISE_DATA: dict = {}
_PERM_SINGLE_NON_TARGET_DATA: dict = {}


def _perm_pairwise_worker(seed: int) -> np.ndarray:
    """Run one pairwise permutation; shared data inherited from parent via fork."""
    d = _PERM_PAIRWISE_DATA
    rng = np.random.default_rng(seed)
    perm_cats = rng.permutation(d["all_cat_vals"])
    perm_leaf_to_cat = dict(zip(d["all_leaves"], perm_cats))
    perm_scores = _compute_scores(
        d["tdata"], d["trees"], perm_leaf_to_cat,
        d["target_cats"], d["aggregate"], d["metric"], d["depth_key"],
    )
    perm_cat_to_leaves: dict = defaultdict(list)
    for leaf, cat in perm_leaf_to_cat.items():
        perm_cat_to_leaves[cat].append(leaf)
    perm_df = _scores_to_linkage_matrix(perm_scores, d["all_cats"], perm_cat_to_leaves)
    return perm_df.reindex(index=d["index"], columns=d["columns"]).values.astype(float)


def _perm_single_target_worker(seed: int) -> dict:
    """Run one single-target permutation; shared data inherited from parent via fork."""
    d = _PERM_SINGLE_DATA
    rng = np.random.default_rng(seed)
    perm = rng.permutation(d["all_cat_vals"])
    perm_leaf_to_cat = dict(zip(d["all_leaves"], perm))
    perm_scores = _compute_scores(
        d["tdata"], d["trees"], perm_leaf_to_cat,
        [d["target"]], d["single_agg"], d["metric"], d["depth_key"],
    )
    perm_score_map = {leaf: s.get(d["target"], np.nan) for leaf, s in perm_scores.items()}
    perm_cat_to_leaves: dict = defaultdict(list)
    for leaf, cat in perm_leaf_to_cat.items():
        perm_cat_to_leaves[cat].append(leaf)
    result: dict = {}
    for cat in d["all_cats"]:
        vals = [
            perm_score_map[l]
            for l in perm_cat_to_leaves[cat]
            if l in perm_score_map and not np.isnan(perm_score_map[l])
        ]
        result[cat] = float(np.mean(vals)) if vals else np.nan
    return result


def _perm_pairwise_non_target_worker(seed: int) -> np.ndarray:
    """Non-target pairwise permutation using precomputed fixed scores.

    Target leaves never move, so scores to each target set are constant across
    permutations.  Each permutation only shuffles source-category labels and
    re-averages precomputed per-leaf scores — no tree computation required.

    Returns a float array of shape ``(n_src, n_tgt)`` aligned to
    ``d["index"] × d["columns"]``.
    """
    d = _PERM_NON_TARGET_PAIRWISE_DATA
    idx = d["index"]
    result = np.full((len(idx), len(d["fixed_scores"])), np.nan)
    for j, (score_map, nt_leaves, nt_cats, tgt_cat, tgt_leaves) in enumerate(zip(
        d["fixed_scores"], d["nt_leaves"], d["nt_cats"], d["columns"], d["target_leaves"],
    )):
        rng = np.random.default_rng([seed, j])
        perm_cats = rng.permutation(nt_cats)
        perm_cat_to_leaves: dict = defaultdict(list)
        for l in tgt_leaves:
            perm_cat_to_leaves[tgt_cat].append(l)
        for l, c in zip(nt_leaves, perm_cats):
            perm_cat_to_leaves[c].append(l)
        for i, src_cat in enumerate(idx):
            vals = [score_map[l] for l in perm_cat_to_leaves.get(src_cat, [])
                    if l in score_map and not np.isnan(score_map[l])]
            result[i, j] = float(np.mean(vals)) if vals else np.nan
    return result


def _perm_single_non_target_worker(seed: int) -> dict:
    """Non-target single-target permutation using precomputed fixed scores.

    Target leaves never move, so their distances to the target set are constant.
    Each permutation only shuffles non-target source labels and re-averages.
    """
    d = _PERM_SINGLE_NON_TARGET_DATA
    rng = np.random.default_rng(seed)
    perm_cats = rng.permutation(d["nt_cats"])
    perm_cat_to_leaves: dict = defaultdict(list)
    for l in d["target_leaves"]:
        perm_cat_to_leaves[d["target"]].append(l)
    for l, c in zip(d["nt_leaves"], perm_cats):
        perm_cat_to_leaves[c].append(l)
    score_map = d["fixed_scores"]
    result: dict = {}
    for cat in d["all_cats"]:
        vals = [score_map[l] for l in perm_cat_to_leaves.get(cat, [])
                if l in score_map and not np.isnan(score_map[l])]
        result[cat] = float(np.mean(vals)) if vals else np.nan
    return result


def _run_parallel(worker_fn: Callable, seeds: np.ndarray, n_threads: int | None) -> list:
    """Run *worker_fn(seed)* for each seed, optionally in parallel via fork processes."""
    max_workers = n_threads if n_threads is not None else 1

    _tqdm_kwargs = {"desc": "Permutations", "leave": not sys.stderr.isatty()}

    if max_workers > 1 and sys.platform == "linux":
        ctx = mp.get_context("fork")
        with ctx.Pool(max_workers) as pool:
            return list(tqdm(pool.imap_unordered(worker_fn, seeds), total=len(seeds), **_tqdm_kwargs))

    # Single-threaded fallback (also used on non-Linux platforms)
    return [worker_fn(seed) for seed in tqdm(seeds, **_tqdm_kwargs)]


def _compute_p_values(
    null_array: np.ndarray,
    obs_values: np.ndarray,
    null_mean: np.ndarray,
    metric: str,
    alternative: str | None,
) -> np.ndarray:
    """Compute permutation p-values given the null distribution and observed values."""
    if alternative == "two-sided":
        deviation = np.abs(null_array - null_mean[np.newaxis])
        obs_deviation = np.abs(obs_values - null_mean)
        return np.nanmean(deviation >= obs_deviation[np.newaxis], axis=0)
    # One-tailed in the "more related" direction
    if metric == "lca":
        return np.nanmean(null_array >= obs_values[np.newaxis], axis=0)
    else:
        return np.nanmean(null_array <= obs_values[np.newaxis], axis=0)


def _run_permutation_test(
    tdata: td.TreeData,
    trees: dict,
    leaf_to_cat: dict,
    all_cats: list,
    target_cats: list,
    observed_df: pd.DataFrame,
    aggregate: str | Callable,
    metric: str,
    depth_key: str,
    n_permutations: int,
    n_threads: int | None,
    alternative: str | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Permutation test: shuffle leaf labels, recompute linkage, return (z_score_df, p_value_df, null_mean_df)."""
    all_leaves = list(leaf_to_cat.keys())
    perm_seeds = np.random.randint(0, 2**31, size=n_permutations)

    _PERM_PAIRWISE_DATA.clear()
    _PERM_PAIRWISE_DATA.update({
        "tdata": tdata,
        "trees": trees,
        "all_leaves": all_leaves,
        "all_cat_vals": [leaf_to_cat[l] for l in all_leaves],
        "target_cats": target_cats,
        "all_cats": all_cats,
        "aggregate": aggregate,
        "metric": metric,
        "depth_key": depth_key,
        "index": observed_df.index,
        "columns": observed_df.columns,
    })

    null_matrices = _run_parallel(_perm_pairwise_worker, perm_seeds, n_threads)

    null_array = np.array(null_matrices)  # (n_permutations, k, k)
    null_mean = np.nanmean(null_array, axis=0)
    null_std = np.nanstd(null_array, axis=0)
    obs_values = observed_df.values.astype(float)

    # Positive z → more related than expected by chance
    sign = 1.0 if metric == "lca" else -1.0
    z_scores = sign * (obs_values - null_mean) / (null_std + 1e-10)

    p_values = _compute_p_values(null_array, obs_values, null_mean, metric, alternative)

    z_score_df = pd.DataFrame(z_scores, index=observed_df.index, columns=observed_df.columns)
    p_value_df = pd.DataFrame(p_values, index=observed_df.index, columns=observed_df.columns)
    null_mean_df = pd.DataFrame(null_mean, index=observed_df.index, columns=observed_df.columns)
    return z_score_df, p_value_df, null_mean_df


def _run_permutation_test_non_target(
    tdata: td.TreeData,
    trees: dict,
    leaf_to_cat: dict,
    all_cats: list,
    observed_df: pd.DataFrame,
    aggregate: str | Callable,
    metric: str,
    depth_key: str,
    n_permutations: int,
    n_threads: int | None,
    alternative: str | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Non-target permutation test: single batch of ``n_permutations`` workers.

    Scores to each fixed target set are precomputed once (same total Dijkstra work
    as the ``all`` mode).  Each permutation then only shuffles source-category labels
    and re-averages precomputed scores — no tree computation inside the permutation loop.
    """
    cat_to_leaves: dict = defaultdict(list)
    for l, c in leaf_to_cat.items():
        cat_to_leaves[c].append(l)
    all_leaves = list(leaf_to_cat.keys())
    cols = observed_df.columns.tolist()

    # Precompute per-leaf scores to each fixed target set (done once, outside the loop)
    fixed_scores = []
    nt_leaves_per_col = []
    nt_cats_per_col = []
    target_leaves_per_col = []
    for target_cat in cols:
        t_leaves = cat_to_leaves.get(target_cat, [])
        scores = _compute_scores(tdata, trees, leaf_to_cat, [target_cat], aggregate, metric, depth_key)
        fixed_scores.append({leaf: s.get(target_cat, np.nan) for leaf, s in scores.items()})
        t_set = set(t_leaves)
        nt = [l for l in all_leaves if l not in t_set]
        nt_leaves_per_col.append(nt)
        nt_cats_per_col.append([leaf_to_cat[l] for l in nt])
        target_leaves_per_col.append(list(t_leaves))

    _PERM_NON_TARGET_PAIRWISE_DATA.clear()
    _PERM_NON_TARGET_PAIRWISE_DATA.update({
        "fixed_scores": fixed_scores,
        "nt_leaves": nt_leaves_per_col,
        "nt_cats": nt_cats_per_col,
        "target_leaves": target_leaves_per_col,
        "index": observed_df.index.tolist(),
        "columns": cols,
    })

    perm_seeds = np.random.randint(0, 2**31, size=n_permutations)
    null_matrices = _run_parallel(_perm_pairwise_non_target_worker, perm_seeds, n_threads)

    null_array = np.array(null_matrices)  # (n_permutations, n_src, n_tgt)
    null_mean = np.nanmean(null_array, axis=0)
    null_std = np.nanstd(null_array, axis=0)
    obs_values = observed_df.values.astype(float)

    sign = 1.0 if metric == "lca" else -1.0
    z_scores = sign * (obs_values - null_mean) / (null_std + 1e-10)
    p_values = _compute_p_values(null_array, obs_values, null_mean, metric, alternative)

    return (
        pd.DataFrame(z_scores, index=observed_df.index, columns=observed_df.columns),
        pd.DataFrame(p_values, index=observed_df.index, columns=observed_df.columns),
        pd.DataFrame(null_mean, index=observed_df.index, columns=observed_df.columns),
    )


# ── public API ────────────────────────────────────────────────────────────────


@overload
def ancestral_linkage(
    tdata: td.TreeData,
    groupby: str,
    target: str | None = None,
    aggregate: Literal["min", "max", "mean"] | Callable | None = None,
    metric: _TreeMetric = "path",
    symmetrize: Literal["mean", "max", "min", None] = None,
    test: Literal["permutation", None] = None,
    alternative: Literal["two-sided", None] = None,
    permutation_mode: Literal["all", "non_target"] = "all",
    n_permutations: int = 100,
    n_threads: int | None = None,
    by_tree: bool = False,
    depth_key: str = "depth",
    random_state: int | None = None,
    key_added: str | None = None,
    tree: str | Sequence[str] | None = None,
    copy: Literal[True] = ...,
) -> pd.DataFrame: ...


@overload
def ancestral_linkage(
    tdata: td.TreeData,
    groupby: str,
    target: str | None = None,
    aggregate: Literal["min", "max", "mean"] | Callable | None = None,
    metric: _TreeMetric = "path",
    symmetrize: Literal["mean", "max", "min", None] = None,
    test: Literal["permutation", None] = None,
    alternative: Literal["two-sided", None] = None,
    permutation_mode: Literal["all", "non_target"] = "all",
    n_permutations: int = 100,
    n_threads: int | None = None,
    by_tree: bool = False,
    depth_key: str = "depth",
    random_state: int | None = None,
    key_added: str | None = None,
    tree: str | Sequence[str] | None = None,
    copy: Literal[False] = ...,
) -> None: ...


def ancestral_linkage(
    tdata: td.TreeData,
    groupby: str,
    target: str | None = None,
    metric: _TreeMetric = "path",
    symmetrize: Literal["mean", "max", "min", None] = None,
    test: Literal["permutation", None] = None,
    alternative: Literal["two-sided", None] = None,
    permutation_mode: Literal["all", "non_target"] = "all",
    n_permutations: int = 100,
    n_threads: int | None = None,
    aggregate: Literal["min", "max", "mean"] | Callable | None = None,
    by_tree: bool = False,
    depth_key: str = "depth",
    random_state: int | None = None,
    key_added: str | None = None,
    tree: str | Sequence[str] | None = None,
    copy: Literal[True, False] = False,
) -> None | pd.DataFrame:
    r"""Measures how closely related cells of different categories are on the lineage tree.

    For each cell, the tree distance to the nearest cell of each target category is
    computed.  These per-cell distances are then averaged across all cells of the same
    source category to produce a directional linkage score: a low path distance (or high
    LCA depth) between two categories means they tend to share recent common ancestors,
    i.e. they are closely related on the tree.

    **Pairwise mode** (``target=None``): computes a category × category matrix of mean
    linkage scores and stores it in ``tdata.uns['{key_added}_linkage']``.

    **Single-target mode** (``target=<category>``): computes the per-cell distance to
    the nearest cell of the given category and stores it in
    ``tdata.obs['{target}_linkage']``.

    Parameters
    ----------
    tdata
        The TreeData object.
    groupby
        Column in ``tdata.obs`` that defines cell categories.
    target
        If specified, compute the per-cell distance to the nearest cell of this
        category and store the result in ``tdata.obs['{target}_linkage']``.
        ``aggregate`` is ignored in this mode.
        If ``None`` (default), compute the full pairwise category × category matrix.
    metric
        How tree distance between two cells is measured:

        - ``'path'`` (default): branch-length path distance
          :math:`d_i + d_j - 2\,d_{\mathrm{LCA}(i,j)}`.  Smaller values mean closer
          relatives.
        - ``'lca'``: depth of the lowest common ancestor
          :math:`d_{\mathrm{LCA}(i,j)}`.  Larger values mean closer relatives.
    symmetrize
        If set, symmetrize the pairwise linkage matrix (pairwise mode only).
        Because linkage is directional (source → target), the raw matrix is generally
        asymmetric; symmetrization combines both directions:

        - ``'mean'``: average of :math:`M[i,j]` and :math:`M[j,i]`.
        - ``'max'`` / ``'min'``: element-wise maximum / minimum.
    test
        Optional significance test:

        - ``'permutation'``: randomly shuffle cell-category labels ``n_permutations``
          times and recompute linkage each time to build a null distribution.
          Z-scores and p-values are added to the stats table.  The stored linkage
          matrix is replaced by ``observed - permuted_mean`` when this test is run.
    alternative
        The alternative hypothesis for the permutation test (ignored when
        ``test=None``):

        - ``None`` (default): one-tailed test in the "more closely related than
          chance" direction — p-value is the fraction of permutations with LCA depth
          ≥ observed (``metric='lca'``) or path distance ≤ observed
          (``metric='path'``).
        - ``'two-sided'``: two-tailed test — p-value is the fraction of permutations
          whose deviation from the null mean is at least as large as the observed
          deviation.
    permutation_mode
        How category labels are shuffled when ``test='permutation'``:

        - ``'all'`` (default): shuffle all cell-category labels across all leaves.
          Tests whether the two categories are more associated on the tree than
          random, but does not control for the target category's cluster structure.
        - ``'non_target'``: fix the target category's leaves at their tree positions
          and shuffle only the non-target labels.  The null therefore reflects
          "random cells near this specific target cluster," which removes inflation
          caused by small, tightly clustered target categories.  In pairwise mode
          each target column gets its own independent null distribution.
    n_permutations
        Number of label permutations used when ``test='permutation'``.
    n_threads
        Number of worker processes for parallel permutation computation.
        ``None`` (default) runs serially.  On Linux, parallel execution uses
        ``fork``-based processes, which copy the parent's memory without
        serialisation overhead.  On other platforms this argument is ignored.
    aggregate
        How per-cell distances to the target category are aggregated into a single
        per-cell score (pairwise mode only).  Defaults to ``'min'`` for
        ``metric='path'`` and ``'max'`` for ``metric='lca'``, both of which
        select the nearest relative:

        - ``'min'``: distance to the closest target cell (natural for ``'path'``).
        - ``'max'``: depth of the shallowest LCA across target cells (natural for ``'lca'``).
        - ``'mean'``: mean distance across all target cells.
        - A callable ``f(array) -> float`` for custom aggregation.
    depth_key
        Node attribute in ``tdata.obst[tree]`` that stores each node's depth.
    random_state
        Random seed for reproducibility of permutation tests.
    key_added
        Base key for output storage.  Defaults to ``groupby``.
    tree
        The ``obst`` key or keys of the trees to use.  If ``None``, all trees are used.
    copy
        If ``True``, return the result as a :class:`DataFrame <pandas.DataFrame>`.

    Returns
    -------
    Returns ``None`` if ``copy=False``, otherwise returns a :class:`DataFrame <pandas.DataFrame>`.

    Sets the following fields:

    * ``tdata.obs['{target}_linkage']`` : :class:`Series <pandas.Series>` (dtype ``float``) – single-target mode only.
        Per-cell distance to the nearest cell of the target category.
    * ``tdata.obs['{target}_norm_linkage']`` : :class:`Series <pandas.Series>` (dtype ``float``) – single-target mode with ``test='permutation'`` only.
        Per-cell z-score: ``sign * (score - null_mean) / null_std``, where the null
        distribution is taken from the cell's source-category permutations.
    * ``tdata.uns['{key_added}_linkage']`` : :class:`DataFrame <pandas.DataFrame>` – pairwise mode only.
        Category × category linkage matrix (source rows, target columns).
        Contains ``observed - permuted_mean`` instead of raw distances when ``test='permutation'``.
    * ``tdata.uns['{key_added}_linkage_params']`` : ``dict`` – pairwise mode only.
        Parameters used to compute the linkage matrix.
    * ``tdata.uns['{key_added}_linkage_stats']`` : :class:`DataFrame <pandas.DataFrame>` – pairwise mode only.
        Long-form table with one row per (source, target) pair containing ``value``,
        ``source_n``, ``target_n``, and (if ``test='permutation'``) ``permuted_value``,
        ``z_score``, ``p_value``.

    Examples
    --------
    Compute pairwise linkage between all cell types using path distance:

    >>> tdata = py.datasets.koblan25()
    >>> py.tl.ancestral_linkage(tdata, groupby="celltype")

    Compute per-cell distance to the closest cell of type "B" with permutation test:

    >>> py.tl.ancestral_linkage(tdata, groupby="celltype", target="B", test="permutation")
    """
    # ── setup ─────────────────────────────────────────────────────────────────
    _set_random_state(random_state)
    key_added = key_added or groupby
    tree_keys = tree
    _check_tree_overlap(tdata, tree_keys)
    trees = get_trees(tdata, tree_keys)

    if groupby not in tdata.obs.columns:
        raise ValueError(f"'{groupby}' not found in tdata.obs.columns.")

    # Resolve default aggregate: 'min' for path, 'max' for lca
    if aggregate is None:
        aggregate = "max" if metric == "lca" else "min"

    if isinstance(aggregate, str) and aggregate not in ("min", "max", "mean"):
        raise ValueError(f"aggregate must be 'min', 'max', 'mean', or a callable; got '{aggregate}'.")

    # Warn about misleading aggregate only in pairwise mode (aggregate is ignored for single target)
    if target is None and metric == "lca" and isinstance(aggregate, str) and aggregate == "min":
        warnings.warn(
            "aggregate='min' with metric='lca' selects the *shallowest* (most distant) "
            "ancestor. To find the most recent common ancestor use aggregate='max'.",
            UserWarning,
            stacklevel=2,
        )

    for t in trees.values():
        check_tree_has_key(t, depth_key)

    # ── build leaf → category mapping ─────────────────────────────────────────
    obs_set = set(tdata.obs_names)
    leaf_to_cat: dict = {}
    for _, t in trees.items():
        for leaf in get_leaves(t):
            if leaf in obs_set:
                cat = tdata.obs.loc[leaf, groupby]
                if pd.notna(cat):
                    leaf_to_cat[leaf] = str(cat)

    all_cats = sorted(set(leaf_to_cat.values()))

    cat_to_leaves: dict = defaultdict(list)
    for l, c in leaf_to_cat.items():
        cat_to_leaves[c].append(l)

    # ── single-target mode ────────────────────────────────────────────────────
    if target is not None:
        if target not in all_cats:
            raise ValueError(f"target '{target}' not found in tdata.obs['{groupby}'].")

        # Always use "closest": min path or max lca
        single_agg: str = "max" if metric == "lca" else "min"
        sign = 1.0 if metric == "lca" else -1.0

        def _run_single_perm(single_tree, tree_lc, tree_sm, tree_cl, extra_row_fields=None):
            """Run the permutation test for one tree (or globally) and return (rows, norm_map)."""
            tree_obs_cat: dict = {}
            for cat in all_cats:
                vals = [tree_sm[l] for l in tree_cl[cat] if l in tree_sm and not np.isnan(tree_sm[l])]
                tree_obs_cat[cat] = float(np.mean(vals)) if vals else np.nan

            tree_leaf_list = list(tree_lc.keys())
            perm_seeds = np.random.randint(0, 2**31, size=n_permutations)

            if permutation_mode == "non_target":
                t_lv = tree_cl[target]
                t_set = set(t_lv)
                nt_lv = [l for l in tree_leaf_list if l not in t_set]
                _PERM_SINGLE_NON_TARGET_DATA.clear()
                _PERM_SINGLE_NON_TARGET_DATA.update({
                    "fixed_scores": tree_sm,
                    "target_leaves": list(t_lv),
                    "nt_leaves": nt_lv,
                    "nt_cats": [tree_lc[l] for l in nt_lv],
                    "target": target,
                    "all_cats": all_cats,
                })
                null_results = _run_parallel(_perm_single_non_target_worker, perm_seeds, n_threads)
            else:
                _PERM_SINGLE_DATA.clear()
                _PERM_SINGLE_DATA.update({
                    "tdata": tdata,
                    "trees": single_tree,
                    "all_leaves": tree_leaf_list,
                    "all_cat_vals": [tree_lc[l] for l in tree_leaf_list],
                    "target": target,
                    "single_agg": single_agg,
                    "metric": metric,
                    "depth_key": depth_key,
                    "all_cats": all_cats,
                })
                null_results = _run_parallel(_perm_single_target_worker, perm_seeds, n_threads)

            null_cat: dict = defaultdict(list)
            for perm_result in null_results:
                for cat in all_cats:
                    null_cat[cat].append(perm_result[cat])

            rows: list = []
            cat_null_mean: dict = {}
            cat_null_std: dict = {}
            for cat in all_cats:
                obs_val = tree_obs_cat[cat]
                null_vals = np.array([v for v in null_cat[cat] if not np.isnan(v)], dtype=float)
                if len(null_vals) > 0:
                    perm_val = float(np.mean(null_vals))
                    null_std = float(np.std(null_vals))
                    z = sign * (obs_val - perm_val) / (null_std + 1e-10)
                    if alternative == "two-sided":
                        p = float(np.mean(np.abs(null_vals - perm_val) >= abs(obs_val - perm_val)))
                    elif metric == "lca":
                        p = float(np.mean(null_vals >= obs_val))
                    else:
                        p = float(np.mean(null_vals <= obs_val))
                else:
                    perm_val, null_std, z, p = np.nan, 0.0, np.nan, np.nan
                cat_null_mean[cat] = perm_val
                cat_null_std[cat] = null_std
                row: dict = {"source": cat, "target": target, "value": obs_val,
                             "permuted_value": perm_val, "z_score": z, "p_value": p}
                if extra_row_fields:
                    row.update(extra_row_fields)
                rows.append(row)

            norm_map: dict = {}
            for leaf, score in tree_sm.items():
                cat = tree_lc.get(leaf)
                if cat is not None and not np.isnan(score):
                    norm_map[leaf] = sign * (score - cat_null_mean.get(cat, np.nan)) / (cat_null_std.get(cat, 0.0) + 1e-10)
                else:
                    norm_map[leaf] = np.nan
            return rows, norm_map

        if by_tree and test == "permutation":
            # Per-tree: compute scores, run permutation, normalize per cell independently
            merged_score_map: dict = {}
            merged_norm_map: dict = {}
            all_rows: list = []

            for tree_key, t in trees.items():
                t_nodes = set(t.nodes())
                single_tree = {tree_key: t}
                tree_lc: dict = {l: c for l, c in leaf_to_cat.items() if l in t_nodes}
                tree_cl: dict = defaultdict(list)
                for l, c in tree_lc.items():
                    tree_cl[c].append(l)

                tree_all_scores = _compute_scores(
                    tdata, single_tree, tree_lc, [target], single_agg, metric, depth_key
                )
                tree_sm: dict = {l: s.get(target, np.nan) for l, s in tree_all_scores.items()}
                merged_score_map.update(tree_sm)

                rows, norm_map = _run_single_perm(single_tree, tree_lc, tree_sm, tree_cl,
                                                  extra_row_fields={"tree": tree_key})
                all_rows.extend(rows)
                merged_norm_map.update(norm_map)

            tdata.obs[f"{target}_linkage"] = tdata.obs.index.map(pd.Series(merged_score_map, dtype=float))
            tdata.obs[f"{target}_norm_linkage"] = tdata.obs.index.map(pd.Series(merged_norm_map, dtype=float))
            test_df = pd.DataFrame(all_rows)
            tdata.uns[f"{key_added}_test"] = test_df
            if copy:
                return test_df

        else:
            # Global (non-by_tree) path
            all_scores = _compute_scores(tdata, trees, leaf_to_cat, [target], single_agg, metric, depth_key)
            score_map = {leaf: scores.get(target, np.nan) for leaf, scores in all_scores.items()}
            tdata.obs[f"{target}_linkage"] = tdata.obs.index.map(pd.Series(score_map, dtype=float))

            if test == "permutation":
                rows, norm_map = _run_single_perm(trees, leaf_to_cat, score_map, cat_to_leaves)
                tdata.obs[f"{target}_norm_linkage"] = tdata.obs.index.map(pd.Series(norm_map, dtype=float))
                test_df = pd.DataFrame(rows)
                tdata.uns[f"{key_added}_test"] = test_df
                if copy:
                    return test_df

            if copy:
                result_series = pd.Series(
                    {cat: float(np.nanmean([score_map.get(l, np.nan) for l in cat_to_leaves[cat]])) for cat in all_cats},
                    name=f"{target}_linkage",
                )
                return result_series.to_frame()

    # ── pairwise mode ─────────────────────────────────────────────────────────
    else:
        # Global linkage across all trees (always computed)
        all_scores = _compute_scores(tdata, trees, leaf_to_cat, all_cats, aggregate, metric, depth_key)
        linkage_df = _scores_to_linkage_matrix(all_scores, all_cats, cat_to_leaves)

        # Global permutation test
        global_z_df: pd.DataFrame | None = None
        global_p_df: pd.DataFrame | None = None
        global_null_mean_df: pd.DataFrame | None = None
        if test == "permutation":
            if permutation_mode == "non_target":
                global_z_df, global_p_df, global_null_mean_df = _run_permutation_test_non_target(
                    tdata, trees, leaf_to_cat, all_cats,
                    linkage_df, aggregate, metric, depth_key, n_permutations, n_threads, alternative,
                )
            else:
                global_z_df, global_p_df, global_null_mean_df = _run_permutation_test(
                    tdata, trees, leaf_to_cat, all_cats, all_cats,
                    linkage_df, aggregate, metric, depth_key, n_permutations, n_threads, alternative,
                )

        # Build stats rows (long format, never symmetrized)
        stats_rows: list = []
        if by_tree:
            for tree_key, t in trees.items():
                t_nodes = set(t.nodes())
                single_tree = {tree_key: t}
                tree_leaf_to_cat = {l: c for l, c in leaf_to_cat.items() if l in t_nodes}
                tree_cat_to_leaves: dict = defaultdict(list)
                for l, c in tree_leaf_to_cat.items():
                    tree_cat_to_leaves[c].append(l)

                tree_scores = _compute_scores(
                    tdata, single_tree, tree_leaf_to_cat, all_cats, aggregate, metric, depth_key
                )
                tree_linkage_df = _scores_to_linkage_matrix(tree_scores, all_cats, tree_cat_to_leaves)

                tree_z_df: pd.DataFrame | None = None
                tree_p_df: pd.DataFrame | None = None
                tree_null_mean_df: pd.DataFrame | None = None
                if test == "permutation":
                    if permutation_mode == "non_target":
                        tree_z_df, tree_p_df, tree_null_mean_df = _run_permutation_test_non_target(
                            tdata, single_tree, tree_leaf_to_cat, all_cats,
                            tree_linkage_df, aggregate, metric, depth_key, n_permutations, n_threads, alternative,
                        )
                    else:
                        tree_z_df, tree_p_df, tree_null_mean_df = _run_permutation_test(
                            tdata, single_tree, tree_leaf_to_cat, all_cats, all_cats,
                            tree_linkage_df, aggregate, metric, depth_key, n_permutations, n_threads, alternative,
                        )

                for src_cat in all_cats:
                    for tgt_cat in all_cats:
                        row: dict = {
                            "source": src_cat,
                            "target": tgt_cat,
                            "tree": tree_key,
                            "value": tree_linkage_df.loc[src_cat, tgt_cat],
                            "source_n": len(tree_cat_to_leaves.get(src_cat, [])),
                            "target_n": len(tree_cat_to_leaves.get(tgt_cat, [])),
                        }
                        if tree_z_df is not None and tree_p_df is not None and tree_null_mean_df is not None:
                            row["permuted_value"] = tree_null_mean_df.loc[src_cat, tgt_cat]
                            row["z_score"] = tree_z_df.loc[src_cat, tgt_cat]
                            row["p_value"] = tree_p_df.loc[src_cat, tgt_cat]
                        stats_rows.append(row)
        else:
            for src_cat in all_cats:
                for tgt_cat in all_cats:
                    row = {
                        "source": src_cat,
                        "target": tgt_cat,
                        "value": linkage_df.loc[src_cat, tgt_cat],
                        "source_n": len(cat_to_leaves.get(src_cat, [])),
                        "target_n": len(cat_to_leaves.get(tgt_cat, [])),
                    }
                    if global_z_df is not None and global_p_df is not None and global_null_mean_df is not None:
                        row["permuted_value"] = global_null_mean_df.loc[src_cat, tgt_cat]
                        row["z_score"] = global_z_df.loc[src_cat, tgt_cat]
                        row["p_value"] = global_p_df.loc[src_cat, tgt_cat]
                    stats_rows.append(row)

        # uns[linkage] = observed - permuted_mean (symmetrized) if test ran, else raw linkage (symmetrized)
        output_df: pd.DataFrame = (linkage_df - global_null_mean_df) if test == "permutation" else linkage_df
        if symmetrize is not None:
            output_df = _symmetrize_matrix(output_df, symmetrize)

        params = {
            "groupby": groupby,
            "aggregate": aggregate,
            "metric": metric,
            "symmetrize": symmetrize,
            "test": test,
            "by_tree": by_tree,
            "depth_key": depth_key,
        }
        stats_df = pd.DataFrame(stats_rows)
        tdata.uns[f"{key_added}_linkage"] = output_df
        tdata.uns[f"{key_added}_linkage_params"] = params
        tdata.uns[f"{key_added}_linkage_stats"] = stats_df

        if copy:
            return stats_df if test is not None else output_df
