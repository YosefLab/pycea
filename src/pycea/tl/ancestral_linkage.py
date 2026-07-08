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

# Categories with fewer than this many cells give noisy linkage estimates (warn only).
_MIN_CELLS_WARN = 10

# ── internal helpers ──────────────────────────────────────────────────────────


def _dijkstra_min_scores(
    tree: nx.DiGraph,
    source_leaves: list,
    target_cats: list,
    cat_to_leaves_in_tree: dict,
    depth_key: str,
) -> dict:
    r"""Per-leaf minimum path distance to each target category via multi-source Dijkstra.

    Uses ``|depth[u] - depth[v]|`` as edge weights, which sums along any root-to-leaf
    path to the correct tree path distance :math:`d_i + d_j - 2\\,d_{LCA(i,j)}` on any
    tree (ultrametric or not).  Self-distances are zero, so a source leaf in the target
    category scores 0 (its own closest target is itself).
    """
    G = nx.Graph(tree)

    def _weight(u, v, _):
        return abs(G.nodes[u][depth_key] - G.nodes[v][depth_key])

    scores: dict = {leaf: {} for leaf in source_leaves}

    for cat in target_cats:
        target_leaves = cat_to_leaves_in_tree.get(cat, [])
        if not target_leaves:
            continue

        dists = nx.multi_source_dijkstra_path_length(G, target_leaves, weight=_weight)
        for leaf in source_leaves:
            if leaf in dists:
                scores[leaf][cat] = dists[leaf]

    return scores


def _max_lca_depth_scores(
    tree: nx.DiGraph,
    source_leaves: list,
    target_cats: list,
    cat_to_leaves_in_tree: dict,
    depth_key: str,
) -> dict:
    r"""Per-leaf maximum LCA depth to each target category (exact for any tree).

    This is the shared "closest relative" primitive for both closest-target aggregates:
    ``lca`` + ``max`` uses it directly, and on an ultrametric tree ``path`` + ``min`` is
    the affine transform ``2D - 2 * score`` of it (see :func:`_compute_scores`).

    For a source leaf ``i``, the deepest (most recent) common ancestor it shares with
    *any* leaf ``j`` of category ``c`` is the deepest ancestor of ``i`` whose subtree
    still contains a ``c`` leaf.  Its depth is therefore :math:`\\max_j d_{LCA(i, j)}`.
    Unlike the path-distance Dijkstra shortcut, this holds regardless of whether the
    tree is ultrametric, so it is exact on non-ultrametric trees too.

    The subtree category membership is built with a single bottom-up pass (reversed
    topological order visits every child before its parent); each source leaf then
    walks up its ancestor chain, recording the first — hence deepest — ancestor that
    covers each still-unresolved category.  A leaf is its own subtree, so a source leaf
    belonging to ``c`` scores its own depth (the maximal possible LCA depth).
    """
    # Map each target leaf to its category (only target categories are relevant).
    leaf_cat: dict = {}
    for cat in target_cats:
        for leaf in cat_to_leaves_in_tree.get(cat, []):
            leaf_cat[leaf] = cat

    # Bottom-up: set of target categories present in each node's subtree.
    subtree_cats: dict = {}
    for node in reversed(list(nx.topological_sort(tree))):
        cats: set = set()
        for child in tree.successors(node):
            cats |= subtree_cats[child]
        own = leaf_cat.get(node)
        if own is not None:
            cats.add(own)
        subtree_cats[node] = cats

    # Walk up from each source leaf; the first ancestor whose subtree contains a
    # category (deepest, since we ascend) gives the maximum LCA depth to that category.
    scores: dict = {leaf: {} for leaf in source_leaves}
    for leaf in source_leaves:
        remaining = set(target_cats)
        node = leaf
        while node is not None and remaining:
            present = subtree_cats[node] & remaining
            if present:
                depth = tree.nodes[node][depth_key]
                for cat in present:
                    scores[leaf][cat] = depth
                remaining -= present
            preds = list(tree.predecessors(node))
            node = preds[0] if preds else None

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

    # Choose strategy for the "closest target" aggregates, computed exactly and without an
    # all-pairs distance matrix:
    #   - max+lca  : deepest ancestor whose subtree covers each category (subtree walk-up).
    #     Exact on any tree and faster than Dijkstra (one bottom-up pass regardless of the
    #     category count), so it is used for all trees.
    #   - min+path : on an ultrametric tree (leaf depth D) path = 2D - 2*lca, so min path =
    #     2D - 2*(max lca): reuse the fast walk-up and apply the affine transform.  On a
    #     non-ultrametric tree leaves differ in depth and this identity fails, so fall back
    #     to multi-source Dijkstra on |Δdepth| edge weights.
    is_named = isinstance(aggregate, str)
    use_path_min = is_named and aggregate == "min" and metric == "path"
    use_lca_max = is_named and aggregate == "max" and metric == "lca"

    if use_path_min or use_lca_max:
        all_scores: dict = {}
        for tree_key, t in trees.items():
            t_leaves = source_leaves_by_tree[tree_key]
            ctl = cat_to_leaves_by_tree[tree_key]
            depths = [t.nodes[l][depth_key] for l in t_leaves]
            ultrametric = (not depths) or np.allclose(depths, depths[0])
            if use_lca_max:
                tree_scores = _max_lca_depth_scores(t, t_leaves, target_cats, ctl, depth_key)
            elif ultrametric:
                # min path = 2D - 2*(max lca); reuse the walk-up and transform.
                D = depths[0]
                lca_scores = _max_lca_depth_scores(t, t_leaves, target_cats, ctl, depth_key)
                tree_scores = {leaf: {c: 2 * D - 2 * v for c, v in s.items()} for leaf, s in lca_scores.items()}
            else:
                tree_scores = _dijkstra_min_scores(t, t_leaves, target_cats, ctl, depth_key)
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
    perm_leaf_to_cat = dict(zip(d["all_leaves"], perm_cats, strict=True))
    perm_scores = _compute_scores(
        d["tdata"],
        d["trees"],
        perm_leaf_to_cat,
        d["target_cats"],
        d["aggregate"],
        d["metric"],
        d["depth_key"],
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
    perm_leaf_to_cat = dict(zip(d["all_leaves"], perm, strict=True))
    perm_scores = _compute_scores(
        d["tdata"],
        d["trees"],
        perm_leaf_to_cat,
        [d["target"]],
        d["single_agg"],
        d["metric"],
        d["depth_key"],
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
    permutations.  Each permutation only shuffles the non-target source labels and
    re-averages the precomputed per-leaf scores — no tree computation required.

    The per-category means are computed with :func:`numpy.bincount` (group-sum and
    group-count over integer category codes) instead of Python loops.  Shuffling the
    integer code array with the same per-column RNG (``default_rng([seed, j])``) yields
    the identical label assignment as shuffling the category labels, so the result is
    unchanged.  Returns a float array of shape ``(n_src, n_tgt)`` aligned to
    ``d["index"] × d["columns"]``.
    """
    d = _PERM_NON_TARGET_PAIRWISE_DATA
    n_src = d["n_src"]
    nt_codes = d["nt_codes"]
    nt_scores = d["nt_scores"]
    nt_finite = d["nt_finite"]
    tgt_code = d["tgt_code"]
    tgt_diag = d["tgt_diag"]
    result = np.full((n_src, len(nt_codes)), np.nan)
    for j in range(len(nt_codes)):
        rng = np.random.default_rng([seed, j])
        perm_codes = rng.permutation(nt_codes[j])
        finite = nt_finite[j]
        codes = perm_codes[finite]
        sums = np.bincount(codes, weights=nt_scores[j][finite], minlength=n_src)
        counts = np.bincount(codes, minlength=n_src)
        with np.errstate(invalid="ignore", divide="ignore"):
            means = np.where(counts > 0, sums / counts, np.nan)
        means[tgt_code[j]] = tgt_diag[j]  # target leaves are fixed → constant diagonal
        result[:, j] = means
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
    for l, c in zip(d["nt_leaves"], perm_cats, strict=True):
        perm_cat_to_leaves[c].append(l)
    score_map = d["fixed_scores"]
    result: dict = {}
    for cat in d["all_cats"]:
        vals = [score_map[l] for l in perm_cat_to_leaves.get(cat, []) if l in score_map and not np.isnan(score_map[l])]
        result[cat] = float(np.mean(vals)) if vals else np.nan
    return result


def _run_parallel(worker_fn: Callable, seeds: np.ndarray, n_threads: int | None) -> list:
    """Run *worker_fn(seed)* for each seed, optionally in parallel via fork processes."""
    max_workers = n_threads if n_threads is not None else 1

    # Hide the progress bar for a single permutation (the ``test=None`` normalization case).
    _tqdm_kwargs = {"desc": "Permutations", "leave": not sys.stderr.isatty(), "disable": len(seeds) <= 1}

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
    alternative: str,
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
    alternative: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Permutation test: shuffle leaf labels, recompute linkage, return (z_score_df, p_value_df, null_mean_df)."""
    all_leaves = list(leaf_to_cat.keys())
    perm_seeds = np.random.randint(0, 2**31, size=n_permutations)

    _PERM_PAIRWISE_DATA.clear()
    _PERM_PAIRWISE_DATA.update(
        {
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
        }
    )

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
    alternative: str,
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
    idx = observed_df.index.tolist()
    code_of = {cat: i for i, cat in enumerate(idx)}  # source category → row index

    # Precompute, once per fixed target column: the non-target leaves' category codes and
    # scores (arrays), a finite-score mask, and the constant target-category diagonal.
    # Each permutation then only shuffles the codes and re-averages via bincount.
    nt_codes: list = []
    nt_scores: list = []
    nt_finite: list = []
    tgt_code: list = []
    tgt_diag: list = []
    for target_cat in cols:
        t_leaves = cat_to_leaves.get(target_cat, [])
        scores = _compute_scores(tdata, trees, leaf_to_cat, [target_cat], aggregate, metric, depth_key)
        score_map = {leaf: s.get(target_cat, np.nan) for leaf, s in scores.items()}
        t_set = set(t_leaves)
        nt = [l for l in all_leaves if l not in t_set]
        nt_codes.append(np.array([code_of[leaf_to_cat[l]] for l in nt], dtype=np.intp))
        s = np.array([score_map.get(l, np.nan) for l in nt], dtype=float)
        nt_scores.append(s)
        nt_finite.append(np.isfinite(s))
        tgt_code.append(code_of[target_cat])
        t_vals = [score_map[l] for l in t_leaves if l in score_map and not np.isnan(score_map[l])]
        tgt_diag.append(float(np.mean(t_vals)) if t_vals else np.nan)

    _PERM_NON_TARGET_PAIRWISE_DATA.clear()
    _PERM_NON_TARGET_PAIRWISE_DATA.update(
        {
            "nt_codes": nt_codes,
            "nt_scores": nt_scores,
            "nt_finite": nt_finite,
            "tgt_code": tgt_code,
            "tgt_diag": tgt_diag,
            "n_src": len(idx),
        }
    )

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
    metric: _TreeMetric = "lca",
    symmetrize: Literal["mean", "max", "min", False] = "mean",
    normalize: bool = True,
    min_size: int = 1,
    test: Literal["permutation", None] = None,
    alternative: Literal["two-sided", "one-sided"] = "one-sided",
    permutation_mode: Literal["all", "non_target"] = "non_target",
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
    metric: _TreeMetric = "lca",
    symmetrize: Literal["mean", "max", "min", False] = "mean",
    normalize: bool = True,
    min_size: int = 1,
    test: Literal["permutation", None] = None,
    alternative: Literal["two-sided", "one-sided"] = "one-sided",
    permutation_mode: Literal["all", "non_target"] = "non_target",
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
    aggregate: Literal["min", "max", "mean"] | Callable | None = None,
    metric: _TreeMetric = "lca",
    symmetrize: Literal["mean", "max", "min", False] = "mean",
    normalize: bool = True,
    min_size: int = 1,
    test: Literal["permutation", None] = None,
    alternative: Literal["two-sided", "one-sided"] = "one-sided",
    permutation_mode: Literal["all", "non_target"] = "non_target",
    n_permutations: int = 100,
    n_threads: int | None = None,
    by_tree: bool = False,
    depth_key: str = "depth",
    random_state: int | None = None,
    key_added: str | None = None,
    tree: str | Sequence[str] | None = None,
    copy: Literal[True, False] = False,
) -> None | pd.DataFrame:
    r"""Quantify relatedness of cells in different categories.

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
    aggregate
        How per-cell distances to the target category are aggregated into a single
        per-cell score (pairwise mode only).  Defaults to ``'min'`` for
        ``metric='path'`` and ``'max'`` for ``metric='lca'``, both of which
        select the nearest relative:

        - ``'min'``: distance to the closest target cell (natural for ``'path'``).
        - ``'max'``: depth of the deepest (most recent) LCA across target cells (natural for ``'lca'``).
        - ``'mean'``: mean distance across all target cells.
        - A callable ``f(array) -> float`` for custom aggregation.
    metric
        How tree distance between two cells is measured:

        - ``'lca'`` (default): depth of the lowest common ancestor
          :math:`d_{\mathrm{LCA}(i,j)}`.  Larger values mean closer relatives.
        - ``'path'``: branch-length path distance
          :math:`d_i + d_j - 2\,d_{\mathrm{LCA}(i,j)}`.  Smaller values mean closer
          relatives.
    symmetrize
        How to symmetrize the pairwise linkage matrix (pairwise mode only). Because linkage
        is directional (source → target), the raw matrix is generally asymmetric;
        symmetrization combines both directions:

        - ``'mean'`` (default): average of :math:`M[i,j]` and :math:`M[j,i]`.
        - ``'max'`` / ``'min'``: element-wise maximum / minimum.
        - ``False``: leave the matrix asymmetric.
    normalize
        If ``True`` (default), subtract the permuted mean from the observed values: pairwise
        linkage matrix becomes ``observed - permuted_mean``; single-target
        ``tdata.obs['{target}_linkage']`` becomes ``cell_score - category_permuted_mean``.
        This works regardless of ``test``: a single permutation is run to estimate the
        permuted mean when ``test=None`` (see ``n_permutations``).
    min_size
        Minimum number of cells a category must have to be included (pairwise mode only).
        Categories with fewer than ``min_size`` cells are dropped before the linkage matrix
        is computed, so they appear as neither source rows nor target columns and their
        cells do not contribute to any score.  Defaults to ``1`` (no filtering).  A warning
        lists any retained categories with fewer than 10 cells, whose linkage will be noisy.
    test
        Optional significance test:

        - ``'permutation'``: randomly shuffle cell-category labels ``n_permutations``
          times and recompute linkage each time to build a null distribution.
          Z-scores and p-values are added to the stats table.
    non
        The alternative hypothesis for the permutation test (ignored when
        ``test=None``):

        - ``'one-sided'`` (default): one-tailed test in the "more closely related than
          chance" direction — p-value is the fraction of permutations with LCA depth
          ≥ observed (``metric='lca'``) or path distance ≤ observed
          (``metric='path'``).
        - ``'two-sided'``: two-tailed test — p-value is the fraction of permutations
          whose deviation from the null mean is at least as large as the observed
          deviation.
    permutation_mode
        How category labels are shuffled to build the permutation null (used both for the
        significance test and for the ``normalize`` permuted mean):

        - ``'non_target'`` (default): fix the target category's leaves at their tree
          positions and shuffle only the non-target labels.  The null therefore reflects
          "random cells near this specific target cluster," which removes inflation
          caused by small, tightly clustered target categories.  In pairwise mode
          each target column gets its own independent null distribution.
        - ``'all'``: shuffle all cell-category labels across all leaves.
          Tests whether the two categories are more associated on the tree than
          random, but does not control for the target category's cluster structure.

    n_permutations
        Number of label permutations used when ``test='permutation'``.  When ``test=None``
        a single permutation is used to estimate the permuted mean for ``normalize``.
    n_threads
        Number of worker processes for parallel permutation computation.
        ``None`` (default) runs serially.  On Linux, parallel execution uses
        ``fork``-based processes, which copy the parent's memory without
        serialisation overhead.  On other platforms this argument is ignored.
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
        Per-cell distance to the nearest cell of the target category.  When
        ``normalize=True``, replaced by ``cell_score - category_permuted_mean``.
    * ``tdata.uns['{key_added}_linkage']`` : :class:`DataFrame <pandas.DataFrame>` – pairwise mode only.
        Category × category linkage matrix (source rows, target columns).
        When ``normalize=True``, contains ``observed - permuted_mean`` instead of raw distances.
    * ``tdata.uns['{key_added}_linkage_params']`` : ``dict`` – pairwise mode only.
        Parameters used to compute the linkage matrix.
    * ``tdata.uns['{key_added}_linkage_stats']`` : :class:`DataFrame <pandas.DataFrame>` – pairwise mode only.
        Long-form table with one row per (source, target) pair containing ``value``,
        ``source_n``, ``target_n``, and ``permuted_value`` (always, from at least one
        permutation), plus ``z_score`` and ``p_value`` when ``test='permutation'``.

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

    if not isinstance(min_size, int) or min_size < 1:
        raise ValueError(f"min_size must be a positive integer; got {min_size!r}.")

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

    # ── exclude small categories (pairwise mode only) ──────────────────────────
    if target is None and min_size > 1:
        kept_cats = {c for c in all_cats if len(cat_to_leaves[c]) >= min_size}
        if not kept_cats:
            raise ValueError(f"No categories in '{groupby}' have at least min_size={min_size} cells.")
        leaf_to_cat = {l: c for l, c in leaf_to_cat.items() if c in kept_cats}
        all_cats = sorted(kept_cats)
        cat_to_leaves = defaultdict(list)
        for l, c in leaf_to_cat.items():
            cat_to_leaves[c].append(l)

    # Warn about categories too small for a reliable linkage estimate.
    small_cats = {c: len(cat_to_leaves[c]) for c in all_cats if len(cat_to_leaves[c]) < _MIN_CELLS_WARN}
    if small_cats:
        warnings.warn(
            f"Categories with fewer than {_MIN_CELLS_WARN} cells give noisy linkage estimates: "
            f"{small_cats}. Consider increasing `min_size` to exclude them.",
            UserWarning,
            stacklevel=2,
        )

    # ── single-target mode ────────────────────────────────────────────────────
    if target is not None:
        if target not in all_cats:
            raise ValueError(f"target '{target}' not found in tdata.obs['{groupby}'].")

        # Always use "closest": min path or max lca
        single_agg: str = "max" if metric == "lca" else "min"
        sign = 1.0 if metric == "lca" else -1.0

        # Run the permutation null whenever a test is requested or normalization is needed;
        # a single permutation suffices to obtain the permuted mean used for normalization.
        run_perm = (test == "permutation") or normalize
        n_perms_effective = n_permutations if test == "permutation" else 1

        def _run_single_perm(single_tree, tree_lc, tree_sm, tree_cl, extra_row_fields=None):
            """Run the permutation test for one tree (or globally) and return (rows, cat_null_mean)."""
            tree_obs_cat: dict = {}
            for cat in all_cats:
                vals = [tree_sm[l] for l in tree_cl[cat] if l in tree_sm and not np.isnan(tree_sm[l])]
                tree_obs_cat[cat] = float(np.mean(vals)) if vals else np.nan

            tree_leaf_list = list(tree_lc.keys())
            perm_seeds = np.random.randint(0, 2**31, size=n_perms_effective)

            if permutation_mode == "non_target":
                t_lv = tree_cl[target]
                t_set = set(t_lv)
                nt_lv = [l for l in tree_leaf_list if l not in t_set]
                _PERM_SINGLE_NON_TARGET_DATA.clear()
                _PERM_SINGLE_NON_TARGET_DATA.update(
                    {
                        "fixed_scores": tree_sm,
                        "target_leaves": list(t_lv),
                        "nt_leaves": nt_lv,
                        "nt_cats": [tree_lc[l] for l in nt_lv],
                        "target": target,
                        "all_cats": all_cats,
                    }
                )
                null_results = _run_parallel(_perm_single_non_target_worker, perm_seeds, n_threads)
            else:
                _PERM_SINGLE_DATA.clear()
                _PERM_SINGLE_DATA.update(
                    {
                        "tdata": tdata,
                        "trees": single_tree,
                        "all_leaves": tree_leaf_list,
                        "all_cat_vals": [tree_lc[l] for l in tree_leaf_list],
                        "target": target,
                        "single_agg": single_agg,
                        "metric": metric,
                        "depth_key": depth_key,
                        "all_cats": all_cats,
                    }
                )
                null_results = _run_parallel(_perm_single_target_worker, perm_seeds, n_threads)

            null_cat: dict = defaultdict(list)
            for perm_result in null_results:
                for cat in all_cats:
                    null_cat[cat].append(perm_result[cat])

            rows: list = []
            cat_null_mean: dict = {}
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
                row: dict = {
                    "source": cat,
                    "target": target,
                    "value": obs_val,
                    "permuted_value": perm_val,
                    "z_score": z,
                    "p_value": p,
                }
                if extra_row_fields:
                    row.update(extra_row_fields)
                rows.append(row)

            return rows, cat_null_mean

        if by_tree and run_perm:
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

                tree_all_scores = _compute_scores(tdata, single_tree, tree_lc, [target], single_agg, metric, depth_key)
                tree_sm: dict = {l: s.get(target, np.nan) for l, s in tree_all_scores.items()}
                merged_score_map.update(tree_sm)

                rows, cat_null_mean = _run_single_perm(
                    single_tree, tree_lc, tree_sm, tree_cl, extra_row_fields={"tree": tree_key}
                )
                all_rows.extend(rows)
                if normalize:
                    for leaf, score in tree_sm.items():
                        cat = tree_lc.get(leaf)
                        perm_val = cat_null_mean.get(cat, np.nan) if cat is not None else np.nan
                        merged_norm_map[leaf] = (score - perm_val) if not np.isnan(score) else np.nan

            tdata.obs[f"{target}_linkage"] = tdata.obs.index.map(pd.Series(merged_score_map, dtype=float))
            if normalize:
                tdata.obs[f"{target}_linkage"] = tdata.obs.index.map(pd.Series(merged_norm_map, dtype=float))
            # Return per-category means from whichever map was written to obs.
            linkage_map = merged_norm_map if normalize else merged_score_map
            if test == "permutation":
                test_df = pd.DataFrame(all_rows)
                tdata.uns[f"{key_added}_test"] = test_df
                if copy:
                    return test_df

            if copy:
                result_series = pd.Series(
                    {
                        cat: float(np.nanmean([linkage_map.get(l, np.nan) for l in cat_to_leaves[cat]]))
                        for cat in all_cats
                    },
                    name=f"{target}_linkage",
                )
                return result_series.to_frame()

        else:
            # Global (non-by_tree) path
            all_scores = _compute_scores(tdata, trees, leaf_to_cat, [target], single_agg, metric, depth_key)
            score_map = {leaf: scores.get(target, np.nan) for leaf, scores in all_scores.items()}
            tdata.obs[f"{target}_linkage"] = tdata.obs.index.map(pd.Series(score_map, dtype=float))

            if run_perm:
                rows, cat_null_mean = _run_single_perm(trees, leaf_to_cat, score_map, cat_to_leaves)
                if normalize:
                    # Replace raw scores with normalized (score - category permuted mean) so the
                    # obs column and the copy=True per-category means below are consistent.
                    score_map = {
                        leaf: (score - cat_null_mean.get(leaf_to_cat.get(leaf), np.nan))
                        if not np.isnan(score)
                        else np.nan
                        for leaf, score in score_map.items()
                    }
                    tdata.obs[f"{target}_linkage"] = tdata.obs.index.map(pd.Series(score_map, dtype=float))
                if test == "permutation":
                    test_df = pd.DataFrame(rows)
                    tdata.uns[f"{key_added}_test"] = test_df
                    if copy:
                        return test_df

            if copy:
                result_series = pd.Series(
                    {
                        cat: float(np.nanmean([score_map.get(l, np.nan) for l in cat_to_leaves[cat]]))
                        for cat in all_cats
                    },
                    name=f"{target}_linkage",
                )
                return result_series.to_frame()

    # ── pairwise mode ─────────────────────────────────────────────────────────
    else:
        # Global linkage across all trees (always computed)
        all_scores = _compute_scores(tdata, trees, leaf_to_cat, all_cats, aggregate, metric, depth_key)
        linkage_df = _scores_to_linkage_matrix(all_scores, all_cats, cat_to_leaves)

        # Global permutation null. At least one permutation is always run so a
        # ``permuted_value`` is available for normalization; z-scores and p-values are only
        # meaningful (and only stored) when a full permutation test is requested.
        n_perms_effective = n_permutations if test == "permutation" else 1
        global_z_df: pd.DataFrame | None = None
        global_p_df: pd.DataFrame | None = None
        if permutation_mode == "non_target":
            _z_df, _p_df, global_null_mean_df = _run_permutation_test_non_target(
                tdata,
                trees,
                leaf_to_cat,
                all_cats,
                linkage_df,
                aggregate,
                metric,
                depth_key,
                n_perms_effective,
                n_threads,
                alternative,
            )
        else:
            _z_df, _p_df, global_null_mean_df = _run_permutation_test(
                tdata,
                trees,
                leaf_to_cat,
                all_cats,
                all_cats,
                linkage_df,
                aggregate,
                metric,
                depth_key,
                n_perms_effective,
                n_threads,
                alternative,
            )
        if test == "permutation":
            global_z_df, global_p_df = _z_df, _p_df

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
                if permutation_mode == "non_target":
                    _tz_df, _tp_df, tree_null_mean_df = _run_permutation_test_non_target(
                        tdata,
                        single_tree,
                        tree_leaf_to_cat,
                        all_cats,
                        tree_linkage_df,
                        aggregate,
                        metric,
                        depth_key,
                        n_perms_effective,
                        n_threads,
                        alternative,
                    )
                else:
                    _tz_df, _tp_df, tree_null_mean_df = _run_permutation_test(
                        tdata,
                        single_tree,
                        tree_leaf_to_cat,
                        all_cats,
                        all_cats,
                        tree_linkage_df,
                        aggregate,
                        metric,
                        depth_key,
                        n_perms_effective,
                        n_threads,
                        alternative,
                    )
                if test == "permutation":
                    tree_z_df, tree_p_df = _tz_df, _tp_df

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
                        if tree_null_mean_df is not None:
                            row["permuted_value"] = tree_null_mean_df.loc[src_cat, tgt_cat]
                        if tree_z_df is not None and tree_p_df is not None:
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
                    if global_null_mean_df is not None:
                        row["permuted_value"] = global_null_mean_df.loc[src_cat, tgt_cat]
                    if global_z_df is not None and global_p_df is not None:
                        row["z_score"] = global_z_df.loc[src_cat, tgt_cat]
                        row["p_value"] = global_p_df.loc[src_cat, tgt_cat]
                    stats_rows.append(row)

        # uns[linkage] = observed - permuted_mean if normalize, else raw linkage (both symmetrized if requested).
        # A single permutation is always run above, so normalization no longer requires test='permutation'.
        output_df: pd.DataFrame = (linkage_df - global_null_mean_df) if normalize else linkage_df
        if symmetrize:
            output_df = _symmetrize_matrix(output_df, symmetrize)

        params = {
            "groupby": groupby,
            "aggregate": aggregate,
            "metric": metric,
            "symmetrize": symmetrize,
            "min_size": min_size,
            "test": test,
            "normalize": normalize,
            "by_tree": by_tree,
            "depth_key": depth_key,
        }
        stats_df = pd.DataFrame(stats_rows)
        tdata.uns[f"{key_added}_linkage"] = output_df
        tdata.uns[f"{key_added}_linkage_params"] = params
        tdata.uns[f"{key_added}_linkage_stats"] = stats_df

        if copy:
            return stats_df if test is not None else output_df
