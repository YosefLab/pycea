"""Tests for tl.ancestral_linkage."""

import warnings
from collections import defaultdict

import networkx as nx
import numpy as np
import pandas as pd
import pytest
import treedata as td

import pycea.tl as tl

# Expected warnings from intentionally small test fixtures / parallel workers.
pytestmark = [
    pytest.mark.filterwarnings("ignore:Categories with fewer than 10 cells"),
    pytest.mark.filterwarnings("ignore:This process .* is multi-threaded"),
]


# ── fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def balanced_tdata():
    """Four-leaf binary tree, two categories A and B.

    Structure (depth values):
        root (depth=0)
        ├── n1 (depth=0.5)
        │   ├── a1 (depth=1.0)   cat=A
        │   └── a2 (depth=1.0)   cat=A
        └── n2 (depth=0.5)
            ├── b1 (depth=1.0)   cat=B
            └── b2 (depth=1.0)   cat=B

    LCA depths:  within = 0.5, between = 0.0
    Path dists:  within = 1.0, between = 2.0
    """
    t = nx.DiGraph()
    nodes = {"root": 0.0, "n1": 0.5, "n2": 0.5, "a1": 1.0, "a2": 1.0, "b1": 1.0, "b2": 1.0}
    for node, depth in nodes.items():
        t.add_node(node, depth=depth)
    t.add_edges_from(
        [("root", "n1"), ("root", "n2"), ("n1", "a1"), ("n1", "a2"), ("n2", "b1"), ("n2", "b2")]
    )
    obs = pd.DataFrame({"celltype": ["A", "A", "B", "B"]}, index=["a1", "a2", "b1", "b2"])
    return td.TreeData(obs=obs, obst={"tree": t})


@pytest.fixture
def three_cat_tdata():
    """Six-leaf tree with categories A, B, C (n=0.4 for internal nodes)."""
    t = nx.DiGraph()
    for node, depth in [
        ("root", 0.0), ("n1", 0.4), ("n2", 0.4), ("n3", 0.4),
        ("a1", 1.0), ("a2", 1.0), ("b1", 1.0), ("b2", 1.0), ("c1", 1.0), ("c2", 1.0),
    ]:
        t.add_node(node, depth=depth)
    for u, v in [
        ("root", "n1"), ("root", "n2"), ("root", "n3"),
        ("n1", "a1"), ("n1", "a2"), ("n2", "b1"), ("n2", "b2"), ("n3", "c1"), ("n3", "c2"),
    ]:
        t.add_edge(u, v)
    obs = pd.DataFrame(
        {"celltype": ["A", "A", "B", "B", "C", "C"]},
        index=["a1", "a2", "b1", "b2", "c1", "c2"],
    )
    return td.TreeData(obs=obs, obst={"tree": t})


@pytest.fixture
def two_tree_tdata():
    """Two separate trees, each with one A leaf and one B leaf."""
    t1 = nx.DiGraph()
    for node, depth in [("r1", 0.0), ("n1", 0.5), ("a1", 1.0), ("b1", 1.0)]:
        t1.add_node(node, depth=depth)
    t1.add_edges_from([("r1", "n1"), ("n1", "a1"), ("n1", "b1")])
    t2 = nx.DiGraph()
    for node, depth in [("r2", 0.0), ("n2", 0.5), ("a2", 1.0), ("b2", 1.0)]:
        t2.add_node(node, depth=depth)
    t2.add_edges_from([("r2", "n2"), ("n2", "a2"), ("n2", "b2")])
    obs = pd.DataFrame({"celltype": ["A", "B", "A", "B"]}, index=["a1", "b1", "a2", "b2"])
    return td.TreeData(obs=obs, obst={"tree1": t1, "tree2": t2})


@pytest.fixture
def small_category_tdata():
    """Star tree with a large category A (12 cells) and a small category B (3 cells)."""
    t = nx.DiGraph()
    t.add_node("root", depth=0.0)
    labels = {}
    for i in range(12):
        t.add_node(f"a{i}", depth=1.0)
        t.add_edge("root", f"a{i}")
        labels[f"a{i}"] = "A"
    for i in range(3):
        t.add_node(f"b{i}", depth=1.0)
        t.add_edge("root", f"b{i}")
        labels[f"b{i}"] = "B"
    obs = pd.DataFrame({"celltype": list(labels.values())}, index=list(labels))
    return td.TreeData(obs=obs, obst={"tree": t})


@pytest.fixture
def non_ultrametric_tdata():
    """Tree with leaves at unequal depths (non-ultrametric), two categories A and B.

        root(0.0)
        ├── n1(0.3): a1(0.5) A, a2(2.0) A
        └── n2(0.6): b1(1.0) B, b2(1.5) B
    """
    t = nx.DiGraph()
    for node, depth in [
        ("root", 0.0), ("n1", 0.3), ("n2", 0.6), ("a1", 0.5), ("a2", 2.0), ("b1", 1.0), ("b2", 1.5),
    ]:
        t.add_node(node, depth=depth)
    t.add_edges_from(
        [("root", "n1"), ("root", "n2"), ("n1", "a1"), ("n1", "a2"), ("n2", "b1"), ("n2", "b2")]
    )
    obs = pd.DataFrame({"celltype": ["A", "A", "B", "B"]}, index=["a1", "a2", "b1", "b2"])
    return td.TreeData(obs=obs, obst={"tree": t})


def _bruteforce_lca_max(tdata, groupby="celltype"):
    """Reference pairwise max-LCA-depth linkage via explicit LCA on the tree."""
    t = list(tdata.obst.values())[0]
    depth = dict(t.nodes(data="depth"))
    cats = defaultdict(list)
    for leaf, c in tdata.obs[groupby].items():
        cats[c].append(leaf)

    def lca_depth(i, j):
        anc_i = nx.ancestors(t, i) | {i}
        anc_j = nx.ancestors(t, j) | {j}
        return max(depth[a] for a in anc_i & anc_j)

    out = {}
    for s in cats:
        row = {}
        for tcat in cats:
            per_cell = [max(depth[a] if a == b else lca_depth(a, b) for b in cats[tcat]) for a in cats[s]]
            row[tcat] = float(np.mean(per_cell))
        out[s] = row
    return pd.DataFrame(out).T


def _bruteforce_path_min(tdata, groupby="celltype"):
    """Reference pairwise min-path-distance linkage via explicit LCA on the tree."""
    t = list(tdata.obst.values())[0]
    depth = dict(t.nodes(data="depth"))
    cats = defaultdict(list)
    for leaf, c in tdata.obs[groupby].items():
        cats[c].append(leaf)

    def path(i, j):
        if i == j:
            return 0.0
        anc_i = nx.ancestors(t, i) | {i}
        anc_j = nx.ancestors(t, j) | {j}
        lca = max(depth[a] for a in anc_i & anc_j)
        return depth[i] + depth[j] - 2 * lca

    out = {}
    for s in cats:
        row = {}
        for tcat in cats:
            per_cell = [min(path(a, b) for b in cats[tcat]) for a in cats[s]]
            row[tcat] = float(np.mean(per_cell))
        out[s] = row
    return pd.DataFrame(out).T


# ── pairwise mode ───────────────────────────────────────────────────────────


def test_pairwise_known_values(balanced_tdata):
    """Exact linkage values and within/between relationships for each metric/aggregate."""
    tdata = balanced_tdata
    # lca + max: within (self) = 1.0, between = root depth 0.0
    lca = tl.ancestral_linkage(tdata, groupby="celltype", aggregate="max", metric="lca",
                               normalize=False, copy=True)
    assert np.isclose(lca.loc["A", "A"], 1.0) and np.isclose(lca.loc["B", "B"], 1.0)
    assert np.isclose(lca.loc["A", "B"], 0.0)
    assert lca.loc["A", "A"] > lca.loc["A", "B"]
    # path + min: within (self) = 0.0, between = 2.0
    path = tl.ancestral_linkage(tdata, groupby="celltype", aggregate="min", metric="path",
                                normalize=False, copy=True)
    assert np.isclose(path.loc["A", "A"], 0.0) and np.isclose(path.loc["A", "B"], 2.0)
    assert path.loc["A", "A"] < path.loc["A", "B"]
    # mean path: between = 2.0, within = mean([0, 1, 1, 0]) = 0.5
    mean = tl.ancestral_linkage(tdata, groupby="celltype", aggregate="mean", metric="path",
                                normalize=False, copy=True)
    assert np.isclose(mean.loc["A", "B"], 2.0) and np.isclose(mean.loc["A", "A"], 0.5)


def test_pairwise_output_structure(balanced_tdata):
    """uns keys, matrix shape, stats columns, key_added, and copy return type."""
    tdata = balanced_tdata
    result = tl.ancestral_linkage(tdata, groupby="celltype", key_added="mykey", copy=True)
    # copy returns the linkage matrix (no test)
    assert isinstance(result, pd.DataFrame)
    assert set(result.index) == {"A", "B"} and set(result.columns) == {"A", "B"}
    # uns keys follow the key_added prefix
    for suffix in ("_linkage", "_linkage_params", "_linkage_stats"):
        assert f"mykey{suffix}" in tdata.uns
    # stats table structure
    stats = tdata.uns["mykey_linkage_stats"]
    assert {"source", "target", "source_n", "target_n", "value"} <= set(stats.columns)
    assert "tree" not in stats.columns  # by_tree=False
    assert len(stats) == 4  # 2 categories -> 4 ordered pairs
    assert stats.loc[stats["source"] == "A"].iloc[0]["source_n"] == 2


def test_custom_callable_aggregate(balanced_tdata):
    """A custom callable aggregate matches the equivalent named aggregate."""
    tdata = balanced_tdata
    mat = tl.ancestral_linkage(tdata, groupby="celltype", aggregate=np.mean, metric="path", copy=True)
    ref = tl.ancestral_linkage(tdata, groupby="celltype", aggregate="mean", metric="path",
                               key_added="ref", copy=True)
    pd.testing.assert_frame_equal(mat, ref)


# ── min_size ──────────────────────────────────────────────────────────────


def test_min_size(three_cat_tdata):
    """min_size drops small categories (matrix + stats), is recorded, and works with a test."""
    tdata = three_cat_tdata
    # default keeps everything
    mat = tl.ancestral_linkage(tdata, groupby="celltype", min_size=1, copy=True)
    assert set(mat.index) == {"A", "B", "C"}
    # relabel so C has a single cell, then exclude it
    tdata.obs["celltype"] = ["A", "A", "B", "B", "C", "B"]
    tl.ancestral_linkage(tdata, groupby="celltype", min_size=2, test="permutation",
                         n_permutations=20, random_state=0)
    mat = tdata.uns["celltype_linkage"]
    stats = tdata.uns["celltype_linkage_stats"]
    assert set(mat.index) == {"A", "B"} and "C" not in mat.columns
    assert "C" not in stats["source"].values and "C" not in stats["target"].values
    assert stats["p_value"].between(0, 1).all()
    assert tdata.uns["celltype_linkage_params"]["min_size"] == 2


def test_min_size_invalid_raises(balanced_tdata, three_cat_tdata):
    """Non-positive min_size and a min_size that excludes everything both raise."""
    with pytest.raises(ValueError, match="min_size"):
        tl.ancestral_linkage(balanced_tdata, groupby="celltype", min_size=0)
    with pytest.raises(ValueError, match="min_size"):
        tl.ancestral_linkage(three_cat_tdata, groupby="celltype", min_size=100)


# ── warnings ────────────────────────────────────────────────────────────────


def test_warnings(balanced_tdata, small_category_tdata):
    """Small categories warn (and can be silenced by min_size); lca+min warns."""
    # Small category B (3 cells) is flagged, large category A (12) is not.
    with pytest.warns(UserWarning, match="noisy linkage") as record:
        tl.ancestral_linkage(small_category_tdata, groupby="celltype")
    msg = next(str(w.message) for w in record if "noisy linkage" in str(w.message))
    assert "'B'" in msg and "'A'" not in msg
    # Excluding the small category via min_size removes the warning.
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        tl.ancestral_linkage(small_category_tdata, groupby="celltype", min_size=10)
    assert not any("noisy linkage" in str(w.message) for w in rec)
    # aggregate='min' with metric='lca' warns about shallowest ancestor.
    with pytest.warns(UserWarning, match="shallowest"):
        tl.ancestral_linkage(balanced_tdata, groupby="celltype", aggregate="min", metric="lca")


# ── symmetrize ──────────────────────────────────────────────────────────────


def test_symmetrize(three_cat_tdata):
    """symmetrize='mean'/'max' make the matrix symmetric; 'max' >= raw; invalid raises."""
    tdata = three_cat_tdata
    raw = tl.ancestral_linkage(tdata, groupby="celltype", aggregate="min", metric="path", copy=True)
    for mode in ("mean", "max"):
        mat = tl.ancestral_linkage(tdata, groupby="celltype", aggregate="min", metric="path",
                                   symmetrize=mode, copy=True)
        for i in mat.index:
            for j in mat.columns:
                assert np.isclose(mat.loc[i, j], mat.loc[j, i])
                if mode == "max":
                    assert mat.loc[i, j] >= raw.loc[i, j] - 1e-9
    with pytest.raises(ValueError, match="symmetrize"):
        tl.ancestral_linkage(tdata, groupby="celltype", symmetrize="meen")


def test_symmetrized_linkage_stats(three_cat_tdata, two_tree_tdata):
    """symmetrize + permutation adds a one-row-per-unordered-pair symmetrized stats table."""
    tdata = three_cat_tdata
    key = "celltype_symmetrized_linkage_stats"
    tl.ancestral_linkage(tdata, groupby="celltype", test="permutation", symmetrize="mean",
                         normalize=False, n_permutations=50, random_state=0)
    assert key in tdata.uns
    sym = tdata.uns[key]
    # one row per unordered pair (upper triangle incl. diagonal): 3 cats -> 6 rows
    assert len(sym) == 6
    assert {"source", "target", "value", "source_n", "target_n",
            "permuted_value", "z_score", "p_value"} <= set(sym.columns)
    assert sym["p_value"].between(0, 1).all()
    # symmetrized value equals the mean of the two directions in the raw stats table
    raw = tdata.uns["celltype_linkage_stats"].set_index(["source", "target"])["value"]
    for _, r in sym.iterrows():
        s, t = r["source"], r["target"]
        assert r["value"] == pytest.approx((raw[(s, t)] + raw[(t, s)]) / 2, abs=1e-9)

    # not created without a permutation test, and a stale table is cleared
    tl.ancestral_linkage(tdata, groupby="celltype", symmetrize="mean")
    assert key not in tdata.uns
    # not created when symmetrize=False
    tl.ancestral_linkage(tdata, groupby="celltype", test="permutation", symmetrize=False,
                         n_permutations=20, random_state=0)
    assert key not in tdata.uns
    # a stale pairwise table is cleared when the same key_added is reused in single-target mode
    tl.ancestral_linkage(tdata, groupby="celltype", test="permutation", symmetrize="mean",
                         n_permutations=20, random_state=0)
    assert key in tdata.uns
    tl.ancestral_linkage(tdata, groupby="celltype", target="B", test="permutation",
                         n_permutations=20, random_state=0)
    assert key not in tdata.uns

    # by_tree gets a per-tree 'tree' column, one row per unordered pair per tree,
    # for both permutation modes
    for mode in ("non_target", "all"):
        tl.ancestral_linkage(two_tree_tdata, groupby="celltype", by_tree=True, test="permutation",
                             permutation_mode=mode, symmetrize="max", n_permutations=20, random_state=0)
        bt = two_tree_tdata.uns[key]
        assert "tree" in bt.columns
        assert set(bt["tree"].unique()) == {"tree1", "tree2"}
        assert bt["p_value"].between(0, 1).all()


# ── single-target mode ────────────────────────────────────────────────────


def test_single_target_known_values(balanced_tdata):
    """Single-target stores per-cell scores in obs with exact values, and copy returns a summary."""
    tdata = balanced_tdata
    # path: a* are 2.0 from nearest B, b* (in target) are 0.0
    tl.ancestral_linkage(tdata, groupby="celltype", target="B", metric="path", normalize=False)
    assert "B_linkage" in tdata.obs.columns and tdata.obs["B_linkage"].notna().all()
    assert np.isclose(tdata.obs.loc["a1", "B_linkage"], 2.0)
    assert np.isclose(tdata.obs.loc["b1", "B_linkage"], 0.0)
    # lca: a* have LCA depth 0.0 to best B; b1 (in B) has LCA with self = depth 1.0
    tl.ancestral_linkage(tdata, groupby="celltype", target="B", metric="lca", normalize=False)
    assert np.isclose(tdata.obs.loc["a1", "B_linkage"], 0.0)
    assert np.isclose(tdata.obs.loc["b1", "B_linkage"], 1.0)
    # copy returns a per-category summary DataFrame
    result = tl.ancestral_linkage(tdata, groupby="celltype", target="B", metric="path", copy=True)
    assert isinstance(result, pd.DataFrame)
    assert "B_linkage" in result.columns and {"A", "B"} <= set(result.index)


# ── by_tree ──────────────────────────────────────────────────────────────


def test_by_tree(two_tree_tdata):
    """by_tree adds a per-tree 'tree' column with per-tree counts; matrix matches global."""
    tdata = two_tree_tdata
    global_mat = tl.ancestral_linkage(tdata, groupby="celltype", copy=True)
    tl.ancestral_linkage(tdata, groupby="celltype", by_tree=True)
    stats = tdata.uns["celltype_linkage_stats"]
    assert "tree" in stats.columns
    assert set(stats["tree"]) == {"tree1", "tree2"}
    assert len(stats) == 8  # 2 trees × 2 × 2
    # each tree has one A leaf and one B leaf
    a_row = stats[(stats["tree"] == "tree1") & (stats["source"] == "A")].iloc[0]
    assert a_row["source_n"] == 1 and a_row["target_n"] == 1
    # the aggregated linkage matrix is identical to the global (by_tree=False) result
    pd.testing.assert_frame_equal(global_mat, tdata.uns["celltype_linkage"])


# ── permutation test ────────────────────────────────────────────────────────


def test_permutation_pairwise(balanced_tdata):
    """Pairwise permutation test: enrichment matrix + stats columns; no separate uns keys."""
    tdata = balanced_tdata
    result = tl.ancestral_linkage(tdata, groupby="celltype", test="permutation",
                                  n_permutations=20, random_state=42, copy=True)
    # copy returns the stats DataFrame when a test is run
    assert isinstance(result, pd.DataFrame)
    assert {"source", "target", "value", "permuted_value", "z_score", "p_value"} <= set(result.columns)
    assert result["p_value"].between(0, 1).all()
    assert result["permuted_value"].notna().all()
    # linkage matrix stored, finite; no stray z/p uns keys
    mat = tdata.uns["celltype_linkage"]
    assert set(mat.index) == {"A", "B"} and np.isfinite(mat.values).all()
    assert "celltype_z_score" not in tdata.uns and "celltype_p_value" not in tdata.uns


def test_permutation_single_target(balanced_tdata):
    """Single-target permutation test returns a long DataFrame scoped to the target."""
    tdata = balanced_tdata
    result = tl.ancestral_linkage(tdata, groupby="celltype", target="B", test="permutation",
                                  n_permutations=20, random_state=0, copy=True)
    assert isinstance(result, pd.DataFrame)
    assert {"source", "target", "value", "z_score", "p_value", "permuted_value"} <= set(result.columns)
    assert (result["target"] == "B").all()
    assert result["permuted_value"].notna().all()


def test_permutation_by_tree(two_tree_tdata):
    """by_tree + permutation stores per-tree z_score/p_value/permuted_value."""
    tdata = two_tree_tdata
    tl.ancestral_linkage(tdata, groupby="celltype", by_tree=True, test="permutation",
                         n_permutations=10, random_state=0)
    stats = tdata.uns["celltype_linkage_stats"]
    assert {"z_score", "p_value", "permuted_value"} <= set(stats.columns)
    assert stats["p_value"].between(0, 1).all()


def test_permutation_reproducible(balanced_tdata):
    """Same random_state is deterministic; parallel matches serial."""
    def run(n_threads):
        return tl.ancestral_linkage(balanced_tdata, groupby="celltype", test="permutation",
                                    n_permutations=20, random_state=42, n_threads=n_threads, copy=True)
    pd.testing.assert_frame_equal(run(None), run(None))  # deterministic
    pd.testing.assert_frame_equal(run(None), run(2))  # parallel == serial


def test_alternative(balanced_tdata):
    """`alternative` p-values stay in range, are symmetric on a symmetric tree, and None==default."""
    tdata = balanced_tdata
    tl.ancestral_linkage(tdata, groupby="celltype", test="permutation", alternative="two-sided",
                         n_permutations=50, random_state=1)
    stats = tdata.uns["celltype_linkage_stats"]
    assert stats["p_value"].between(0, 1).all()
    ab = stats[(stats["source"] == "A") & (stats["target"] == "B")]["p_value"].values[0]
    ba = stats[(stats["source"] == "B") & (stats["target"] == "A")]["p_value"].values[0]
    assert np.isclose(ab, ba)
    # two-sided also works in single-target mode
    result = tl.ancestral_linkage(tdata, groupby="celltype", target="B", test="permutation",
                                  alternative="two-sided", n_permutations=20, random_state=0, copy=True)
    assert result["p_value"].between(0, 1).all()
    # alternative=None matches omitting the parameter (the default)
    explicit = tl.ancestral_linkage(balanced_tdata, groupby="celltype", test="permutation",
                                    alternative=None, n_permutations=20, random_state=42, copy=True)
    default = tl.ancestral_linkage(balanced_tdata, groupby="celltype", test="permutation",
                                   n_permutations=20, random_state=42, copy=True)
    pd.testing.assert_frame_equal(explicit, default)


def test_permutation_mode_non_target(balanced_tdata, two_tree_tdata):
    """permutation_mode='non_target' yields valid p-values in every mode combination."""
    tdata = balanced_tdata
    # pairwise: 'all' and 'non_target' produce the same stats schema and valid p-values
    all_df = tl.ancestral_linkage(tdata, groupby="celltype", test="permutation",
                                  permutation_mode="all", n_permutations=20, random_state=42, copy=True)
    nt_df = tl.ancestral_linkage(tdata, groupby="celltype", test="permutation",
                                 permutation_mode="non_target", n_permutations=20, random_state=42, copy=True)
    assert set(all_df.columns) == set(nt_df.columns)
    assert {"z_score", "permuted_value"} <= set(nt_df.columns)
    assert nt_df["p_value"].between(0, 1).all()
    # single-target
    result = tl.ancestral_linkage(tdata, groupby="celltype", target="B", test="permutation",
                                  permutation_mode="non_target", n_permutations=20, random_state=0, copy=True)
    assert result["p_value"].between(0, 1).all()
    # by_tree
    tl.ancestral_linkage(two_tree_tdata, groupby="celltype", by_tree=True, test="permutation",
                         permutation_mode="non_target", n_permutations=10, random_state=0)
    bt_stats = two_tree_tdata.uns["celltype_linkage_stats"]
    assert bt_stats["p_value"].between(0, 1).all()


# ── validation ──────────────────────────────────────────────────────────────


def test_invalid_params(balanced_tdata):
    """Invalid groupby, aggregate, and target all raise ValueError."""
    with pytest.raises(ValueError, match="not found"):
        tl.ancestral_linkage(balanced_tdata, groupby="nonexistent")
    with pytest.raises(ValueError, match="aggregate"):
        tl.ancestral_linkage(balanced_tdata, groupby="celltype", aggregate="median")
    with pytest.raises(ValueError, match="not found"):
        tl.ancestral_linkage(balanced_tdata, groupby="celltype", target="Z")


# ── correctness vs brute force (ultrametric & non-ultrametric) ────────────────


@pytest.mark.parametrize("fixture_name", ["three_cat_tdata", "non_ultrametric_tdata"])
def test_path_min_matches_bruteforce(fixture_name, request):
    """path+min matches an explicit min-path brute force (walk-up and Dijkstra paths)."""
    tdata = request.getfixturevalue(fixture_name)
    mat = tl.ancestral_linkage(tdata, groupby="celltype", aggregate="min", metric="path",
                               normalize=False, symmetrize=False, copy=True)
    ref = _bruteforce_path_min(tdata)
    pd.testing.assert_frame_equal(mat, ref.loc[mat.index, mat.columns])


@pytest.mark.parametrize("fixture_name", ["three_cat_tdata", "non_ultrametric_tdata"])
def test_lca_max_matches_bruteforce(fixture_name, request):
    """lca+max matches an explicit LCA brute force on ultrametric and non-ultrametric trees."""
    tdata = request.getfixturevalue(fixture_name)
    mat = tl.ancestral_linkage(tdata, groupby="celltype", aggregate="max", metric="lca",
                               normalize=False, symmetrize=False, copy=True)
    ref = _bruteforce_lca_max(tdata)
    pd.testing.assert_frame_equal(mat, ref.loc[mat.index, mat.columns])
    if fixture_name == "non_ultrametric_tdata":
        # within-A mean(0.5, 2.0) = 1.25, within-B mean(1.0, 1.5) = 1.25, between = root 0.0
        assert np.isclose(mat.loc["A", "A"], 1.25) and np.isclose(mat.loc["B", "B"], 1.25)
        assert np.isclose(mat.loc["A", "B"], 0.0)


# ── normalize ────────────────────────────────────────────────────────────────


def test_permuted_value_and_normalize_without_test(balanced_tdata):
    """permuted_value is populated without a test; normalize uses it; z/p are absent."""
    tdata = balanced_tdata
    tl.ancestral_linkage(tdata, groupby="celltype", normalize=False)
    stats = tdata.uns["celltype_linkage_stats"]
    linkage = tdata.uns["celltype_linkage"]
    assert stats["permuted_value"].notna().all()
    assert "z_score" not in stats.columns and "p_value" not in stats.columns
    # normalize=False stores the raw linkage
    for _, row in stats.iterrows():
        assert linkage.loc[row["source"], row["target"]] == pytest.approx(row["value"], abs=1e-9)
    # normalize=True (no test) stores value - permuted_value
    tl.ancestral_linkage(tdata, groupby="celltype", normalize=True)
    linkage = tdata.uns["celltype_linkage"]
    stats = tdata.uns["celltype_linkage_stats"]
    for _, row in stats.iterrows():
        assert linkage.loc[row["source"], row["target"]] == pytest.approx(row["value"] - row["permuted_value"], abs=1e-9)
    # single-target normalize without a test creates no _test key
    tl.ancestral_linkage(tdata, groupby="celltype", target="B", normalize=True)
    assert "B_linkage" in tdata.obs.columns and "celltype_test" not in tdata.uns


def test_pairwise_normalize_with_test(balanced_tdata):
    """With test='permutation', normalize toggles between enrichment and raw linkage."""
    tdata = balanced_tdata
    tl.ancestral_linkage(tdata, groupby="celltype", test="permutation", normalize=True,
                         n_permutations=20, random_state=0)
    linkage, stats = tdata.uns["celltype_linkage"], tdata.uns["celltype_linkage_stats"]
    for _, row in stats.iterrows():
        assert linkage.loc[row["source"], row["target"]] == pytest.approx(row["value"] - row["permuted_value"], abs=1e-9)
    tl.ancestral_linkage(tdata, groupby="celltype", test="permutation", normalize=False,
                         n_permutations=20, random_state=0)
    linkage, stats = tdata.uns["celltype_linkage"], tdata.uns["celltype_linkage_stats"]
    for _, row in stats.iterrows():
        assert linkage.loc[row["source"], row["target"]] == pytest.approx(row["value"], abs=1e-9)


def test_single_target_normalize(balanced_tdata):
    """Single-target normalize overwrites obs with score - category_permuted_mean."""
    tdata = balanced_tdata
    tl.ancestral_linkage(tdata, groupby="celltype", target="B", metric="path", test="permutation",
                         normalize=False, n_permutations=30, random_state=1)
    raw = tdata.obs["B_linkage"].copy()
    assert raw.loc["a1"] == pytest.approx(2.0)  # raw min-path score
    tl.ancestral_linkage(tdata, groupby="celltype", target="B", metric="path", test="permutation",
                         normalize=True, n_permutations=30, random_state=1)
    norm = tdata.obs["B_linkage"]
    test_df = tdata.uns["celltype_test"]
    for cell in ["a1", "a2", "b1", "b2"]:
        cat = tdata.obs.loc[cell, "celltype"]
        perm_val = test_df.loc[test_df["source"] == cat, "permuted_value"].iloc[0]
        assert norm[cell] == pytest.approx(raw[cell] - perm_val, abs=1e-9)
    assert "B_norm_linkage" not in tdata.obs.columns


def test_single_target_by_tree(three_cat_tdata, balanced_tdata):
    """by_tree single-target: per-tree test rows, normalized copy matches obs means, non_target."""
    tdata = three_cat_tdata
    # copy per-category means match the (normalized) obs column
    result = tl.ancestral_linkage(tdata, groupby="celltype", target="A", metric="path",
                                  normalize=True, by_tree=True, random_state=1, copy=True)
    for cat in ["A", "B", "C"]:
        cells = tdata.obs.index[tdata.obs["celltype"] == cat]
        expected = float(np.nanmean(tdata.obs.loc[cells, "A_linkage"]))
        assert result.loc[cat, "A_linkage"] == pytest.approx(expected, abs=1e-9)
    # by_tree + permutation writes a per-tree test table
    tl.ancestral_linkage(tdata, groupby="celltype", target="A", test="permutation", by_tree=True,
                         n_permutations=20, random_state=7)
    test_df = tdata.uns["celltype_test"]
    assert "tree" in test_df.columns
    assert set(tdata.obst) <= set(test_df["tree"].values)
    assert "A_norm_linkage" not in tdata.obs.columns
    # by_tree + non_target permutation mode also produces a tree column
    tl.ancestral_linkage(balanced_tdata, groupby="celltype", target="B", test="permutation",
                         by_tree=True, permutation_mode="non_target", n_permutations=20, random_state=3)
    assert "tree" in balanced_tdata.uns["celltype_test"].columns


if __name__ == "__main__":
    pytest.main(["-v", __file__])
