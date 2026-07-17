"""Tests for pl.ancestral_linkage."""

import matplotlib

matplotlib.use("Agg")
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import pytest
import treedata as td

import pycea
import pycea.pl as pl
import pycea.tl as tl

plot_path = Path(__file__).parent / "plots"

# Expected warnings from intentionally small test fixtures / parallel workers.
pytestmark = [
    pytest.mark.filterwarnings("ignore:Categories with fewer than 10 cells"),
    pytest.mark.filterwarnings("ignore:This process .* is multi-threaded"),
]


@pytest.fixture
def tdata():
    """Six-leaf ultrametric tree with three categories A, B, C."""
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
    obs = pd.DataFrame({"celltype": ["A", "A", "B", "B", "C", "C"]}, index=["a1", "a2", "b1", "b2", "c1", "c2"])
    return td.TreeData(obs=obs, obst={"tree": t})


@pytest.fixture
def linkage_tdata(tdata):
    """tdata with a pairwise permutation-tested linkage stored in uns."""
    tl.ancestral_linkage(tdata, groupby="celltype", test="permutation", n_permutations=20, random_state=0)
    return tdata


def test_returns_axes(linkage_tdata):
    ax = pl.ancestral_linkage(linkage_tdata, groupby="celltype")
    assert isinstance(ax, plt.Axes)
    # square heatmap with all three categories on both axes
    assert len(ax.get_xticklabels()) == 3
    assert set(t.get_text() for t in ax.get_yticklabels()) == {"A", "B", "C"}
    plt.close("all")


def test_groupby_inferred_from_uns(linkage_tdata):
    """groupby=None infers the single linkage result present in uns."""
    ax = pl.ancestral_linkage(linkage_tdata)
    assert isinstance(ax, plt.Axes)
    plt.close("all")


def test_ambiguous_groupby_raises(linkage_tdata):
    """Multiple linkage results require an explicit groupby."""
    tl.ancestral_linkage(linkage_tdata, groupby="celltype", key_added="other")
    with pytest.raises(ValueError, match="Specify groupby"):
        pl.ancestral_linkage(linkage_tdata)


def test_normalize_bool(linkage_tdata):
    for normalize in (True, False):
        ax = pl.ancestral_linkage(linkage_tdata, groupby="celltype", normalize=normalize)
        assert isinstance(ax, plt.Axes)
    plt.close("all")


def test_normalize_without_test(tdata):
    """normalize=True works without a full test: permuted_value is always populated."""
    tl.ancestral_linkage(tdata, groupby="celltype")  # no test
    assert "permuted_value" in tdata.uns["celltype_linkage_stats"].columns
    ax = pl.ancestral_linkage(tdata, groupby="celltype", normalize=True)
    assert isinstance(ax, plt.Axes)
    plt.close("all")


def test_normalize_false_without_permutation(tdata):
    """normalize=False colors by the raw linkage value."""
    tl.ancestral_linkage(tdata, groupby="celltype")
    ax = pl.ancestral_linkage(tdata, groupby="celltype", normalize=False)
    assert isinstance(ax, plt.Axes)
    plt.close("all")


def test_symmetrize_options(linkage_tdata):
    for sym in ("mean", "max", "min", False, None):
        ax = pl.ancestral_linkage(linkage_tdata, groupby="celltype", symmetrize=sym)
        assert isinstance(ax, plt.Axes)
    plt.close("all")


def test_normalize_inferred_from_uns(tdata):
    """normalize=None mirrors how tl.ancestral_linkage was run (via the cmap default)."""
    # tl run with normalize=False -> plot infers False -> sequential viridis
    tl.ancestral_linkage(tdata, groupby="celltype", normalize=False)
    ax = pl.ancestral_linkage(tdata, groupby="celltype")
    assert ax.collections[0].get_cmap().name == "viridis"
    plt.close("all")
    # tl run with normalize=True -> plot infers True -> diverging RdBu_r
    tl.ancestral_linkage(tdata, groupby="celltype", normalize=True)
    ax = pl.ancestral_linkage(tdata, groupby="celltype")
    assert ax.collections[0].get_cmap().name == "RdBu_r"
    plt.close("all")


def test_symmetrize_inferred_from_uns():
    """symmetrize=None mirrors how tl.ancestral_linkage was run."""
    # Asymmetric tree: A={a1,a2}, B={b1}; a2 and b1 are siblings so A->B != B->A.
    t = nx.DiGraph()
    for n, d in [("root", 0.0), ("n1", 0.5), ("a1", 1.0), ("a2", 1.0), ("b1", 1.0)]:
        t.add_node(n, depth=d)
    t.add_edges_from([("root", "a1"), ("root", "n1"), ("n1", "a2"), ("n1", "b1")])
    obs = pd.DataFrame({"celltype": ["A", "A", "B"]}, index=["a1", "a2", "b1"])
    tdata = td.TreeData(obs=obs, obst={"tree": t})

    def plotted(ax):
        return np.asarray(ax.collections[0].get_array(), dtype=float).reshape(2, 2)

    # tl left asymmetric (symmetrize=False) -> plot infers False -> asymmetric matrix
    tl.ancestral_linkage(tdata, groupby="celltype", metric="path", normalize=False, symmetrize=False)
    arr = plotted(pl.ancestral_linkage(tdata, groupby="celltype"))
    assert not np.allclose(arr, arr.T)
    plt.close("all")
    # tl symmetrized with 'mean' -> plot infers 'mean' -> symmetric matrix
    tl.ancestral_linkage(tdata, groupby="celltype", metric="path", normalize=False, symmetrize="mean")
    arr = plotted(pl.ancestral_linkage(tdata, groupby="celltype"))
    assert np.allclose(arr, arr.T)
    plt.close("all")


def test_default_cmap_by_normalize(linkage_tdata):
    """cmap=None resolves to RdBu_r when normalize=True and viridis when False."""
    ax = pl.ancestral_linkage(linkage_tdata, groupby="celltype", normalize=True)
    assert ax.collections[0].get_cmap().name == "RdBu_r"
    plt.close("all")
    ax = pl.ancestral_linkage(linkage_tdata, groupby="celltype", normalize=False)
    assert ax.collections[0].get_cmap().name == "viridis"
    plt.close("all")


def test_explicit_cmap(linkage_tdata):
    ax = pl.ancestral_linkage(linkage_tdata, groupby="celltype", cmap="magma")
    assert ax.collections[0].get_cmap().name == "magma"
    plt.close("all")


def test_vmin_vmax(linkage_tdata):
    ax = pl.ancestral_linkage(linkage_tdata, groupby="celltype", vmin=-3, vmax=3)
    norm = ax.collections[0].norm
    assert norm.vmin == -3 and norm.vmax == 3
    plt.close("all")


def test_center_diverging_only(linkage_tdata):
    """center produces a TwoSlopeNorm when normalize=True, but is ignored when False."""
    from matplotlib.colors import Normalize, TwoSlopeNorm

    ax = pl.ancestral_linkage(linkage_tdata, groupby="celltype", normalize=True, center=0)
    assert isinstance(ax.collections[0].norm, TwoSlopeNorm)
    assert ax.collections[0].norm.vcenter == 0
    plt.close("all")

    ax = pl.ancestral_linkage(linkage_tdata, groupby="celltype", normalize=False, center=0)
    norm = ax.collections[0].norm
    assert isinstance(norm, Normalize) and not isinstance(norm, TwoSlopeNorm)
    plt.close("all")


def test_center_none_disables_centering(linkage_tdata):
    from matplotlib.colors import Normalize, TwoSlopeNorm

    ax = pl.ancestral_linkage(linkage_tdata, groupby="celltype", normalize=True, center=None)
    norm = ax.collections[0].norm
    assert isinstance(norm, Normalize) and not isinstance(norm, TwoSlopeNorm)
    plt.close("all")


def test_cluster_methods_and_none(linkage_tdata):
    for cluster in ("ward", "average", None):
        ax = pl.ancestral_linkage(linkage_tdata, groupby="celltype", cluster=cluster)
        assert isinstance(ax, plt.Axes)
    plt.close("all")


def _block_stats():
    """4-category similarity table: {A,B} and {C,D} are the tight pairs."""
    vals = {("A", "B"): 8, ("C", "D"): 8, ("A", "C"): 1, ("A", "D"): 1, ("B", "C"): 1, ("B", "D"): 1}
    cats = ["A", "B", "C", "D"]
    rows = []
    for s in cats:
        for t in cats:
            v = 10.0 if s == t else float(vals.get((s, t), vals.get((t, s))))
            rows.append({"source": s, "target": t, "value": v})
    return pd.DataFrame(rows)


def test_cluster_mode_negation_changes_order(linkage_tdata):
    """similarity vs dissimilarity give different leaf orders for the same matrix."""
    stats = _block_stats()
    kw = dict(data=stats, normalize=False, symmetrize=False, cluster="average")
    order_sim = [t.get_text() for t in pl.ancestral_linkage(linkage_tdata, cluster_mode="similarity", **kw).get_xticklabels()]
    plt.close("all")
    order_dis = [t.get_text() for t in pl.ancestral_linkage(linkage_tdata, cluster_mode="dissimilarity", **kw).get_xticklabels()]
    plt.close("all")
    assert order_sim != order_dis


def test_cluster_mode_none_defaults_similarity_without_params(linkage_tdata):
    """With no recorded metric, cluster_mode=None behaves like 'similarity'."""
    stats = _block_stats()
    kw = dict(data=stats, normalize=False, symmetrize=False, cluster="average")
    order_none = [t.get_text() for t in pl.ancestral_linkage(linkage_tdata, cluster_mode=None, **kw).get_xticklabels()]
    plt.close("all")
    order_sim = [t.get_text() for t in pl.ancestral_linkage(linkage_tdata, cluster_mode="similarity", **kw).get_xticklabels()]
    plt.close("all")
    assert order_none == order_sim


def test_cluster_mode_inferred_from_metric(tdata):
    """cluster_mode=None matches the mode implied by the recorded metric."""
    stats = _block_stats()
    # path -> dissimilarity
    tdata.uns["celltype_linkage_stats"] = stats
    tdata.uns["celltype_linkage_params"] = {"metric": "path"}
    order_none = [t.get_text() for t in pl.ancestral_linkage(tdata, groupby="celltype", normalize=False, symmetrize=False, cluster="average").get_xticklabels()]
    plt.close("all")
    order_dis = [t.get_text() for t in pl.ancestral_linkage(tdata, groupby="celltype", normalize=False, symmetrize=False, cluster="average", cluster_mode="dissimilarity").get_xticklabels()]
    plt.close("all")
    assert order_none == order_dis
    # lca -> similarity
    tdata.uns["celltype_linkage_params"] = {"metric": "lca"}
    order_none = [t.get_text() for t in pl.ancestral_linkage(tdata, groupby="celltype", normalize=False, symmetrize=False, cluster="average").get_xticklabels()]
    plt.close("all")
    order_sim = [t.get_text() for t in pl.ancestral_linkage(tdata, groupby="celltype", normalize=False, symmetrize=False, cluster="average", cluster_mode="similarity").get_xticklabels()]
    plt.close("all")
    assert order_none == order_sim


def test_invalid_cluster_mode_raises(linkage_tdata):
    with pytest.raises(ValueError, match="cluster_mode"):
        pl.ancestral_linkage(linkage_tdata, groupby="celltype", cluster_mode="nope")


def test_explicit_order(linkage_tdata):
    ax = pl.ancestral_linkage(linkage_tdata, groupby="celltype", order=["C", "A", "B"])
    assert [t.get_text() for t in ax.get_xticklabels()] == ["C", "A", "B"]
    plt.close("all")


def test_invalid_order_raises(linkage_tdata):
    with pytest.raises(ValueError, match="not in the linkage matrix"):
        pl.ancestral_linkage(linkage_tdata, groupby="celltype", order=["A", "B", "Z"])


def test_missing_groupby_raises(tdata):
    with pytest.raises(KeyError, match="linkage_stats"):
        pl.ancestral_linkage(tdata, groupby="celltype")


def test_no_linkage_in_uns_raises(tdata):
    with pytest.raises(KeyError, match="linkage_stats"):
        pl.ancestral_linkage(tdata)


def test_ticklabels_toggle(linkage_tdata):
    ax = pl.ancestral_linkage(linkage_tdata, groupby="celltype", xticklabels=False, yticklabels=False)
    assert [t.get_text() for t in ax.get_xticklabels()] == []
    assert [t.get_text() for t in ax.get_yticklabels()] == []
    plt.close("all")


def test_labelsize(linkage_tdata):
    ax = pl.ancestral_linkage(linkage_tdata, groupby="celltype", labelsize=4)
    assert ax.get_xticklabels()[0].get_fontsize() == 4
    plt.close("all")


def test_data_override(linkage_tdata):
    stats = linkage_tdata.uns["celltype_linkage_stats"]
    ax = pl.ancestral_linkage(linkage_tdata, data=stats)
    assert isinstance(ax, plt.Axes)
    plt.close("all")


def test_accepts_existing_ax(linkage_tdata):
    fig, ax = plt.subplots()
    out = pl.ancestral_linkage(linkage_tdata, groupby="celltype", ax=ax, cbar=False)
    assert out is ax
    plt.close("all")


def test_by_tree_tree_selection():
    """by_tree stats can be plotted for a single tree or averaged across trees."""
    t1 = nx.DiGraph()
    for n, d in [("r1", 0.0), ("n1", 0.5), ("a1", 1.0), ("b1", 1.0)]:
        t1.add_node(n, depth=d)
    t1.add_edges_from([("r1", "n1"), ("n1", "a1"), ("n1", "b1")])
    t2 = nx.DiGraph()
    for n, d in [("r2", 0.0), ("n2", 0.5), ("a2", 1.0), ("b2", 1.0)]:
        t2.add_node(n, depth=d)
    t2.add_edges_from([("r2", "n2"), ("n2", "a2"), ("n2", "b2")])
    obs = pd.DataFrame({"celltype": ["A", "B", "A", "B"]}, index=["a1", "b1", "a2", "b2"])
    tdata = td.TreeData(obs=obs, obst={"tree1": t1, "tree2": t2})
    tl.ancestral_linkage(tdata, groupby="celltype", by_tree=True)

    ax = pl.ancestral_linkage(tdata, groupby="celltype", normalize=False, tree="tree1")
    assert isinstance(ax, plt.Axes)
    ax = pl.ancestral_linkage(tdata, groupby="celltype", normalize=False)  # averaged
    assert isinstance(ax, plt.Axes)
    with pytest.raises(ValueError, match="not found"):
        pl.ancestral_linkage(tdata, groupby="celltype", normalize=False, tree="tree9")
    plt.close("all")


def test_plot_matches_expected(linkage_tdata):
    """Smoke image regression."""
    ax = pl.ancestral_linkage(linkage_tdata, groupby="celltype")
    ax.get_figure().savefig(plot_path / "ancestral_linkage.png")
    plt.close("all")
    assert (plot_path / "ancestral_linkage.png").exists()
