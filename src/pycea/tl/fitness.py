from __future__ import annotations

import math
from collections import defaultdict
from typing import Literal, overload

import networkx as nx
import pandas as pd
import treedata as td

from pycea.get import node_df
from pycea.utils import check_tree_has_key, get_trees

from ._utils import _check_tree_overlap


def _infer_fitness_lbi(
    G: nx.DiGraph,
    depth_key: str,
    tau: float | None = None,
    fitness_attr: str = "fitness",
    zscore: bool = False,
) -> dict:
    """Compute Local Branching Index for all nodes and set attribute."""
    if len(G) == 0:
        return {}
    for n in G.nodes:
        if depth_key not in G.nodes[n]:
            raise ValueError(f"Node {n!r} is missing required '{depth_key}' attribute.")
    roots = [n for n in G.nodes if G.in_degree(n) == 0]
    if len(roots) != 1:
        raise ValueError(f"Tree must have exactly one root (found {len(roots)}).")
    root = roots[0]
    parent: dict = {}
    children: defaultdict = defaultdict(list)
    try:
        order = list(nx.topological_sort(G))
    except nx.NetworkXUnfeasible as e:  # pragma: no cover - networkx errors
        raise ValueError("Graph must be a DAG (a rooted tree/DAG).") from e
    if order[0] != root:
        order = list(nx.bfs_tree(G, root).nodes())
    time = {n: float(G.nodes[n][depth_key]) for n in G.nodes}
    b = {root: 0.0}
    for u in order:
        for v in G.successors(u):
            parent[v] = u
            children[u].append(v)
            b[v] = abs(time[v] - time[u])
    depth = {root: 0.0}
    for u in order:
        for v in children[u]:
            depth[v] = depth[u] + b[v]
    leaves = [n for n in G.nodes if G.out_degree(n) == 0]
    if not leaves:
        return {}

    def lca(u, v):
        seen = set()
        a = u
        while True:
            seen.add(a)
            if a == root:
                break
            a = parent.get(a)
            if a is None:
                break
        bnode = v
        while bnode not in seen:
            bnode = parent.get(bnode)
            if bnode is None:
                return root
        return bnode

    def mean_pairwise_patristic(nodes):
        m, s = 0, 0.0
        L = len(nodes)
        if L < 2:
            return 0.0
        for i in range(L):
            for j in range(i + 1, L):
                u, v = nodes[i], nodes[j]
                a = lca(u, v)
                dist = (depth[u] - depth[a]) + (depth[v] - depth[a])
                s += dist
                m += 1
        return s / m

    if tau is None:
        mean_pair = mean_pairwise_patristic(leaves)
        tau = 0.0625 * mean_pair if mean_pair > 0 else 1e-6
    if tau <= 0:
        raise ValueError("tau must be > 0")

    post = list(reversed(order))
    m_up: dict = {}
    for i in post:
        sum_child_up = sum(m_up[c] for c in children[i]) if children[i] else 0.0
        bi = b.get(i, 0.0)
        e = math.exp(-bi / tau) if bi > 0 else 1.0
        m_up[i] = tau * (1.0 - e) + e * sum_child_up

    m_down = {root: 0.0}
    sum_up_children = {i: sum(m_up.get(c, 0.0) for c in children[i]) for i in G.nodes}
    for i in order:
        for c in children[i]:
            bc = b[c]
            e = math.exp(-bc / tau) if bc > 0 else 1.0
            siblings_contrib = sum_up_children[i] - m_up[c]
            m_down[c] = tau * (1.0 - e) + e * (m_down[i] + siblings_contrib)

    lbi = {i: m_down[i] + sum_up_children[i] for i in G.nodes}
    if zscore and len(leaves) >= 2:
        vals = [lbi[i] for i in leaves]
        mu = sum(vals) / len(vals)
        var = sum((x - mu) ** 2 for x in vals) / (len(vals) - 1)
        sd = math.sqrt(var) if var > 0 else 1.0
        for i in leaves:
            lbi[i] = (lbi[i] - mu) / sd

    nx.set_node_attributes(G, lbi, fitness_attr)
    return lbi


@overload
def fitness(
    tdata: td.TreeData,
    tree: str,
    depth_key: str = "depth",
    key_added: str = "fitness",
    copy: Literal[True] = True,
) -> pd.DataFrame: ...


@overload
def fitness(
    tdata: td.TreeData,
    tree: str,
    depth_key: str = "depth",
    key_added: str = "fitness",
    copy: Literal[False] = False,
) -> None: ...


def fitness(
    tdata: td.TreeData,
    tree: str,
    depth_key: str = "depth",
    key_added: str = "fitness",
    copy: bool = False,
) -> pd.DataFrame | None:
    """
    Infer node fitness using Local Branching Index.

    Parameters
    ----------
    tdata
        TreeData object.
    tree
        Key identifying the tree in `tdata.obst`.
    depth_key
        Node attribute storing depth.
    key_added
        Attribute name to store inferred fitness.
    copy
        If True, return a DataFrame with node fitness.

    Returns
    -------
    node_df - DataFrame of node fitness if `copy` is True, otherwise `None`.
    """
    tree_keys = tree
    _check_tree_overlap(tdata, tree_keys)
    trees = get_trees(tdata, tree_keys)
    if len(trees) != 1:
        raise ValueError("`tree` must refer to a single tree.")
    key, G = next(iter(trees.items()))
    check_tree_has_key(G, depth_key)
    scores = _infer_fitness_lbi(G, depth_key=depth_key, fitness_attr=key_added)

    leaves = [n for n in G.nodes if G.out_degree(n) == 0]
    obs_scores = {n: scores[n] for n in leaves if n in tdata.obs_names}
    if obs_scores:
        tdata.obs.loc[list(obs_scores.keys()), key_added] = list(obs_scores.values())

    if copy:
        df = node_df(tdata, tree=key)
        return df[[key_added]]
    return None
