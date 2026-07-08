from __future__ import annotations

import itertools
from collections.abc import Callable, Sequence
from typing import Any, Literal, overload

import networkx as nx
import numpy as np
import pandas as pd
import treedata as td

from pycea.utils import _check_tree_overlap, check_tree_has_key, get_keyed_obs_data, get_root, get_trees

from .ancestral_states import _fitch_hartigan_downpass, ancestral_states


@overload
def parsimony(
    tdata: td.TreeData,
    key: str,
    method: str | Callable = "fitch_hartigan",
    missing_state: str | None = None,
    costs: pd.DataFrame | None = None,
    reconstruct: bool = True,
    tree: str | Sequence[str] | None = None,
    key_added: str = "parsimony",
    copy: Literal[True] = True,
) -> int | pd.Series: ...
@overload
def parsimony(
    tdata: td.TreeData,
    key: str,
    method: str | Callable = "fitch_hartigan",
    missing_state: str | None = None,
    costs: pd.DataFrame | None = None,
    reconstruct: bool = True,
    tree: str | Sequence[str] | None = None,
    key_added: str = "parsimony",
    copy: Literal[False] = False,
) -> None: ...
def parsimony(
    tdata: td.TreeData,
    key: str,
    method: str | Callable = "fitch_hartigan",
    missing_state: str | None = None,
    costs: pd.DataFrame | None = None,
    reconstruct: bool = True,
    tree: str | Sequence[str] | None = None,
    key_added: str = "parsimony",
    copy: Literal[True, False] = True,
) -> int | pd.Series | None:
    """Computes the small-parsimony score of a tree.

    The parsimony score is the number of edges along which the state of a
    categorical attribute changes. Ancestral states can be reconstructed on the
    fly (default) or read from states already present on the tree.

    Parameters
    ----------
    tdata
        TreeData object.
    key
        An `obs.keys()` corresponding to a categorical variable.
    method
        Method used to reconstruct ancestral states when ``reconstruct=True``.
        Passed to :func:`pycea.tl.ancestral_states`. Typically ``'fitch_hartigan'``
        or ``'sankoff'``.
    missing_state
        The state to consider as missing data. Edges touching a missing state are
        not counted.
    costs
        A :class:`DataFrame <pandas.DataFrame>` with the costs of changing states
        (from rows to columns). Only used if ``method='sankoff'``.
    reconstruct
        If True, reconstruct ancestral states with :func:`pycea.tl.ancestral_states`
        before scoring. If False, use states already stored under ``key`` on the tree.
    tree
        The `obst` key or keys of the trees to use. If `None`, all trees are used.
    key_added
        Key in ``tdata.uns`` where the parsimony score(s) are stored.
    copy
        If True, returns the parsimony score(s).

    Returns
    -------
    Returns `None` if `copy=False`. Otherwise returns an ``int`` for a single tree,
    or a :class:`Series <pandas.Series>` of scores indexed by tree for multiple trees.

    Sets the following fields:

    * `tdata.uns[key_added]` : `int` | :class:`Series <pandas.Series>`
        - Parsimony score for a single tree, or a Series of scores for multiple trees.

    Examples
    --------
    Compute the parsimony score of a categorical character:

    >>> py.tl.parsimony(tdata, key="clone")
    """
    tree_keys = tree
    _check_tree_overlap(tdata, tree_keys)
    trees = get_trees(tdata, tree_keys)
    if reconstruct:
        ancestral_states(tdata, keys=key, method=method, missing_state=missing_state, costs=costs, tree=tree_keys)
    scores = {}
    for name, t in trees.items():
        if not reconstruct:
            check_tree_has_key(t, key)
        score = 0
        for parent, child in nx.dfs_edges(t, source=get_root(t)):
            parent_state = t.nodes[parent].get(key)
            child_state = t.nodes[child].get(key)
            if parent_state is None or child_state is None:
                continue
            if missing_state is not None and (parent_state == missing_state or child_state == missing_state):
                continue
            if parent_state != child_state:
                score += 1
        scores[name] = score
    result: int | pd.Series
    result = next(iter(scores.values())) if len(scores) == 1 else pd.Series(scores, name=key_added)
    tdata.uns[key_added] = result
    if copy:
        return result


@overload
def fitch_count(
    tdata: td.TreeData,
    key: str,
    states: Sequence[str] | None = None,
    missing_state: str | None = None,
    root: str | None = None,
    tree: str | Sequence[str] | None = None,
    key_added: str = "fitch_count",
    copy: Literal[True] = True,
) -> pd.DataFrame: ...
@overload
def fitch_count(
    tdata: td.TreeData,
    key: str,
    states: Sequence[str] | None = None,
    missing_state: str | None = None,
    root: str | None = None,
    tree: str | Sequence[str] | None = None,
    key_added: str = "fitch_count",
    copy: Literal[False] = False,
) -> None: ...
def fitch_count(
    tdata: td.TreeData,
    key: str,
    states: Sequence[str] | None = None,
    missing_state: str | None = None,
    root: str | None = None,
    tree: str | Sequence[str] | None = None,
    key_added: str = "fitch_count",
    copy: Literal[True, False] = True,
) -> pd.DataFrame | None:
    """Runs the FitchCount algorithm.

    Performs the FitchCount algorithm for inferring the number of times that two
    states transition to one another across all equally-parsimonious solutions
    returned by the Fitch-Hartigan algorithm. The original algorithm was described
    in :cite:`Quinn_2021`. The output is an MxM count matrix, where the values
    indicate the number of times that ``m1`` transitioned to ``m2`` along an edge in
    a Fitch-Hartigan solution. To obtain probabilities ``P(m1 -> m2)``, divide each
    row by its row-sum.

    This procedure will only work on categorical data.

    Parameters
    ----------
    tdata
        TreeData object.
    key
        An `obs.keys()` corresponding to a categorical variable.
    states
        State space that can be optionally provided by the user. If not provided,
        the unique non-missing values in ``tdata.obs[key]`` are used.
    missing_state
        The state to consider as missing data. Missing leaves are treated as
        wildcards (may take on any state in ``states``).
    root
        Node to treat as the root. Only the subtree below this node is considered.
        Only valid when a single tree is used.
    tree
        The `obst` key or keys of the trees to use. If `None`, all trees are used.
    key_added
        Key in ``tdata.uns`` where the count matrix is stored.
    copy
        If True, returns the count matrix.

    Returns
    -------
    Returns `None` if `copy=False`. Otherwise returns an MxM
    :class:`DataFrame <pandas.DataFrame>`. When multiple trees are used, the
    per-tree count matrices are summed into a single matrix.

    Sets the following fields:

    * `tdata.uns[key_added]` : :class:`DataFrame <pandas.DataFrame>`
        - Count matrix, summed across all trees used.

    Examples
    --------
    Count state transitions across equally-parsimonious solutions:

    >>> py.tl.fitch_count(tdata, key="tissue", copy=True)
    """
    tree_keys = tree
    _check_tree_overlap(tdata, tree_keys)
    trees = get_trees(tdata, tree_keys)
    if root is not None and len(trees) > 1:
        raise ValueError("`root` can only be specified when a single tree is used.")
    data, _, _ = get_keyed_obs_data(tdata, [key])
    values = data[key]
    observed = pd.unique(values[values != missing_state].dropna())
    if states is None:
        states = list(observed)
    else:
        states = list(states)
        if len(np.setdiff1d(np.asarray(observed), np.asarray(states))) > 0:
            raise ValueError("Specified `states` do not span the set of states that appear in the data.")

    matrices = []
    for _name, t in trees.items():
        g = t
        if root is not None:
            g = nx.DiGraph(g.subgraph(nx.descendants(g, root) | {root}).copy())
        actual_root = root if root is not None else get_root(g)
        # Set leaf states then compute Fitch-Hartigan state sets. Pandas NaNs are
        # coerced to the missing sentinel so missing leaves are treated as wildcards.
        leaf_states = {}
        for node in g.nodes:
            if g.out_degree(node) == 0 and node in values.index:
                val = values[node]
                leaf_states[node] = missing_state if pd.isna(val) else val
        nx.set_node_attributes(g, leaf_states, key)
        _fitch_hartigan_downpass(g, key, missing_state, set_key="_fitch_set")
        # Treat missing (wildcard) sets as the full state space
        for node in g.nodes:
            if g.nodes[node]["_fitch_set"] == missing_state:
                g.nodes[node]["_fitch_set"] = set(states)

        bfs_nodes = [actual_root] + [v for _, v in nx.bfs_edges(g, actual_root)]
        node_to_i = dict(zip(bfs_nodes, range(len(bfs_nodes)), strict=False))
        label_to_j = dict(zip(states, range(len(states)), strict=False))

        N = _N_fitch_count(g, states, node_to_i, label_to_j, "_fitch_set")
        C = _C_fitch_count(g, N, states, node_to_i, label_to_j, "_fitch_set")

        M = pd.DataFrame(np.zeros((len(states), len(states))), index=states, columns=states)
        for s1 in states:
            for s2 in states:
                M.loc[s1, s2] = np.sum(C[node_to_i[actual_root], :, label_to_j[s1], label_to_j[s2]])
        matrices.append(M)

    result = sum(matrices)
    tdata.uns[key_added] = result
    if copy:
        return result  # type: ignore


def _N_fitch_count(
    g: nx.DiGraph,
    unique_states: Sequence[str],
    node_to_i: dict[Any, int],
    label_to_j: dict[str, int],
    state_key: str = "S1",
) -> np.ndarray:
    """Fill in the dynamic programming table N for FitchCount.

    Computes N[v, s], corresponding to the number of solutions below
    a node v in the tree given v takes on the state s.

    Parameters
    ----------
    g
        Directed graph with state_key attribute set on all nodes.
    unique_states
        The state space that a node can take on.
    node_to_i
        Mapping of each node to a unique integer.
    label_to_j
        Mapping of each unique state to a unique integer.
    state_key
        Node attribute storing the possible states for each node.

    Returns
    -------
    A 2-dimensional array storing N[v, s].
    """

    def _fill(v: str, s: str) -> float:
        if g.out_degree(v) == 0:
            return 1
        children = list(g.successors(v))
        A = np.zeros(len(children))
        for i, u in enumerate(children):
            if s not in g.nodes[u][state_key]:
                legal_states = g.nodes[u][state_key]
            else:
                legal_states = [s]
            A[i] = np.sum([N[node_to_i[u], label_to_j[sp]] for sp in legal_states])
        return float(np.prod(A))

    N = np.full((len(g.nodes), len(unique_states)), 0.0)
    root = next(n for n in g.nodes if g.in_degree(n) == 0)
    for n in nx.dfs_postorder_nodes(g, source=root):
        for s in g.nodes[n][state_key]:
            N[node_to_i[n], label_to_j[s]] = _fill(n, s)

    return N


def _C_fitch_count(
    g: nx.DiGraph,
    N: np.ndarray,
    unique_states: Sequence[str],
    node_to_i: dict[Any, int],
    label_to_j: dict[str, int],
    state_key: str = "S1",
) -> np.ndarray:
    """Fill in the dynamic programming table C for FitchCount.

    Computes C[v, s, s1, s2], the number of transitions from state s1 to
    state s2 in the subtree rooted at v, given that state v takes on the
    state s.

    Parameters
    ----------
    g
        Directed graph with state_key attribute set on all nodes.
    N
        N array computed during FitchCount.
    unique_states
        The state space that a node can take on.
    node_to_i
        Mapping of each node to a unique integer.
    label_to_j
        Mapping of each unique state to a unique integer.
    state_key
        Node attribute storing the possible states for each node.

    Returns
    -------
    A 4-dimensional array storing C[v, s, s1, s2].
    """

    def _fill(v: str, s: str, s1: str, s2: str) -> float:
        if g.out_degree(v) == 0:
            return 0

        children = list(g.successors(v))
        A = np.zeros(len(children))
        LS = [[]] * len(children)

        for i, u in enumerate(children):
            if s in g.nodes[u][state_key]:
                LS[i] = [s]
            else:
                LS[i] = g.nodes[u][state_key]

            A[i] = np.sum([C[node_to_i[u], label_to_j[sp], label_to_j[s1], label_to_j[s2]] for sp in LS[i]])

            if s1 == s and s2 in LS[i]:
                A[i] += N[node_to_i[u], label_to_j[s2]]

        parts = []
        for i, u in enumerate(children):
            prod = 1
            for k, up in enumerate(children):
                if up == u:
                    continue
                prod *= sum(N[node_to_i[up], label_to_j[sp]] for sp in LS[k])
            parts.append(A[i] * prod)

        return np.sum(parts)

    C = np.zeros((len(g.nodes), N.shape[1], N.shape[1], N.shape[1]))
    root = next(n for n in g.nodes if g.in_degree(n) == 0)
    for n in nx.dfs_postorder_nodes(g, source=root):
        for s in g.nodes[n][state_key]:
            for s1, s2 in itertools.product(unique_states, repeat=2):
                C[node_to_i[n], label_to_j[s], label_to_j[s1], label_to_j[s2]] = _fill(n, s, s1, s2)

    return C
