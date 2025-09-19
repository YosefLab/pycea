from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Literal, overload

import networkx as nx
import numpy as np
import pandas as pd
import treedata as td
from scipy.interpolate import interp1d

from pycea.utils import (
    _check_tree_overlap,
    check_tree_has_key,
    get_keyed_leaf_data,
    get_keyed_node_data,
    get_leaves,
    get_root,
    get_trees,
)

from ._metrics import _path_distance
from ._utils import _set_random_state
from .tree_distance import _tree_distance

non_negativity_cutoff = 1e-20


def _integrate_rk4(df, f0, T, dt, args):
    """Self made fixed time step rk4 integration."""
    sol = np.zeros([len(T)] + list(f0.shape))
    sol[0] = f0
    f = f0.copy()
    t = T[0]
    for ti, tnext in enumerate(T[1:]):
        while t < tnext:
            h = min(dt, tnext - t)
            k1 = df(f, t, *args)
            k2 = df(f + 0.5 * h * k1, t + 0.5 * h, *args)
            k3 = df(f + 0.5 * h * k2, t + 0.5 * h, *args)
            k4 = df(f + h * k3, t + h, *args)
            t += h
            f += h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
            f[f < non_negativity_cutoff] = non_negativity_cutoff
        sol[ti + 1] = f
    return sol


class survival_gen_func:
    """Survival generating function.

    Solves the generating function in the traveling fitness wave. These
    generating functions are used to calculate lineage propagators conditional
    on not having sampled other branches.
    """

    def __init__(self, fitness_grid=None):
        """Instantiate the class.

        Parameters
        ----------
        fitness_grid
            Discretization used for the solution of the ODEs.
        """
        if fitness_grid is None:
            self.fitness_grid = np.linspace(-5, 8, 101)
        else:
            self.fitness_grid = fitness_grid

        self.L = len(self.fitness_grid)
        # precompute values necessary for the numerical evalutation of the ODE
        self.dx = self.fitness_grid[1] - self.fitness_grid[0]
        self.dxinv = 1.0 / self.dx
        self.dxsqinv = self.dxinv**2
        # dictionary to save interpolation objects of the numerical solutions
        self.phi_solutions = {}

    def integrate_phi(self, gamma, eps, T, save_sol=True, dt=None):
        """Solve the equation for the generating function.

        Parameters
        ----------
        gamma
            Dimensionless diffusion constant. This is connected with the
            population size via ``v = (24 gamma^2 log N)**(1/3)``.
        eps
            Initial condition for the generating function.
        T
            treerid of times on which the generating function is to be evaluated.
        """
        phi0 = np.ones(self.L) * eps
        # if dt is not provided, use a heuristic that decreases with increasing diffusion constant
        if dt is None:
            dt = min(0.01, 0.001 / gamma)
        # set non-negative or very small values to non_negativity_cutoff
        sol = np.maximum(non_negativity_cutoff, _integrate_rk4(self.dphi, phi0, T, dt, args=(gamma,)))
        if save_sol:
            # produce and interpolation object to evaluate the solution at arbitrary time
            self.phi_solutions[(gamma, eps)] = interp1d(T, sol, axis=0)
        return sol

    def dphi(self, phi, t, gamma):
        """Time derivative of the generating function."""
        dp = np.zeros_like(phi)
        dp[1:-1] = (
            gamma * (phi[:-2] + phi[2:] - 2 * phi[1:-1]) * self.dxsqinv
            + (self.fitness_grid[1:-1]) * phi[1:-1]
            - phi[1:-1] ** 2
            - (phi[2:] - phi[:-2]) * 0.5 * self.dxinv
        )
        dp[0] = (
            0 * (phi[0] + phi[2] - 2 * phi[1]) * self.dxsqinv
            + (self.fitness_grid[0]) * phi[0]
            - phi[0] ** 2
            - (phi[1] - phi[0]) * self.dxinv
        )
        dp[-1] = (
            0 * (phi[-3] + phi[-1] - 2 * phi[-2]) * self.dxsqinv
            + (self.fitness_grid[-1]) * phi[-1]
            - phi[-1] ** 2
            - (phi[-1] - phi[-2]) * self.dxinv
        )
        return dp

    def integrate_prop(self, gamma, eps, x, t1, t2, dt=None):
        """Integrate the lineage propagator using RK4.

        Parameters
        ----------
        gamma
            Dimensionless diffusion constant.
        eps
            Initial condition for the generating function, corresponding to the sampling probability.
        x
            Fitness at the "closer to the present" end of the branch.
        t1
            Time closer to the present.
        t2
            Times after which to evaluate the propagator, either a float or iterable of floats.

        Returns
        -------
        np.ndarray - Propagator values.
        """
        if not np.iterable(t2):
            t2 = [t2]
        else:
            t2 = list(t2)

        if not np.iterable(x):
            x = [x]
        if dt is None:
            dt = min(0.05, 0.01 / gamma)

        sol = np.zeros((len(t2) + 1, self.L, len(x)))  # type: ignore
        prop0 = np.zeros((self.L, len(x)))  # type: ignore
        for ii, x_val in enumerate(x):
            xi = np.argmin(x_val > self.fitness_grid)
            prop0[xi, ii] = self.dxinv

        sol[:, :, :] = _integrate_rk4(self.dprop_backward, prop0, [t1] + t2, dt, args=((gamma, eps),))
        return np.maximum(non_negativity_cutoff, sol.swapaxes(1, 2))

    def dprop_backward(self, prop, t, params):
        """Time derivative of the propagator.

        Parameters
        ----------
        prop
            Value of the propagator.
        t
            Time to evaluate the generating function.
        params
            Parameters used to calculate the generating function ``(gamma, eps)``.
        """
        dp = np.zeros_like(prop)
        gamma = params[0]
        if params not in self.phi_solutions:
            raise ValueError("parameters not in phi_solutions")

        # evaluate at t if 1e-6 < t < T[-2], boundaries otherwise
        tmp_phi = self.phi_solutions[params](min(max(1e-6, t), self.phi_solutions[params].x[-2]))
        # if propagator is 2 dimensional, repeat the generating function along the missing axis
        if len(prop.shape) == 2:
            tmp_phi = tmp_phi.repeat(prop.shape[1]).reshape([-1, prop.shape[1]])
            fitness_grid = self.fitness_grid.repeat(prop.shape[1]).reshape([-1, prop.shape[1]])
        else:
            fitness_grid = self.fitness_grid
        dp[1:-1] = (
            gamma * (prop[:-2] + prop[2:] - 2 * prop[1:-1]) * self.dxsqinv
            + (fitness_grid[1:-1] - 2 * tmp_phi[1:-1]) * prop[1:-1]
            - (prop[2:] - prop[:-2]) * 0.5 * self.dxinv
        )
        dp[0] = (
            0 * (prop[0] + prop[2] - 2 * prop[1]) * self.dxsqinv
            + (fitness_grid[0] - 2 * tmp_phi[0]) * prop[0]
            - (prop[1] - prop[0]) * self.dxinv
        )
        dp[-1] = (
            0 * (prop[-3] + prop[-1] - 2 * prop[-2]) * self.dxsqinv
            + (fitness_grid[-1] - 2 * tmp_phi[-1]) * prop[-1]
            - (prop[-1] - prop[-2]) * self.dxinv
        )
        return dp


def _estimate_time_scale(tree, leaves, depth_key, sample_n):
    """Estimate time scale using sampled leaf pairs."""
    pairs = []
    n_leaves = len(leaves)
    if sample_n > n_leaves * (n_leaves - 1) / 2:
        sample_n = n_leaves * (n_leaves - 1) // 2
    for _ in range(sample_n):
        i, j = np.random.choice(n_leaves, size=2, replace=False)
        pairs.append((leaves[i], leaves[j]))
    distance = _tree_distance(tree, depth_key=depth_key, metric=_path_distance, pairs=pairs)[2]
    return float(np.mean(distance) / 2.0)


def _infer_fitness_sbd(
    tree: nx.DiGraph,
    gamma: float = 0.2,
    sample_frac: float = 0.01,
    depth_key: str = "depth",
    fit_grid: np.ndarray | None = None,
    eps_branch_length: float = 1e-7,
    time_scale: float | None = None,
    key_added: str = "fitness",
    attach_posteriors: bool = False,
    boundary_layer: int = 4,
    sample_n: int = 200,
) -> None:
    """
    Infer node fitness using survival branching dynamics.

    Parameters
    ----------
    tree
        Tree as a directed graph.
    gamma
        Dimensionless diffusion constant.
    sample_frac
        Sampling fraction.
    depth_key
        Node attribute storing depth.
    fit_grid
        Optional fitness grid for ODE solver.
    eps_branch_length
        Minimum branch length to avoid zero.
    time_scale
        Optional time-scale override.
    key_added
        Attribute name to store inferred fitness.
    attach_posteriors
        If True, attach posterior distributions to nodes.
    boundary_layer
        Number of grid points ignored at each boundary.
    sample_n
        Maximum number of leaf pairs for time-scale estimation.
    """
    # ---- initialization ----
    root = get_root(tree)
    preorder = list(nx.dfs_preorder_nodes(tree, root))
    leaves = get_leaves(tree)
    leaf_counts, mean_leaf_depth = {}, {}
    for n in reversed(preorder):
        if tree.out_degree(n) == 0:
            leaf_counts[n] = 1
            mean_leaf_depth[n] = tree.nodes[n][depth_key]
        else:
            s_count = sum(leaf_counts[c] for c in tree.successors(n))
            s_depth = sum(mean_leaf_depth[c] * leaf_counts[c] for c in tree.successors(n))
            leaf_counts[n] = s_count
            mean_leaf_depth[n] = s_depth / s_count
    time_to_present = {n: mean_leaf_depth[n] - tree.nodes[n][depth_key] for n in tree}
    if time_scale is None:
        time_scale = _estimate_time_scale(tree, leaves, depth_key, sample_n=sample_n) * gamma

    # ---- survival / propagators ----
    sgf = survival_gen_func(fit_grid if fit_grid is not None else None)
    fitness_grid = sgf.fitness_grid
    Lg = len(fitness_grid)
    bnd = boundary_layer

    T = np.concatenate([np.linspace(0, 10, 201), np.linspace(10, 200, 20)])
    sgf.integrate_phi(gamma, sample_frac, T)

    up_msg = {n: np.zeros(Lg) for n in tree.nodes}
    down_msg = {n: np.zeros(Lg) for n in tree.nodes}
    posterior = {n: np.zeros(Lg) for n in tree.nodes}
    propagator = {}

    down_msg[root] = np.exp(-0.5 * fitness_grid**2)  # type: ignore
    down_msg[root][down_msg[root] < non_negativity_cutoff] = non_negativity_cutoff

    # ---- UPWARD PASS ----
    for n in reversed(preorder):
        if tree.successors(n):
            lp = np.zeros(Lg)
            for c in tree.successors(n):
                lp += np.log(np.clip(up_msg[c], non_negativity_cutoff, None))
        else:
            lp = np.zeros(Lg)

        lp -= np.max(lp)
        p_node = np.exp(lp)
        s = p_node.sum()
        p_node = (p_node / s) if s > 0 else np.ones(Lg) / Lg

        t1 = time_to_present[n] / time_scale
        l = tree.nodes[n][depth_key] - tree.nodes[next(tree.predecessors(n))][depth_key] if n != root else 0.0
        t2 = (time_to_present[n] + (l + eps_branch_length)) / time_scale
        P = sgf.integrate_prop(gamma, sample_frac, fitness_grid[bnd:-bnd], t1, t2)[-1]
        propagator[n] = P

        p_x = p_node[bnd:-bnd]
        up = P.T @ p_x
        up[up < non_negativity_cutoff] = non_negativity_cutoff
        up_msg[n] = up

    # ---- DOWNWARD PASS ----
    for n in preorder:
        if not tree.successors(n):
            continue
        child_logs = [(c, np.log(np.clip(up_msg[c], non_negativity_cutoff, None))) for c in tree.successors(n)]
        log_sums = None
        for _, lc in child_logs:
            log_sums = lc if log_sums is None else (log_sums + lc)

        for c, log_c in child_logs:
            lp = np.log(np.clip(down_msg[n], non_negativity_cutoff, None)) + (log_sums - log_c)
            lp -= np.max(lp)
            p_parent = np.exp(lp)
            s = p_parent.sum()
            p_parent = (p_parent / s) if s > 0 else np.ones(Lg) / Lg

            dm = propagator[c] @ p_parent
            full = np.full(Lg, non_negativity_cutoff)
            full[bnd:-bnd] = np.clip(dm, non_negativity_cutoff, None)
            down_msg[c] = full

    # ---- MARGINALS ----
    means, vars_ = {}, {}
    for n in preorder:
        lp = np.log(np.clip(down_msg[n], non_negativity_cutoff, None))
        for c in tree.successors(n):
            lp += np.log(np.clip(up_msg[c], non_negativity_cutoff, None))
        lp -= np.max(lp)
        p = np.exp(lp)
        Z = p.sum()
        p = (p / Z) if Z > 0 else np.ones(Lg) / Lg
        posterior[n] = np.asarray(p).reshape(-1)

        mu = float(np.sum(fitness_grid * p))
        var = float(np.sum((fitness_grid**2) * p) - mu**2)
        means[n] = mu
        vars_[n] = max(var, 0.0)

    # ---- FINALIZE ----
    nx.set_node_attributes(tree, means, key_added)
    if attach_posteriors:
        nx.set_node_attributes(tree, posterior, f"{key_added}_posterior")
        nx.set_node_attributes(tree, vars_, f"{key_added}_var")


def _infer_fitness_lbi(
    tree: nx.DiGraph,
    depth_key: str,
    tau: float | None = None,
    key_added: str = "fitness",
    sample_n: int = 200,
) -> None:
    """
    Compute Local Branching Index for all nodes and set attribute.

    Parameters
    ----------
    tree
        Tree as a directed graph.
    depth_key
        Node attribute storing depth.
    tau
        Time scale. If ``None`` it is estimated from the tree.
    key_added
        Attribute name to store inferred fitness.
    sample_n
        Number of leaf pairs to use for time-scale estimation.
    """
    # ---- initialization ----
    root = get_root(tree)
    leaves = get_leaves(tree)
    post = list(nx.dfs_postorder_nodes(tree, root))
    if tau is None:
        time_scale = _estimate_time_scale(tree, leaves, depth_key, sample_n)
        tau = 0.125 * time_scale if time_scale > 0 else 1e-6

    # ---- UPWARD PASS ----
    m_up: dict = {}
    for i in post:
        sum_child_up = sum(m_up[c] for c in tree.successors(i)) if tree.successors(i) else 0.0
        bi = tree.nodes[i][depth_key] - tree.nodes[next(tree.predecessors(i))][depth_key] if i != root else 0.0
        e = math.exp(-bi / tau) if bi > 0 else 1.0
        m_up[i] = tau * (1.0 - e) + e * sum_child_up

    # ---- DOWNWARD PASS ----
    m_down = {root: 0.0}
    sum_up_children = {i: sum(m_up.get(c, 0.0) for c in tree.successors(i)) for i in tree.nodes}
    for i in reversed(post):
        for c in tree.successors(i):
            bc = tree.nodes[c][depth_key] - tree.nodes[i][depth_key]
            e = math.exp(-bc / tau) if bc > 0 else 1.0
            siblings_contrib = sum_up_children[i] - m_up[c]
            m_down[c] = tau * (1.0 - e) + e * (m_down[i] + siblings_contrib)

    # ---- FINALIZE ----
    lbi = {i: m_down[i] + sum_up_children[i] for i in tree.nodes}
    nx.set_node_attributes(tree, lbi, key_added)


@overload
def fitness(
    tdata: td.TreeData,
    depth_key: str = "depth",
    key_added: str = "fitness",
    method: Literal["sbd", "lbi"] = "sbd",
    method_kwargs: Mapping | None = None,
    sample_n: int = 200,
    tree: str | list[str] | None = None,
    random_state: int | None = None,
    copy: Literal[True, False] = True,
) -> pd.DataFrame: ...


@overload
def fitness(
    tdata: td.TreeData,
    depth_key: str = "depth",
    key_added: str = "fitness",
    method: Literal["sbd", "lbi"] = "sbd",
    method_kwargs: Mapping | None = None,
    sample_n: int = 200,
    tree: str | list[str] | None = None,
    random_state: int | None = None,
    copy: Literal[True, False] = False,
) -> None: ...


def fitness(
    tdata: td.TreeData,
    depth_key: str = "depth",
    key_added: str = "fitness",
    method: Literal["sbd", "lbi"] = "sbd",
    method_kwargs: Mapping | None = None,
    sample_n: int = 200,
    tree: str | list[str] | None = None,
    random_state: int | None = None,
    copy: Literal[True, False] = False,
) -> pd.DataFrame | None:
    """
    Estimates node fitness.

    This function implements two algorithms proposed by :cite:p:`Neher_2014` for estimating
    relative fitness from the tree topology and branch lengths:

    * ``method="sbd"``
        Selection-Biased Diffusion (SBD), a message-passing algorithm that propagates
        information up and down the tree to infer posterior distributions of fitness
        at each node. This corresponds to the probabilistic framework described by
        Neher et al. (2014) and yields posterior mean fitness values for the tree's
        nodes.

    * ``method="lbi"``
        Local Branching Index (LBI), a heuristic that measures node fitness based on
        the density of branching in its local neighborhood. Higher LBI values
        correspond to nodes with more prolific descendant lineages.

    Parameters
    ----------
    tdata
        TreeData object.
    tree
        Key identifying the tree in ``tdata.obst``. If ``None`` use all trees.
    depth_key
        Node attribute storing depth.
    key_added
        Attribute name to store inferred fitness.
    method
        Method to use for fitness inference.

        - `'sbd'`: Selection-Biased Diffusion.
        - `'lbi'`: Local Branching Index.
    method_kwargs
        Additional keyword arguments passed to the selected method. For example:

        - `gamma` (float, default=0.2): Dimensionless diffusion constant (for SBD).
        - `attach_posteriors` (bool, default=False): If True, attach posterior distributions to nodes (for SBD).
        - `tau` (float, default=None): Time scale (for LBI).
    sample_n
        Number of leaf pairs to use for time-scale estimation.
    tree
        The `obst` key or keys of the trees to use. If `None`, all trees are used.
    random_state
        Random seed.
    copy
        If ``True``, return a DataFrame with node fitness.

    Returns
    -------
    Returns `None` if ``copy=False``, otherwise returns a :class:`pandas.DataFrame` with fitness values.

    Sets the following fields:

    * tdata.obst[tree].nodes[key_added] : `float`
        - Inferred fitness values for each node.
    * tdata.obs[key_added] : `float`
        - Inferred fitness values for each leaf.
    """
    tree_keys = tree
    _check_tree_overlap(tdata, tree_keys)
    _set_random_state(random_state)
    trees = get_trees(tdata, tree_keys)
    for t in trees.values():
        check_tree_has_key(t, depth_key)
        if method == "sbd":
            _infer_fitness_sbd(t, depth_key=depth_key, key_added=key_added, sample_n=sample_n, **(method_kwargs or {}))
        elif method == "lbi":
            _infer_fitness_lbi(t, depth_key=depth_key, key_added=key_added, sample_n=sample_n, **(method_kwargs or {}))
        else:
            raise ValueError(f"method {method!r} not recognized, use 'sbd' or 'lbi'")
    leaf_fitness = get_keyed_leaf_data(tdata, key_added, tree_keys)
    tdata.obs[key_added] = tdata.obs.index.map(leaf_fitness[key_added])
    if copy:
        return get_keyed_node_data(tdata, key_added, tree_keys)
    return None
    return None
