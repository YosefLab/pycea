from __future__ import annotations

import math
from collections import defaultdict
from typing import Literal, overload

import networkx as nx
import numpy as np
import pandas as pd
import treedata as td
from scipy.integrate import odeint
from scipy.interpolate import interp1d

from pycea.utils import check_tree_has_key, get_keyed_leaf_data, get_keyed_node_data, get_trees

from ._utils import _check_tree_overlap

non_negativity_cutoff = 1e-20


def integrate_rk4(df, f0, T, dt, args):
    """
    Self made fixed time step rk4 integration
    """
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
    """
    solves for the generating function of the copy number distribution of lineages
    in the traveling fitness wave. These generating functions are used to calculate
    lineage propagators conditional on not having sampled other branches.
    """

    def __init__(self, fitness_grid=None):
        """
        Instantiates the class
        arguments:
            fitness_grid    discretization used for the solution of the ODEs
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

    def integrate_phi(self, D, eps, T, save_sol=True, dt=None):
        """
        Solve the equation for the generating function
        arguments:
        D   -- dimensionless diffusion constant. this is connected with the popsize
               via v = (24 D^2 log N)**(1/3)
        eps -- initial condition for the generating function
        T   -- grid of times on which the generating function is to be evaluated
        """
        phi0 = np.ones(self.L) * eps
        # if dt is not provided, use a heuristic that decreases with increasing diffusion constant
        if dt is None:
            dt = min(0.01, 0.001 / D)

        # set non-negative or very small values to non_negativity_cutoff
        sol = np.maximum(non_negativity_cutoff, integrate_rk4(self.dphi, phi0, T, dt, args=(D,)))
        if save_sol:
            # produce and interpolation object to evaluate the solution at arbitrary time
            self.phi_solutions[(D, eps)] = interp1d(T, sol, axis=0)
        return sol

    def dphi(self, phi, t, D):
        """
        Time derivative of the generating function
        """
        dp = np.zeros_like(phi)
        dp[1:-1] = (
            D * (phi[:-2] + phi[2:] - 2 * phi[1:-1]) * self.dxsqinv
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

    def integrate_prop_odeint(self, D, eps, x, t1, t2):
        """
        Integrate the lineage propagator, accounting for non branching. THIS USES THE SCIPY ODE INTEGRATOR
        parameters:
        D   --  dimensionless diffusion constant
        eps --  initial condition for the generating function, corresponding to the sampling probability
        x   --  fitness at the "closer to the present" end of the branch
        t1  --  time closer to the present
        t2  --  times after which to evaluate the propagator, either a float or iterable of floats
        """
        if not np.iterable(t2):  # if only one value is provided, produce a list with this value
            t2 = [t2]
        else:  # otherwise, cast to list. This is necessary to concatenate with with t1
            t2 = list(t2)

        if np.iterable(x):
            xdim = len(x)
        else:
            xdim = 1
            x = [x]

        # allocate array for solution: dimensions: #time points, #late fitness values, #fitness grid points
        sol = np.zeros((len(t2) + 1, len(x), self.L))
        # loop over late fitness values
        for ii, x_val in enumerate(x):
            # find index in fitness grid
            xi = np.argmin(x_val > self.fitness_grid)
            # init as delta function, normized
            prop0 = np.zeros(self.L)
            prop0[xi] = self.dxinv
            # propagate backwards and save in correct row
            sol[:, ii, :] = odeint(
                self.dprop_backward,
                prop0,
                [t1] + t2,
                args=((D, eps),),
                rtol=0.001,
                atol=1e-5,
                h0=1e-2,
                hmin=1e-4,
                printmessg=False,
            )

        return np.maximum(non_negativity_cutoff, sol)

    def integrate_prop(self, D, eps, x, t1, t2, dt=None):
        """
        Integrate the lineage propagator, accounting for non branching. THIS USES THE SELF-MADE RK4.
        this is usually superior odeint, since we are integrating a system of ODEs with very different
        convergence criteria along the vector
        parameters:
        D   --  dimensionless diffusion constant
        eps --  initial condition for the generating function, corresponding to the sampling probability
        x   --  fitness at the "closer to the present" end of the branch
        t1  --  time closer to the present
        t2  --  times after which to evaluate the propagator, either a float or iterable of floats
        """
        if not np.iterable(t2):
            t2 = [t2]
        else:
            t2 = list(t2)

        if np.iterable(x):
            xdim = len(x)
        else:
            xdim = 1
            x = [x]
        # allocate array for solution: dimensions: #time points, #late fitness values, #fitness grid points
        if dt is None:
            dt = min(0.05, 0.01 / D)

        sol = np.zeros((len(t2) + 1, self.L, len(x)))
        # loop over late fitness values
        prop0 = np.zeros((self.L, len(x)))
        for ii, x_val in enumerate(x):
            # find index in fitness grid
            xi = np.argmin(x_val > self.fitness_grid)
            # init as delta function, normalized
            prop0[xi, ii] = self.dxinv

        # solve the entire matrix simultaneously.
        sol[:, :, :] = integrate_rk4(self.dprop_backward, prop0, [t1] + t2, dt, args=((D, eps),))
        return np.maximum(non_negativity_cutoff, sol.swapaxes(1, 2))

    def dprop_backward(self, prop, t, params):
        """
        Time derivative of the propagator.
        arguments:
            prop    value of the propagator
            t       time to evaluate the generating function
            params  parameters used to calculate the generating function (D,eps)
        """
        dp = np.zeros_like(prop)
        D = params[0]
        if params in self.phi_solutions:  # evaluate the generating function at the this time
            if t > self.phi_solutions[params].x[-1]:
                print("t larger than interpolation range! t=", t, ">", self.phi_solutions[params].x[-1])
            # evaluate at t if 1e-6<t<T[-2], boundaries otherwise
            tmp_phi = self.phi_solutions[params](min(max(1e-6, t), self.phi_solutions[params].x[-2]))
            # if propagator is 2 dimensional, repeat the generating function along the missing axis
            if len(prop.shape) == 2:
                tmp_phi = tmp_phi.repeat(prop.shape[1]).reshape([-1, prop.shape[1]])
                fitness_grid = self.fitness_grid.repeat(prop.shape[1]).reshape([-1, prop.shape[1]])
            else:
                fitness_grid = self.fitness_grid
        else:
            print("solve_survival.dprop_backwards: parameters not in phi_solutions")
            return None
        dp[1:-1] = (
            D * (prop[:-2] + prop[2:] - 2 * prop[1:-1]) * self.dxsqinv
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


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from matplotlib import cm

    mysol = survival_gen_func(np.linspace(-4, 8, 61))

    # calculate the generating function for different values of D and eps
    # the result is saved in mysol
    t_initial = 0.0
    for D in [0.2, 0.5]:
        plt.figure()
        plt.title("D=" + str(D))
        for eps, ls in zip([0.0001, 0.01, 1.0], ["-", "--", "-."], strict=False):
            sol = mysol.integrate_phi(D, eps, np.linspace(0, 20, 40))
            plt.plot(mysol.fitness_grid, mysol.phi_solutions[(D, eps)].y.T, ls=ls)

        plt.yscale("log")
        plt.xlabel("fitness")

    # use the precomputed generating functions to calculate the lineage propagators
    import time

    tstart = time.time()
    x_array = mysol.fitness_grid[5:-5]
    t_array = np.linspace(0.01, 7, 18)
    eps = 0.0001
    for D in [0.2]:  # , 0.5, 1.0]:
        sol_bwd = mysol.integrate_prop(D, eps, x_array, t_initial, t_initial + t_array)

        plt.figure()
        plt.suptitle("propagator D=" + str(D))
        for ti, t in enumerate(t_array[::2]):
            plt.subplot(3, 3, ti + 1)
            plt.title("t=" + str(np.round(t, 2)))
            plt.imshow(np.log(sol_bwd[2 * ti, :, :].T), interpolation="nearest")

        # plot mean offspring fitness as a function of time for difference ancestor fitness
        plt.figure()
        plt.title("Child fitness D=" + str(D) + " omega=" + str(eps))
        for yi in range(10, 50, 5):
            child_means = np.array([np.sum(sol_bwd[i, :, yi] * x_array) for i in range(sol_bwd.shape[0])])
            child_means /= np.sum(sol_bwd[:, :, yi], axis=1)
            child_std = np.array(
                [np.sum(sol_bwd[i, :, yi] * (x_array - child_means[i]) ** 2) for i in range(sol_bwd.shape[0])]
            )
            child_std /= np.sum(sol_bwd[:, :, yi], axis=1)
            child_std = np.sqrt(child_std)
            plt.plot(t_array, child_means[1:], ls="-", c=cm.jet(yi / 50.0), lw=2)
            plt.plot(t_array, child_std[1:], ls="--", c=cm.jet(yi / 50.0), lw=2)
        plt.xlabel("time to ancestor, child at t=" + str(t_initial))
        plt.ylabel("child mean fitness/std")

        # plot mean offspring fitness as a function of time for difference ancestor fitness
        plt.figure()
        plt.title("Ancestor fitness D=" + str(D) + " omega=" + str(eps))
        tmp_gauss = np.exp(-0 * mysol.fitness_grid**2 / 2)
        for xi in range(10, 45, 5):
            plt.plot(
                t_array,
                [
                    np.sum(sol_bwd[i, xi, :] * mysol.fitness_grid * tmp_gauss) / np.sum(sol_bwd[i, xi, :] * tmp_gauss)
                    for i in range(1, sol_bwd.shape[0])
                ],
            )
        plt.xlabel("time to ancestor, child at t=" + str(t_initial))
        plt.ylabel("ancestor fitness")
    print(np.round(time.time() - tstart, 3), "seconds")


def _infer_fitness_sbd(
    G: nx.DiGraph,
    D: float = 0.2,
    samp_frac: float = 0.01,
    depth_key: str = "depth",
    fit_grid: np.ndarray | None = None,
    eps_branch_length: float = 1e-7,
    time_scale: float | None = None,
    fitness_attr: str = "fitness",
    attach_posteriors: bool = False,
    boundary_layer: int = 4,
    wave_velocity: float = 1.0,
    zscore: bool = False,
    *,
    rng_seed: int = 0,
    num_pairs: int = 200,
) -> dict:
    if len(G) == 0:
        return {}

    for n in G.nodes:
        if depth_key not in G.nodes[n]:
            raise ValueError(f"Node {n!r} missing required '{depth_key}' attribute")

    roots = [n for n in G.nodes if G.in_degree(n) == 0]
    if len(roots) != 1:
        raise ValueError(f"Tree must have exactly one root (found {len(roots)})")
    root = roots[0]

    # ---- deterministic traversal & children order ----
    time = {n: float(G.nodes[n][depth_key]) for n in G.nodes}

    def sorted_successors(u):
        succ = list(G.successors(u))
        succ.sort(key=lambda x: (time[x], str(x)))
        return succ

    topo, seen = [], set()
    stack = [root]
    while stack:
        u = stack.pop()
        if u in seen:
            continue
        seen.add(u)
        topo.append(u)
        succ = sorted_successors(u)
        for v in reversed(succ):
            stack.append(v)

    parent = {}
    children = {n: [] for n in G.nodes}
    b = {root: 0.0}
    for u in topo:
        for v in sorted_successors(u):
            parent[v] = u
            children[u].append(v)
            # directed difference (no abs)
            b[v] = max(0.0, time[v] - time[u])

    leaves = [n for n in G.nodes if len(children[n]) == 0]
    if not leaves:
        return {}

    # ---- patristic depth from root ----
    depth = {root: 0.0}
    for u in topo:
        for v in children[u]:
            depth[v] = depth[u] + b[v]

    # ---- mean leaf depth per subtree & time-to-present ----
    def postorder_nodes():
        out, st = [], [(root, 0)]
        while st:
            n, state = st.pop()
            if state == 0:
                st.append((n, 1))
                for c in children[n]:
                    st.append((c, 0))
            else:
                out.append(n)
        return out

    leaf_counts, mean_leaf_depth = {}, {}
    for n in postorder_nodes():
        if not children[n]:
            leaf_counts[n] = 1
            mean_leaf_depth[n] = depth[n]
        else:
            s_count = sum(leaf_counts[c] for c in children[n])
            s_depth = sum(mean_leaf_depth[c] * leaf_counts[c] for c in children[n])
            leaf_counts[n] = s_count
            mean_leaf_depth[n] = s_depth / s_count

    time_to_present = {n: mean_leaf_depth[n] - depth[n] for n in G.nodes}

    # ---- LCA helper ----
    def lca(u, v):
        anc = set()
        a = u
        while True:
            anc.add(a)
            if a == root or a not in parent:
                break
            a = parent[a]
        bnode = v
        while bnode not in anc:
            if bnode == root or bnode not in parent:
                return root
            bnode = parent[bnode]
        return bnode

    # ---- seeded, sampled time_scale estimate (no replacement) ----
    def estimate_time_scale_sampled() -> float:
        L = len(leaves)
        if L < 2:
            return 1.0

        # total unordered pairs
        total_pairs = L * (L - 1) // 2
        sample_k = min(num_pairs, total_pairs)

        # map pair index -> (i,j) in lexicographic order, or sample indices directly
        if sample_k == total_pairs:
            # small case: compute all, still deterministic
            pairs = list(itertools.combinations(range(L), 2))
        else:
            # sample without replacement from the index set of pairs
            # Encode pair indices as triangular numbers for O(1) mapping
            # idx in [0, total_pairs)
            rng = np.random.default_rng(rng_seed)
            chosen = rng.choice(total_pairs, size=sample_k, replace=False)
            chosen.sort()  # deterministic order of evaluation
            pairs = []
            # map linear index -> (i,j) for upper-triangular enumeration
            # We invert triangular numbers: idx = i*(L-1) - i*(i+1)//2 + (j - i - 1)
            # Use a deterministic search for clarity (L is #leaves, typically modest)
            for idx in chosen:
                # find i such that cumulative pairs up to row i exceeds idx
                # cumulative up to (but not including) row i: cum(i) = i*(2L - i - 1)/2
                # do a small integer search; O(sqrt(total_pairs))
                lo, hi = 0, L - 1
                while lo < hi:
                    mid = (lo + hi) // 2
                    cum_mid = mid * (2 * L - mid - 1) // 2
                    if cum_mid <= idx:
                        lo = mid + 1
                    else:
                        hi = mid
                i = lo - 1
                cum_i = i * (2 * L - i - 1) // 2
                offset = idx - cum_i
                j = i + 1 + offset
                pairs.append((i, j))

        t2s = []
        for i, j in pairs:
            ui, vj = leaves[i], leaves[j]
            a = lca(ui, vj)
            t2s.append(min(depth[ui] - depth[a], depth[vj] - depth[a]))

        mean_t2 = float(np.mean(t2s)) if t2s else 1.0
        return (mean_t2 * D / wave_velocity) if mean_t2 > 0 else 1.0

    if time_scale is None:
        time_scale = estimate_time_scale_sampled()

    # ---- survival / propagators ----
    sgf = survival_gen_func(fit_grid if fit_grid is not None else None)
    fitness_grid = sgf.fitness_grid
    Lg = len(fitness_grid)
    bnd = boundary_layer

    T = np.concatenate([np.linspace(0, 10, 201), np.linspace(10, 200, 20)])
    sgf.integrate_phi(D, samp_frac, T)

    up_msg = {n: np.zeros(Lg) for n in G.nodes}
    down_msg = {n: np.zeros(Lg) for n in G.nodes}
    posterior = {n: np.zeros(Lg) for n in G.nodes}
    propagator = {}

    down_msg[root] = np.exp(-0.5 * fitness_grid**2)
    down_msg[root][down_msg[root] < non_negativity_cutoff] = non_negativity_cutoff

    # ---- UPWARD PASS ----
    for n in reversed(topo):
        if children[n]:
            lp = np.zeros(Lg)
            for c in children[n]:
                lp += np.log(np.clip(up_msg[c], non_negativity_cutoff, None))
        else:
            lp = np.zeros(Lg)

        lp -= np.max(lp)
        p_node = np.exp(lp)
        s = p_node.sum()
        p_node = (p_node / s) if s > 0 else np.ones(Lg) / Lg

        t1 = time_to_present[n] / time_scale
        t2 = (time_to_present[n] + (b.get(n, 0.0) + eps_branch_length)) / time_scale
        P = sgf.integrate_prop(D, samp_frac, fitness_grid[bnd:-bnd], t1, t2)[-1]
        propagator[n] = P

        p_x = p_node[bnd:-bnd]
        up = P.T @ p_x
        up[up < non_negativity_cutoff] = non_negativity_cutoff
        up_msg[n] = up

    # ---- DOWNWARD PASS ----
    for n in topo:
        if not children[n]:
            continue
        child_logs = [(c, np.log(np.clip(up_msg[c], non_negativity_cutoff, None))) for c in children[n]]
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
    for n in topo:
        lp = np.log(np.clip(down_msg[n], non_negativity_cutoff, None))
        for c in children[n]:
            lp += np.log(np.clip(up_msg[c], non_negativity_cutoff, None))
        lp -= np.max(lp)
        p = np.exp(lp)
        Z = p.sum()
        p = (p / Z) if Z > 0 else np.ones(Lg) / Lg
        posterior[n] = p

        mu = float(np.sum(fitness_grid * p))
        var = float(np.sum((fitness_grid**2) * p) - mu**2)
        means[n] = mu
        vars_[n] = max(var, 0.0)

    leaf_means = {n: means[n] for n in leaves}
    if zscore and len(leaf_means) >= 2:
        vals = np.array(list(leaf_means.values()), dtype=float)
        mu = float(vals.mean())
        sd = float(vals.std(ddof=1))
        leaf_means = {k: (0.0 if sd == 0 else float((v - mu) / sd)) for k, v in leaf_means.items()}

    for n in G.nodes:
        if len(children[n]) == 0:
            G.nodes[n][fitness_attr] = float(leaf_means[n])
        if attach_posteriors:
            G.nodes[n]["fitness_posterior"] = posterior[n]
            G.nodes[n]["fitness_mean"] = float(means[n])
            G.nodes[n]["fitness_var"] = float(vars_[n])

    return {n: G.nodes[n][fitness_attr] for n in leaves}


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
    copy: Literal[True, False] = True,
) -> pd.DataFrame: ...


@overload
def fitness(
    tdata: td.TreeData,
    tree: str,
    depth_key: str = "depth",
    key_added: str = "fitness",
    copy: Literal[True, False] = False,
) -> None: ...


def fitness(
    tdata: td.TreeData,
    tree: str | list[str] | None = None,
    depth_key: str = "depth",
    key_added: str = "fitness",
    method: Literal["sbd", "lbi"] = "sbd",
    copy: Literal[True, False] = False,
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
    for t in trees.values():
        check_tree_has_key(t, depth_key)
        if method == "sbd":
            scores = _infer_fitness_sbd(t, depth_key=depth_key, fitness_attr=key_added)
        elif method == "lbi":
            scores = _infer_fitness_lbi(t, depth_key=depth_key, fitness_attr=key_added)
    leaf_to_clade = get_keyed_leaf_data(tdata, key_added, tree_keys)
    tdata.obs[key_added] = tdata.obs.index.map(leaf_to_clade[key_added])
    if copy:
        return get_keyed_node_data(tdata, key_added, tree_keys)
    return None
    return None
