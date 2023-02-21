"""An implementation of the Wasserstein metric (Earth Mover's distance)
between two weighted distributions. Aka Earth Mover's distance, the
appropriate metric for comparing two PDDs.

Copyright (C) 2020 Cameron Hargreaves. This code is adapted from the 
Element Movers Distance project https://github.com/lrcfmd/ElMD.
"""

from typing import Tuple

import numba
import numpy as np


@numba.njit(cache=True)
def network_simplex(
        source_demands: np.ndarray,
        sink_demands: np.ndarray,
        network_costs: np.ndarray
) -> Tuple[float, np.ndarray]:
    """Given two sets of weights and a cost matrix on two
    distributions, calculate the Earth mover's distance 
    (Wasserstien metric).

    This is a port of the network simplex algorithm implented by Loïc
    Séguin-C for the networkx package to allow acceleration via the
    numba package. Copyright (C) 2010 Loïc Séguin-C.
    loicseguin@gmail.com. All rights reserved. BSD license.

    Parameters
    ----------
    source_demands : :class:`numpy.ndarray`
        Weights of the first distribution.
    sink_demands : :class:`numpy.ndarray`
        Weights of the second distribution.
    network_costs : :class:`numpy.ndarray`
        Cost matrix of distances between elements of the two 
        distributions. Shape (len(source_demands), len(sink_demands)).

    Returns
    -------
    (emd, plan) : Tuple[float, :class:`numpy.ndarray`]
        A tuple of the Earth mover's distance and the optimal matching.

    References
    ----------
    [1] Z. Kiraly, P. Kovacs.
        Efficient implementation of minimum-cost flow algorithms.
        Acta Universitatis Sapientiae, Informatica 4(1), 67--118
        (2012).
    [2] R. Barr, F. Glover, D. Klingman.
        Enhancement of spanning tree labeling procedures for network
        optimization. INFOR 17(1), 16--34 (1979).
    """

    n_sources, n_sinks = source_demands.shape[0], sink_demands.shape[0]
    network_costs = network_costs.ravel()
    n = n_sources + n_sinks
    e = n_sources * n_sinks
    B = np.int64(np.ceil(np.sqrt(e)))
    fp_multiplier = np.int64(1_000_000)

    # Add one additional node for a dummy source and sink
    source_d_int = (source_demands * fp_multiplier).astype(np.int64)
    sink_d_int = (sink_demands * fp_multiplier).astype(np.int64)

    # FP conversion error correction
    source_sum = np.sum(source_d_int)
    sink_sum = np.sum(sink_d_int)
    sink_source_sum_diff = sink_sum - source_sum

    if sink_source_sum_diff > 0:
        source_d_int[np.argmax(source_d_int)] += sink_source_sum_diff
    elif sink_source_sum_diff < 0:
        sink_d_int[np.argmax(sink_d_int)] -= sink_source_sum_diff

    demands = np.empty(n, dtype=np.int64)
    demands[:n_sources] = -source_d_int
    demands[n_sources:] = sink_d_int

    tails = np.empty(e + n, dtype=np.int64)
    heads = np.empty(e + n, dtype=np.int64)

    for i in range(n_sources):
        for j in range(n_sinks):
            ind = i * n_sinks + j
            tails[ind] = i
            heads[ind] = j + n_sources

    for i, demand in enumerate(demands):
        if demand > 0:
            tails[e + i] = -1
            heads[e + i] = -1
        else:
            tails[e + i] = i
            heads[e + i] = i

    # Create costs and capacities for the arcs between nodes
    network_costs = network_costs * fp_multiplier

    network_capac = np.array([min(source_demands[i], sink_demands[j])
                              for i in range(n_sources)
                              for j in range(n_sinks)],
                             dtype=np.float64) * fp_multiplier

    faux_inf = 3 * np.max(np.array((
        np.sum(network_capac),
        np.sum(np.abs(network_costs)),
        np.amax(source_d_int),
        np.amax(sink_d_int)),
        dtype=np.int64))

    # allocate arrays
    costs = np.empty(e + n, dtype=np.int64)
    costs[:e] = network_costs
    costs[e:] = faux_inf

    capac = np.empty(e + n, dtype=np.int64)
    capac[:e] = network_capac
    capac[e:] = fp_multiplier

    flows = np.empty(e + n, dtype=np.int64)
    flows[:e] = 0
    flows[e:e+n_sources] = source_d_int
    flows[e+n_sources:] = sink_d_int

    potentials = np.empty(n, dtype=np.int64)
    demands_neg_mask = demands <= 0
    potentials[demands_neg_mask] = faux_inf
    potentials[~demands_neg_mask] = -faux_inf

    parent = np.empty(n + 1, dtype=np.int64)
    parent[:-1] = -1
    parent[-1] = -2

    size = np.empty(n + 1, dtype=np.int64)
    size[:-1] = 1
    size[-1] = n + 1

    next_node = np.arange(1, n + 2, dtype=np.int64)
    next_node[-2] = -1
    next_node[-1] = 0

    last_node = np.arange(n + 1, dtype=np.int64)
    last_node[-1] = n - 1

    prev_node = np.arange(-1, n, dtype=np.int64)
    edge = np.arange(e, e + n, dtype=np.int64)

    ### Pivot loop ###

    f = 0
    while True:
        i, p, q, f = find_entering_edges(B, e, f, tails, heads, costs, potentials, flows)
        if p == -1: # If no entering edges then the optimal score is found
            break

        cycle_nodes, cycle_edges = find_cycle(i, p, q, size, edge, parent)
        j, s, t = find_leaving_edge(cycle_nodes, cycle_edges, capac, flows, tails, heads)
        augment_flow(cycle_nodes, cycle_edges, residual_capacity(j, s, capac, flows, tails), tails, flows)

        if i != j:  # Do nothing more if the entering edge is the same as the leaving edge.
            if parent[t] != s:
                # Ensure that s is the parent of t.
                s, t = t, s

            # Ensure that q is in the subtree rooted at t.
            for val in cycle_edges:
                if val == j:
                    p, q = q, p
                    break
                if val == i:
                    break

            remove_edge(s, t, size, prev_node, last_node, next_node, parent, edge)
            make_root(q, parent, size, last_node, prev_node, next_node, edge)
            add_edge(i, p, q, next_node, prev_node, last_node, size, parent, edge)
            update_potentials(i, p, q, heads, potentials, costs, last_node, next_node)

    final_flows = flows[:e] / fp_multiplier
    edge_costs = costs[:e] / fp_multiplier
    final = final_flows.dot(edge_costs)

    return final, final_flows.reshape((n_sources, n_sinks))


@numba.njit(cache=True)
def reduced_cost(i, costs, potentials, tails, heads, flows):
    """Return the reduced cost of an edge i.
    """
    c = costs[i] - potentials[tails[i]] + potentials[heads[i]]
    if flows[i] == 0:
        return c
    return -c


@numba.njit(cache=True)
def find_entering_edges(B, e, f, tails, heads, costs, potentials, flows):
    """Yield entering edges until none can be found. Entering edges are
    found by combining Dantzig's rule and Bland's rule. The edges are
    cyclically grouped into blocks of size B. Within each block,
    Dantzig's rule is applied to find an entering edge. The blocks to
    search is determined following Bland's rule.
    """

    M = (e + B - 1) // B # number of blocks needed to cover all edges
    m = 0

    while m < M:
        # Determine the next block of edges.
        l = f + B
        if l <= e:
            edge_inds = np.arange(f, l)
        else:
            l -= e
            edge_inds = np.empty(e - f + l, dtype=np.int64)
            for i, v in enumerate(range(f, e)):
                edge_inds[i] = v
            for i in range(l):
                edge_inds[e - f + i] = i

        f = l

        # Find the first edge with the lowest reduced cost.
        i = edge_inds[0]
        c = reduced_cost(i, costs, potentials, tails, heads, flows)
        for ind in edge_inds[1:]:
            cost = reduced_cost(ind, costs, potentials, tails, heads, flows)
            if cost < c:
                c = cost
                i = ind

        p = q = -1

        if c >= 0:
            m += 1

        # Entering edge found.
        else:
            if flows[i] == 0:
                p = tails[i]
                q = heads[i]
            else:
                p = heads[i]
                q = tails[i]

            return i, p, q, f

    # All edges have nonnegative reduced costs. The flow is optimal.
    return -1, -1, -1, -1


@numba.njit(cache=True)
def find_apex(p, q, size, parent):
    """Find the lowest common ancestor of nodes p and q in the spanning
    tree.
    """
    size_p = size[p]
    size_q = size[q]

    while True:
        while size_p < size_q:
            p = parent[p]
            size_p = size[p]
        while size_p > size_q:
            q = parent[q]
            size_q = size[q]
        if size_p == size_q:
            if p != q:
                p = parent[p]
                size_p = size[p]
                q = parent[q]
                size_q = size[q]
            else:
                return p


@numba.njit(cache=True)
def trace_path(p, w, edge, parent):
    """Return the nodes and edges on the path from node p to its
    ancestor w.
    """
    cycle_nodes = [p]
    cycle_edges = []

    while p != w:
        cycle_edges.append(edge[p])
        p = parent[p]
        cycle_nodes.append(p)

    return cycle_nodes, cycle_edges


@numba.njit(cache=True)
def find_cycle(i, p, q, size, edge, parent):
    """Return the nodes and edges on the cycle containing edge 
    i == (p, q) when the latter is added to the spanning tree.
    The cycle is oriented in the direction from p to q.
    """

    w = find_apex(p, q, size, parent)
    cycle_nodes, cycle_edges = trace_path(p, w, edge, parent)
    cycle_nodes_rev, cycle_edges_rev = trace_path(q, w, edge, parent)
    append_i_to_edges = (len(cycle_edges) < 1 or cycle_edges[-1] != i)
    add_to_c_nodes = max(len(cycle_nodes_rev) - 1, 0)
    cycle_nodes_ = np.empty(len(cycle_nodes) + add_to_c_nodes, dtype=np.int64)

    for j in range(len(cycle_nodes)):
        cycle_nodes_[j] = cycle_nodes[-(j+1)]
    for j in range(add_to_c_nodes):
        cycle_nodes_[len(cycle_nodes) + j] = cycle_nodes_rev[j]

    if append_i_to_edges:
        cycle_edges_ = np.empty(len(cycle_edges) + len(cycle_edges_rev) + 1, dtype=np.int64)
        cycle_edges_[len(cycle_edges)] = i
    else:
        cycle_edges_ = np.empty(len(cycle_edges) + len(cycle_edges_rev), dtype=np.int64)

    for j in range(len(cycle_edges)):
        cycle_edges_[j] = cycle_edges[-(j+1)]
    for j in range(1, len(cycle_edges_rev) + 1):
        cycle_edges_[-j] = cycle_edges_rev[-j]

    return cycle_nodes_, cycle_edges_


@numba.njit(cache=True)
def residual_capacity(i, p, capac, flows, tails):
    """Return the residual capacity of an edge i in the direction away
    from its endpoint p.
    """
    if tails[i] == p:
        return capac[i] - flows[i]
    return flows[i]


@numba.njit(cache=True)
def find_leaving_edge(cycle_nodes, cycle_edges, capac, flows, tails, heads):
    """Return the leaving edge in a cycle represented by cycle_nodes
    and cycle_edges.
    """
    j, s = cycle_edges[0], cycle_nodes[0]
    res_caps_min = residual_capacity(j, s, capac, flows, tails)
    for ind in range(1, cycle_edges.shape[0]):
        j_, s_ = cycle_edges[ind], cycle_nodes[ind]
        res_cap = residual_capacity(j_, s_, capac, flows, tails)
        if res_cap < res_caps_min:
            res_caps_min = res_cap
            j, s = j_, s_

    t = heads[j] if tails[j] == s else tails[j]
    return j, s, t


@numba.njit(cache=True)
def augment_flow(cycle_nodes, cycle_edges, f, tails, flows):
    """Augment f units of flow along a cycle representing Wn with
    cycle_edges.
    """
    for i, p in zip(cycle_edges, cycle_nodes):
        if tails[i] == p:
            flows[i] += f
        else:
            flows[i] -= f


@numba.njit(cache=True)
def remove_edge(s, t, size, prev, last, next_node, parent, edge):
    """Remove an edge (s, t) where parent[t] == s from the spanning
    tree.
    """
    size_t = size[t]
    prev_t = prev[t]
    last_t = last[t]
    next_last_t = next_node[last_t]
    # Remove (s, t).
    parent[t] = -2
    edge[t] = -2
    # Remove the subtree rooted at t from the depth-first thread.
    next_node[prev_t] = next_last_t
    prev[next_last_t] = prev_t
    next_node[last_t] = t
    prev[t] = last_t

    # Update the subtree sizes & last descendants of the (old) ancestors of t.
    while s != -2:
        size[s] -= size_t
        if last[s] == last_t:
            last[s] = prev_t
        s = parent[s]


@numba.njit(cache=True)
def make_root(q, parent, size, last, prev, next_node, edge):
    """Make a node q the root of its containing subtree.
    """
    ancestors = []
    # -2 means node is checked
    while q != -2:
        ancestors.insert(0, q)
        q = parent[q]

    for i in range(len(ancestors) - 1):
        p = ancestors[i]
        q = ancestors[i+1]
        size_p = size[p]
        last_p = last[p]
        prev_q = prev[q]
        last_q = last[q]
        next_last_q = next_node[last_q]

        # Make p a child of q.
        parent[p] = q
        parent[q] = -2
        edge[p] = edge[q]
        edge[q] = -2
        size[p] = size_p - size[q]
        size[q] = size_p

        # Remove the subtree rooted at q from the depth-first thread.
        next_node[prev_q] = next_last_q
        prev[next_last_q] = prev_q
        next_node[last_q] = q
        prev[q] = last_q

        if last_p == last_q:
            last[p] = prev_q
            last_p = prev_q

        # Add the remaining parts of the subtree rooted at p as a subtree
        # of q in the depth-first thread.
        prev[p] = last_q
        next_node[last_q] = p
        next_node[last_p] = q
        prev[q] = last_p
        last[q] = last_p


@numba.njit(cache=True)
def add_edge(i, p, q, next_node, prev, last, size, parent, edge):
    """Add an edge (p, q) to the spanning tree where q is the root of a
    subtree.
    """
    last_p = last[p]
    next_last_p = next_node[last_p]
    size_q = size[q]
    last_q = last[q]
    # Make q a child of p.
    parent[q] = p
    edge[q] = i
    # Insert the subtree rooted at q into the depth-first thread.
    next_node[last_p] = q
    prev[q] = last_p
    prev[next_last_p] = last_q
    next_node[last_q] = next_last_p

    # Update the subtree sizes and last descendants of the (new) ancestors
    # of q.
    while p != -2:
        size[p] += size_q
        if last[p] == last_p:
            last[p] = last_q
        p = parent[p]


@numba.njit(cache=True)
def update_potentials(i, p, q, heads, potentials, costs, last_node, next_node):
    """Update the potentials of the nodes in the subtree rooted at a
    node q connected to its parent p by an edge i.
    """
    if q == heads[i]:
        d = potentials[p] - costs[i] - potentials[q]
    else:
        d = potentials[p] + costs[i] - potentials[q]

    potentials[q] += d
    l = last_node[q]
    while q != l:
        q = next_node[q]
        potentials[q] += d
