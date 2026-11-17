"""Path-constrained informative routing (Binney, Krause & Sukhatme, IJRR 2013).

GRG (Greedy-Recursive-Growth) algorithm for the WELL-E GP sampling testbed.
Vertex-only measurements, static field. All functions are torch-free (numpy + networkx).

Public API
----------
build_complete_travel_graph  -- complete weighted graph over candidate nodes
travel_time_matrix           -- precomputed n×n shortest-path time matrix
gp_mutual_information        -- I(y_A; f_V | y_X), monotone submodular GP objective
grg_plan                     -- GRG recursive planner (Algorithm 1, Binney 2013)
nearest_neighbor_tour_fixed_end -- greedy NN baseline with fixed start/end
"""

import numpy as np
import networkx as nx


# ── Graph helpers ─────────────────────────────────────────────────────────────

def build_complete_travel_graph(candidate_points: np.ndarray, speed: float) -> nx.Graph:
    """Complete weighted graph; edge weight = Euclidean distance / speed (seconds)."""
    n = len(candidate_points)
    G = nx.Graph()
    for i in range(n):
        G.add_node(i, pos=tuple(float(x) for x in candidate_points[i]))
    for i in range(n):
        for j in range(i + 1, n):
            dist = float(np.linalg.norm(candidate_points[i] - candidate_points[j]))
            G.add_edge(i, j, weight=dist / speed)
    return G


def travel_time_matrix(G: nx.Graph) -> np.ndarray:
    """n×n matrix of shortest pairwise travel times (Dijkstra on G)."""
    n = G.number_of_nodes()
    T = np.full((n, n), np.inf)
    for src, lengths in nx.all_pairs_dijkstra_path_length(G, weight='weight'):
        for dst, d in lengths.items():
            T[src, dst] = d
    return T


# ── MI objective (order-independent) ──────────────────────────────────────────

def gp_mutual_information(
    A_indices,
    K_all: np.ndarray,
    noise_variance: float,
    X_indices=None,
) -> float:
    """I(y_A; f_V | y_X) — GP entropy-reduction objective, order-independent.

    Measures how much observing A (given already-committed X) reduces posterior
    uncertainty about the FULL candidate set V. This is the f_X(P, τ) objective
    used by GRG (Binney, Krause & Sukhatme 2013): monotone, submodular, and
    well-defined even when A = V.

    Formula: 0.5 × [log det Σ_{V|X} − log det Σ_{V|A∪X}], where Σ_{V|C} is the
    GP posterior field covariance of all n nodes given noisy observations at C.

    Args:
        A_indices: indices of the new measurement set A (iterable of ints).
        K_all: n×n prior kernel matrix over all n candidate nodes.
        noise_variance: GP observation noise σ² (scalar).
        X_indices: already-committed indices (iterable of ints or None).

    Returns:
        Non-negative float; 0.0 if A is empty or equal to X.
    """
    nv = noise_variance
    n = len(K_all)
    A = sorted(set(int(i) for i in A_indices))
    X = sorted(set(int(i) for i in (X_indices or [])))
    AX = sorted(set(A) | set(X))

    if not A:
        return 0.0

    def _posterior_field_cov_V(cond: list) -> np.ndarray:
        """n×n posterior field covariance of V given noisy obs at cond."""
        if not cond:
            return K_all.copy()
        K_cc = K_all[np.ix_(cond, cond)] + nv * np.eye(len(cond))
        K_vc = K_all[:, cond]               # n × len(cond)
        return K_all - K_vc @ np.linalg.solve(K_cc, K_vc.T)

    jitter = 1e-10 * np.eye(n)
    Sv_X = _posterior_field_cov_V(X)
    Sv_AX = _posterior_field_cov_V(AX)

    _, ld1 = np.linalg.slogdet(Sv_X + jitter)
    _, ld2 = np.linalg.slogdet(Sv_AX + jitter)

    return max(0.0, 0.5 * (ld1 - ld2))


# ── GRG planner ───────────────────────────────────────────────────────────────

def grg_plan(
    s: int,
    t: int,
    tau_start: float,
    tau_end: float,
    tt: np.ndarray,
    K_all: np.ndarray,
    noise_variance: float,
    depth: int,
    n_tau_splits: int = 5,
    _X_frozen=None,
    _mi_cache=None,
    objective_fn=None,
):
    """GRG recursive path planner (Binney, Krause & Sukhatme 2013, Algorithm 1).

    Recursively splits the time budget [tau_start, tau_end] over an intermediate
    waypoint v and split time τ, selecting the combination that maximises
    gp_mutual_information of the combined path.

    Args:
        s, t: fixed start and end node indices.
        tau_start, tau_end: time budget window [seconds].
        tt: n×n precomputed travel-time matrix (from travel_time_matrix()).
        K_all: n×n prior kernel matrix over all n candidate nodes.
        noise_variance: GP observation noise σ².
        depth: recursion depth; 0 = return direct s→t path (no intermediate stops).
        n_tau_splits: number of interior τ split points tried per recursion level.
        _X_frozen: internal — frozenset of already-committed node indices.
        _mi_cache: optional dict for memoising gp_mutual_information results.
            Pass an empty dict {} from the top-level call to enable caching.
            The cache key is (frozenset(A), frozenset(X)); K_all and
            noise_variance are assumed fixed within a single planning call.
            None (default) disables caching (backward-compatible behaviour).
        objective_fn: optional callable with signature
            ``f(A_indices, K_all, noise_variance, X_indices) -> float``.
            When provided, replaces ``gp_mutual_information`` as the planning
            objective. Must be monotone submodular for Theorem 1 to hold.
            None (default) uses ``gp_mutual_information`` (backward-compatible).

    Returns:
        Ordered list of node indices (the planned path), or None if infeasible.
    """
    X = frozenset() if _X_frozen is None else _X_frozen
    budget = tau_end - tau_start

    # Infeasibility: cannot reach t from s within remaining budget
    if tt[s, t] > budget + 1e-9:
        return None

    _obj_fn = objective_fn if objective_fn is not None else gp_mutual_information

    def _mi(A_set, X_set):
        """Objective (gp_mutual_information or custom) with optional memoisation."""
        if _mi_cache is None:
            return _obj_fn(A_set, K_all, noise_variance, X_set)
        key = (frozenset(A_set), frozenset(X_set))
        if key not in _mi_cache:
            _mi_cache[key] = _obj_fn(A_set, K_all, noise_variance, X_set)
        return _mi_cache[key]

    # Base path: direct s → t (trivially feasible)
    base = [s] if s == t else [s, t]
    A_base = frozenset(base) - X
    best_score = _mi(A_base, X)
    best_path = base

    if depth == 0:
        return best_path

    n = len(K_all)

    for v in range(n):
        if v == s or v == t:
            continue  # degenerate splits add no new information
        # Per-waypoint feasible split window:
        #   tau_lo = tau_start + tt[s,v]  (earliest s can reach v)
        #   tau_hi = tau_end   - tt[v,t]  (latest v can depart and still reach t)
        # A single global grid over [tau_start, tau_end] silently skips waypoints
        # whose feasible window [tau_lo, tau_hi] is narrower than one grid step —
        # common near tight budgets and for collinear nodes (e.g. diagonal of a
        # regular grid where the detour cost is exactly zero but the window is
        # only ~0.17 s wide at τ=3.0 s). Generating candidates per-waypoint
        # guarantees every feasible v gets N_TAU_SPLITS split-time candidates.
        tau_lo = tau_start + tt[s, v]
        tau_hi = tau_end   - tt[v, t]
        if tau_lo > tau_hi + 1e-9:
            continue  # v is infeasible regardless of split time
        tau_grid_v = np.unique(np.linspace(tau_lo, tau_hi, n_tau_splits + 2)[1:-1])
        for tau_mid in tau_grid_v:

            P1 = grg_plan(s, v, tau_start, tau_mid, tt, K_all, noise_variance,
                          depth - 1, n_tau_splits, X, _mi_cache)
            if P1 is None:
                continue

            # P2 is planned with P1's measurements already committed
            X1 = X | frozenset(P1)
            P2 = grg_plan(v, t, tau_mid, tau_end, tt, K_all, noise_variance,
                          depth - 1, n_tau_splits, X1, _mi_cache)
            if P2 is None:
                continue

            combined = P1 + P2[1:]   # remove duplicate v at junction
            A_combined = frozenset(combined) - X
            score = _mi(A_combined, X)

            if score > best_score:
                best_score = score
                best_path = combined

    return best_path


# ── Nearest-neighbour baseline ─────────────────────────────────────────────────

def nearest_neighbor_tour_fixed_end(
    start: int,
    end: int,
    tt: np.ndarray,
    time_budget: float,
) -> list:
    """Greedy nearest-neighbour tour from start to end within time_budget.

    At each step visits the closest unvisited node that still leaves enough
    time to reach `end`. Returns the ordered list of visited node indices
    (including start; includes end if reachable after the last intermediate stop).
    """
    n = len(tt)
    tour = [start]
    # Mark end as 'reserved' so we never visit it as an intermediate stop
    visited = {start, end}
    current = start
    elapsed = 0.0

    while True:
        best_j, best_d = None, float('inf')
        for j in range(n):
            if j in visited:
                continue
            d_to_j = tt[current, j]
            # Can we visit j and still reach end within budget?
            if elapsed + d_to_j + tt[j, end] <= time_budget + 1e-9:
                if d_to_j < best_d:
                    best_j, best_d = j, d_to_j

        if best_j is None:
            break

        tour.append(best_j)
        visited.add(best_j)
        elapsed += best_d
        current = best_j

    # Append end if reachable from current position
    if tt[current, end] <= time_budget - elapsed + 1e-9:
        tour.append(end)

    return tour
