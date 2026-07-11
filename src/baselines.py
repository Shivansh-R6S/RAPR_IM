import random
import logging
import networkx as nx
import numpy as np
from collections import deque

logger = logging.getLogger(__name__)


# --- Signed IC diffusion ---

def run_signed_ic(G: nx.DiGraph, seed_set: set, mc_runs: int = 10) -> dict:
    total_aligned = 0
    total_opposed = 0
    total_immune = 0
    for _ in range(mc_runs):
        r = _signed_ic_single(G, seed_set)
        total_aligned += r["aligned"]
        total_opposed += r["opposed"]
        total_immune += r["immune"]
    return {
        "avg_aligned": total_aligned / mc_runs,
        "avg_opposed": total_opposed / mc_runs,
        "avg_immune": total_immune / mc_runs,
    }


def _signed_ic_single(G: nx.DiGraph, seed_set: set) -> dict:
    """
    Signed IC: standard IC propagation where negative edges flip the
    influence direction. No conformity/reactance coefficients - activation
    probability is uniform 1/in_degree(v) regardless of sign. Negative
    edges propagate the opposite information type (Pro->Anti, Anti->Pro).
    """
    state = {n: "Neutral" for n in G.nodes()}
    info_type = {n: None for n in G.nodes()}
    in_degree = dict(G.in_degree())

    for n in seed_set:
        if n in G:
            state[n] = "Aligned"
            info_type[n] = "Pro"

    queue = deque(n for n in seed_set if n in G)
    attempted = set(n for n in seed_set if n in G)

    while queue:
        u = queue.popleft()
        if state[u] not in ("Aligned", "Opposed"):
            continue
        u_info = info_type[u]

        for _, v, d in G.out_edges(u, data=True):
            if state[v] != "Neutral":
                continue
            sign = d.get("sign", 1)
            prob = 1.0 / in_degree[v] if in_degree[v] > 0 else 0.0
            if random.random() < prob:
                new_info = u_info if sign == 1 else ("Anti" if u_info == "Pro" else "Pro")
                state[v] = "Aligned" if new_info == "Pro" else "Opposed"
                info_type[v] = new_info
                if v not in attempted:
                    queue.append(v)
                    attempted.add(v)

    aligned = sum(1 for s in state.values() if s == "Aligned")
    opposed = sum(1 for s in state.values() if s == "Opposed")
    immune = sum(1 for s in state.values() if s == "Immune")

    for n, s in state.items():
        if s in ("Aligned", "Opposed"):
            state[n] = "Immune"

    return {"aligned": aligned, "opposed": opposed, "immune": immune}


# --- Signed LT diffusion ---

def run_signed_lt(G: nx.DiGraph, seed_set: set, mc_runs: int = 10) -> dict:
    total_aligned = 0
    total_opposed = 0
    total_immune = 0
    for _ in range(mc_runs):
        r = _signed_lt_single(G, seed_set)
        total_aligned += r["aligned"]
        total_opposed += r["opposed"]
        total_immune += r["immune"]
    return {
        "avg_aligned": total_aligned / mc_runs,
        "avg_opposed": total_opposed / mc_runs,
        "avg_immune": total_immune / mc_runs,
    }


def _signed_lt_single(G: nx.DiGraph, seed_set: set) -> dict:
    """
    Signed LT: each node v has a random threshold theta_v ~ Uniform[0,1].
    For each active neighbor u of v, positive edges contribute +w(u,v) to
    v's Pro influence accumulator, negative edges contribute +w(u,v) to v's
    Anti accumulator. v activates as Aligned if Pro_sum >= theta_v, Opposed
    if Anti_sum >= theta_v (Pro takes priority if both exceed threshold).
    Edge weight w(u,v) = 1/in_degree(v) (standard LT normalisation).
    """
    state = {n: "Neutral" for n in G.nodes()}
    info_type = {n: None for n in G.nodes()}
    in_degree = dict(G.in_degree())
    threshold = {n: random.random() for n in G.nodes()}

    pro_sum = {n: 0.0 for n in G.nodes()}
    anti_sum = {n: 0.0 for n in G.nodes()}

    for n in seed_set:
        if n in G:
            state[n] = "Aligned"
            info_type[n] = "Pro"

    active = set(n for n in seed_set if n in G)
    newly_active = set(active)

    while newly_active:
        next_newly_active = set()
        for u in newly_active:
            u_info = info_type[u]
            for _, v, d in G.out_edges(u, data=True):
                if state[v] != "Neutral":
                    continue
                sign = d.get("sign", 1)
                w = 1.0 / in_degree[v] if in_degree[v] > 0 else 0.0
                if sign == 1:
                    if u_info == "Pro":
                        pro_sum[v] += w
                    else:
                        anti_sum[v] += w
                else:
                    if u_info == "Pro":
                        anti_sum[v] += w
                    else:
                        pro_sum[v] += w

                # check threshold
                if pro_sum[v] >= threshold[v] and state[v] == "Neutral":
                    state[v] = "Aligned"
                    info_type[v] = "Pro"
                    next_newly_active.add(v)
                elif anti_sum[v] >= threshold[v] and state[v] == "Neutral":
                    state[v] = "Opposed"
                    info_type[v] = "Anti"
                    next_newly_active.add(v)

        newly_active = next_newly_active

    aligned = sum(1 for s in state.values() if s == "Aligned")
    opposed = sum(1 for s in state.values() if s == "Opposed")
    immune = sum(1 for s in state.values() if s == "Immune")

    for n, s in state.items():
        if s in ("Aligned", "Opposed"):
            state[n] = "Immune"

    return {"aligned": aligned, "opposed": opposed, "immune": immune}


# --- Seed selection baselines ---

def pagerank_seeds(G: nx.DiGraph, k: int) -> list:
    """Top-k nodes by PageRank."""
    pr = nx.pagerank(G, alpha=0.85)
    return sorted(pr, key=pr.get, reverse=True)[:k]


def degree_centrality_seeds(G: nx.DiGraph, k: int) -> list:
    """Top-k nodes by out-degree."""
    return [n for n, _ in sorted(G.out_degree(), key=lambda x: x[1], reverse=True)[:k]]


def random_seeds(G: nx.DiGraph, k: int, seed: int = 42) -> list:
    """k randomly selected nodes."""
    rng = random.Random(seed)
    return rng.sample(list(G.nodes()), k)


def celf_greedy_baseline(G: nx.DiGraph, k: int, diffusion_fn,
                          mc_runs: int = 5, n_candidates: int = 1000) -> list:
    """
    CELF-accelerated greedy on the original graph G (no realizations).
    Candidate pool restricted to top n_candidates by out-degree for speed.
    Uses diffusion_fn to evaluate sigma+ (allows plugging in any model).

    Speedups vs naive greedy:
    - Candidate pool capped at n_candidates (top out-degree nodes)
    - CELF lazy evaluation skips recomputation for most candidates per round
    - mc_runs=5 during selection (lower than final evaluation)
    """
    import heapq, itertools

    candidates = [n for n, _ in sorted(
        G.out_degree(), key=lambda x: x[1], reverse=True
    )[:n_candidates]]

    def _sigma(seed_set):
        if not seed_set:
            return 0.0
        return diffusion_fn(G, seed_set, mc_runs=mc_runs)["avg_aligned"]

    S = []
    S_set = set()
    current_sigma = 0.0
    heap = []
    counter = itertools.count()

    for x in candidates:
        gain = _sigma({x}) - current_sigma
        heapq.heappush(heap, (-gain, next(counter), x, 0))

    for j in range(k):
        if not heap:
            break
        best_node = None
        best_gain = None
        recompute_count = 0

        while heap:
            neg_gain, _, x, round_computed = heapq.heappop(heap)
            if x in S_set:
                continue
            if round_computed == len(S):
                best_node = x
                best_gain = -neg_gain
                break
            recompute_count += 1
            fresh_gain = _sigma(S_set | {x}) - current_sigma
            heapq.heappush(heap, (-fresh_gain, next(counter), x, len(S)))

        if best_node is None:
            break

        S.append(best_node)
        S_set.add(best_node)
        current_sigma = _sigma(S_set)
        logger.info(f"CELF baseline step {j+1}/{k}: node={best_node}, "
                    f"sigma+={current_sigma:.2f}, recomputations={recompute_count}")

    return S


# --- Evaluation ---

def evaluate(G: nx.DiGraph, seed_set: list, diffusion_fn, mc_runs: int = 10) -> dict:
    """
    Runs diffusion_fn on G with seed_set and returns the 4 reported metrics:
    avg_sigma_plus, polarization_index, net_influence, seed_efficiency.
    """
    k = len(seed_set)
    if k == 0:
        return {"avg_sigma_plus": 0.0, "polarization_index": 0.0,
                "net_influence": 0.0, "seed_efficiency": 0.0}

    result = diffusion_fn(G, set(seed_set), mc_runs=mc_runs)
    a = result["avg_aligned"]
    o = result["avg_opposed"]

    return {
        "avg_sigma_plus": round(a, 2),
        "polarization_index": round(a / (a + o), 4) if (a + o) > 0 else 0.0,
        "net_influence": round(a - o, 2),
        "seed_efficiency": round(a / k, 4),
    }