import heapq
import itertools
import logging
import networkx as nx

from src.diffusion import run_paic_dgbc

logger = logging.getLogger(__name__)


def sigma_plus(G: nx.DiGraph, seed_set: set, mc_runs: int) -> float:
    """
    sigma+_G(S): expected number of Aligned nodes when S is seeded on G,
    estimated via mc_runs Monte Carlo simulations of PaIC-DGBC.
    """
    if not seed_set:
        return 0.0
    result = run_paic_dgbc(G, seed_set, mc_runs=mc_runs)
    return result["avg_aligned"]


def _avg_marginal_gain(realizations: list, S_set: set, x, current_sigma: list, mc_runs: int) -> float:
    """
    Average marginal gain of adding x to S across realizations where x exists.
    Divides by count of realizations containing x, not always R, to avoid
    underestimating gain for nodes absent from some realizations.
    """
    total_gain = 0.0
    count = 0
    for i, G_prime in enumerate(realizations):
        if x not in G_prime:
            continue
        candidate_sigma = sigma_plus(G_prime, S_set | {x}, mc_runs)
        total_gain += candidate_sigma - current_sigma[i]
        count += 1
    return total_gain / count if count > 0 else 0.0


def get_top_candidates(G: nx.DiGraph, n: int = 1000) -> list:
    """
    Returns top n nodes by out-degree as the candidate pool for greedy.
    Defensible for IM since high out-degree nodes have the most direct
    influence propagation potential. Restricting candidates this way keeps
    greedy tractable on the full graph without sacrificing result integrity
    since low-degree nodes are rarely optimal seeds anyway.
    """
    return [node for node, _ in sorted(
        G.out_degree(), key=lambda x: x[1], reverse=True
    )[:n]]


def modified_greedy(realizations: list, k: int, mc_runs: int = 5,
                    candidate_nodes: list = None, use_celf: bool = True) -> list:
    """
    Algorithm 2: Modified Greedy (CELF-accelerated).

    Greedily builds seed set S by repeatedly selecting the node with the
    highest average marginal sigma+ gain across all R realizations:

        x* = argmax_{x in V\S} (1/R) * sum_{i=1}^{R} [sigma+_{G'_i}(S U {x})
                                                       - sigma+_{G'_i}(S)]

    CELF acceleration: since sigma+ is submodular, marginal gains can only
    shrink as S grows. Candidates kept in a max-heap tagged with the round
    their gain was computed. Stale entries are recomputed lazily, skipping
    the vast majority of recomputations after round 1.

    Parameters
    ----------
    realizations : list of nx.DiGraph
    k : int - target seed set size
    mc_runs : int - MC simulations per sigma+ estimate during selection
    candidate_nodes : list, optional - restrict candidate pool
    use_celf : bool - False falls back to naive greedy (for validation only)

    Returns
    -------
    list - selected seed nodes in order of selection
    """
    if not realizations:
        raise ValueError("modified_greedy requires at least one realization")

    R = len(realizations)

    if candidate_nodes is None:
        candidate_nodes = set()
        for G_prime in realizations:
            candidate_nodes.update(G_prime.nodes())
        candidate_nodes = list(candidate_nodes)

    if not use_celf:
        return _naive_greedy(realizations, k, mc_runs, candidate_nodes)

    S = []
    S_set = set()
    current_sigma = [0.0] * R

    # max-heap: (-gain, tie_breaker, node, round_last_computed)
    heap = []
    counter = itertools.count()

    logger.info(f"CELF: computing initial gains for {len(candidate_nodes)} candidates "
                f"(R={R}, mc_runs={mc_runs})...")
    for x in candidate_nodes:
        gain = _avg_marginal_gain(realizations, S_set, x, current_sigma, mc_runs)
        heapq.heappush(heap, (-gain, next(counter), x, 0))

    for j in range(1, k + 1):
        if not heap:
            logger.warning(f"No candidates left at step {j}, stopping at |S|={len(S)}")
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

            # stale - recompute against current S and re-insert
            recompute_count += 1
            fresh_gain = _avg_marginal_gain(realizations, S_set, x, current_sigma, mc_runs)
            heapq.heappush(heap, (-fresh_gain, next(counter), x, len(S)))

        if best_node is None:
            logger.warning(f"No valid candidate at step {j}, stopping at |S|={len(S)}")
            break

        S.append(best_node)
        S_set.add(best_node)

        # update baseline sigma for next round
        for i, G_prime in enumerate(realizations):
            current_sigma[i] = sigma_plus(G_prime, S_set, mc_runs)

        logger.info(
            f"Step {j}/{k}: node={best_node}, "
            f"avg_gain={best_gain:.4f}, "
            f"avg_sigma+={sum(current_sigma)/R:.4f}, "
            f"celf_recomputations={recompute_count}"
        )

    return S


def evaluate_seed_set(G: nx.DiGraph, seed_set: list, mc_runs: int = 10) -> dict:
    """
    Full evaluation of a seed set on graph G. Runs mc_runs diffusion
    simulations and computes all metrics derivable from the final state map.

    Metrics
    -------
    avg_aligned         : expected Aligned nodes (sigma+, primary IM objective)
    avg_opposed         : expected Opposed nodes (backfire effect magnitude)
    avg_immune          : expected Immune nodes
    aligned_ratio       : sigma+ / |V|
    opposed_ratio       : sigma- / |V|
    polarization_index  : sigma+ / (sigma+ + sigma-), share of active nodes on Pro side
    influence_coverage  : (aligned + opposed + immune) / |V|, total cascade reach
    net_influence       : sigma+ - sigma-, signed net benefit
    immune_rate         : immune / |V|
    seed_efficiency     : sigma+ / k, aligned nodes per seed
    """
    n_nodes = G.number_of_nodes()
    k = len(seed_set)
    result = run_paic_dgbc(G, set(seed_set), mc_runs=mc_runs)

    avg_aligned = result["avg_aligned"]
    avg_opposed = result["avg_opposed"]
    avg_immune = result["avg_immune"]
    total_active = avg_aligned + avg_opposed + avg_immune

    polarization_index = (
        avg_aligned / (avg_aligned + avg_opposed)
        if (avg_aligned + avg_opposed) > 0 else 0.0
    )

    return {
        "k": k,
        "avg_aligned": round(avg_aligned, 4),
        "avg_opposed": round(avg_opposed, 4),
        "avg_immune": round(avg_immune, 4),
        "aligned_ratio": round(avg_aligned / n_nodes, 4),
        "opposed_ratio": round(avg_opposed / n_nodes, 4),
        "polarization_index": round(polarization_index, 4),
        "influence_coverage": round(total_active / n_nodes, 4),
        "net_influence": round(avg_aligned - avg_opposed, 4),
        "immune_rate": round(avg_immune / n_nodes, 4),
        "seed_efficiency": round(avg_aligned / k, 4) if k > 0 else 0.0,
    }


def _naive_greedy(realizations: list, k: int, mc_runs: int, candidate_nodes: list) -> list:
    """Naive greedy - recomputes every candidate's gain every round. Validation only."""
    R = len(realizations)
    S = []
    S_set = set()
    current_sigma = [0.0] * R

    for j in range(1, k + 1):
        best_node = None
        best_avg_gain = float("-inf")

        for x in candidate_nodes:
            if x in S_set:
                continue
            avg_gain = _avg_marginal_gain(realizations, S_set, x, current_sigma, mc_runs)
            if avg_gain > best_avg_gain:
                best_avg_gain = avg_gain
                best_node = x

        if best_node is None:
            logger.warning(f"No candidate at step {j}, stopping at |S|={len(S)}")
            break

        S.append(best_node)
        S_set.add(best_node)

        for i, G_prime in enumerate(realizations):
            current_sigma[i] = sigma_plus(G_prime, S_set, mc_runs)

        logger.info(
            f"[naive] Step {j}/{k}: node={best_node}, "
            f"avg_gain={best_avg_gain:.4f}, "
            f"avg_sigma+={sum(current_sigma)/R:.4f}"
        )

    return S