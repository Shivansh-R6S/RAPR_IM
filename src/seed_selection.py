import heapq
import itertools
import logging
import networkx as nx

from src.diffusion import run_paic_dgbc

logger = logging.getLogger(__name__)


def sigma_plus(G: nx.DiGraph, seed_set: set, mc_runs: int) -> float:
    """
    sigma+_G(S): expected number of Aligned (Pro) nodes reached when S is
    seeded on G, estimated via mc_runs Monte Carlo simulations of PaIC-DGBC.
    """
    if not seed_set:
        return 0.0
    result = run_paic_dgbc(G, seed_set, mc_runs=mc_runs)
    return result["avg_aligned"]


def _avg_marginal_gain(realizations: list, S_set: set, x, current_sigma: list, mc_runs: int) -> float:
    """
    Average, across all R realizations, of sigma+_{G'_i}(S U {x}) - sigma+_{G'_i}(S),
    where sigma+_{G'_i}(S) is passed in via current_sigma (already computed)
    rather than recomputed here.
    """
    R = len(realizations)
    total_gain = 0.0
    for i, G_prime in enumerate(realizations):
        if x not in G_prime:
            continue
        candidate_sigma = sigma_plus(G_prime, S_set | {x}, mc_runs)
        total_gain += candidate_sigma - current_sigma[i]
    return total_gain / R


def modified_greedy(realizations: list, k: int, mc_runs: int = 10,
                     candidate_nodes: list = None, use_celf: bool = True) -> list:
    """
    Algorithm 2: Modified Greedy (CELF-accelerated).

    Given R signed-randomized realizations G'_1, ..., G'_R (from
    src.randomization.generate_realizations) and a target seed size k,
    greedily builds a seed set S by repeatedly adding the node with the
    highest average marginal gain in sigma+ across all R realizations:

        x* = argmax_{x in (V minus S)} (1/R) * sum_{i=1}^{R} [sigma+_{G'_i}(S U {x})
                                                        - sigma+_{G'_i}(S)]
        S <- S U {x*}

    CELF acceleration (Cost-Effective Lazy Forward selection): naive greedy
    recomputes every candidate's marginal gain in every round, i.e.
    O(k * |candidates| * R) diffusion estimates. Since sigma+ is (at least
    approximately) submodular, a candidate's marginal gain can only shrink
    as S grows across rounds. This lets us keep gains in a max-heap tagged
    with the round they were last computed in; if the top of the heap was
    computed during the current round, it is still valid and can be
    selected immediately without recomputation. Only when the top entry is
    stale (computed in an earlier round) do we recompute it, re-insert it,
    and check the heap again. In practice this skips the vast majority of
    candidate evaluations after the first round, while still selecting
    exactly the same nodes naive greedy would (assuming submodularity holds,
    modulo Monte Carlo noise in the sigma+ estimates).

    Parameters
    ----------
    realizations : list of nx.DiGraph
        The R realizations produced by signed randomization.
    k : int
        Target seed set size.
    mc_runs : int
        Monte Carlo simulations per sigma+ estimate (per realization).
    candidate_nodes : list, optional
        Restrict the candidate pool (V minus S) to these nodes (e.g. for a
        faster pass on a dense subgraph). Defaults to the union of nodes
        across all realizations.
    use_celf : bool
        If True (default), use CELF lazy evaluation. If False, fall back to
        naive greedy (recompute every candidate's gain every round) - useful
        for validating CELF against the exact naive result on small graphs.

    Returns
    -------
    list
        The selected seed set S, in the order nodes were added.
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

    # sigma+_{G'_i}(S) for the current S, cached per realization so we only
    # pay for one diffusion estimate per realization per round instead of
    # recomputing it for every candidate.
    current_sigma = [0.0] * R

    # Max-heap of (-gain, tie_breaker, node, round_computed). Python's heapq
    # is a min-heap, so gains are negated. tie_breaker (a counter) avoids
    # comparing nodes directly when gains are equal.
    heap = []
    counter = itertools.count()

    logger.info(f"CELF: computing initial marginal gains for {len(candidate_nodes)} candidates (round 0)")
    for x in candidate_nodes:
        gain = _avg_marginal_gain(realizations, S_set, x, current_sigma, mc_runs)
        heapq.heappush(heap, (-gain, next(counter), x, 0))

    for j in range(1, k + 1):
        if not heap:
            logger.warning(f"No candidates left at step {j}; stopping early with |S|={len(S)}")
            break

        best_node = None
        best_gain = None
        recompute_count = 0

        while heap:
            neg_gain, _, x, round_computed = heapq.heappop(heap)

            if x in S_set:
                continue

            if round_computed == len(S):
                # Gain was computed against the current S - still valid.
                best_node = x
                best_gain = -neg_gain
                break

            # Stale entry - recompute against current S and re-insert.
            recompute_count += 1
            fresh_gain = _avg_marginal_gain(realizations, S_set, x, current_sigma, mc_runs)
            heapq.heappush(heap, (-fresh_gain, next(counter), x, len(S)))

        if best_node is None:
            logger.warning(f"No candidate found at step {j}; stopping early with |S|={len(S)}")
            break

        S.append(best_node)
        S_set.add(best_node)

        # Refresh current_sigma for the newly grown S, one estimate per
        # realization, to use as the baseline for the next round.
        for i, G_prime in enumerate(realizations):
            current_sigma[i] = sigma_plus(G_prime, S_set, mc_runs)

        logger.info(f"Greedy step {j}/{k}: selected node {best_node} "
                    f"(avg marginal gain={best_gain:.4f}, "
                    f"avg sigma+={sum(current_sigma) / R:.4f}, "
                    f"celf recomputations={recompute_count}/{len(candidate_nodes) - len(S) + 1})")

    return S


def _naive_greedy(realizations: list, k: int, mc_runs: int, candidate_nodes: list) -> list:
    """Exact naive greedy - recomputes every candidate's gain every round."""
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
            logger.warning(f"No candidate found at step {j}; stopping early with |S|={len(S)}")
            break

        S.append(best_node)
        S_set.add(best_node)

        for i, G_prime in enumerate(realizations):
            current_sigma[i] = sigma_plus(G_prime, S_set, mc_runs)

        logger.info(f"[naive] Greedy step {j}/{k}: selected node {best_node} "
                    f"(avg marginal gain={best_avg_gain:.4f}, "
                    f"avg sigma+={sum(current_sigma) / R:.4f})")

    return S