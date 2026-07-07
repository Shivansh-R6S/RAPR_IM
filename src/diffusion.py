import random
import logging
from collections import deque
import networkx as nx

logger = logging.getLogger(__name__)


def run_paic_dgbc(G: nx.DiGraph, seed_set: set, mc_runs: int = 100) -> dict:
    total_aligned = 0
    total_opposed = 0
    total_immune = 0

    for _ in range(mc_runs):
        result = _single_diffusion(G, seed_set)
        total_aligned += result["aligned"]
        total_opposed += result["opposed"]
        total_immune += result["immune"]

    return {
        "avg_aligned": total_aligned / mc_runs,
        "avg_opposed": total_opposed / mc_runs,
        "avg_immune": total_immune / mc_runs,
    }


def _single_diffusion(G: nx.DiGraph, seed_set: set) -> dict:
    """
    Single run of PaIC-DGBC.

    Key invariants:
    - A node activates only if new prob > current activation_level
    - Each active node gets exactly one chance to influence its neighbors
    - Immune transition fires when same-polarity exposure > half of in-degree
    """
    state = {node: "Neutral" for node in G.nodes()}
    info_type = {node: None for node in G.nodes()}
    activation_level = {node: 0.0 for node in G.nodes()}
    same_polarity_count = {node: 0 for node in G.nodes()}
    in_degree = dict(G.in_degree())

    for node in seed_set:
        state[node] = "Aligned"
        info_type[node] = "Pro"
        activation_level[node] = 1.0

    queue = deque(seed_set)
    queued = set(seed_set)

    while queue:
        u = queue.popleft()
        u_state = state[u]
        u_info = info_type[u]

        if u_state not in ("Aligned", "Opposed"):
            continue

        for _, v, edge_data in G.out_edges(u, data=True):
            if state[v] == "Immune":
                continue

            sign = edge_data.get("sign", 1)
            weight = edge_data.get("weight", 1)

            new_info, prob = _resolve_influence(
                u_info, sign,
                G.nodes[v].get("conformity", 0.5),
                G.nodes[v].get("reactance", 0.4),
                weight
            )

            if new_info is None or prob <= activation_level[v]:
                continue

            if random.random() < prob:
                prev_state = state[v]
                new_state = "Aligned" if new_info == "Pro" else "Opposed"

                if prev_state == "Neutral":
                    state[v] = new_state
                    info_type[v] = new_info
                    activation_level[v] = prob
                    same_polarity_count[v] += 1
                    _check_immune(v, state, same_polarity_count, in_degree)
                    if state[v] != "Immune" and v not in queued:
                        queue.append(v)
                        queued.add(v)

                elif prev_state == new_state:
                    same_polarity_count[v] += 1
                    activation_level[v] = prob
                    _check_immune(v, state, same_polarity_count, in_degree)

                else:
                    # state flip - backfire or distrust gateway
                    state[v] = new_state
                    info_type[v] = new_info
                    activation_level[v] = prob
                    same_polarity_count[v] = 0
                    if state[v] != "Immune" and v not in queued:
                        queue.append(v)
                        queued.add(v)

    # Counts of interest (aligned/opposed/immune) are captured based on each
    # node's polarity *before* the termination sweep below. This is what the
    # paper's sigma+ (aligned/Pro count) refers to in the Modified Greedy
    # objective, and matches the polarities shown in the paper's worked
    # example (Fig. 2) at the final timestep.
    aligned_count = sum(1 for s in state.values() if s == "Aligned")
    opposed_count = sum(1 for s in state.values() if s == "Opposed")
    immune_count = sum(1 for s in state.values() if s == "Immune")

    # Termination phase: once no new nodes are activated (queue empty),
    # all remaining active nodes (Aligned/Opposed) become Immune, per the
    # paper's Reinforcement & Immune Phase spec. This updates state_map
    # (the per-node state), but does not change the aligned/opposed/immune
    # counts returned above, which reflect final polarity, not final state.
    for node, s in state.items():
        if s in ("Aligned", "Opposed"):
            state[node] = "Immune"

    return {
        "aligned": aligned_count,
        "opposed": opposed_count,
        "immune": immune_count,
        "state_map": state,
    }


def _check_immune(node, state, same_polarity_count, in_degree):
    if in_degree[node] > 0 and same_polarity_count[node] > in_degree[node] / 2:
        state[node] = "Immune"


def _resolve_influence(u_info: str, edge_sign: int,
                        conformity: float, reactance: float,
                        weight: float) -> tuple:
    """
    Backfire cascade resolution:
      Pro  + positive edge -> Pro,  prob = conformity * weight
      Pro  + negative edge -> Anti, prob = reactance * weight   (distrust gateway)
      Anti + positive edge -> Anti, prob = conformity * weight
      Anti + negative edge -> Pro,  prob = reactance * weight   (backfire cascade)
    """
    if u_info == "Pro":
        if edge_sign == 1:
            return "Pro", conformity * weight
        else:
            return "Anti", reactance * weight
    elif u_info == "Anti":
        if edge_sign == 1:
            return "Anti", conformity * weight
        else:
            return "Pro", reactance * weight

    return None, 0.0