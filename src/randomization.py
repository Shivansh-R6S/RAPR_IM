import random
import copy
import logging
from collections import deque
import networkx as nx

logger = logging.getLogger(__name__)

MAX_CYCLE_DEPTH = 10
MAX_START_ATTEMPTS = 100
ROUNDS_PER_REALIZATION = 5

# standard deviation for per-realization node attribute perturbation
# small enough to keep attributes recognizable, large enough to affect diffusion
NODE_ATTR_PERTURB_STD = 0.05


def generate_realizations(G: nx.DiGraph, R: int) -> list:
    """
    Generates R randomized signed graphs. Each realization differs via:
    1. Multi-round single-use sign randomization (ROUNDS_PER_REALIZATION passes)
    2. Per-realization node attribute perturbation on conformity and reactance
    Both preserve total signed degree (in+out) per node.
    """
    realizations = []
    for i in range(R):
        G_prime = signed_randomization(G, realization_seed=i)
        realizations.append(G_prime)
        logger.info(f"Realization {i+1}/{R} complete.")
    return realizations


def signed_randomization(G: nx.DiGraph, realization_seed: int = 0) -> nx.DiGraph:
    """
    Runs ROUNDS_PER_REALIZATION sequential single-use sign randomization rounds,
    then applies per-realization node attribute perturbation.

    realization_seed is used to make node perturbation deterministic and unique
    per realization while keeping the sign randomization stochastic.
    """
    G_current = copy.deepcopy(G)

    for round_num in range(1, ROUNDS_PER_REALIZATION + 1):
        logger.info(f"  round {round_num}/{ROUNDS_PER_REALIZATION}: "
                    f"{G_current.number_of_edges()} edges")
        G_current = _single_round(G_current, round_num)

    _perturb_node_attributes(G_current, realization_seed)
    return G_current


def _perturb_node_attributes(G: nx.DiGraph, realization_seed: int):
    """
    Adds small Gaussian noise (std=NODE_ATTR_PERTURB_STD) to each node's
    conformity and reactance, clamped to [0, 1].

    Uses a seeded RNG so perturbations are reproducible per realization index
    but differ across realizations. This models behavioral uncertainty — the
    paper provides no method for estimating these from real data, so treating
    them as uncertain across realizations is methodologically consistent.
    """
    rng = random.Random(realization_seed * 1000 + 7)  # unique seed per realization

    for node in G.nodes():
        k = G.nodes[node].get("conformity", 0.5)
        r = G.nodes[node].get("reactance", 0.5)

        G.nodes[node]["conformity"] = max(0.0, min(1.0, k + rng.gauss(0, NODE_ATTR_PERTURB_STD)))
        G.nodes[node]["reactance"] = max(0.0, min(1.0, r + rng.gauss(0, NODE_ATTR_PERTURB_STD)))


def _single_round(G: nx.DiGraph, round_num: int) -> nx.DiGraph:
    """
    One single-use randomization pass over G.
    Returns G' with some edges sign-flipped, all topology and total degree preserved.
    """
    G_working = copy.deepcopy(G)
    G_prime = nx.DiGraph()
    G_prime.add_nodes_from(G.nodes(data=True))
    used_nodes = set()

    G_working, G_prime, used_nodes = _find_desired_state(G_working, G_prime, used_nodes)

    iteration = 0
    while G_working.number_of_edges() > 0:
        iteration += 1
        path = _find_alternating_cycle(G_working, used_nodes)

        if path is None:
            break

        cycle_nodes = {u for u, v in path} | {v for u, v in path}
        _flip_and_transfer(G_working, G_prime, path)
        used_nodes.update(cycle_nodes)
        G_working, G_prime, used_nodes = _retire_nodes(G_working, G_prime, cycle_nodes, used_nodes)

    for u, v, data in G_working.edges(data=True):
        G_prime.add_edge(u, v, **data)

    logger.info(f"    {iteration} cycles found, {len(used_nodes)} nodes touched")
    return G_prime


def _retire_nodes(G_working: nx.DiGraph, G_prime: nx.DiGraph, cycle_nodes: set, used_nodes: set):
    """
    Dumps all remaining edges of used nodes into G_prime unchanged and removes
    them from G_working. Neighbor nodes keep their other edges in G_working.
    """
    for node in cycle_nodes:
        if not G_working.has_node(node):
            continue
        for u, v, data in list(G_working.out_edges(node, data=True)):
            G_prime.add_edge(u, v, **data)
        for u, v, data in list(G_working.in_edges(node, data=True)):
            if not G_prime.has_edge(u, v):
                G_prime.add_edge(u, v, **data)
        G_working.remove_node(node)

    return G_working, G_prime, used_nodes


def _find_desired_state(G_working: nx.DiGraph, G_prime: nx.DiGraph, used_nodes: set):
    """
    Initial pass: removes leaf nodes and uniform-sign nodes to G_prime.
    These can never participate in a valid alternating cycle.
    """
    check_set = deque(G_working.nodes())

    while check_set:
        node = check_set.popleft()
        if not G_working.has_node(node):
            continue

        in_edges = list(G_working.in_edges(node, data=True))
        out_edges = list(G_working.out_edges(node, data=True))

        if not in_edges and not out_edges:
            continue

        neighbors = set([u for u, _, _ in in_edges] + [v for _, v, _ in out_edges])

        remove = False
        if len(neighbors) <= 1:
            remove = True
        else:
            all_signs = [d.get("sign") for _, _, d in out_edges] + \
                        [d.get("sign") for _, _, d in in_edges]
            all_signs = [s for s in all_signs if s is not None]
            if all_signs and (all(s == 1 for s in all_signs) or all(s == -1 for s in all_signs)):
                remove = True

        if remove:
            affected = set(neighbors)
            for u, v, data in list(G_working.out_edges(node, data=True)):
                G_prime.add_edge(u, v, **data)
            for u, v, data in list(G_working.in_edges(node, data=True)):
                if not G_prime.has_edge(u, v):
                    G_prime.add_edge(u, v, **data)
            G_working.remove_node(node)
            used_nodes.add(node)
            check_set.extend(affected)

    return G_working, G_prime, used_nodes


def _find_alternating_cycle(G_working: nx.DiGraph, used_nodes: set):
    """
    Finds a short alternating-sign cycle restricted to unused nodes.
    Tries up to MAX_START_ATTEMPTS random start nodes.
    """
    nodes = [n for n in G_working.nodes() if n not in used_nodes]
    if not nodes:
        return None

    random.shuffle(nodes)
    attempts = min(MAX_START_ATTEMPTS, len(nodes))

    for start in nodes[:attempts]:
        path = _bfs_alternating_cycle(G_working, start, used_nodes)
        if path is not None:
            return path

    return None


def _bfs_alternating_cycle(G: nx.DiGraph, start, used_nodes: set):
    """
    BFS from `start` finding a cycle back to `start` with alternating edge signs.
    Never traverses through used nodes. Tracks (node, last_sign) to stay
    polynomial on high-degree hubs. Validates wraparound sign to ensure
    start node's total degree is preserved by the flip.
    """
    out_edges = [(s, t, d) for s, t, d in G.out_edges(start, data=True)
                 if t not in used_nodes or t == start]
    if not out_edges:
        return None

    visited = set()
    queue = deque()

    for _, neighbor, data in out_edges:
        sign = data.get("sign")
        queue.append((neighbor, sign, sign, [(start, neighbor)]))
        visited.add((neighbor, sign))

    while queue:
        current, last_sign, first_sign, edge_path = queue.popleft()

        if current == start:
            if last_sign != first_sign:
                return edge_path
            continue

        if len(edge_path) >= MAX_CYCLE_DEPTH:
            continue

        for _, neighbor, data in G.out_edges(current, data=True):
            if neighbor != start and neighbor in used_nodes:
                continue

            next_sign = data.get("sign")

            if next_sign == last_sign:
                continue

            if neighbor == start:
                if next_sign != first_sign:
                    return edge_path + [(current, neighbor)]
                continue

            key = (neighbor, next_sign)
            if key not in visited:
                visited.add(key)
                queue.append((neighbor, next_sign, first_sign, edge_path + [(current, neighbor)]))

    return None


def _flip_and_transfer(G_working: nx.DiGraph, G_prime: nx.DiGraph, path: list):
    """Flips sign on every edge in path, transfers to G_prime, removes from G_working."""
    for u, v in path:
        if not G_working.has_edge(u, v):
            continue
        data = dict(G_working[u][v])
        data["sign"] = -data["sign"]
        data["weight"] = abs(data["sign"])
        G_prime.add_edge(u, v, **data)
        G_working.remove_edge(u, v)


def verify_total_degree_preserved(G_original: nx.DiGraph, G_prime: nx.DiGraph) -> bool:
    """
    Verifies each node's TOTAL signed degree (in+out combined) is preserved.
    Note: node attributes (conformity, reactance) are intentionally perturbed
    and are NOT checked here - only edge sign structure is verified.
    """
    def total_signed_degree(G, node):
        out_s = [d['sign'] for _, _, d in G.out_edges(node, data=True)]
        in_s = [d['sign'] for _, _, d in G.in_edges(node, data=True)]
        all_s = out_s + in_s
        return sum(1 for s in all_s if s == 1), sum(1 for s in all_s if s == -1)

    for node in G_original.nodes():
        orig = total_signed_degree(G_original, node)
        new = total_signed_degree(G_prime, node)
        if orig != new:
            logger.warning(f"Node {node}: original total degree {orig} vs realization {new}")
            return False

    return True