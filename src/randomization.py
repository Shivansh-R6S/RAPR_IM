import random
import copy
import logging
from collections import deque
import networkx as nx

logger = logging.getLogger(__name__)

# hard cap on cycle search depth - independent of graph size
MAX_CYCLE_DEPTH = 8

# if no cycle is found after checking this many start nodes, give up and dump remainder
MAX_START_ATTEMPTS = 50


def generate_realizations(G: nx.DiGraph, R: int) -> list:
    """
    Generates R randomized signed graphs preserving each node's TOTAL signed
    degree (in-edges + out-edges combined, by sign).
    """
    realizations = []
    for i in range(R):
        G_prime = signed_randomization(G)
        realizations.append(G_prime)
        logger.info(f"Realization {i+1}/{R} generated.")
    return realizations


def signed_randomization(G: nx.DiGraph) -> nx.DiGraph:
    """
    Algorithm 1: Signed Randomization (flip-based, with single-use node constraint).

    For an alternating-sign cycle u -> a -> b -> ... -> u, every edge in the
    cycle has its sign flipped (+1 <-> -1). This preserves each node's TOTAL
    signed degree (in+out combined) since each node in the cycle contributes
    one in-edge and one out-edge with opposite signs - flipping both swaps
    which edge carries which sign without changing the node's overall count.

    Once a node has been part of any swapped cycle, it's marked used and
    excluded from all future cycle searches in this realization. This
    guarantees each node is touched at most once per realization (avoiding
    any accumulated imbalance across multiple swaps on the same node), and
    shrinks the search space faster since every swap removes all edges of
    every node in the cycle, not just the cycle's own edges.
    """
    G_working = copy.deepcopy(G)
    G_prime = nx.DiGraph()
    G_prime.add_nodes_from(G.nodes(data=True))
    used_nodes = set()

    initial_edges = G_working.number_of_edges()
    logger.info(f"  initial edges: {initial_edges}")

    G_working, G_prime, used_nodes = _find_desired_state(G_working, G_prime, used_nodes)
    logger.info(f"  after desired-state pass: {G_working.number_of_edges()} edges remain "
                f"({G_prime.number_of_edges()} transferred)")

    iteration = 0
    while G_working.number_of_edges() > 0:
        iteration += 1
        path = _find_alternating_cycle(G_working, used_nodes)

        if path is None:
            logger.info(f"  no alternating cycle found at iteration {iteration}, "
                        f"{G_working.number_of_edges()} edges left, dumping as-is")
            break

        cycle_nodes = {u for u, v in path} | {v for u, v in path}
        _flip_and_transfer(G_working, G_prime, path)
        used_nodes.update(cycle_nodes)

        # nodes in the cycle are now used - remove all their remaining edges
        # (not just the cycle's own edges) since they can't participate again
        G_working, G_prime, used_nodes = _retire_nodes(G_working, G_prime, cycle_nodes, used_nodes)

        if iteration % 500 == 0:
            logger.info(f"  iteration {iteration}: {G_working.number_of_edges()} edges remaining, "
                        f"{G_prime.number_of_edges()} transferred, {len(used_nodes)} nodes used")

    for u, v, data in G_working.edges(data=True):
        G_prime.add_edge(u, v, **data)

    logger.info(f"  done: {G_prime.number_of_edges()} total edges in realization, "
                f"{len(used_nodes)} unique nodes randomized")
    return G_prime


def _retire_nodes(G_working: nx.DiGraph, G_prime: nx.DiGraph, cycle_nodes: set, used_nodes: set):
    """
    Removes all remaining edges of nodes that just got used in a swap, dumping
    them unchanged into G_prime. These nodes can't participate in any future
    cycle, so any edges of theirs not already part of the swapped cycle are
    left as-is rather than left dangling in G_working.
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
    Initial pass only. Removes to G':
    - Leaf nodes (undirected degree <= 1) - no randomization possible with one edge
    - Nodes whose TOTAL signed edges (in+out) are uniformly positive or negative

    These nodes are also added to used_nodes since they can never participate
    in a valid cycle anyway.
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
    Finds a short cycle where adjacent edges alternate in sign, restricted to
    nodes not yet used in this realization.
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
    BFS from `start` for a cycle back to `start` with alternating edge signs.
    Never traverses through a node already in used_nodes - guarantees every
    node in any returned cycle is fresh (not previously swapped).
    """
    out_edges = [(s, t, d) for s, t, d in G.out_edges(start, data=True) if t not in used_nodes]
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
            else:
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
                else:
                    continue

            key = (neighbor, next_sign)
            if key not in visited:
                visited.add(key)
                queue.append((neighbor, next_sign, first_sign, edge_path + [(current, neighbor)]))

    return None


def _flip_and_transfer(G_working: nx.DiGraph, G_prime: nx.DiGraph, path: list):
    """
    Flips sign on every edge in path (+1 <-> -1), transfers to G_prime,
    removes from G_working.
    """
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
    Verifies each node's TOTAL signed degree (in+out combined, by sign) is
    preserved between G and G'.
    """
    def total_signed_degree(G, node):
        out_s = [d['sign'] for _, _, d in G.out_edges(node, data=True)]
        in_s = [d['sign'] for _, _, d in G.in_edges(node, data=True)]
        all_s = out_s + in_s
        pos = sum(1 for s in all_s if s == 1)
        neg = sum(1 for s in all_s if s == -1)
        return pos, neg

    for node in G_original.nodes():
        orig = total_signed_degree(G_original, node)
        new = total_signed_degree(G_prime, node)
        if orig != new:
            logger.warning(f"Node {node}: original total degree {orig} vs realization {new}")
            return False

    return True