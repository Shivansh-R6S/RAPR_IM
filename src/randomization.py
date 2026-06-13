import random
import copy
import logging
import networkx as nx

logger = logging.getLogger(__name__)


def generate_realizations(G: nx.DiGraph, R: int) -> list:
    """
    Generates R randomized signed graphs preserving each node's signed degree.
    """
    realizations = []
    for i in range(R):
        G_prime = signed_randomization(G)
        realizations.append(G_prime)
        logger.info(f"Realization {i+1}/{R} generated.")
    return realizations


def signed_randomization(G: nx.DiGraph) -> nx.DiGraph:
    """
    Algorithm 1: Signed Randomization.
    Returns G' with randomized signs, preserving signed degree and topology.
    """
    G_working = copy.deepcopy(G)
    G_prime = nx.DiGraph()
    G_prime.add_nodes_from(G.nodes(data=True))

    flag = True
    while flag:
        G_working, G_prime = _find_desired_state(G_working, G_prime)

        if G_working.number_of_edges() == 0:
            flag = False
            break

        path = _find_alternating_cycle(G_working)

        if path is None:
            flag = False
        else:
            _swap_and_transfer(G_working, G_prime, path)

        if G_working.number_of_edges() == 0:
            flag = False

    # transfer remaining edges unchanged
    for u, v, data in G_working.edges(data=True):
        G_prime.add_edge(u, v, **data)

    return G_prime


def _find_desired_state(G_working: nx.DiGraph, G_prime: nx.DiGraph):
    """
    Iteratively removes to G':
    - Leaf nodes (undirected degree == 1)
    - Fully positive or fully negative nodes

    Repeats until stable — remaining graph is the desired state.
    """
    changed = True
    while changed:
        changed = False
        nodes_to_remove = []

        for node in list(G_working.nodes()):
            in_edges = list(G_working.in_edges(node, data=True))
            out_edges = list(G_working.out_edges(node, data=True))

            if not in_edges and not out_edges:
                continue

            neighbors = set(
                [u for u, _, _ in in_edges] + [v for _, v, _ in out_edges]
            )
            if len(neighbors) <= 1:
                nodes_to_remove.append(node)
                changed = True
                continue

            signs = [d.get("sign") for _, _, d in out_edges] + \
                    [d.get("sign") for _, _, d in in_edges]
            signs = [s for s in signs if s is not None]

            if signs and (all(s == 1 for s in signs) or all(s == -1 for s in signs)):
                nodes_to_remove.append(node)
                changed = True

        for node in nodes_to_remove:
            for u, v, data in list(G_working.out_edges(node, data=True)):
                G_prime.add_edge(u, v, **data)
            for u, v, data in list(G_working.in_edges(node, data=True)):
                if not G_prime.has_edge(u, v):
                    G_prime.add_edge(u, v, **data)
            G_working.remove_node(node)

    return G_working, G_prime


def _find_alternating_cycle(G_working: nx.DiGraph):
    """
    Finds a cycle where adjacent edges alternate in sign.
    Randomly shuffles start nodes for diversity across realizations.
    Returns list of (u, v) edge tuples or None.
    """
    nodes = list(G_working.nodes())
    random.shuffle(nodes)

    for start in nodes:
        path = _dfs_alternating_cycle(G_working, start)
        if path is not None:
            return path

    return None


def _dfs_alternating_cycle(G: nx.DiGraph, start):
    """
    Iterative DFS from `start` finding a cycle back to `start`
    with alternating edge signs. Caps path length to avoid runaway searches.
    """
    out_edges = list(G.out_edges(start, data=True))
    if not out_edges:
        return None

    max_path_len = min(20, G.number_of_nodes() * 2)

    # stack: (current_node, last_sign, edge_path)
    stack = []
    for _, neighbor, data in out_edges:
        sign = data.get("sign")
        stack.append((neighbor, sign, [(start, neighbor)]))

    while stack:
        current, last_sign, edge_path = stack.pop()

        if len(edge_path) > 1 and current == start:
            return edge_path

        if len(edge_path) >= max_path_len:
            continue

        visited_in_path = {u for u, v in edge_path} | {v for u, v in edge_path}

        for _, neighbor, data in G.out_edges(current, data=True):
            next_sign = data.get("sign")

            if next_sign == last_sign:
                continue

            if neighbor != start and neighbor in visited_in_path:
                continue

            stack.append((neighbor, next_sign, edge_path + [(current, neighbor)]))

    return None


def _swap_and_transfer(G_working: nx.DiGraph, G_prime: nx.DiGraph, path: list):
    """
    Flips signs on all edges in path, transfers to G_prime, removes from G_working.
    """
    for u, v in path:
        if not G_working.has_edge(u, v):
            continue
        data = dict(G_working[u][v])
        data["sign"] = -data["sign"]
        data["weight"] = abs(data["sign"])
        G_prime.add_edge(u, v, **data)
        G_working.remove_edge(u, v)


def verify_signed_degree_preserved(G_original: nx.DiGraph, G_prime: nx.DiGraph) -> bool:
    """
    Verifies d+ and d- are preserved for every node between G and G'.
    """
    for node in G_original.nodes():
        orig_pos = sum(1 for _, _, d in G_original.out_edges(node, data=True) if d.get("sign") == 1)
        orig_neg = sum(1 for _, _, d in G_original.out_edges(node, data=True) if d.get("sign") == -1)

        new_pos = sum(1 for _, _, d in G_prime.out_edges(node, data=True) if d.get("sign") == 1)
        new_neg = sum(1 for _, _, d in G_prime.out_edges(node, data=True) if d.get("sign") == -1)

        if orig_pos != new_pos or orig_neg != new_neg:
            logger.warning(
                f"Node {node}: original ({orig_pos}+, {orig_neg}-) "
                f"vs realization ({new_pos}+, {new_neg}-)"
            )
            return False

    return True