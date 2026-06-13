import networkx as nx
import numpy as np
import yaml
import logging
import random

logger = logging.getLogger(__name__)


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_graph(config: dict) -> nx.DiGraph:
    path = config["data"]["path"]
    delimiter = config["data"]["delimiter"]

    G = nx.DiGraph()

    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split(delimiter)
            if len(parts) != 3:
                continue
            u, v, sign = int(parts[0]), int(parts[1]), int(parts[2])
            if sign not in (1, -1):
                logger.warning(f"Unexpected sign value {sign} on edge ({u},{v}), skipping.")
                continue
            G.add_edge(u, v, sign=sign, weight=abs(sign))

    logger.info(f"Loaded graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G


def assign_node_attributes(G: nx.DiGraph, config: dict, seed: int = 42) -> nx.DiGraph:
    """
    Assigns conformity (k) and reactance (r) to each node via uniform random sampling in [0, 1].
    Seed ensures reproducibility across runs.
    """
    rng = random.Random(seed)

    for node in G.nodes():
        G.nodes[node]["conformity"] = round(rng.uniform(0, 1), 4)
        G.nodes[node]["reactance"] = round(rng.uniform(0, 1), 4)
        G.nodes[node]["state"] = "Neutral"

    logger.info(f"Node attributes assigned randomly (seed={seed}).")
    return G


def extract_subgraph(G: nx.DiGraph, n_nodes: int = 5000, strategy: str = "degree") -> nx.DiGraph:
    """
    Extracts a subgraph of n_nodes nodes.
    strategy='degree': top n_nodes by out-degree — dense, well-connected, representative for diffusion.
    strategy='random': random sample.
    """
    if strategy == "degree":
        top_nodes = sorted(G.nodes(), key=lambda x: G.out_degree(x), reverse=True)[:n_nodes]
    else:
        top_nodes = random.sample(list(G.nodes()), n_nodes)

    subgraph = G.subgraph(top_nodes).copy()
    logger.info(f"Subgraph: {subgraph.number_of_nodes()} nodes, {subgraph.number_of_edges()} edges")

    pos = sum(1 for _, _, d in subgraph.edges(data=True) if d.get("sign") == 1)
    neg = subgraph.number_of_edges() - pos
    logger.info(f"Subgraph signs: pos={pos}, neg={neg}, ratio={round(pos / subgraph.number_of_edges(), 4)}")

    return subgraph


def graph_stats(G: nx.DiGraph) -> dict:
    edges = G.edges(data=True)
    pos_edges = sum(1 for _, _, d in edges if d.get("sign") == 1)
    neg_edges = G.number_of_edges() - pos_edges

    stats = {
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "positive_edges": pos_edges,
        "negative_edges": neg_edges,
        "pos_ratio": round(pos_edges / G.number_of_edges(), 4),
        "avg_out_degree": round(sum(d for _, d in G.out_degree()) / G.number_of_nodes(), 4),
    }

    for k, v in stats.items():
        logger.info(f"  {k}: {v}")

    return stats