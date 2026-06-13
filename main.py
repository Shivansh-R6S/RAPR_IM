import logging
from src.graph import load_config, load_graph, assign_node_attributes, graph_stats, extract_subgraph
from src.diffusion import run_paic_dgbc
from src.randomization import generate_realizations, verify_signed_degree_preserved

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

if __name__ == "__main__":
    config = load_config("config.yaml")

    G = load_graph(config)
    G = assign_node_attributes(G, config)
    stats = graph_stats(G)

    print("\nGraph Summary:")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    # subgraph for validation
    G_sub = extract_subgraph(G, n_nodes=5000)

    # diffusion sanity check on subgraph
    seed_set = set(list(G_sub.nodes())[:5])
    result = run_paic_dgbc(G_sub, seed_set, mc_runs=config["model"]["mc_simulations"])
    print("\nDiffusion Result (subgraph):")
    for k, v in result.items():
        print(f"  {k}: {v:.2f}")

    # signed randomization on subgraph
    print("\nGenerating 2 realizations on subgraph...")
    realizations = generate_realizations(G_sub, R=2)

    print("\nVerifying signed degree preservation:")
    for i, G_prime in enumerate(realizations):
        preserved = verify_signed_degree_preserved(G_sub, G_prime)
        pos = sum(1 for _, _, d in G_prime.edges(data=True) if d.get("sign") == 1)
        neg = sum(1 for _, _, d in G_prime.edges(data=True) if d.get("sign") == -1)
        print(f"  Realization {i+1}: edges={G_prime.number_of_edges()}, "
              f"pos={pos}, neg={neg}, degree_preserved={preserved}")