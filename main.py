import logging
from src.graph import load_config, load_graph, assign_node_attributes, graph_stats, extract_subgraph
from src.diffusion import run_paic_dgbc
from src.randomization import generate_realizations, verify_total_degree_preserved
from src.seed_selection import modified_greedy, sigma_plus

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

    # diffusion sanity check on full graph
    seed_set = set(list(G.nodes())[:5])
    result = run_paic_dgbc(G, seed_set, mc_runs=config["model"]["mc_simulations"])
    print("\nDiffusion Result (full graph):")
    for k, v in result.items():
        print(f"  {k}: {v:.2f}")

    # signed randomization on full graph - generates R realizations per config
    R = config["model"]["R"]
    print(f"\nGenerating {R} realizations on full graph...")
    realizations = generate_realizations(G, R=R)

    print("\nVerifying total signed degree preservation:")
    for i, G_prime in enumerate(realizations):
        preserved = verify_total_degree_preserved(G, G_prime)
        pos = sum(1 for _, _, d in G_prime.edges(data=True) if d.get("sign") == 1)
        neg = sum(1 for _, _, d in G_prime.edges(data=True) if d.get("sign") == -1)
        print(f"  Realization {i+1}: edges={G_prime.number_of_edges()}, "
              f"pos={pos}, neg={neg}, degree_preserved={preserved}")

    # --- Modified Greedy seed selection (Algorithm 2) ---
    # Greedy is O(k * |candidates| * R) diffusion estimates, so run it on a
    # dense subgraph rather than the full graph for tractability. Realizations
    # are regenerated on the subgraph so candidate nodes and the realizations
    # they're evaluated on are consistent with each other.
    subgraph_size = config.get("model", {}).get("greedy_subgraph_size", 2000)
    G_sub = extract_subgraph(G, n_nodes=subgraph_size, strategy="degree")
    sub_realizations = generate_realizations(G_sub, R=R)

    mc_simulations = config["model"]["mc_simulations"]
    seed_sizes = config["model"]["seed_sizes"]
    max_k = max(seed_sizes)

    print(f"\nRunning Modified Greedy on subgraph "
          f"({G_sub.number_of_nodes()} nodes) for k up to {max_k}...")
    full_seed_order = modified_greedy(
        sub_realizations, k=max_k, mc_runs=mc_simulations
    )

    print("\nSeed sets by size (prefix of greedy order):")
    for k in seed_sizes:
        S_k = full_seed_order[:k]
        avg_sigma = sum(
            sigma_plus(G_prime, set(S_k), mc_simulations)
            for G_prime in sub_realizations
        ) / len(sub_realizations)
        print(f"  k={k}: seeds={S_k}, avg sigma+ across realizations={avg_sigma:.2f}")