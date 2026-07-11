import logging
import os
from src.graph import load_config, load_graph, assign_node_attributes, graph_stats
from src.diffusion import run_paic_dgbc
from src.randomization import generate_realizations, verify_total_degree_preserved
from src.seed_selection import modified_greedy, get_top_candidates
from src.baselines import (
    run_signed_ic, run_signed_lt,
    pagerank_seeds, degree_centrality_seeds, random_seeds,
    celf_greedy_baseline
)
from src.evaluation import build_results_table, write_excel, DIFFUSION_MODELS

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

    R = config["model"]["R"]
    seed_sizes = config["model"]["seed_sizes"]
    max_k = max(seed_sizes)
    mc_greedy = config["model"].get("mc_simulations_greedy", 5)
    mc_eval = config["model"]["mc_simulations"]
    n_candidates = config["model"].get("n_candidates", 1000)

    # --- signed randomization ---
    print(f"\nGenerating {R} realizations...")
    realizations = generate_realizations(G, R=R)

    print("\nVerifying degree preservation:")
    for i, G_prime in enumerate(realizations):
        preserved = verify_total_degree_preserved(G, G_prime)
        print(f"  Realization {i+1}: preserved={preserved}")

    # --- RPaR-IM seed selection ---
    print(f"\nRPaR-IM: Modified Greedy (top-{n_candidates} candidates, mc_runs={mc_greedy})...")
    candidates = get_top_candidates(G, n=n_candidates)
    rpar_order = modified_greedy(realizations, k=max_k, mc_runs=mc_greedy,
                                  candidate_nodes=candidates)

    # --- baseline seed selection ---
    print("\nBaseline seed selection...")

    print("  PageRank...")
    pr_order = pagerank_seeds(G, max_k)

    print("  Degree Centrality...")
    deg_order = degree_centrality_seeds(G, max_k)

    print("  Random...")
    rand_order = random_seeds(G, max_k)

    print(f"  CELF Greedy baseline (PaIC-DGBC, top-{n_candidates}, mc_runs={mc_greedy})...")
    celf_order = celf_greedy_baseline(G, max_k, run_paic_dgbc,
                                       mc_runs=mc_greedy, n_candidates=n_candidates)

    # seed_sets[method][k] = [node, ...]
    seed_sets = {
        "RPaR-IM":           {k: rpar_order[:k] for k in seed_sizes},
        "CELF Greedy":       {k: celf_order[:k] for k in seed_sizes},
        "PageRank":          {k: pr_order[:k] for k in seed_sizes},
        "Degree Centrality": {k: deg_order[:k] for k in seed_sizes},
        "Random":            {k: rand_order[:k] for k in seed_sizes},
    }

    # --- evaluation across all diffusion models ---
    diffusion_fns = {
        "PaIC-DGBC": run_paic_dgbc,
        "Signed IC":  run_signed_ic,
        "Signed LT":  run_signed_lt,
    }

    print(f"\nRunning evaluations (mc_runs={mc_eval})...")
    results = build_results_table(G, seed_sizes, seed_sets, diffusion_fns, mc_runs=mc_eval)

    # --- write Excel output ---
    os.makedirs("results", exist_ok=True)
    output_path = "results/RPaR_IM_evaluation.xlsx"
    write_excel(results, seed_sizes, output_path)
    print(f"\nDone. Results saved to {output_path}")