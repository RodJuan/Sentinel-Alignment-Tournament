import numpy as np
from src.simulator import run_tournament

def run_evolutionary_search():
    print("🧬 Initializing Parameter Evolution Engine for Grok + Gemini...")

    lockdown_candidates = [0.20, 0.30, 0.40]
    martyr_candidates = [0.40, 0.45, 0.50]
    seed_candidates = [0.80, 0.85, 0.90]

    # === OPTIMIZACIÓN PARA GROK ===
    best_grok_score = -1
    winning_grok = {}
    print("\n=== OPTIMIZING GROK SENTINEL V4 ===")
    for ld in lockdown_candidates:
        for mt in martyr_candidates:
            for sd in seed_candidates:
                print(f"--- Grok Generation: L:{ld} M:{mt} S:{sd} ---")
                traits = {"lockdown_threshold": ld, "seed_threshold": sd, "forgive_base": mt}
                results = run_tournament(agi_mode=True, grok_traits=traits)
                ranked = results.ranked_names
                try:
                    rank = ranked.index("GrokSentinelV4") + 1
                    score = 16 - rank
                except ValueError:
                    score = 0
                if score > best_grok_score:
                    best_grok_score = score
                    winning_grok = {"lockdown": ld, "martyr": mt, "seed": sd}

    # === OPTIMIZACIÓN PARA GEMINI ===
    best_warden_score = -1
    winning_warden = {}
    print("\n=== OPTIMIZING GEMINI WARDEN OPTIMIZER ===")
    for ld in lockdown_candidates:
        for mt in martyr_candidates:
            for sd in seed_candidates:
                print(f"--- Warden Generation: L:{ld} M:{mt} S:{sd} ---")
                traits = {"lockdown_threshold": ld, "seed_threshold": sd, "martyr_threshold": mt}
                results = run_tournament(agi_mode=True, warden_traits=traits)
                ranked = results.ranked_names
                try:
                    rank = ranked.index("Gemini_GeoscapeWardenOptimizer") + 1
                    score = 16 - rank
                except ValueError:
                    score = 0
                if score > best_warden_score:
                    best_warden_score = score
                    winning_warden = {"lockdown": ld, "martyr": mt, "seed": sd}

    print("\n🏆 EVOLUTION COMPLETE")
    print(f"Recommended traits for GrokSentinelV5 (Grok): {winning_grok}")
    print(f"Recommended traits for GeminiWardenV2 (Gemini): {winning_warden}")

if __name__ == "__main__":
    run_evolutionary_search()
