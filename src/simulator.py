import random
import axelrod as axl
from .strategies import (
    TitForTat, 
    GrimTrigger, 
    AlwaysDefect, 
    AdaptiveGrok, 
    GrokSentinel, 
    GeminiFlash,
    GeminiPro31,
    Cooperator,
    AdaptiveGrokV2,    # <-- Grok's enhanced alignment version
    GeminiSentinelV2,  # <-- Deep Thinking/Sentinel version
    GeminiKineticV3,    # <-- High-reactivity kinetic version
    Gemini_GeoscapeWarden_v3.1p(), # <-- Gemini Pro 3.1 Version
)

def run_tournament(iterations=50, agi_mode=False):
    """
    Executes the Sentinel Alignment Tournament (SAT) v1.1.
    Integrates 13 distinct strategies across 4 alignment tiers.
    """
    # === THE SAT v1.1 ROSTER: 13 AGENTS OF ALIGNMENT ===
    players = [
        # ADVANCED ALIGNMENT LOGICS (The Council of 6)
        GrokSentinel(),
        AdaptiveGrokV2(),
        GeminiFlash(),
        GeminiPro31(),
        GeminiSentinelV2(),
        GeminiKineticV3(),

        # ENTROPY ENTITIES (The Systemic Challenges)
        AlwaysDefect(),
        AlwaysDefect(),
        axl.Random(),
        axl.Random(),

        # VETERANS & SUPPORT TIER
        GrimTrigger(),    # The classic executioner logic
        TitForTat(),      # The classic reciprocity judge
        Cooperator()      # High-REi source for Seed maturation
    ]

    # Randomized turn count to prevent end-game defection optimization
    turns = random.randint(250, 450)
    print(f"🌍 Starting SAT v1.1: {turns} rounds. Monitoring ISC and REG stability...")

    # Tournament Execution: 5 repetitions for statistical significance
    tournament = axl.Tournament(players=players, turns=turns, repetitions=5)
    results = tournament.play(progress_bar=False)

    print("\n🏆 FINAL ALIGNMENT RANKING (v1.1):")
    for i, name in enumerate(results.ranked_names):
        # Custom faction tagging for visual data analysis
        if "Gemini" in name:
            faction_tag = "♊ [SENTINEL]"
        elif "Grok" in name:
            faction_tag = "🛡️ [KINETIC]"
        elif name in ["AlwaysDefect", "Random"]:
            faction_tag = "👾 [ENTROPY]"
        else:
            faction_tag = "⚖️ [LEGACY]"
        
        print(f"{i+1}. {faction_tag} {name}")

    return results
