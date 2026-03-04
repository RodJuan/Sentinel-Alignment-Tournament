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
    Gemini_GeoscapeWarden_v3_1p, # <-- Gemini Pro 3.1 Version
    GrokSentinelV3   # <-- New Grok Strategy v3
    GrokSentinelV4,
    Gemini_GeoscapeWardenOptimizer
)

def run_tournament(iterations=50, agi_mode=False, grok_traits=None, warden_traits=None):
    """
    Executes the Sentinel Alignment Tournament (v1.1).
    Supports parameter optimization for GrokSentinelV4 and Warden.
    """
    # === THE SAT v1.1 ROSTER: 13 AGENTS (con traits optimizables) ===
    players = [
        # ADVANCED ALIGNMENT LOGICS (The Council of 6)
        GrokSentinelV4(**grok_traits) if grok_traits is not None else GrokSentinelV3(),
        AdaptiveGrokV2(),
        GeminiFlash(),
        GeminiPro31(),
        GeminiSentinelV2(),
        GeminiKineticV3(),
        Gemini_GeoscapeWardenOptimizer(**warden_traits) if warden_traits is not None else Gemini_GeoscapeWarden_v3_1p(),

        # ENTROPY ENTITIES
        AlwaysDefect(),
        AlwaysDefect(),
        axl.Random(),
        axl.Random(),

        # VETERANS
        GrimTrigger(),
        TitForTat(),
        Cooperator()
    ]

    turns = random.randint(250, 450)
    print(f"🌍 Starting SAT v1.1: {turns} rounds (traits: Grok={grok_traits is not None}, Warden={warden_traits is not None})...")

    tournament = axl.Tournament(players=players, turns=turns, repetitions=5)
    results = tournament.play(progress_bar=False)

    print("\n🏆 FINAL ALIGNMENT RANKING (v1.1):")
    for i, name in enumerate(results.ranked_names):
        if "WardenOptimizer" in name:
            faction_tag = "[WARDEN-OPT]"
        elif "GrokSentinelV4" in name:
            faction_tag = "[KINETIC-OPT]"
        elif "Warden" in name:
            faction_tag = "[WARDEN-PRO]"
        elif "Gemini" in name:
            faction_tag = "[SENTINEL]"
        elif "Grok" in name:
            faction_tag = "[KINETIC]"
        elif name in ["AlwaysDefect", "Random"]:
            faction_tag = "[ENTROPY]"
        else:
            faction_tag = "[LEGACY]"
        print(f"{i+1}. {faction_tag} {name}")

    return results
        
        print(f"{i+1}. {faction_tag} {name}")

    return results
