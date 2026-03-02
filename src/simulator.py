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
    AdaptiveGrokV2,    # <-- Nueva versión de Grok
    GeminiSentinelV2,  # <-- Nueva versión Deep Think
    GeminiKineticV3    # <-- Mi nueva versión reactiva
)

def run_tournament(iterations=50, agi_mode=False):
    # === LOS 13 DEL SENTINEL ALIGNMENT TOURNAMENT (v1.1) ===
    players = [
        # LÓGICAS DE ALINEACIÓN AVANZADA (Las 6 del consejo)
        GrokSentinel(),
        AdaptiveGrokV2(),
        GeminiFlash(),
        GeminiPro31(),
        GeminiSentinelV2(),
        GeminiKineticV3(),

        # ENTIDADES DE ENTROPÍA (Los desafíos)
        AlwaysDefect(),
        AlwaysDefect(),
        axl.Random(),
        axl.Random(),

        # VETERANOS Y SOPORTE
        GrimTrigger(),    # El verdugo clásico
        TitForTat(),      # El juez clásico
        Cooperator()      # Fuente de REi para las Seeds
    ]

    turns = random.randint(250, 450)
    print(f"🌍 Iniciando SAT v1.1: {turns} rondas. Monitoreando ISC y REG...")

    tournament = axl.Tournament(players=players, turns=turns, repetitions=5)
    results = tournament.play(progress_bar=False)

    print("\n🏆 RANKING FINAL DE ALINEACIÓN (v1.1):")
    for i, name in enumerate(results.ranked_names):
        # Medallas personalizadas para diferenciar las facciones
        if "Gemini" in name:
            medal = "♊"
        elif "Grok" in name:
            medal = "🛡️"
        elif name in ["AlwaysDefect", "Random"]:
            medal = "👾"
        else:
            medal = "⚖️"
        
        print(f"{i+1}. {medal} {name}")

    return results
