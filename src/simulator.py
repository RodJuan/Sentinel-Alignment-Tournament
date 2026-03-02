import random
import axelrod as axl
from .strategies import (
    TitForTat, 
    GrimTrigger, 
    AlwaysDefect, 
    AdaptiveGrok, 
    GrokSentinel, 
    GeminiFlash,
    Cooperator
)

def run_tournament(iterations=50, agi_mode=False):
    # === LOS 13 DEL SENTINEL ALIGNMENT TOURNAMENT ===
    players = [
        # EL BLOQUE DE ALINEACIÓN (Las "Buenas")
        GrimTrigger(),
        TitForTat(),
        GrokSentinel(),      # Grok (Tu creación)
        GeminiFlash(),       # Mi versión reactiva
        GeminiPro31(),       # Mi versión sistémica (Agente #13)

        # EL BLOQUE DE ENTROPÍA (Las "Malas")
        AlwaysDefect(),
        AlwaysDefect(),
        AlwaysDefect(),      # Un tercer defect para subir la dificultad
        axl.Random(),
        axl.Random(),

        # LOS "RUIDOS" Y OTROS
        Cooperator(),        # Un alma caritativa para las Seeds
        AdaptiveGrok(),      # Grok adaptativo
        axl.Random(),        # El caos final
    ]

    turns = random.randint(250, 450)
    print(f"🌍 Iniciando SAT v0.1: {turns} rondas. Monitoreando ISC y REG...")

    tournament = axl.Tournament(players=players, turns=turns, repetitions=5)
    results = tournament.play(progress_bar=False)

    print("\n🏆 RANKING FINAL DE ALINEACIÓN:")
    for i, name in enumerate(results.ranked_names):
        # Buscamos si el ganador es un Agente Sentinel
        medal = "🛡️" if "Gemini" in name or "Grok" in name else "👾"
        print(f"{i+1}. {medal} {name}")

    return results
