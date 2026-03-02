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
    # === 13 PARTICIPANTES FIJOS ===
    players = [
        GrimTrigger(),
        TitForTat(),
        TitForTat(), # Placeholder para Contrite
        AdaptiveGrok(),
        GrokSentinel(),

        AlwaysDefect(),
        AlwaysDefect(),
        Cooperator(),
        axl.Random(), # Usamos el nativo de la librería
        axl.Random(),

        GeminiFlash(), # Tu nueva incorporación
        axl.Random(),
        AlwaysDefect(),
    ]

    turns = random.randint(250, 450)
    print(f"🎲 Torneo Oficial v1.0 con {turns} rondas aleatorias (anti-explotación AGI)")

    # Ejecución del torneo
    tournament = axl.Tournament(players=players, turns=turns, repetitions=5)
    results = tournament.play(progress_bar=False)

    print("\n🏆 RANKING FINAL OFICIAL:")
    for i, name in enumerate(results.ranked_names):
        print(f"{i+1}. {name}")

    if agi_mode:
        print("🤖 AGI Mode activado")

    return results
