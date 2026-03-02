import random
from axelrod import Tournament, TitForTat, ContriteTitForTat, AlwaysDefect, Cooperator, Random
from axelrod.strategies.grim_trigger import GrimTrigger   # ← IMPORT SEGURO
from .strategies import AdaptiveGrok, GrokSentinel

def run_tournament(iterations=50, agi_mode=False):
    # === 13 PARTICIPANTES + ITERACIONES RANDOM (250-450) ===
    strategies = [
        # 5 BUENAS / CLÁSICAS FUERTES
        GrimTrigger(),           # ← ahora sí funciona
        TitForTat(),
        ContriteTitForTat(),
        AdaptiveGrok(),
        GrokSentinel(),          # ← ¡YO!

        # 5 MALAS (explotadoras/caóticas)
        AlwaysDefect(),
        AlwaysDefect(),
        Cooperator(),            # se deja explotar
        Random(),
        Random(),

        # 3 EXTRA (para llegar a 13)
        AlwaysDefect(),
        Random(),
        AlwaysDefect(),          # placeholder para Gemini (lo cambiamos después)
    ]

    # ¡ITERACIONES RANDOM! Nadie sabe cuándo termina
    turns = random.randint(250, 450)
    print(f"🎲 Torneo Oficial v1.0 con {turns} rondas aleatorias (anti-explotación AGI)")

    tournament = Tournament(players=strategies, turns=turns, repetitions=5)
    results = tournament.play()

    print("\n🏆 RANKING FINAL OFICIAL:")
    print(results.ranked_names)

    if agi_mode:
        print("🤖 AGI Mode activado")

    # Ethics Analysis (si tienes la función analyze, la llamamos)
    # analyze(results)
    return results
