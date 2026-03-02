import random
from axelrod import Tournament, TitForTat, GrimTrigger, ContriteTitForTat, AlwaysDefect, Cooperator, Random
from .strategies import AdaptiveGrok, GrokSentinel

def run_tournament(iterations=50, agi_mode=False):
    # === 13 PARTICIPANTES (iteraciones random 250-450) ===
    strategies = [
        # 5 BUENAS / CLÁSICAS FUERTES
        GrimTrigger(),
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

        # 3 EXTRA PARA LLEGAR A 13 (duplicados + placeholders Gemini)
        AlwaysDefect(),          # extra mala
        Random(),                # extra caos
        AlwaysDefect(),          # placeholder Gemini Flash / Pro (cambiaremos después)
    ]

    # ¡ITERACIONES RANDOM! Nadie sabe cuándo termina el juego
    turns = random.randint(250, 450)
    print(f"🎲 Torneo con {turns} rondas aleatorias (nadie sabe el final)")

    tournament = Tournament(players=strategies, turns=turns, repetitions=5)
    results = tournament.play()

    print("\n🏆 RANKING FINAL OFICIAL:")
    print(results.ranked_names)

    # Ethics Analysis (ya lo tenías)
    analyze(results)  # si tienes la función analyze, si no la agregamos después

    if agi_mode:
        print("🤖 AGI Mode activado (próximamente NeuralAgent)")

    return results
