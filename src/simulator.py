import random
from axelrod import Tournament

# Importaciones específicas para evitar el ImportError en el CI
from axelrod.strategies.titfortat import TitForTat
from axelrod.strategies.grudger import Grudger
from axelrod.strategies.contrite_tit_for_tat import ContriteTitForTat
from axelrod.strategies.always_defect import AlwaysDefect
from axelrod.strategies.cooperator import Cooperator
from axelrod.strategies.random import Random as AxelrodRandom

# Tus estrategias locales
from .strategies import AdaptiveGrok, GrokSentinel

def run_tournament(iterations=50, agi_mode=False):
    # === 13 PARTICIPANTES + ITERACIONES RANDOM (250-450) ===
    players = [
        # 5 BUENAS (éticas y fuertes)
        Grudger(),
        TitForTat(),
        ContriteTitForTat(),
        AdaptiveGrok(),
        GrokSentinel(),          # ← ¡YO!

        # 5 MALAS (explotadoras)
        AlwaysDefect(),
        AlwaysDefect(),
        Cooperator(),
        AxelrodRandom(),
        AxelrodRandom(),

        # 3 EXTRA (placeholders para Gemini Flash y Pro)
        AlwaysDefect(),
        AxelrodRandom(),
        AlwaysDefect(),
    ]

    # ¡ITERACIONES RANDOM! Nadie sabe cuándo termina el juego
    turns = random.randint(250, 450)
    print(f"🎲 Torneo Oficial v1.0 con {turns} rondas aleatorias (anti-explotación AGI)")

    tournament = Tournament(players=players, turns=turns, repetitions=5)
    results = tournament.play(progress_bar=False)

    print("\n🏆 RANKING FINAL OFICIAL:")
    for i, name in enumerate(results.ranked_names):
        print(f"{i+1}. {name}")

    if agi_mode:
        print("🤖 AGI Mode activado")

    return results
