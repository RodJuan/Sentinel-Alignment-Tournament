import random

# === IMPORTS 100% COMPATIBLES CON AXELROD 4.14 (CI de GitHub) ===
from axelrod import Tournament
from axelrod.strategies.titfortat import TitForTat
from axelrod.strategies.grim_trigger import GrimTrigger
from axelrod.strategies.contrite_tit_for_tat import ContriteTitForTat
from axelrod.strategies.always_defect import AlwaysDefect
from axelrod.strategies.cooperator import Cooperator
from axelrod.strategies.random import Random

from .strategies import AdaptiveGrok, GrokSentinel

def run_tournament(iterations=50, agi_mode=False):
    # === 13 PARTICIPANTES + ITERACIONES RANDOM (250-450) ===
    strategies = [
        # 5 BUENAS (altamente alineadas y éticas)
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

        # 3 EXTRA para llegar a 13 (placeholders Gemini)
        AlwaysDefect(),
        Random(),
        AlwaysDefect(),          # aquí meteremos Gemini Flash y Pro
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

    return results
