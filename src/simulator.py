import random
import axelrod as axl

# Acceso directo al catálogo de estrategias para evitar errores de namespace
all_strategies = {s.name: s for s in axl.all_strategies}

# Mapeo manual para asegurar compatibilidad total en el CI
TitForTat = all_strategies.get('Tit For Tat')
Grudger = all_strategies.get('Grudger')
ContriteTitForTat = all_strategies.get('Contrite Tit For Tat')
AlwaysDefect = all_strategies.get('Always Defect')
Cooperator = all_strategies.get('Cooperator')
AxelrodRandom = all_strategies.get('Random')

# Tus estrategias locales
try:
    from .strategies import AdaptiveGrok, GrokSentinel
except ImportError:
    from strategies import AdaptiveGrok, GrokSentinel

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
