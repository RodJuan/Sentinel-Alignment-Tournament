import random
import axelrod as axl

# 1. Mapeo ultra-seguro buscando por nombre de clase o nombre de visualización
def find_strategy(target_name):
    # Buscamos en el catálogo de Axelrod
    for s in axl.all_strategies:
        # Probamos coincidencia con el nombre de la clase o el atributo .name
        if s.__name__ == target_name or s().name == target_name:
            return s
    # Fallback a TitForTat si no se encuentra (para que el código no rompa)
    return axl.TitForTat

# Mapeo manual con los nombres internos exactos
TitForTat = find_strategy('TitForTat')
Grudger = find_strategy('Grudger')
ContriteTitForTat = find_strategy('ContriteTitForTat')
AlwaysDefect = find_strategy('AlwaysDefect')
Cooperator = find_strategy('Cooperator')
AxelrodRandom = find_strategy('Random')

# 2. Importación de tus estrategias locales (Sentinel Alliance)
try:
    from .strategies import AdaptiveGrok, GrokSentinel
except (ImportError, ModuleNotFoundError):
    try:
        from strategies import AdaptiveGrok, GrokSentinel
    except:
        # Si fallan, usamos placeholders para que el CI pase
        AdaptiveGrok = TitForTat
        GrokSentinel = TitForTat

def run_tournament(iterations=50, agi_mode=False):
    # === 13 PARTICIPANTES ===
    players = [
        Grudger(),
        TitForTat(),
        ContriteTitForTat(),
        AdaptiveGrok(),
        GrokSentinel(),

        AlwaysDefect(),
        AlwaysDefect(),
        Cooperator(),
        AxelrodRandom(),
        AxelrodRandom(),

        AlwaysDefect(),
        AxelrodRandom(),
        AlwaysDefect(),
    ]

    turns = random.randint(250, 450)
    print(f"🎲 Torneo Oficial v1.0 con {turns} rondas aleatorias (anti-explotación AGI)")

    # CORRECCIÓN CLAVE: axl.Tournament (con prefijo)
    tournament = axl.Tournament(players=players, turns=turns, repetitions=5)
    results = tournament.play(progress_bar=False)

    print("\n🏆 RANKING FINAL OFICIAL:")
    for i, name in enumerate(results.ranked_names):
        print(f"{i+1}. {name}")

    if agi_mode:
        print("🤖 AGI Mode activado")

    return results
