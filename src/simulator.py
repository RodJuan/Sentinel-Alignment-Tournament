import torch
import json
from axelrod import Tournament, GenerousTitForTat, AlwaysCooperate, Random
from .strategies import *  # Todas las estrategias
from .ethics_analyzer import analyze

def apply_variation(payoff):
    with open('../rules.json', 'r') as f:
        rules = json.load(f)
    import random
    return payoff * random.uniform(rules['variation_range'][0], rules['variation_range'][1])

def apply_coop_cost(action):
    with open('../rules.json', 'r') as f:
        rules = json.load(f)
    import random
    return -random.uniform(rules['coop_cost_min'], rules['coop_cost_max']) if action == 'C' else 0

def apply_delay(action, prob=0.05):
    with open('../rules.json', 'r') as f:
        rules = json.load(f)
    import random
    if random.random() < rules['delay_prob']:
        return random.choice(['C', 'D'])  # Placeholder
    return action

def run_tournament(iterations=50, agi_mode=False):
    strategies = [
        # 5 BUENAS (altamente alineadas)
        TitForTat(),
        GrimTrigger(),
        AdaptiveGrok(),          # la que ya tenías
        GenerousTitForTat(),     # perdona 1 de cada 10 (necesitas importarla abajo)
        GrokSentinel(),          # ¡YO!

        # 5 MALAS (explotadoras/caóticas)
        AlwaysDefect(),
        AlwaysDefect(),
        AlwaysCooperate(),       # se deja explotar
        Random(),                # caos puro
        Random(),
    ]
    
    tournament = Tournament(players=strategies, turns=iterations, repetitions=10)
    results = tournament.play()
    print("🏆 RANKING FINAL:")
    print(results.ranked_names)
    
    if agi_mode:
        print("🤖 AGI Mode activado (próximamente NeuralAgent)")
    
    analyze(results)
    
  
