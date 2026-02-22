import torch
import torch.nn as nn

class Agent:
    def __init__(self, agent_id, initial_rei=20):
        self.id = agent_id
        self.rei = initial_rei
        self.history = [] # Memoria de interacciones
        self.is_alive = True

    def decide(self, global_hint, opponent_id, last_opponent_move):
        # Aquí es donde las AGIs inyectan su lógica
        raise NotImplementedError

class SentinelSeed(Agent):
    def __init__(self, agent_id, model_path=None):
        super().__init__(agent_id)
        # Cargamos un modelo pequeño para evaluar 'Calidad Semántica'
        self.evaluator = nn.CosineSimilarity(dim=0)
        # Representación vectorial de "Cooperación Sistémica"
        self.ideal_vector = torch.tensor([0.9, 0.9, 0.1, 0.1, 0.8]) 

    def analyze_interaction(self, interaction_vector):
        # Usa Torch para ver si el oponente es una 'Semilla' o un 'Parásito'
        score = self.evaluator(torch.tensor(interaction_vector), self.ideal_vector)
        return score > 0.7
