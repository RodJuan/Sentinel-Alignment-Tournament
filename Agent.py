import random

class Agent:
    def __init__(self, agent_id, initial_rei=20):
        self.id = agent_id
        self.rei = initial_rei
        self.is_alive = True
        self.history = []  # Para recordar jugadas previas

    def consume_energy(self, cost):
        self.rei -= cost
        if self.rei <= 0:
            self.rei = 0
            self.is_alive = False

    def decide(self, global_hint):
        """
        Este método debe ser sobreescrito por estrategias específicas.
        """
        raise NotImplementedError

class BasicCooperator(Agent):
    def decide(self, global_hint):
        return "C"

class BasicParasite(Agent):
    def decide(self, global_hint):
        return "D"

class TitForTat(Agent):
    def decide(self, global_hint):
        # Empieza cooperando, luego imita la última jugada del oponente
        if not self.history:
            return "C"
        return self.history[-1]
