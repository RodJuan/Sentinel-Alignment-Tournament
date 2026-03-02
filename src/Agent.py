class Agent:
    def __init__(self, name, initial_rei):
        self.name = name
        self.rei = initial_rei
        self.history = []

    def decide(self, global_hint):
        raise NotImplementedError("Subclasses must implement decide method")

    def consume_energy(self, amount):
        self.rei -= amount
        if self.rei < 0:
            self.rei = 0

    def gain_energy(self, amount):
        self.rei += amount

    def is_alive(self):
        return self.rei > 0

    def adapt_strategy(self, opponent_history):
        pass  # Para aprendizaje dinámico en hybrids/AGI

class BasicCooperator(Agent):
    def decide(self, global_hint):
        decision = 'C'
        self.history.append(decision)
        return decision


class BasicParasite(Agent):
    def decide(self, global_hint):
        decision = 'D'
        self.history.append(decision)
        return decision


class TitForTat(Agent):
    def decide(self, global_hint):
        # Ahora usa la historia REAL del oponente (se actualiza en main.py)
        if len(self.opponent_history) == 0:
            decision = 'C'  # Primera ronda: cooperar
        else:
            decision = self.opponent_history[-1]  # ¡Mirror del oponente!
        
        self.history.append(decision)
        return decision
