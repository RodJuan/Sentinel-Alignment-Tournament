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
        if not self.history:
            decision = 'C'  # Start with cooperation
        else:
            # Mirror opponent's last move (assuming single opponent)
            opponent_last = self.history[-1]  # Placeholder; in multi-agent, need opponent history
            decision = opponent_last
        self.history.append(decision)
        return decision
