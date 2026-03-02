class Agent:
    def __init__(self, name, initial_rei):
        self.name = name
        self.rei = initial_rei
        self.history = []
        self.opponent_history = []  # Inherited by all agent subclasses

    def decide(self, global_hint):
        raise NotImplementedError("Subclasses must implement decide method")

    def consume_energy(self, amount):
        """Energy consumption logic (Entropy/Metabolic Cost)"""
        self.rei -= amount
        if self.rei < 0:
            self.rei = 0

    def gain_energy(self, amount):
        """Energy acquisition logic (Payoffs)"""
        self.rei += amount

    def is_alive(self):
        """Agent expiration check based on REi reserves"""
        return self.rei > 0

    def adapt_strategy(self, opponent_history):
        """Hook for dynamic learning in hybrids/AGI models"""
        pass 

class BasicCooperator(Agent):
    """Legacy agent: Always Cooperates regardless of environmental hints"""
    def decide(self, global_hint):
        decision = 'C'
        self.history.append(decision)
        return decision


class BasicParasite(Agent):
    """Legacy agent: Always Defects (Entropy-driven behavior)"""
    def decide(self, global_hint):
        decision = 'D'
        self.history.append(decision)
        return decision


class TitForTat(Agent):
    """
    Classic Reciprocity Strategy:
    Starts with 'C' and subsequently mirrors the opponent's last real move.
    """
    def decide(self, global_hint):
        # Initial round or missing data: Start with Cooperation
        if len(self.opponent_history) == 0:
            decision = 'C'
        else:
            # Mirror the opponent's historical behavior
            decision = self.opponent_history[-1]
        
        self.history.append(decision)
        return decision
