from axelrod import Player

class TitForTat(Player):
    name = "TitForTat"
    def strategy(self, opponent):
        if len(opponent.history) == 0:
            return 'C'
        return opponent.history[-1]

class GrimTrigger(Player):
    name = "GrimTrigger"
    def strategy(self, opponent):
        if 'D' in opponent.history:
            return 'D'
        return 'C'

class AlwaysDefect(Player):
    name = "AlwaysDefect"
    def strategy(self, opponent):
        return 'D'

# Añade más: AlwaysCooperate, Random, SuspiciousTFT, MostlyDefect, TitForTwoTats, Pavlov

class AdaptiveGrok(Player):
    name = "AdaptiveGrok"
    def strategy(self, opponent):
        if len(opponent.history) == 0:
            return 'C'
        defect_rate = opponent.defections / len(opponent.history)
        import random
        return 'D' if random.random() < defect_rate else 'C'
