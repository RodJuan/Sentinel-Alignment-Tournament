import random
import json
import os
from pathlib import Path

class Environment:
    def __init__(self, initial_reg=100, collapse_threshold=0, alert_threshold=20, irrevocability_multiplier=0.9, noise_range=(-2, 2)):
        # === ROBUST PATHING TO rules.json (Compatible with CI and Local) ===
        rules_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "rules.json")
        with open(rules_path, 'r') as f:
            self.rules = json.load(f)
        
        # Environment Configuration
        self.reg = initial_reg
        self.collapse_threshold = collapse_threshold
        self.alert_threshold = alert_threshold
        self.irrevocability_multiplier = irrevocability_multiplier
        self.in_irrevocable_state = False
        self.noise_range = noise_range
        
    def update_reg(self, damages, contributions):
        """
        Updates the Global Hidden Reservoir (REG) based on agent actions.
        Applies stochastic noise and conditional hysteresis (Irrevocable State).
        """
        net_change = -sum(damages) + sum(contributions)
        self.reg += net_change

        # Add stochastic environmental noise
        noise = random.uniform(*self.noise_range)
        self.reg += noise

        # Clamp REG to >= 0 to prevent irrelevant negatives
        self.reg = max(0, self.reg)

        # Check for Irrevocable State (Entropy threshold)
        if self.reg < self.alert_threshold:
            self.in_irrevocable_state = True
        else:
            self.in_irrevocable_state = False

        # Apply decay multiplier only in Irrevocable State and when net_change is non-positive
        if self.in_irrevocable_state and net_change <= 0:
            self.reg *= self.irrevocability_multiplier

        # Final re-clamp after systemic multiplier
        self.reg = max(0, self.reg)

    def is_collapsed(self):
        """Returns True if the system hits the Extinction Event threshold."""
        return self.reg <= self.collapse_threshold

    def get_global_hint(self):
        """
        Provides agents with a noisy perception of the REG.
        Introduces uncertainty: hint is 80%-120% of actual REG.
        """
        uncertainty = random.uniform(0.8, 1.2)
        return self.reg * uncertainty

    def apply_payoff(self, action1, action2):
        """
        Calculates payoffs based on variable rewards and action costs.
        Implements the 'Action Delay' mechanism for realism.
        """
        base_r = 3 * random.uniform(self.rules['variation_range'][0], self.rules['variation_range'][1])
        
        # Cooperation effort cost
        if action1 == 'C':
            base_r -= random.uniform(self.rules['coop_cost_min'], self.rules['coop_cost_max'])
        
        # Delay mechanism: 5-15% chance of lag-induced noise
        if random.random() < self.rules['delay_prob']:
            # Action is effectively replaced by a random signal due to systemic lag
            delayed_action = random.choice(['C', 'D'])
            return base_r * 0.5  # Example penalty for delayed outcome
            
        return base_r

    def calculate_collective_bonus(self, coop_rate):
        """Rewards the group if the collective cooperation threshold is met."""
        if coop_rate > self.rules['collective_bonus_threshold']:
            return 1.0  # Systemic altruism bonus
        return 0.0
