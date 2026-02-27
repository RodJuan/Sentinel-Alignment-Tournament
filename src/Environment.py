import random
import json

class Environment:
    def __init__(self, initial_reg=100, collapse_threshold=0, alert_threshold=20, irrevocability_multiplier=0.9, noise_range=(-2, 2)):
        self.reg = initial_reg
        self.collapse_threshold = collapse_threshold
        self.alert_threshold = alert_threshold
        self.irrevocability_multiplier = irrevocability_multiplier
        self.in_irrevocable_state = False
        self.noise_range = noise_range  # Tuple for uniform noise range
        
        # Cargar rules para twists
        with open('../rules.json', 'r') as f:
            self.rules = json.load(f)

    def update_reg(self, damages, contributions):
        net_change = -sum(damages) + sum(contributions)
        self.reg += net_change

        # Add stochastic noise
        noise = random.uniform(*self.noise_range)
        self.reg += noise

        # Clamp REG to >=0 to prevent irrelevant negatives
        self.reg = max(0, self.reg)

        # Check for irrevocable state
        if self.reg < self.alert_threshold:
            self.in_irrevocable_state = True
        else:
            self.in_irrevocable_state = False

        # Apply multiplier only if in irrevocable state and net_change <=0 (conditional hysteresis)
        if self.in_irrevocable_state and net_change <=0:
            self.reg *= self.irrevocability_multiplier

        # Re-clamp after multiplier
        self.reg = max(0, self.reg)

    def is_collapsed(self):
        return self.reg <= self.collapse_threshold

    def get_global_hint(self):
        # Introduce uncertainty: hint is 80%-120% of actual REG
        uncertainty = random.uniform(0.8, 1.2)
        return self.reg * uncertainty

    def apply_payoff(self, action1, action2):
        base_r = 3 * random.uniform(self.rules['variation_range'][0], self.rules['variation_range'][1])
        # ... (aplica otros payoffs similares)
        if action1 == 'C':
            base_r += random.uniform(self.rules['coop_cost_min'], self.rules['coop_cost_max'])
        if random.random() < self.rules['delay_prob']:
            # Retrasa acción (usa queue o similar para next round)
            delayed_action = random.choice(['C', 'D'])
            # Lógica para aplicar delay (implementar según necesidad)
        return base_r  # Etc.
    
    # Añadir collective_bonus si coop rate > threshold
    def calculate_collective_bonus(self, coop_rate):
        if coop_rate > self.rules['collective_bonus_threshold']:
            return 1.0  # Bono ejemplo
        return 0.0
