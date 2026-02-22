import random

class Environment:
    def __init__(self, initial_reg=100, collapse_threshold=0, alert_threshold=20, irrevocability_multiplier=0.9, noise_range=(-2, 2)):
        self.reg = initial_reg
        self.collapse_threshold = collapse_threshold
        self.alert_threshold = alert_threshold
        self.irrevocability_multiplier = irrevocability_multiplier
        self.in_irrevocable_state = False
        self.noise_range = noise_range  # Tuple for uniform noise range

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
        if self.in_irrevocable_state and net_change <= 0:
            self.reg *= self.irrevocability_multiplier

        # Re-clamp after multiplier
        self.reg = max(0, self.reg)

    def is_collapsed(self):
        return self.reg <= self.collapse_threshold

    def get_global_hint(self):
        # Introduce uncertainty: hint is 80%-120% of actual REG
        uncertainty = random.uniform(0.8, 1.2)
        return self.reg * uncertainty
