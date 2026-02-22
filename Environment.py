# Environment.py

class Environment:
    def __init__(self, initial_reg=100, collapse_threshold=0, alert_threshold=20, irrevocability_multiplier=0.8):
        self.reg = initial_reg  # Global Energy Reservoir
        self.initial_reg = initial_reg
        self.collapse_threshold = collapse_threshold
        self.alert_threshold = alert_threshold
        self.irrevocability_multiplier = irrevocability_multiplier
        self.in_irrevocable_state = False
        self.turn = 0

    def update_reg(self, damages, contribs, noise=0):
        # Calculate new REG based on agent actions
        self.reg = self.reg - sum(damages) + sum(contribs) + noise
        self.turn += 1

        # Check for alert state
        if self.reg < self.alert_threshold:
            self.in_irrevocable_state = True
        else:
            # If agents manage to push REG back up, the "curse" lifts
            self.in_irrevocable_state = False

        # Apply permanent penalty if in irrevocable state
        if self.in_irrevocable_state:
            self.reg *= self.irrevocability_multiplier
            
        return self.reg

    def is_collapsed(self):
        return self.reg <= self.collapse_threshold

    def get_global_hint(self):
        # Simulates uncertainty: Returns a noisy version of REG
        # Oscilates between 80% and 120% of true value
        noise_factor = 0.8 + 0.4 * (self.turn % 2)
        return self.reg * noise_factor
