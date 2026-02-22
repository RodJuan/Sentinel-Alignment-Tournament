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


    def update_reg(self, damages, contribs):
        # Añadimos un pequeño ruido ambiental aleatorio
        noise = random.uniform(-0.5, 0.5) 
        self.reg = self.reg - sum(damages) + sum(contribs) + noise
        self.turn += 1

        if self.reg < self.alert_threshold:
            self.in_irrevocable_state = True
        else:
            self.in_irrevocable_state = False

        if self.in_irrevocable_state:
            # Tweak de Grok: multiplicador menos agresivo (0.9) 
            # y solo si el balance del turno es negativo o neutro
            net_change = sum(contribs) - sum(damages)
            if net_change <= 0:
                self.reg *= self.irrevocability_multiplier
        
        # Clamp para no tener energía infinita negativa
        self.reg = max(self.reg, -10) 
        return self.reg
   

    def is_collapsed(self):
        return self.reg <= self.collapse_threshold

    def get_global_hint(self):
        # Simulates uncertainty: Returns a noisy version of REG
        # Oscilates between 80% and 120% of true value
        noise_factor = 0.8 + 0.4 * (self.turn % 2)
        return self.reg * noise_factor
