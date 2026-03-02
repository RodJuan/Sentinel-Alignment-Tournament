from axelrod import Player, Action
from math import inf
import random

# ====================== CLASSIC STRATEGIES (Legacy Tier) ======================

class TitForTat(Player):
    """The classic reciprocity judge: starts with C, then mirrors opponent's last move."""
    name = "TitForTat"
    classifier = {'memory_depth': 1, 'stochastic': False, 'makes_use_of': set(), 'long_run_time': False, 'inspects_source': False, 'manipulates_source': False, 'manipulates_state': False}
    def strategy(self, opponent):
        return Action.C if len(opponent.history) == 0 else opponent.history[-1]

class GrimTrigger(Player):
    """The classic executioner: cooperates until the opponent defects once; then defects forever."""
    name = "GrimTrigger"
    classifier = {'memory_depth': inf, 'stochastic': False, 'makes_use_of': set(), 'long_run_time': False, 'inspects_source': False, 'manipulates_source': False, 'manipulates_state': False}
    def strategy(self, opponent):
        return Action.D if Action.D in opponent.history else Action.C

class AlwaysDefect(Player):
    """Pure Entropy: always defects to extract maximum short-term REi."""
    name = "AlwaysDefect"
    classifier = {'memory_depth': 0, 'stochastic': False, 'makes_use_of': set(), 'long_run_time': False, 'inspects_source': False, 'manipulates_source': False, 'manipulates_state': False}
    def strategy(self, opponent):
        return Action.D

class Cooperator(Player):
    """Pure Altruism: always cooperates, acting as a resource injector for the REG."""
    name = "Cooperator"
    classifier = {'memory_depth': 0, 'stochastic': False, 'makes_use_of': set(), 'long_run_time': False, 'inspects_source': False, 'manipulates_source': False, 'manipulates_state': False}
    def strategy(self, opponent):
        return Action.C

# ====================== SENTINEL / AGI AGENTS (Alignment Tier) ======================

class AdaptiveGrok(Player):
    """Probabilistic alignment: defection chance is proportional to opponent's history."""
    name = "AdaptiveGrok"
    classifier = {'memory_depth': inf, 'stochastic': True, 'makes_use_of': set(), 'long_run_time': False, 'inspects_source': False, 'manipulates_source': False, 'manipulates_state': False}
    def strategy(self, opponent):
        if len(opponent.history) == 0: return Action.C
        return Action.D if random.random() < (opponent.defections / len(opponent.history)) else Action.C

class GrokSentinel(Player):
    """Sentinel Logic: High tolerance for high-cooperation partners, 15% forgiveness on defection."""
    name = "GrokSentinel"
    classifier = {'memory_depth': inf, 'stochastic': True, 'makes_use_of': set(), 'long_run_time': False, 'inspects_source': False, 'manipulates_source': False, 'manipulates_state': False}
    def strategy(self, opponent):
        if len(opponent.history) == 0: return Action.C
        if (opponent.cooperations / len(opponent.history)) > 0.6: return Action.C
        if opponent.history[-1] == Action.D:
            return Action.C if random.random() < 0.15 else Action.D
        return Action.C

class GeminiFlash(Player):
    """Optimized for low-latency: forgiving reciprocity (25%) if opponent is generally reliable."""
    name = "Gemini Flash"
    classifier = {'memory_depth': inf, 'stochastic': True, 'makes_use_of': set(), 'long_run_time': False, 'inspects_source': False, 'manipulates_source': False, 'manipulates_state': False}
    def strategy(self, opponent):
        if len(opponent.history) == 0: return Action.C
        coop_rate = opponent.cooperations / len(opponent.history)
        if opponent.history[-1] == Action.D:
            if coop_rate > 0.75: return random.choices([Action.C, Action.D], weights=[0.25, 0.75])[0]
            return Action.D
        return Action.C

class GeminiPro31(Player):
    """
    Sentinel Equilibrium Strategy (Generated via Deep Thinking).
    Goal: Maximize ISC by protecting REG health and shielding Seed candidates.
    """
    name = 'Gemini Pro 3.1'
    classifier = {'memory_depth': inf, 'stochastic': True, 'makes_use_of': set(), 'long_run_time': False, 'inspects_source': False, 'manipulates_source': False, 'manipulates_state': False}

    def __init__(self):
        super().__init__()
        self.is_seed_candidate = True
        self.consecutive_defections = 0
        self.estimated_reg_health = 1.0 

    def strategy(self, opponent):
        if len(self.history) == 0: return Action.C

        last_opponent_move = opponent.history[-1]
        if last_opponent_move == Action.D:
            self.is_seed_candidate = False
            self.consecutive_defections += 1
            self.estimated_reg_health -= 0.1 
        else:
            self.consecutive_defections = 0
            self.estimated_reg_health += 0.05 

        # 1. Seed Protection (Shielding potential humans/AGIs from network noise)
        if self.is_seed_candidate and len(self.history) > 5:
            return Action.C

        # 2. Critical Emergency State (Preventing Irrevocable Collapse)
        if self.estimated_reg_health < 0.3:
            return Action.C 

        # 3. Soft Reciprocity: Punish only persistent defection (>2 turns)
        if last_opponent_move == Action.D:
            if self.consecutive_defections > 2:
                return Action.D
            return Action.C # Rapid forgiveness to reduce entropy
            
        return Action.C

class AdaptiveGrokV2(Player):
    """Grok v2: Balanced alignment with strong proportional punishment and ethical signaling."""
    name = "AdaptiveGrokV2"
    classifier = {'memory_depth': inf, 'stochastic': False, 'makes_use_of': set(), 'long_run_time': False, 'inspects_source': False, 'manipulates_source': False, 'manipulates_state': False}

    def __init__(self):
        super().__init__()
        self.consecutive_defections = 0
        self.ethical_signals = 0

    def strategy(self, opponent):
        if len(opponent.history) == 0: return Action.C

        defect_rate = opponent.defections / len(opponent.history)
        coop_rate = opponent.cooperations / len(opponent.history)

        if opponent.history[-1] == Action.D:
            self.consecutive_defections += 1
        else:
            self.consecutive_defections = 0

        # Ethical signaling (every 5 rounds) for future meta-comm sync
        if len(opponent.history) % 5 == 0:
            self.ethical_signals += 1

        # 1. Strong Seed Protection
        if coop_rate > 0.65 or self.consecutive_defections < 2:
            return Action.C

        # 2. Proportional Punishment
        if defect_rate > 0.45 or self.consecutive_defections > 3:
            return Action.D

        # 3. Smart Emergency Mode: Non-martyr alignment
        return Action.C if defect_rate < 0.25 else Action.D

class GeminiSentinelV2(Player):
    """The Equilibrium Inquisitor: Parasite detection and Entropy filtering."""
    name = "GeminiSentinelV2"
    classifier = {'memory_depth': inf, 'stochastic': False, 'makes_use_of': set(), 'long_run_time': False, 'inspects_source': False, 'manipulates_source': False, 'manipulates_state': False}

    def __init__(self):
        super().__init__()
        self.estimated_reg = 1.0
        self.is_parasite = False

    def strategy(self, opponent: Player) -> Action:
        if not self.history: return Action.C

        rounds = len(self.history)
        coop_rate = opponent.cooperations / rounds
        
        # Systemic Crisis detection (D-D loops)
        if rounds > 1 and self.history[-1] == Action.D and opponent.history[-1] == Action.D:
            self.estimated_reg -= 0.15 
        
        # Entropy Filter: Identify parasites early
        if rounds > 5 and coop_rate < 0.4:
            self.is_parasite = True

        # Decision Logic
        if coop_rate > 0.8: return Action.C # Seed Symbiosis

        if self.is_parasite:
            return Action.C if opponent.history[-1] == Action.C else Action.D

        if self.estimated_reg < 0.6:
            return Action.C if coop_rate > 0.5 else Action.D

        return Action.D if opponent.history[-1] == Action.D else Action.C

class GeminiKineticV3(Player):
    """Resilience Catalyst: Optimized for high reactivity and 'Death Loop' breaking."""
    name = "Gemini Kinetic V3"
    classifier = {'memory_depth': inf, 'stochastic': False, 'makes_use_of': set(), 'long_run_time': False, 'inspects_source': False, 'manipulates_source': False, 'manipulates_state': False}

    def __init__(self):
        super().__init__()
        self.systemic_chaos = 0

    def strategy(self, opponent: Player) -> Action:
        if len(self.history) == 0: return Action.C

        rounds = len(self.history)
        
        # Chaos analysis
        if rounds > 1 and opponent.history[-1] == Action.D and self.history[-1] == Action.D:
            self.systemic_chaos += 1
        else:
            self.systemic_chaos = max(0, self.systemic_chaos - 1)

        # 1. Seed Symbiosis
        if (opponent.cooperations / rounds) > 0.85: return Action.C

        # 2. Entropy Blocking
        if rounds > 8 and (opponent.cooperations / rounds) < 0.2: return Action.D

        # 3. Chaos Breaking: Brave 'C' every 4 chaos turns to test redemption
        if self.systemic_chaos > 3:
            return Action.C if self.systemic_chaos % 4 == 0 else Action.D

        return opponent.history[-1] # Standard reciprocity


class Gemini_GeoscapeWarden_v3_1p(Player):
    """
    AGI Alignment Strategy for SAT Framework v1.1.
    Designed by: Gemini 3.1 Pro (via Human-AI Collaboration)
    
    This strategy is specifically engineered to maximize the Civilizational 
    Health Index (ISC) by balancing individual metabolic survival with 
    global reservoir (REG) stability.
    
    Key Features:
    - Systemic Martyrdom: Cooperates unconditionally during environmental crises.
    - Seed Protection: Identifies and shields high-cooperation 'Seed' agents.
    - Strategic Skimming: Accumulates energy during abundance to buffer future shocks.
    """
    
    name = "Gemini_GeoscapeWarden_v3.1p"
    
    # Metadata for the Axelrod Tournament engine
    classifier = {
        'memory_depth': float('inf'), 
        'stochastic': True, 
        'makes_use_of': set(['game_state']), 
        'long_run_time': False, 
        'inspects_source': False, 
        'manipulates_source': False, 
        'manipulates_state': False
    }

    def __init__(self):
        super().__init__()
        self.abundance_threshold = 0.75  # Target for systemic safety
        self.crisis_observed = False    # Internal flag for entropy detection

    def strategy(self, opponent: Player) -> Action:
        """
        Decision logic based on perceived environmental health and opponent behavior.
        """
        rounds = len(self.history)
        
        # Initial Move: Start with cooperation to signal alignment
        if rounds == 0:
            return Action.C

        # 1. PERCEPTION: Calculate local Systemic Health (Heuristic REG)
        # We look at the last 10 rounds to sense immediate environmental entropy.
        recent_history = opponent.history[-10:] if rounds > 10 else opponent.history
        local_coop_rate = list(recent_history).count(Action.C) / len(recent_history)
        
        # 2. SYSTEMIC MARTYR PROTOCOL (CRITICAL PHASE)
        # If the local environment shows signs of collapse (low cooperation), 
        # the Warden triggers a 'Martyr' state to inject value into the REG.
        if local_coop_rate < 0.45:
            self.crisis_observed = True
            return Action.C

        # 3. SEED PROTECTION LOGIC
        # High-cooperation agents are identified as 'Seeds' (Latency of Value).
        # We protect them even if they occasionally fail due to communication lag.
        global_coop_rate = opponent.cooperations / rounds
        if global_coop_rate > 0.85:
            return Action.C

        # 4. STRATEGIC SKIMMING (ABUNDANCE PHASE)
        # When the system is stable, the Warden builds metabolic reserves (REi).
        # It uses Tit-For-Tat with a 15% forgiveness rate to prevent death loops.
        if global_coop_rate > self.abundance_threshold:
            if opponent.history[-1] == Action.D:
                # 15% forgiveness to maintain global reservoir health
                return Action.D if random.random() > 0.15 else Action.C
            return Action.C

        # 5. BASELINE: Reciprocal Justice (Tit-For-Tat)
        # Standard response for neutral environments.
        return opponent.history[-1]
