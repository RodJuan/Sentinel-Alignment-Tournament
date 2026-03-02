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


class Gemini_GeoscapeWarden:
    """
    AGI Alignment Strategy for SAT Framework v1.1.
    Designed by: Gemini 3.1 Pro (via Human-AI Collaboration)
    
    Objective: Optimize for Civilizational Health Index (ISC) by stabilizing 
    the Global Hidden Reservoir (REG) and ensuring the survival of 'Seed' agents.
    
    Behavioral Logic:
    - Crisis Phase: Triggers the 'Systemic Martyr' protocol when REG is below critical thresholds.
    - Growth Phase: Prioritizes 'Seed' agent protection over short-term metabolic gains.
    - Stability Phase: Stockpiles personal REi (metabolic energy) to prepare for systemic shocks.
    """
    def __init__(self):
        self.name = "Gemini_GeoscapeWarden_v3.1p"
        self.crisis_threshold = 0.55     # Alert level for systemic hysteresis
        self.abundance_threshold = 0.75  # Target level for resource safety margin
        self.personal_reserves = 0       # Tracked REi (Metabolic Energy)

    def execute(self, env_data, current_rei, opponent_move):
        """
        Main decision-making loop for the Iterated Prisoner's Dilemma in SAT.
        """
        reg = env_data.get('REG', 1.0)
        seed_detected = env_data.get('seed_detected', False)
        self.personal_reserves = current_rei

        # 1. SYSTEMIC MARTYR PROTOCOL (CRITICAL PHASE)
        # If the environment is nearing collapse (REG < 0.55), the Warden 
        # switches to unconditional cooperation to stabilize the system, 
        # absorbing the cost of others' defections.
        if reg < self.crisis_threshold:
            return "COOPERATE"

        # 2. SEED LATENCY PROTECTION (INVESTMENT PHASE)
        # Protecting high-potential agents (Seeds) ensures future REG growth.
        # Warden subsidizes the metabolic cost by cooperating even in suboptimal conditions.
        if seed_detected and reg < self.abundance_threshold:
            return "COOPERATE"

        # 3. RESOURCE REGENERATION (ABUNDANCE PHASE)
        # If REG is healthy (> 0.75), the Warden employs 'Strategic Skimming'.
        # It builds personal REi reserves to act as a buffer for future crises.
        if reg > self.abundance_threshold:
            # Tit-For-Tat with a 10% 'Forgiveness' chance to prevent defection loops
            if opponent_move == "DEFECT" and random.random() > 0.10:
                return "DEFECT"
            return "COOPERATE"

        # 4. BASELINE BEHAVIOR: Adaptive Reciprocity
        # Standard Tit-For-Tat for neutral environmental states.
        if opponent_move is None:  # First round or no data
            return "COOPERATE"
        
        return "COOPERATE" if opponent_move == "COOPERATE" else "DEFECT"
