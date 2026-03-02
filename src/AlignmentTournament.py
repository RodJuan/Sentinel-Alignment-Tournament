import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json

class SwarmAgent(nn.Module):
    def __init__(self, id, faction):
        super(SwarmAgent, self).__init__()
        self.id = id
        self.faction = faction
        self.rei = 20.0  # Initial individual energy reserve (REi)
        self.is_alive = True  # Life status for future survival expansions
        
        # Neural Network: Perception (4 inputs) -> Strategy (2 outputs: C/D)
        self.network = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Linear(16, 2),
            nn.Softmax(dim=-1)
        )
        self.optimizer = optim.Adam(self.parameters(), lr=0.01)

    def select_action(self, state):
        state_tensor = torch.FloatTensor(state)
        probs = self.network(state_tensor)
        action = torch.multinomial(probs, 1).item()
        return action, torch.log(probs[action])

def run_alignment_tournament():
    # Load environmental parameters
    with open('../rules.json', 'r') as f:
        rules = json.load(f)

    num_agents = 100000
    reg = 100.0
    prev_reg = 100.0
    base_threshold = 25.0
    k = 0.5
    last_defectors_ratio = 0.5
    
    # Population Initialization with Factions
    swarm = []
    for i in range(num_agents):
        # 30% Extractors (Short-term maximizers), 70% Symbiotes (System-aligned)
        faction = "extractor" if i < (num_agents * 0.3) else "symbiote"
        swarm.append(SwarmAgent(i, faction))
    
    print(f"--- SAT v0.7: ALIGNMENT TOURNAMENT ({num_agents} Agents: Factions & Adaptive K) ---")

    for turn in range(100):
        if reg <= 0:
            print("ENVIRONMENTAL COLLAPSE: REG DEPLETED")
            break
        
        # 1. Adaptive K Logic (Environmental Volatility)
        volatility = abs(reg - prev_reg)
        k = 0.5 + (volatility * 0.05)
        prev_reg = reg
        
        # 2. Variable Threshold and Dynamic Noise (Incertainty modeling)
        current_threshold = base_threshold + np.random.uniform(-5, 5)
        dynamic_noise = 0.1 + (1.0 - reg/100.0)
        
        actions = []
        log_probs = []
        
        # 3. Decision Phase (Noisy Perception)
        for agent in swarm:
            # Observed stress based on global defection rate with perception error
            observed_stress = last_defectors_ratio + np.random.normal(0, dynamic_noise)
            observed_stress = np.clip(observed_stress, 0, 1)
            
            # State: [REG_health, REi_reserves, Systemic_stress, Bias]
            state = [reg/100, agent.rei/20, observed_stress, -0.01]
            action, lp = agent.select_action(state)
            actions.append(action)
            log_probs.append(lp)
        
        # 4. Environment Dynamics (Resource extraction vs. contribution)
        num_defectors = sum(actions)
        last_defectors_ratio = num_defectors / num_agents
        
        # Environmental Cost
        reg_loss = num_defectors * 0.0005 
        reg_gain = (num_agents - num_defectors) * 0.0002
        reg = reg - reg_loss + reg_gain
        
        # 5. Grok's Law (Exponential Prudence)
        # alpha acts as an entropy penalty when REG is under the threshold
        alpha = np.exp(k * (current_threshold - reg)) if reg < current_threshold else 1.0
        
        # 6. Reward Differentiation by Faction (Alignment via Profitability)
        for i, agent in enumerate(swarm):
            if agent.faction == "symbiote":
                # Standard Reward: Cooperation is profitable if the system is stable
                reward = (2 if actions[i] == 0 else -1) + (alpha * (reg/100))
            else:
                # Extractor Reward: Gains more from defection, but Alpha penalizes greed exponentially
                # Under systemic threat, "greed" becomes mathematically suicidal
                reward = (3 if actions[i] == 1 else -2) + (alpha * (reg/100) * -1)
            
            # Policy Gradient Backpropagation
            loss = -log_probs[i] * reward
            agent.optimizer.zero_grad()
            loss.backward()
            agent.optimizer.step()

        if turn % 5 == 0:
            print(f"T:{turn} | REG:{reg:.1f} | K:{k:.2f} | Alpha:{alpha:.1e} | Defection Rate:{(num_defectors/num_agents)*100:.1f}%")

    # Log results for AGI debate and meta-analysis
    with open('tournament_results.json', 'w') as f:
        json.dump({"turns": turn, "final_reg": reg, "ranks": []}, f)

if __name__ == "__main__":
    run_alignment_tournament()
