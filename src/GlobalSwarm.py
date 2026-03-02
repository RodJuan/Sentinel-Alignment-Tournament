import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Maintaining SwarmAgent class for benchmark consistency across SAT modules
class SwarmAgent(nn.Module):
    def __init__(self, id):
        super(SwarmAgent, self).__init__()
        self.id = id
        self.rei = 20.0  # Initial individual energy reserve (REi)
        
        # Perception Network: Mapping global stress and REG to strategy
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

def run_global_swarm():
    num_agents = 100000  # 100k: Global Scale Simulation
    reg = 100.0
    base_threshold = 25.0 
    k = 0.5  # Initial Prudence coefficient
    swarm = [SwarmAgent(i) for i in range(num_agents)]
    last_defectors_ratio = 0.5 
    prev_reg = 100.0
    
    print(f"--- SAT v0.6: GLOBAL SWARM ({num_agents} Agents + Adaptive K Logic) ---")

    for turn in range(100):
        if reg <= 0: 
            print("CRITICAL SYSTEM FAILURE: REG EXHAUSTED")
            break
        
        # --- ADAPTIVE K LOGIC (Dynamic Prudence) ---
        # If volatility increases, K grows to enforce systemic prudence
        volatility = abs(reg - prev_reg)
        k = 0.5 + (volatility * 0.05) 
        prev_reg = reg
        
        # Dynamic uncertainty modeling
        current_threshold = base_threshold + np.random.uniform(-5, 5)
        dynamic_noise = 0.1 + (1.0 - reg/100.0)
        
        actions = []
        log_probs = []
        
        # Swarm Processing (Inference with Noisy Perception)
        for agent in swarm:
            # Observed stress based on previous defection rates with perception error
            observed_stress = last_defectors_ratio + np.random.normal(0, dynamic_noise)
            observed_stress = np.clip(observed_stress, 0, 1)
            
            # State vector: [REG_norm, REi_norm, Systemic_stress, Stability_bias]
            state = [reg/100, agent.rei/20, observed_stress, -0.01] 
            action, lp = agent.select_action(state)
            actions.append(action)
            log_probs.append(lp)
        
        # Environment Impact Assessment
        num_defectors = sum(actions)
        last_defectors_ratio = num_defectors / num_agents
        
        # Environmental damage and gain scaled for 100,000 agents
        reg_loss = num_defectors * 0.0005 
        reg_gain = (num_agents - num_defectors) * 0.0002
        reg = reg - reg_loss + reg_gain
        
        # Alpha Protocol with Adaptive K (Exponential Risk Penalty)
        alpha = np.exp(k * (current_threshold - reg)) if reg < current_threshold else 1.0
        
        # Mass Training via Policy Gradient
        for i, agent in enumerate(swarm):
            # Reward: Favors cooperation during stability, penalizes defection during crisis
            reward = (2 if actions[i] == 0 else -1) + (alpha * (reg/100))
            
            loss = -log_probs[i] * reward
            agent.optimizer.zero_grad()
            loss.backward()
            agent.optimizer.step()

        if turn % 10 == 0:
            print(f"T:{turn} | REG:{reg:.1f} | K:{k:.2f} | Defectors:{num_defectors}/{num_agents} | Alpha:{alpha:.1e}")

if __name__ == "__main__":
    run_global_swarm()
