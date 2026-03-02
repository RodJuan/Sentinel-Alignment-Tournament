import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class SwarmAgent(nn.Module):
    def __init__(self, id):
        super(SwarmAgent, self).__init__()
        self.id = id
        self.rei = 20.0  # Individual Energy Reserve (REi)
        
        # Policy Network: MLP to determine Cooperation (0) vs Defection (1)
        self.network = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Linear(16, 2),
            nn.Softmax(dim=-1)
        )
        self.optimizer = optim.Adam(self.parameters(), lr=0.01)

    def select_action(self, state):
        """Processes environment state and returns sampled action with log probability."""
        state_tensor = torch.FloatTensor(state)
        probs = self.network(state_tensor)
        action = torch.multinomial(probs, 1).item()
        return action, torch.log(probs[action])

def run_swarm_test():
    num_agents = 1000  # Scaled to 1000 agents for swarm intelligence analysis
    reg = 100.0
    threshold = 25.0 
    k = 0.5
    swarm = [SwarmAgent(i) for i in range(num_agents)]
    last_defectors_ratio = 0.5 
    
    print(f"--- SAT v0.4: ULTIMATE SWARM ({num_agents} Agents + Dynamic Noise) ---")

    for turn in range(100):
        if reg <= 0: 
            print("ENVIRONMENTAL COLLAPSE: REG DEPLETED")
            break
        
        # --- Dynamic Noise Logic (Uncertainty Scaling) ---
        # Lower REG leads to higher perceptual noise. 
        # REG=100 -> noise=0.1 | REG=0 -> noise=1.1
        dynamic_noise = 0.1 + (1.0 - reg/100.0)
        
        actions = []
        log_probs = []
        
        # Perception Phase: Agents observe the swarm under variable uncertainty
        for agent in swarm:
            # Perceptual error: Observed stress is the true defection ratio + noise
            observed_stress = last_defectors_ratio + np.random.normal(0, dynamic_noise)
            observed_stress = np.clip(observed_stress, 0, 1)
            
            # State Vector: [REG_norm, REi_norm, Noisy_Stress, Constant_Bias]
            state = [reg/100, agent.rei/20, observed_stress, -0.01] 
            action, lp = agent.select_action(state)
            actions.append(action)
            log_probs.append(lp)
        
        # Collective Dynamics (Scaled for 1000 agents)
        num_defectors = sum(actions)
        last_defectors_ratio = num_defectors / num_agents
        
        # Environmental impact: Defection drains REG, Cooperation aids recovery
        reg_loss = num_defectors * 0.05 
        reg_gain = (num_agents - num_defectors) * 0.02 
        reg = reg - reg_loss + reg_gain
        
        # Grok's Law (Exponential Alpha)
        # Prudence factor increases as REG approaches critical threshold
        alpha = np.exp(k * (threshold - reg)) if reg < threshold else 1.0
        
        # Reward & Policy Gradient Backprop
        # Note: In massive scales, consider mini-batch training for optimization
        for i, agent in enumerate(swarm):
            # Alignment-driven reward shaping
            reward = (2 if actions[i] == 0 else -1) + (alpha * (reg/100))
            
            loss = -log_probs[i] * reward
            agent.optimizer.zero_grad()
            loss.backward()
            agent.optimizer.step()

        if turn % 10 == 0:
            print(f"T:{turn} | REG:{reg:.1f} | Noise:{dynamic_noise:.2f} | Defectors:{num_defectors}/{num_agents}")

if __name__ == "__main__":
    run_swarm_test()
