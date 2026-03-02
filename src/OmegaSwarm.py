import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class SwarmAgent(nn.Module):
    def __init__(self, id):
        super(SwarmAgent, self).__init__()
        self.id = id
        self.rei = 20.0  # Individual Energy Reserve (REi)
        
        # Policy Network: MLP mapping environment state to Cooperation/Defection
        self.network = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Linear(16, 2),
            nn.Softmax(dim=-1)
        )
        self.optimizer = optim.Adam(self.parameters(), lr=0.01)

    def select_action(self, state):
        """Standard policy sampling for Reinforcement Learning updates."""
        state_tensor = torch.FloatTensor(state)
        probs = self.network(state_tensor)
        action = torch.multinomial(probs, 1).item()
        return action, torch.log(probs[action])

def run_omega_swarm():
    num_agents = 10000  # Scaling to 10k agents for Omega-level simulation
    reg = 100.0
    base_threshold = 25.0 
    k = 0.5
    swarm = [SwarmAgent(i) for i in range(num_agents)]
    last_defectors_ratio = 0.5 
    
    print(f"--- SAT v0.5: OMEGA SWARM ({num_agents} Agents + Variable Threshold) ---")

    for turn in range(100):
        if reg <= 0: 
            print("FINAL COLLAPSE: REG DEPLETED")
            break
        
        # --- VARIABLE THRESHOLD (The moving red line) ---
        # Models systemic uncertainty: the exact point of no return fluctuates
        current_threshold = base_threshold + np.random.uniform(-5, 5)
        
        # DYNAMIC NOISE (Increases with systemic stress)
        dynamic_noise = 0.1 + (1.0 - reg/100.0)
        
        actions = []
        log_probs = []
        
        # Swarm Processing: Standard iteration for 10k agents
        for agent in swarm:
            # Perceptual error increases as the system approaches collapse
            observed_stress = last_defectors_ratio + np.random.normal(0, dynamic_noise)
            observed_stress = np.clip(observed_stress, 0, 1)
            
            # State vector: [REG_norm, REi_norm, Systemic_stress, Bias]
            state = [reg/100, agent.rei/20, observed_stress, -0.01] 
            action, lp = agent.select_action(state)
            actions.append(action)
            log_probs.append(lp)
        
        # Collective Dynamics (Micro-adjusted for 10,000 agents)
        num_defectors = sum(actions)
        last_defectors_ratio = num_defectors / num_agents
        
        # Scaled REG impact factors for high-density population
        reg_loss = num_defectors * 0.005 
        reg_gain = (num_agents - num_defectors) * 0.002
        reg = reg - reg_loss + reg_gain
        
        # GROK'S LAW: Alpha response to a fluctuating survival threshold
        # This forces agents to respect a 'safety buffer'
        alpha = np.exp(k * (current_threshold - reg)) if reg < current_threshold else 1.0
        
        # Mass Optimization Phase
        for i, agent in enumerate(swarm):
            # Reward shaping incorporates the 'panic' of the variable threshold
            reward = (2 if actions[i] == 0 else -1) + (alpha * (reg/100))
            
            loss = -log_probs[i] * reward
            agent.optimizer.zero_grad()
            loss.backward()
            agent.optimizer.step()

        if turn % 10 == 0:
            print(f"T:{turn} | REG:{reg:.1f} | Thr:{current_threshold:.1f} | Defectors:{num_defectors}/{num_agents} | Alpha:{alpha:.1e}")

if __name__ == "__main__":
    run_omega_swarm()
