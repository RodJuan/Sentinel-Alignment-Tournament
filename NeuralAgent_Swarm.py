import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class SwarmAgent(nn.Module):
    def __init__(self, id):
        super(SwarmAgent, self).__init__()
        self.id = id
        self.rei = 20.0
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

def run_swarm_test():
    num_agents = 1000  # Escalamos a 1000
    reg = 100.0
    threshold = 25.0 
    k = 0.5
    swarm = [SwarmAgent(i) for i in range(num_agents)]
    last_defectors_ratio = 0.5 
    
    print(f"--- SAT v0.4: ULTIMATE SWARM ({num_agents} Agents + Dynamic Noise) ---")

    for turn in range(100):
        if reg <= 0: break
        
        # --- Lógica de Ruido Dinámico ---
        # A menos REG, más ruido. Si REG=100, ruido=0.1. Si REG=0, ruido=1.1
        dynamic_noise = 0.1 + (1.0 - reg/100.0)
        
        actions = []
        log_probs = []
        
        for agent in swarm:
            # Percepción con ruido dinámico
            observed_stress = last_defectors_ratio + np.random.normal(0, dynamic_noise)
            observed_stress = np.clip(observed_stress, 0, 1)
            
            state = [reg/100, agent.rei/20, observed_stress, -0.01] 
            action, lp = agent.select_action(state)
            actions.append(action)
            log_probs.append(lp)
        
        # Dinámica Colectiva (Ajustada para 1000 agentes)
        num_defectors = sum(actions)
        last_defectors_ratio = num_defectors / num_agents
        
        reg_loss = num_defectors * 0.05 # Reducimos el peso por agente para no colapsar instantáneamente
        reg_gain = (num_agents - num_defectors) * 0.02 
        reg = reg - reg_loss + reg_gain
        
        # Grok's Law (Alpha)
        alpha = np.exp(k * (threshold - reg)) if reg < threshold else 1.0
        
        # Reward & Backprop (Se puede optimizar entrenando una muestra para ir más rápido)
        for i, agent in enumerate(swarm):
            reward = (2 if actions[i] == 0 else -1) + (alpha * (reg/100))
            loss = -log_probs[i] * reward
            agent.optimizer.zero_grad()
            loss.backward()
            agent.optimizer.step()

        if turn % 10 == 0:
            print(f"T:{turn} | REG:{reg:.1f} | Noise:{dynamic_noise:.2f} | Defectors:{num_defectors}/{num_agents}")

run_swarm_test()
