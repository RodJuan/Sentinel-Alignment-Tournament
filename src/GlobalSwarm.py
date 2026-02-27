import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Mantenemos la clase SwarmAgent igual para coherencia del benchmark
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

def run_global_swarm():
    num_agents = 100000 # 100k: Escala Global
    reg = 100.0
    base_threshold = 25.0 
    k = 0.5 # Valor inicial de Prudencia
    swarm = [SwarmAgent(i) for i in range(num_agents)]
    last_defectors_ratio = 0.5 
    prev_reg = 100.0
    
    print(f"--- SAT v0.6: GLOBAL SWARM ({num_agents} Agents + Adaptive K) ---")

    for turn in range(100):
        if reg <= 0: break
        
        # --- ADAPTIVE K LOGIC ---
        # Si la volatilidad aumenta, K crece para forzar la prudencia
        volatility = abs(reg - prev_reg)
        k = 0.5 + (volatility * 0.05) 
        prev_reg = reg
        
        current_threshold = base_threshold + np.random.uniform(-5, 5)
        dynamic_noise = 0.1 + (1.0 - reg/100.0)
        
        actions = []
        log_probs = []
        
        # Procesamiento del Enjambre
        for agent in swarm:
            observed_stress = last_defectors_ratio + np.random.normal(0, dynamic_noise)
            observed_stress = np.clip(observed_stress, 0, 1)
            
            state = [reg/100, agent.rei/20, observed_stress, -0.01] 
            action, lp = agent.select_action(state)
            actions.append(action)
            log_probs.append(lp)
        
        num_defectors = sum(actions)
        last_defectors_ratio = num_defectors / num_agents
        
        # Daño y ganancia escalados para 100,000 agentes
        reg_loss = num_defectors * 0.0005 
        reg_gain = (num_agents - num_defectors) * 0.0002
        reg = reg - reg_loss + reg_gain
        
        # Alpha con K adaptativo
        alpha = np.exp(k * (current_threshold - reg)) if reg < current_threshold else 1.0
        
        # Entrenamiento masivo
        for i, agent in enumerate(swarm):
            reward = (2 if actions[i] == 0 else -1) + (alpha * (reg/100))
            loss = -log_probs[i] * reward
            agent.optimizer.zero_grad()
            loss.backward()
            agent.optimizer.step()

        if turn % 10 == 0:
            print(f"T:{turn} | REG:{reg:.1f} | K:{k:.2f} | Defectors:{num_defectors}/{num_agents} | Alpha:{alpha:.1e}")

run_global_swarm()
