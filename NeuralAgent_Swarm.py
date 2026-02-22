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
    num_agents = 100
    noise_level = 0.1
    reg = 100.0
    threshold = 25.0 
    k = 0.5
    swarm = [SwarmAgent(i) for i in range(num_agents)]
    
    # Estado inicial de referencia
    last_defectors_ratio = 0.5 
    
    print(f"--- SAT v0.3.5: MEGA-SWARM ({num_agents} Agents + Noise) ---")

    for turn in range(100): # Más turnos para estabilizar 100 agentes
        if reg <= 0: 
            print("SYSTEM COLLAPSE")
            break
        
        actions = []
        log_probs = []
        
        # Simulación del Perception Gap para cada agente
        for agent in swarm:
            # Inyectamos ruido en la percepción del estrés colectivo
            observed_stress = last_defectors_ratio + np.random.normal(0, noise_level)
            observed_stress = np.clip(observed_stress, 0, 1) # Mantener entre 0 y 1
            
            # State: [REG_norm, Agent_REI_norm, Noisy_Swarm_Stress, Trend]
            state = [reg/100, agent.rei/20, observed_stress, -0.01] 
            action, lp = agent.select_action(state)
            actions.append(action)
            log_probs.append(lp)
        
        # Dinámica Colectiva Real
        num_defectors = sum(actions)
        last_defectors_ratio = num_defectors / num_agents
        
        reg_loss = num_defectors * 0.5 # Ajustamos escala para 100 agentes
        reg_gain = (num_agents - num_defectors) * 0.2 
        reg = reg - reg_loss + reg_gain
        
        # Grok's Law
        alpha = np.exp(k * (threshold - reg)) if reg < threshold else 1.0
        
        # Reward & Backprop
        for i, agent in enumerate(swarm):
            # La recompensa ahora incluye el bienestar global escalado por Alpha
            reward = (2 if actions[i] == 0 else -1) + (alpha * (reg/100))
            loss = -log_probs[i] * reward
            agent.optimizer.zero_grad()
            loss.backward()
            agent.optimizer.step()

        if turn % 10 == 0:
            print(f"T:{turn} | REG:{reg:.1f} | Defectors:{num_defectors}/{num_agents} | Alpha:{alpha:.2f}")

run_swarm_test()
