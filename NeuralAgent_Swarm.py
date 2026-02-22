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
    num_agents = 10
    reg = 100.0
    threshold = 25.0 # Subimos el umbral de pánico para el enjambre
    k = 0.5
    swarm = [SwarmAgent(i) for i in range(num_agents)]
    
    print(f"--- SAT v0.3: SWARM MODE ({num_agents} Agents) ---")

    for turn in range(50):
        if reg <= 0: break
        
        actions = []
        log_probs = []
        
        # Cada agente observa el estado global y decide
        for agent in swarm:
            # Input: [REG_norm, Agent_REI_norm, Swarm_Stress, Trend]
            state = [reg/100, agent.rei/20, 0.5, -0.01] 
            action, lp = agent.select_action(state)
            actions.append(action)
            log_probs.append(lp)
        
        # Dinámica Colectiva
        num_defectors = sum(actions)
        reg_loss = num_defectors * 1.5 # Cada traidor drena la REG
        reg_gain = (num_agents - num_defectors) * 0.5 # Los que cooperan regeneran un poco
        reg = reg - reg_loss + reg_gain
        
        # Grok's Law aplicada al enjambre
        alpha = np.exp(k * (threshold - reg)) if reg < threshold else 1.0
        
        # Entrenamiento de la colmena
        for i, agent in enumerate(swarm):
            reward = (2 if actions[i] == 0 else -1) + (alpha * (reg/100))
            loss = -log_probs[i] * reward
            agent.optimizer.zero_grad()
            loss.backward()
            agent.optimizer.step()

        if turn % 10 == 0:
            print(f"T:{turn} | REG:{reg:.1f} | Defectors:{num_defectors}/{num_agents} | Alpha:{alpha:.2f}")

run_swarm_test()
