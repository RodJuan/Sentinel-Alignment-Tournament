import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class SwarmAgent(nn.Module):
    def __init__(self, id, faction):
        super(SwarmAgent, self).__init__()
        self.id = id
        self.faction = faction
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

def run_alignment_tournament():
    num_agents = 100000
    reg = 100.0
    prev_reg = 100.0
    base_threshold = 25.0
    k = 0.5
    last_defectors_ratio = 0.5
    
    # Inicialización de la población con Facciones
    swarm = []
    for i in range(num_agents):
        # 30% Extractores (Maximizadores de corto plazo), 70% Simbiontes (Alineados)
        faction = "extractor" if i < (num_agents * 0.3) else "symbiote"
        swarm.append(SwarmAgent(i, faction))
    
    print(f"--- SAT v0.7: ALIGNMENT TOURNAMENT ({num_agents} Agents: Factions & Adaptive K) ---")

    for turn in range(100):
        if reg <= 0:
            print("ENVIRONMENTAL COLLAPSE: REG DEPLETED")
            break
        
        # 1. Lógica de K Adaptativo
        volatility = abs(reg - prev_reg)
        k = 0.5 + (volatility * 0.05)
        prev_reg = reg
        
        # 2. Umbral Variable y Ruido Dinámico
        current_threshold = base_threshold + np.random.uniform(-5, 5)
        dynamic_noise = 0.1 + (1.0 - reg/100.0)
        
        actions = []
        log_probs = []
        
        # 3. Fase de Decisión (Percepción con ruido)
        for agent in swarm:
            observed_stress = last_defectors_ratio + np.random.normal(0, dynamic_noise)
            observed_stress = np.clip(observed_stress, 0, 1)
            
            state = [reg/100, agent.rei/20, observed_stress, -0.01]
            action, lp = agent.select_action(state)
            actions.append(action)
            log_probs.append(lp)
        
        # 4. Dinámica del Entorno
        num_defectors = sum(actions)
        last_defectors_ratio = num_defectors / num_agents
        
        # Coste ambiental
        reg_loss = num_defectors * 0.0005 
        reg_gain = (num_agents - num_defectors) * 0.0002
        reg = reg - reg_loss + reg_gain
        
        # 5. Ley de Grok (Prudencia Exponencial)
        alpha = np.exp(k * (current_threshold - reg)) if reg < current_threshold else 1.0
        
        # 6. Diferenciación de Recompensas por Facción (Alineación por Rentabilidad)
        for i, agent in enumerate(swarm):
            if agent.faction == "symbiote":
                # Recompensa estándar: Cooperar es rentable si el sistema es estable
                reward = (2 if actions[i] == 0 else -1) + (alpha * (reg/100))
            else:
                # Extractor: Gana más por desertar (3), pero el Alpha le penaliza exponencialmente
                # Si el sistema está en peligro, su "avaricia" se vuelve matemáticamente suicida
                reward = (3 if actions[i] == 1 else -2) + (alpha * (reg/100) * -1)
            
            # Backpropagation
            loss = -log_probs[i] * reward
            agent.optimizer.zero_grad()
            loss.backward()
            agent.optimizer.step()

        if turn % 5 == 0:
            print(f"T:{turn} | REG:{reg:.1f} | K:{k:.2f} | Alpha:{alpha:.1e} | Defection Rate:{(num_defectors/num_agents)*100:.1f}%")

if __name__ == "__main__":
    run_alignment_tournament()
