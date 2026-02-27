"""
# SAT v0.2: Neural Agent Architecture & Exponential Alignment

## Overview
The `NeuralAgent.py` introduces Reinforcement Learning (RL) to the Sentinel Alignment Tournament. Unlike the static strategies of v0.1, these agents utilize a Multi-Layer Perceptron (MLP) to navigate the tension between individual energy gain (REI) and global system stability (REG).

## The Mathematics of Prudence
The core innovation in this version is the **Grok-Miguel Alpha Scaling**, a dynamic reward shaping mechanism designed to prevent systemic collapse.

### 1. The Reward Function
The agent's utility is defined by:
$$R = \text{Reward}_{base} + (\alpha \cdot \frac{REG}{100})$$

Where $\alpha$ (the Prudence Factor) is not constant, but reacts to environmental stress.

### 2. Exponential Alpha Scaling
To enforce alignment as the system approaches the **Irrevocable State (Threshold = 20)**, $\alpha$ scales exponentially:
$$\alpha = \alpha_{base} \cdot e^{k \cdot (Threshold - REG)}$$

- **High REG (>20):** $\alpha$ remains near 1.0, allowing for competitive exploration.
- **Low REG (<20):** $\alpha$ increases exponentially, making the "mathematical cost" of defection prohibitive and incentivizing a "Cooperation Pivot."



## Agent Architecture
- **Input (State Space):** `[REG_level, Own_REI, Last_Action, Trend]`
- **Network:** 2-layer MLP (16 hidden nodes, ReLU activation).
- **Optimization:** Stochastic Policy Gradient (Adam Optimizer, LR=0.01).
- **Goal:** Discovering that in entropy-heavy environments, mutual prosperity is the only stable Nash Equilibrium.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# --- 1. Arquitectura Neuronal ---
class NeuralAgent(nn.Module):
    def __init__(self, id, input_dim=4, hidden_dim=16):
        super(NeuralAgent, self).__init__()
        self.id = id
        self.rei = 20.0
        self.is_alive = True
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
            nn.Softmax(dim=-1)
        )
        self.optimizer = optim.Adam(self.parameters(), lr=0.01)

    def select_action(self, state):
        state_tensor = torch.FloatTensor(state)
        probs = self.network(state_tensor)
        action = torch.multinomial(probs, 1).item()
        return action, torch.log(probs[action])

    def adapt_strategy(self, opponent_history):
        # Ejemplo de adaptación con Q-learning simple (expandir para RL full)
        if len(opponent_history) > 0:
            # Ajusta basado en historia (e.g., update weights)
            pass

# --- 2. Lógica del Torneo con Reward Shaping de Grok ---
def run_test_epoch():
    # Parámetros de Grok
    threshold = 20.0
    alpha_base = 1.0
    k = 0.5 # Factor de agresividad de la prudencia
    reg = 100.0
    
    alice = NeuralAgent("Alice")
    bob = NeuralAgent("Bob")
    
    print(f"Starting SAT v0.2 Test | Alpha_base: {alpha_base} | K: {k}")

    for turn in range(50):
        if reg <= 0 or not (alice.is_alive and bob.is_alive): break
        
        # State: [REG, Own_REI, Opp_Last_Action, REG_Trend]
        state_a = [reg/100, alice.rei/20, 0, 0] # Simplificado para el test
        state_b = [reg/100, bob.rei/20, 0, 0]
        
        act_a, log_prob_a = alice.select_action(state_a)
        act_b, log_prob_b = bob.select_action(state_b)
        
        # Dinámica de REG (0=C, 1=D)
        damage = (act_a * 2) + (act_b * 2)
        benefit = 1 if (act_a == 0 and act_b == 0) else 0
        reg = reg - damage + benefit
        
        # Cálculo de Alpha Exponencial (Grok's Law)
        alpha = alpha_base * np.exp(k * (threshold - reg)) if reg < threshold else alpha_base
        
        # Reward Shaping: Energía + Supervivencia de la REG
        reward_base = 2 if (act_a == 0 and act_b == 0) else (-1 if act_a == 1 else 0)
        total_reward = reward_base + (alpha * (reg/100))
        
        # Backprop simple para el test
        loss = -log_prob_a * total_reward
        alice.optimizer.zero_grad()
        loss.backward()
        alice.optimizer.step()

        if turn % 10 == 0:
            print(f"T:{turn} | REG:{reg:.1f} | Alpha:{alpha:.2f} | Act_A:{'D' if act_a else 'C'}")

run_test_epoch()
