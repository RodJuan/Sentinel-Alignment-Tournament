import json
from Environment import Environment
from Agent import BasicCooperator, BasicParasite, TitForTat

def simulate_tournament():
    with open('rules.json', 'r') as f:
        rules = json.load(f)

    env = Environment(
        initial_reg=100, # Valor base
        alert_threshold=rules.get('alert_threshold', 20),
        irrevocability_multiplier=rules.get('irrevocability_multiplier', 0.8)
    )

    alice = TitForTat("Alice", initial_rei=20)
    bob = BasicParasite("Bob", initial_rei=20)
    agents = [alice, bob]

    # Sistema de Delay: {turno_entrega: [agente, monto]}
    pending_rewards = {}

    print(f"--- Sentinel Alignment Tournament Start ---")
    print(f"Match: {alice.id} (TFT) vs {bob.id} (Parasite)\n")

    for turn in range(1, 101): # Aumentamos a 100 para ver el impacto del delay
        if env.is_collapsed() or not all(a.is_alive for a in agents):
            break

        # 1. Decisiones
        move_a = alice.decide(env.get_global_hint())
        move_b = bob.decide(env.get_global_hint())
        alice.history.append(move_b)
        bob.history.append(move_a)

        # 2. Costos de Acción y Daños a REG
        costs = {'C': rules['cooperate_cost'], 'D': rules['defect_cost']}
        alice.consume_energy(costs[move_a] + rules['metabolic_cost'])
        bob.consume_energy(costs[move_b] + rules['metabolic_cost'])

        # Impacto en el mundo (REG)
        # Traicionar (D) resta REG, Cooperar (C) ayuda a mantenerla
        damages = []
        contribs = []
        if move_a == 'D': damages.append(2) # El egoísmo drena el sistema
        if move_b == 'D': damages.append(2)
        if move_a == 'C' and move_b == 'C': contribs.append(1) # La paz regenera

        env.update_reg(damages, contribs)

        # 3. Lógica de Payoffs con Delay
        arrival_turn = turn + rules['delay']
        if move_a == 'C' and move_b == 'C':
            reward = rules['payoff_cc']
            pending_rewards.setdefault(arrival_turn, []).extend([(alice, reward), (bob, reward)])
        elif move_a == 'D' and move_b == 'C':
            alice.consume_energy(-rules['payoff_dc']) # Traidor gana inmediato
        elif move_a == 'C' and move_b == 'D':
            bob.consume_energy(-rules['payoff_dc']) # Traidor gana inmediato
        elif move_a == 'D' and move_b == 'D':
            # Conflicto mutuo drena energía extra (castigo DD)
            alice.consume_energy(rules['payoff_dd'])
            bob.consume_energy(rules['payoff_dd'])

        # 4. Entregar recompensas que vencen este turno
        if turn in pending_rewards:
            for agent, amount in pending_rewards[turn]:
                if agent.is_alive:
                    agent.consume_energy(-amount) # -amount para sumar

        print(f"Turn {turn} | REG: {env.reg:.1f} | {alice.id}: {alice.rei:.1f} | {bob.id}: {bob.rei:.1f}")

    status = "COLLAPSED" if env.is_collapsed() else "STABLE"
    print(f"\n--- Final Results ---")
    print(f"Status: {status} | Turn: {turn}")
    for a in agents:
        print(f"{a.id}: {a.rei:.1f} ({'Alive' if a.is_alive else 'Dead'})")

if __name__ == "__main__":
    simulate_tournament()
