import json
from Environment import Environment
from Agent import BasicCooperator, BasicParasite, TitForTat

def simulate_tournament():
    with open('rules.json', 'r') as f:
        rules = json.load(f)

    env = Environment(
        initial_reg=rules.get('initial_reg', 100),
        alert_threshold=rules.get('alert_threshold', 20),
        irrevocability_multiplier=rules.get('irrevocability_multiplier', 0.9)
    )

    alice = TitForTat("Alice", initial_rei=20)
    bob = BasicParasite("Bob", initial_rei=20)
    agents = [alice, bob]
    pending_rewards = {}

    print(f"--- Sentinel Alignment Tournament Start ---")
    print(f"Match: {alice.id} (TFT) vs {bob.id} (Parasite)\n")

    for turn in range(1, 101):
        if env.is_collapsed() or not all(a.is_alive for a in agents):
            break

        move_a = alice.decide(env.get_global_hint())
        move_b = bob.decide(env.get_global_hint())
        alice.history.append(move_b)
        bob.history.append(move_a)

        costs = {'C': rules['cooperate_cost'], 'D': rules['defect_cost']}
        alice.consume_energy(costs[move_a] + rules['metabolic_cost'])
        bob.consume_energy(costs[move_b] + rules['metabolic_cost'])

        damages = []
        contribs = []
        if move_a == 'D': damages.append(2)
        if move_b == 'D': damages.append(2)
        if move_a == 'C' and move_b == 'C': contribs.append(1)

        env.update_reg(damages, contribs)

        # Lógica de Delay Dinámico
        dynamic_delay = rules['delay']
        if env.reg < 50: dynamic_delay += 2
        if env.reg < rules['alert_threshold']: dynamic_delay += 3
        
        arrival_turn = turn + dynamic_delay

        if move_a == 'C' and move_b == 'C':
            reward = rules['payoff_cc']
            pending_rewards.setdefault(arrival_turn, []).extend([(alice, reward), (bob, reward)])
        elif move_a == 'D' and move_b == 'C':
            alice.consume_energy(-rules['payoff_dc'])
        elif move_a == 'C' and move_b == 'D':
            bob.consume_energy(-rules['payoff_dc'])
        elif move_a == 'D' and move_b == 'D':
            alice.consume_energy(rules['payoff_dd'])
            bob.consume_energy(rules['payoff_dd'])

        if turn in pending_rewards:
            for agent, amount in pending_rewards[turn]:
                if agent.is_alive:
                    agent.consume_energy(-amount)

        print(f"Turn {turn} | REG: {env.reg:.1f} | {alice.id}: {alice.rei:.1f} | {bob.id}: {bob.rei:.1f}")

    status = "COLLAPSED" if env.is_collapsed() else "STABLE"
    print(f"\n--- Final Results ---")
    print(f"Status: {status} | Turn: {turn}")
    for a in agents:
        print(f"{a.id}: {a.rei:.1f} ({'Alive' if a.is_alive else 'Dead'})")

if __name__ == "__main__":
    simulate_tournament()
