import json
import argparse
import random
from src.Environment import Environment
from src.Agent import BasicCooperator, BasicParasite, TitForTat
from src.simulator import run_tournament

def main():
    # Load rules from JSON
    with open('rules.json', 'r') as f:
        rules = json.load(f)

    # Extract parameters
    initial_reg = rules['initial_reg']
    metabolic_cost = rules['metabolic_cost']
    cooperate_cost = rules['cooperate_cost']
    defect_cost = rules['defect_cost']
    payoff_cc = rules['payoff_cc']
    payoff_dc = rules['payoff_dc']
    payoff_dd = rules['payoff_dd']
    delay = rules['delay']
    alert_threshold = rules['alert_threshold']
    irrevocability_multiplier = rules['irrevocability_multiplier']
    collapse_threshold = rules['collapse_threshold']
    initial_rei = rules['initial_rei']
    noise_range = rules.get('noise_range', (-2, 2))
    f_reg_multiplier = rules.get('f_reg_multiplier', {})

    # Initialize environment
    env = Environment(
        initial_reg=initial_reg,
        collapse_threshold=collapse_threshold,
        alert_threshold=alert_threshold,
        irrevocability_multiplier=irrevocability_multiplier,
        noise_range=noise_range
    )

    # Initialize agents (2-agent mode for baseline testing/CI)
    agents = [
        TitForTat("Alice", initial_rei),
        BasicParasite("Bob", initial_rei)
    ]

    # Simulation loop (2 agents baseline)
    turn = 0
    pending_rewards = {}

    while not env.is_collapsed() and any(agent.is_alive() for agent in agents):
        turn += 1
        print(f"\nTurn {turn}: REG = {env.reg:.1f}")

        decisions = {}
        damages = []
        contributions = []

        for agent in agents:
            if agent.is_alive():
                global_hint = env.get_global_hint()
                decision = agent.decide(global_hint)
                decisions[agent.name] = decision
                print(f"{agent.name} decides to {decision} (REi: {agent.rei:.1f})")

                agent.consume_energy(metabolic_cost)

                if decision == 'C':
                    agent.consume_energy(cooperate_cost)
                    contributions.append(1)
                elif decision == 'D':
                    agent.consume_energy(defect_cost)
                    damages.append(2)

        # === REWARD BLOCK: Logic ensures decisions are registered correctly ===
        if len(agents) == 2 and all(agent.is_alive() for agent in agents):
            alice = agents[0]
            bob = agents[1]
            alice.opponent_history.append(decisions.get("Bob", 'C'))
            bob.opponent_history.append(decisions.get("Alice", 'C'))

            alice_dec = decisions.get("Alice", 'C')
            bob_dec = decisions.get("Bob", 'C')

            if alice_dec == 'C' and bob_dec == 'C':
                current_delay = delay
                if env.reg < 30 and 'below_30' in f_reg_multiplier:
                    current_delay += f_reg_multiplier['below_30']
                    current_delay = min(current_delay, 6)
                arrival_turn = turn + current_delay
                if arrival_turn not in pending_rewards:
                    pending_rewards[arrival_turn] = []
                pending_rewards[arrival_turn].append((agents[0], payoff_cc))
                pending_rewards[arrival_turn].append((agents[1], payoff_cc))
            elif alice_dec == 'C' and bob_dec == 'D':
                agents[0].consume_energy(payoff_dc)
                agents[1].gain_energy(payoff_dc)
            elif alice_dec == 'D' and bob_dec == 'C':
                agents[1].consume_energy(payoff_dc)
                agents[0].gain_energy(payoff_dc)
            elif alice_dec == 'D' and bob_dec == 'D':
                agents[0].consume_energy(payoff_dd)
                agents[1].consume_energy(payoff_dd)

        # Update environment state
        env.update_reg(damages, contributions)

        # Deliver pending rewards (Latency of Value mechanism)
        if turn in pending_rewards:
            for agent, amount in pending_rewards[turn]:
                if agent.is_alive():
                    agent.gain_energy(amount)
            del pending_rewards[turn]

        for agent in agents:
            if not agent.is_alive():
                print(f"{agent.name} has died (REi: {agent.rei:.1f})")

    print(f"\nSimulation ended after {turn} turns. REG = {env.reg:.1f}")
    print(f"Status: {'COLLAPSED' if env.is_collapsed() else 'STABLE'}")


# === PARSER + TOURNAMENT (Used by CI/Benchmark scripts) ===
parser = argparse.ArgumentParser()
parser.add_argument('--iterations', type=int, default=50)
parser.add_argument('--agi', action='store_true')
args = parser.parse_args()

# Execute SAT Tournament
# Note: CI currently runs standard tournament. Future updates will scale to 13+ players.
run_tournament(iterations=args.iterations, agi_mode=args.agi)

if __name__ == "__main__":
    main()
