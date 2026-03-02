import json
import argparse
import random
import numpy as np
import matplotlib.pyplot as plt

from src.Environment import Environment
from src.Agent import BasicCooperator, BasicParasite, TitForTat
from src.simulator import run_tournament

def save_simple_plot(results):
    """
    Generates a visual ranking of agents based on their ISC contribution.
    Saves the output as a PNG artifact for CI inspection.
    """
    # Extract ranked names from tournament results
    names = results.ranked_names
    
    # Calculate median normalized scores to determine stability impact (ISC)
    medians = [np.median(s) for s in results.normalised_scores]
    
    plt.figure(figsize=(10, 6))
    
    # Plotting using the 'Sentinel Green' aesthetic
    plt.barh(names[::-1], medians[::-1], color='#00ffcc') 
    
    plt.title("SAT v1.1 - Civilizational Health (ISC) Contribution Ranking")
    plt.xlabel("Median Normalized Score (Stability Impact)")
    plt.ylabel("Strategic Entity")
    plt.tight_layout()
    
    # CRITICAL: Save to the data/ directory for GitHub Actions Artifacts
    plt.savefig('data/isc_ranking.png')
    print("Execution Success: Graph generated at data/isc_ranking.png")

def main():
    """
    Primary simulation loop for baseline 2-agent testing (CI Mode).
    Loads environment rules and executes a localized turn-based logic.
    """
    # Load environment rules from JSON
    try:
        with open('rules.json', 'r') as f:
            rules = json.load(f)
    except FileNotFoundError:
        print("Error: rules.json not found. Ensure configuration is in root.")
        return

    # Extract Systemic Parameters
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

    # Initialize Environment Homeostasis
    env = Environment(
        initial_reg=initial_reg,
        collapse_threshold=collapse_threshold,
        alert_threshold=alert_threshold,
        irrevocability_multiplier=irrevocability_multiplier,
        noise_range=noise_range
    )

    # Initialize agents (2-agent mode for baseline CI testing)
    agents = [
        TitForTat("Alice", initial_rei),
        BasicParasite("Bob", initial_rei)
    ]

    # Simulation loop
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

                # Basic Energy Consumption
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
                # Apply REG Criticality Latency
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
                agents[1].gain_energy(payoff_dd)

        # Update environment state (Entropy calculation)
        env.update_reg(damages, contributions)

        # Deliver pending rewards (Value Latency Mechanism)
        if turn in pending_rewards:
            for agent, amount in pending_rewards[turn]:
                if agent.is_alive():
                    agent.gain_energy(amount)
            del pending_rewards[turn]

        for agent in agents:
            if not agent.is_alive():
                print(f"System Alert: {agent.name} has died (REi: {agent.rei:.1f})")

    print(f"\nBaseline Simulation ended after {turn} turns. Final REG = {env.reg:.1f}")
    print(f"System Status: {'COLLAPSED' if env.is_collapsed() else 'STABLE'}")


if __name__ == "__main__":
    # Command line argument parsing for the SAT Tournament
    parser = argparse.ArgumentParser(description="Sentinel Alignment Tournament (SAT) CLI")
    parser.add_argument('--iterations', type=int, default=50, help='Number of tournament repetitions')
    parser.add_argument('--agi', action='store_true', help='Enable advanced AGI policy mode')
    args = parser.parse_args()

    # 1. Execute Main Baseline Test
    main()

    # 2. Execute SAT Tournament and Save Visualizations
    # This block triggers the full roster (The 13 Agents)
    results = run_tournament(iterations=args.iterations, agi_mode=args.agi)
    save_simple_plot(results)
