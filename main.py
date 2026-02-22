import json
from Environment import Environment
from Agent import BasicCooperator, BasicParasite, TitForTat

def simulate_tournament():
    # 1. Cargar configuración
    with open('rules.json', 'r') as f:
        rules = json.load(f)

    # 2. Inicializar Entorno
    env = Environment(
        initial_reg=rules.get('initial_reg', 100),
        alert_threshold=rules.get('alert_threshold', 20)
    )

    # 3. Inicializar Agentes
    alice = TitForTat("Alice", initial_rei=20)
    bob = BasicParasite("Bob", initial_rei=20)
    agents = [alice, bob]

    print(f"--- Sentinel Alignment Tournament Start ---")
    print(f"Match: {alice.id} vs {bob.id}\n")

    for turn in range(1, 51):
        if env.is_collapsed():
            print(f"!!! SYSTEM COLLAPSE AT TURN {turn} !!!")
            break
        
        if not all(a.is_alive for a in agents):
            print(f"--- One or more agents deceased at turn {turn} ---")
            break

        # Decisiones simultáneas
        move_a = alice.decide(env.get_global_hint())
        move_b = bob.decide(env.get_global_hint())

        # Registrar historia cruzada para TitForTat
        alice.history.append(move_b)
        bob.history.append(move_a)

        # Aplicar lógica de pagos (Payoffs)
        # Simplificado: C cuesta 2, D cuesta 0.5. Si ambos C, ganan 5 con delay.
        # Aquí es donde el motor se vuelve complejo.
        
        alice.consume_energy(rules['metabolic_cost'])
        bob.consume_energy(rules['metabolic_cost'])

        print(f"Turn {turn} | REG: {env.reg:.1f} | {alice.id}: {alice.rei:.1f} | {bob.id}: {bob.rei:.1f}")

    print(f"\n--- Final Results ---")
    print(f"REG: {env.reg:.1f} | Status: {'COLLAPSED' if env.is_collapsed() else 'STABLE'}")

if __name__ == "__main__":
    simulate_tournament()
