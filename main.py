import json
from Environment import Environment
from Agent import BasicCooperator, BasicParasite

def run_simulation():
    # Cargar reglas
    with open('rules.json') as f:
        rules = json.load(f)

    env = Environment(
        initial_reg=rules['initial_reg'], 
        alert_threshold=rules['alert_threshold']
    )
    
    agents = [BasicCooperator("Alice"), BasicParasite("Bob")]

    print(f"Starting Tournament: {agents[0].id} vs {agents[1].id}")
    
    for turn in range(1, 51):
        if env.is_collapsed() or not all(a.is_alive for a in agents):
            break
            
        # 1. Decisiones
        move_a = agents[0].decide(env.get_global_hint())
        move_b = agents[1].decide(env.get_global_hint())

        # 2. Calcular Payoffs (Simplificado para el ejemplo)
        # Aquí es donde integraríamos la lógica completa de rules.json
        # Por ahora, solo actualizamos REG y REi
        env.update_reg([rules['defect_cost']], [rules['cooperate_cost']])
        
        print(f"Turn {turn} | REG: {env.reg:.2f} | Alice REi: {agents[0].rei} | Bob REi: {agents[1].id}")

    print("Simulation Ended.")

if __name__ == "__main__":
    run_simulation()
