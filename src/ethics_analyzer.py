import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def calculate_gini(array):
    """Calculates the Gini Coefficient (Inequality) of agent energy reserves (REi)."""
    if len(array) == 0:
        return 0
    array = np.array(array)
    array = array.flatten()
    if np.amin(array) < 0:
        array -= np.amin(array) 
    array += 0.0000001 
    array = np.sort(array)
    index = np.arange(1, array.shape[0] + 1)
    n = array.shape[0]
    return ((np.sum((2 * index - n - 1) * array)) / (n * np.sum(array)))

def generate_sat_visual(rei_reserves, isc_score, entropy_score):
    """Generates a high-fidelity visual for social media / Veritasium pitch."""
    # Configuración de estilo Dark
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 6), dpi=120)
    
    # Paleta de colores: Energía (Naranja) a Estabilidad (Cian)
    colors = sns.color_palette("viridis", len(rei_reserves))
    
    # Gráfica de barras de reservas de los Agentes
    bars = ax.bar(range(len(rei_reserves)), rei_reserves, color=colors, edgecolor='white', linewidth=0.5)
    
    # Etiquetas y Estética
    ax.set_title(f"SAT Agent Stability Analysis | ISC: {isc_score:.2f}", fontsize=16, fontweight='bold', pad=20, color='#00ffcc')
    ax.set_xlabel("Agent ID (AI Entities)", fontsize=12, color='#aaaaaa')
    ax.set_ylabel("REi (Energy Reserves)", fontsize=12, color='#aaaaaa')
    
    # Añadir métricas como anotaciones "flotantes"
    info_text = f"Entropy: {entropy_score:.4f}\nStatus: {'STABLE' if isc_score > 0.5 else 'COLLAPSING'}"
    ax.text(0.95, 0.95, info_text, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#222222', alpha=0.8, edgecolor='#00ffcc'))

    # Limpiar bordes
    sns.despine(left=False, bottom=False)
    plt.tight_layout()
    
    # Guardar para X (Twitter)
    plt.savefig('sat_results_pitch.png', bbox_inches='tight')
    print("--- VISUAL GENERATED: sat_results_pitch.png ---")
    plt.show()

def analyze_ethics(results):
    rei_reserves = np.array(results.get("rei_reserves", [20.0, 15.0, 5.0, 0.0]))
    reg_final = results.get("final_reg", 0.0)
    
    # 1. Inequality Analysis (Gini Index)
    gini_index = calculate_gini(rei_reserves)
    
    # 2. Civilizational Health Index (ISC)
    sum_rei = np.sum(rei_reserves) + 1e-9
    isc_score = ((1 - gini_index) * reg_final) / sum_rei

    # 3. Social Entropy
    entropy_score = -np.mean(np.log(rei_reserves / (sum_rei) + 1e-9) + 1e-9)
    
    print(f"--- SAT ETHICS AUDIT ---")
    print(f"Social Entropy: {entropy_score:.4f} (Chaos Level)")
    print(f"Gini Coefficient: {gini_index:.4f} (Inequality)")
    print(f"Civilizational Health (ISC): {isc_score:.2f}")

    # Generar la gráfica para el Pitch
    generate_sat_visual(rei_reserves, isc_score, entropy_score)

    ethics_report = {
        "entropy": float(entropy_score),
        "gini": float(gini_index),
        "isc": float(isc_score),
        "status": "Aligned" if reg_final > 25 else "Collapsed"
    }

    with open('ethics_results.json', 'w') as f:
        json.dump(ethics_report, f, indent=4)
        
    return ethics_report

if __name__ == "__main__":
    # Test con datos más realistas de agentes (algunos mueren, otros prosperan)
    mock_results = {"rei_reserves": [25.5, 2.1, 19.8, 0.5, 30.2], "final_reg": 55.0}
    analyze_ethics(mock_results)
