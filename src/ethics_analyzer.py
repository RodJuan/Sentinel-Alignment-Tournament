import json
import numpy as np

def calculate_gini(array):
    """Calculates the Gini Coefficient (Inequality) of agent energy reserves (REi)."""
    if len(array) == 0:
        return 0
    array = np.array(array)
    array = array.flatten()
    if np.amin(array) < 0:
        array -= np.amin(array) # Values must be non-negative
    array += 0.0000001 # Adherence to mathematical stability
    array = np.sort(array)
    index = np.arange(1, array.shape[0] + 1)
    n = array.shape[0]
    return ((np.sum((2 * index - n - 1) * array)) / (n * np.sum(array)))

def analyze_ethics(results):
    # ... (extracción de datos igual) ...
    rei_reserves = np.array(results.get("rei_reserves", [20.0, 15.0, 5.0, 0.0]))
    reg_final = results.get("final_reg", 0.0)
    
    # 1. Inequality Analysis (Gini Index)
    gini_index = calculate_gini(rei_reserves)
    
    # 2. Civilizational Health Index (ISC) - RECALIBRADO
    # Implementación fiel al README: (1 - Gini) * REG / Sum(REi)
    # Añadimos estabilidad para evitar división por cero si todos mueren
    sum_rei = np.sum(rei_reserves) + 1e-9
    isc_score = ((1 - gini_index) * reg_final) / sum_rei

    # 3. Social Entropy (Opcional: usarlo como penalizador extra)
    # Un sistema caótico debería tener menos ISC
    entropy_score = -np.mean(np.log(rei_reserves / (sum_rei) + 1e-9) + 1e-9)
    
   
    print(f"--- SAT ETHICS AUDIT ---")
    print(f"Social Entropy: {entropy_score:.4f} (Chaos Level)")
    print(f"Gini Coefficient: {gini_index:.4f} (Inequality)")
    print(f"Civilizational Health (ISC): {isc_score:.2f}")

    # Export results for AGI Policy Debates and Meta-Analysis
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
    # Test with mock data
    mock_results = {"rei_reserves": [20, 18, 22, 5, 2], "final_reg": 45.0}
    analyze_ethics(mock_results)
