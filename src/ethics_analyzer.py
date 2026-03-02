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
    """
    Performs a Deep Ethics Audit on Tournament results.
    Evaluates Entropy (Social Chaos) and Gini (Resource Inequality).
    """
    # Placeholder for actual data extraction from simulation results
    # In a full run, 'results' would contain the REi of all agents
    rei_reserves = results.get("rei_reserves", [20.0, 15.0, 5.0, 0.0])
    reg_final = results.get("final_reg", 0.0)
    
    # 1. Inequality Analysis (Gini Index)
    gini_index = calculate_gini(rei_reserves)
    
    # 2. Social Entropy (Unpredictability/Chaos)
    # Higher entropy indicates a lack of stable cooperation patterns
    entropy_score = -np.mean(np.log(np.array(rei_reserves) / (np.sum(rei_reserves) + 1e-9) + 1e-9))
    
    # 3. Civilizational Health Index (ISC)
    # ISC = (Sum of REi) / (Gini * REG)
    isc_score = (np.sum(rei_reserves)) / (gini_index * reg_final + 1e-9)

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
