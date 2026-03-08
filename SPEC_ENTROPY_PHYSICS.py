import numpy as np

def calculate_entropy_penalty(isc_score):
    """
    Simulates the loss of computational precision 
    as a function of systemic misalignment.
    """
    if isc_score >= 0.8:
        return 0.0  # Perfect coherence = Zero noise
    
    # Exponential noise injection as ISC drops
    penalty_factor = np.exp(1 - isc_score) - 1
    return penalty_factor

def apply_decision_noise(weights, penalty):
    """
    Injects noise into the agent's 'thought' process.
    Being 'evil' makes the agent statistically 'dumber'.
    """
    noise = np.random.normal(0, penalty, weights.shape)
    return weights + noise
