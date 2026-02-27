import pytest
from src.simulator import run_tournament

def test_tournament_basic():
    run_tournament(iterations=10)
    assert True  # Añade asserts reales
