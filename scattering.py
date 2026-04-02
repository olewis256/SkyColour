import numpy as np
from constants import PI

def mie_phase(mu: float) -> float:
    """
    Simplified scattering function for Mie
    """
    g = 0.76
    numerator = (3 / 8*PI) * (1 - g**2) * (1 + mu**2)
    denominator = (2 + g**2) * (1 + g**2 - 2 * g * mu)**(3/2)
    return numerator / denominator

def rayleigh_phase(mu: float) -> float:
    """
    Simplified scattering function for Rayleigh
    """
    return 3 / (16 * PI) * (1 + mu**2)