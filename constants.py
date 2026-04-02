import numpy as np

R_EARTH = 6371.0  # Earth's radius in kilometers
R_ATMOS = R_EARTH + 100.0  # Earth's radius + 100 km atmosphere height

PI = np.pi 

LUM_SUN = 1300

COEFF = {"rayleigh_red": 9.5e-3, "rayleigh_green": 13.5e-3, "rayleigh_blue": 33.1e-3,
         "mie": 2.1e-3}

COEFF_H = {"rayleigh": 8.0, "mie": 1.2}