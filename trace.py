import numpy as np
from constants import R_ATMOS, R_EARTH, COEFF, COEFF_H, LUM_SUN
from scattering import rayleigh_phase, mie_phase

from tqdm import tqdm

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

sphere_origin = np.zeros(3)
view_origin=np.array([0, 0, R_EARTH])

def sky_line_dist(view_dir: np.ndarray) -> float:
    """
    Returns the distance to the sky dome for a given elevation angle (in degrees).
    """
    
    a = np.dot(view_dir, view_dir)
    b = 2 * np.dot(view_origin - sphere_origin, view_dir)
    c = np.dot(view_origin - sphere_origin, view_origin - sphere_origin) - R_ATMOS**2

    discriminant = b**2 - 4 * a * c
    if discriminant < 0:
        return ValueError("No intersection with sky dome")

    t = (-b + np.sqrt(discriminant)) / (2 * a)

    if t < 0:
        return ValueError("Looking at ground!")

    return t

def optical_depth(A: np.ndarray, B: np.ndarray, coeff: float, steps: int=10) -> float:
    """
    Tranmittance between two points A and B, based on the distance through the atmosphere
    """
    distance = np.linalg.norm(B - A)
    ds = distance / steps

    coeff_H = COEFF_H[coeff.split("_")[0]]

    h = [np.linalg.norm(A + (B - A) * (i / steps)) - R_EARTH for i in range(steps)]
    depth = sum([np.exp(-h[i] / coeff_H)*ds for i in range(int(steps))])

    return depth

def light_line(view_dir: np.ndarray, sun_dir: np.ndarray, steps: int=10) -> np.array:
    """
    Returns scattered light from sun
    """
    dist_atmo = sky_line_dist(view_dir)
    ds_point = dist_atmo / steps
    sample_points = view_origin + np.outer(ds_point * np.arange(steps), view_dir)

    transmittances = {'red': [], 'green': [], 'blue': [], 'mie': []}

    for coeff in COEFF.keys():

        coeff_H = COEFF_H[coeff.split("_")[0]]
        coeff_val = COEFF[coeff]
        transmittance = 0.0

        for point in sample_points:
            
            h = np.linalg.norm(point) - R_EARTH

            depth_view = optical_depth(view_origin, point, coeff, steps)

            dist_sun = sky_line_dist(sun_dir)
            depth_light = optical_depth(point, point + (sun_dir * dist_sun), coeff, steps)

            

            transmittance += coeff_val * (np.exp(-h / coeff_H)
                                       * np.exp(-coeff_val * (depth_view + depth_light)) * ds_point)
            
        transmittances[coeff.split("_")[-1]] = transmittance

    return transmittances

def colour(view_dir: np.ndarray, sun_dir: np.ndarray) -> np.ndarray:
    """
    Returns intensity of each colour (W/m^2)
    """

    transmittance = light_line(view_dir, sun_dir)  
    mu = np.dot(view_dir, sun_dir)

    colours = ['red', 'green', 'blue']
    transmittance = {k: v * rayleigh_phase(mu) if k in colours 
                     else v * mie_phase(mu) for k, v in transmittance.items()}

    colour = {k: ((transmittance[k] + transmittance['mie']) * LUM_SUN) for k in colours}


    return colour


def sky_image(colour_fn, azim_range: tuple=(-180, 180), elev_range: tuple=(0, 90),
              width: int=200, height: int=100) -> np.ndarray:
    """
    Generate sky image, computing colour for each pixel
    """

    azims = np.linspace(*azim_range, width)
    elevs = np.linspace(*elev_range, height)

    image = np.zeros((height, width, 3))

    for j, elev in tqdm(enumerate(elevs), total=len(elevs)):
        for i, azim in enumerate(azims):
            az = np.radians(azim)
            el = np.radians(elev)
            view_dir = np.array([
                np.cos(el) * np.sin(az),
                np.cos(el) * np.cos(az),
                np.sin(el)
            ])
            rgb = colour_fn(view_dir, sun_dir)
            image[height - 1 - j, i] = [rgb["red"], rgb["green"], rgb["blue"]]

    image = image / (image.max() + 1e-10)
    image = np.clip(image, 0, 1)

    return image


def display_sky(colour_fn, sun_azim: float=0, sun_elev: float=3, width: int=200, height: int=100) -> None:
    """"
    Plotting tool for sky image
    """
    sa, se = np.radians(sun_azim), np.radians(sun_elev)
    sun_dir = np.array([np.cos(se)*np.sin(sa), np.cos(se)*np.cos(sa), np.sin(se)])

    img = sky_image(colour_fn, sun_dir=sun_dir, width=width, height=height)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(img, extent=[-180, 180, 0, 90], aspect='auto')
    ax.set_xlabel("Azimuth (°)")
    ax.set_ylabel("Elevation (°)")
    ax.set_title(f"Sky — sun az={sun_azim}° el={sun_elev}°")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":

    elev = 89
    azim = 0

    view_dir = np.array([np.sin(np.radians(azim)) * np.sin(np.radians(elev)), np.cos(np.radians(azim)) * np.sin(np.radians(elev)), np.cos(np.radians(elev))])
    sun_dir = np.array([0, 0, np.sin(np.radians(0))])  

    display_sky(colour)