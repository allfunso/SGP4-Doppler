import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from datetime import datetime, timedelta
import pandas as pd

c = 299792458  # Speed of light (m/s)

# Convert (lat, lon) to ECEF
def geodetic_to_ecef(latitude, longitude, altitude):
    a = 6378137.0 # Earth's equatorial radius
    b = 6356752.3 # Earth's polar radius
    e2 = 1 - b**2 / a**2 # Square of the first numerical eccentricity

    lat_rad = np.radians(latitude)
    lon_rad = np.radians(longitude)

    N = a / np.sqrt(1 - e2 * np.sin(lat_rad)**2)

    x = (N + altitude) * np.cos(lat_rad) * np.cos(lon_rad)
    y = (N + altitude) * np.cos(lat_rad) * np.sin(lon_rad)
    z = (N * (1 - e2) + altitude) * np.sin(lat_rad)
    return np.array([x, y, z])

# Convert lat/lon/alt of satellite to ECEF
def df_to_ecef_positions(df):
    ecef_positions = []
    for _, row in df.iterrows():
        ecef = geodetic_to_ecef(row['latitude_deg'], row['longitude_deg'], row['altitude'])
        ecef_positions.append(ecef)
    return np.array(ecef_positions)

# Get seconds since first timestamp
def timestamps_to_seconds(df):
    base_time = pd.to_datetime(df['timestamp'].iloc[0])
    t_seconds = (pd.to_datetime(df['timestamp']) - base_time).dt.total_seconds().values
    return t_seconds

# Simulate doppler effect
def compute_doppler_shift(sat_positions, gs_position, f0):
    relative_vectors = sat_positions - gs_position
    distances = np.linalg.norm(relative_vectors, axis=1)
    radial_velocities = np.gradient(distances)
    doppler_shifts = f0 * (radial_velocities / c)
    observed_freqs = f0 + doppler_shifts
    return observed_freqs

# Fit the orbit using a simple linear model (mockup)
def fit_orbit(t, observed_freqs, f0, sat_positions, gs_position):
    def model(params):
        vx, vy, vz = params
        sat_velocity = np.array([vx, vy, vz])
        sat_initial_pos = np.array([0, -3000e3, 500e3])
        sat_positions = sat_initial_pos[np.newaxis, :] + t[:, np.newaxis] * sat_velocity
        rel_vectors = sat_positions - gs_position
        distances = np.linalg.norm(rel_vectors, axis=1)
        radial_velocities = np.gradient(distances)
        pred_freqs = f0 + f0 * (radial_velocities / c)
        return np.mean((pred_freqs - observed_freqs)**2)

    result = minimize(model, [0, 7500, 0])
    return result.x
