import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from datetime import datetime, timedelta
import pandas as pd

# Constants
c = 299792458  # Speed of light (m/s)
f0 = 437e6     # Base frequency (Hz), e.g. UHF

# Station location
latitude = 26
longitude = -100
altitude = 6378137.0 + 2100

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

# Ground station position (ECEF, meters)
gs_position = geodetic_to_ecef(latitude, longitude, altitude)

#---------------------------

df = pd.read_csv('ground_track.csv')


communication_df = df.loc[df['comm_window'] == 1]
communication_df = communication_df.copy()
communication_df['timestamp'] = pd.to_datetime(communication_df['timestamp'])
communication_df['altitude'] = communication_df['altitude_km'] * 1000

# Convert lat/lon/alt of satellite to ECEF
def df_to_ecef_positions(df):
    ecef_positions = []
    for _, row in df.iterrows():
        ecef = geodetic_to_ecef(row['latitude_deg'], row['longitude_deg'], row['altitude'])
        ecef_positions.append(ecef)
    return np.array(ecef_positions)

# Verify the data belongs to the same pass
pass_df = pd.DataFrame(columns=communication_df.columns)
idx = 0
while idx < len(communication_df.index):
    if (communication_df.iloc[idx + 1]['timestamp'] - communication_df.iloc[idx]['timestamp']) > timedelta(minutes=60):
        break
    pass_df.loc[idx] = communication_df.iloc[idx]
    idx += 1

def timestamps_to_seconds(df):
    base_time = pd.to_datetime(df['timestamp'].iloc[0])
    t_seconds = (pd.to_datetime(df['timestamp']) - base_time).dt.total_seconds().values
    return t_seconds

# Get satellite positions and time vector from the dataframe
sat_positions = df_to_ecef_positions(pass_df)
t = timestamps_to_seconds(pass_df)

def compute_doppler_shift(sat_positions, gs_position):
    relative_vectors = sat_positions - gs_position
    distances = np.linalg.norm(relative_vectors, axis=1)
    radial_velocities = np.gradient(distances)
    doppler_shifts = f0 * (radial_velocities / c)
    observed_freqs = f0 + doppler_shifts
    return observed_freqs

# Fit the orbit using a simple linear model (mockup)
def fit_orbit(t, observed_freqs):
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

# Main simulation
observed_freqs = compute_doppler_shift(sat_positions, gs_position)

# Add noise
observed_freqs += np.random.normal(0, 1, size=observed_freqs.shape)  # Hz-level noise

# Fit back the velocity
estimated_velocity = fit_orbit(t, observed_freqs)
print("Estimated Velocity (m/s):", estimated_velocity)

# Plot
plt.plot(t, observed_freqs, label='Observed Frequency')
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.title('Simulated Doppler Shift')
plt.grid()
plt.legend()
plt.show()

# Stub for generating a TLE
now = datetime.utcnow()
tle_template = f"""1 99999U 25001A   {now.strftime('%y%j')}.00000000  .00000000  00000-0  00000-0 0  9990\n2 99999 {98.0000:8.4f} {0.0000:8.4f} 0000001  0.0000  0.0000 {15.00000000:11.8f}    00"""
print(f"Generated TLE set (stub):\n {tle_template}")