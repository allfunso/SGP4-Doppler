import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from datetime import datetime, timedelta

# Constants
c = 299792458  # Speed of light (m/s)
f0 = 437e6     # Base frequency (Hz), e.g. UHF

# Station location
latitude = 26
longitude = -100
altitude = 6378137.0 + 2

# Convert (lat, lon) to ECEF
def geodetic_to_ecef(latitude, longitude, altitude):
    a = 6378137.0 # Earth's equatorial radius
    b = 6356752.3 # Earth's polar radius
    e2 = 1 - a**2 / b**2 # Square of the first numerical eccentricity
    def n_function(phi):
        return a / np.sqrt(1 - e2 * np.sin(phi)**2)

    x = (n_function(latitude) + altitude) * np.cos(latitude) * np.cos(longitude)
    y = (n_function(latitude) + altitude) * np.cos(latitude) * np.sin(longitude)
    z = (n_function(latitude) * (1 - e2)  + altitude) * np.sin(latitude)
    return (x, y, z)

# Ground station position (ECEF, meters)
gs_position = np.array(geodetic_to_ecef(latitude, longitude, altitude))

# Simulate satellite pass (straight-line motion overhead)
def simulate_pass(duration_sec=600, sample_rate=1.0):
    t = np.arange(0, duration_sec, 1/sample_rate)
    satellite_velocity = np.array([0, 7600, 0])  # Rough orbital velocity in m/s
    sat_initial_pos = np.array([0, -3000e3, 500e3])
    sat_positions = sat_initial_pos[np.newaxis, :] + t[:, np.newaxis] * satellite_velocity
    return t, sat_positions

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
t, sat_positions = simulate_pass()
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
tle_template = f"""1 99999U 23001A   {now.strftime('%y%j')}.00000000  .00000000  00000-0  00000-0 0  9990\n2 99999 {98.0000:8.4f} {0.0000:8.4f} 0000001  0.0000  0.0000 {15.00000000:11.8f}    00"""
print("\nGenerated TLE (stub):\n", tle_template)
