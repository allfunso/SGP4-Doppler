import doppler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from datetime import datetime, timedelta, timezone

# Constants
f0 = 437e6     # Base frequency (Hz), e.g. UHF

# Station location
gs_latitude = 26
gs_longitude = -100
gs_altitude = 6378137.0 + 2100

# Ground station position (ECEF, meters)
gs_position_ecef = doppler.geodetic_to_ecef(gs_latitude, gs_longitude, gs_altitude)

# Get positions
df = pd.read_csv('ground_track.csv')

communication_df = df.loc[df['comm_window'] == 1]
communication_df = communication_df.copy()
communication_df['timestamp'] = pd.to_datetime(communication_df['timestamp'])
communication_df['altitude'] = communication_df['altitude_km'] * 1000

# Verify the data belongs to the same pass
pass_df = pd.DataFrame(columns=communication_df.columns)
idx = 0
while idx < len(communication_df.index):
    if (communication_df.iloc[idx + 1]['timestamp'] - communication_df.iloc[idx]['timestamp']) > timedelta(minutes=60):
        break
    pass_df.loc[idx] = communication_df.iloc[idx]
    idx += 1

# Get satellite positions and time vector from the dataframe
sat_positions = doppler.df_to_ecef_positions(pass_df)
t = doppler.timestamps_to_seconds(pass_df)

# Main simulation
observed_freqs = doppler.compute_doppler_shift(sat_positions, gs_position_ecef, f0)

# Add noise
observed_freqs += np.random.normal(0, 1, size=observed_freqs.shape)  # Hz-level noise

# Fit back the velocity
estimated_velocity = doppler.fit_orbit(t, observed_freqs, f0, sat_positions, gs_position_ecef)
print(f"Estimated Velocity (m/s): {estimated_velocity}\n")

# Plot
plt.plot(t, observed_freqs, label='Observed Frequency')
plt.hlines(f0, t[0], t[-1], colors='red', linestyles='dashed')
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.title('Simulated Doppler Shift')
plt.grid()
plt.legend()
plt.show()

# Stub for generating a TLE
orbit_altitude_km = 650 # To calculate
eccentricity      = 0 # Should stay the same
inclination_deg   = 97.5 # Should stay the same
raan_deg          = 190 # To calculate
argp_deg          = 45 # To calculate
mean_anomaly_deg  = 0 # To calculate

earth_radius_m = 6371 * 1000
semi_major_axis = (earth_radius_m + orbit_altitude_km * 1000)
GM = 3.986004418e14

orbital_period = np.sqrt(4 * np.pi**2 * semi_major_axis**3 / GM)
orbits_per_day = (24 * 3600) / orbital_period

epoch = (datetime.now(timezone.utc).strftime("%y%j.%f") + "00")[:14]
tle_1 = f"1 99999U 25001A   {epoch}  .00000000  00000-0  00000-0 0  9990"
tle_2 = f"2 99999 {inclination_deg:8.4f} {raan_deg:8.4f} {int(eccentricity * 1e7):07d} {argp_deg:8.4f} {mean_anomaly_deg:8.4f} {orbits_per_day}"
print(f"Generated TLE set (stub): \n\t{tle_1}\n\t{tle_2}")
