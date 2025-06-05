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
epoch = datetime.now(timezone.utc).strftime("%y%j.%f")[:14]
tle_1 = f"1 99999U 25001A   {epoch}  .00000000  .00000000  00000-0  00000-0 0  9990"
tle_2 = f"2 99999 {98.0000:8.4f} {0.0000:8.4f} 0000001  0.0000  0.0000 {15.00000000:11.8f}    00"
print(f"Generated TLE set (stub): \n\t{tle_1}\n\t{tle_2}")
