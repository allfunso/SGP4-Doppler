from propagator import Orbit_Propagation, Deliver_Data
from doppler import geodetic_to_ecef, df_to_ecef_positions
import pandas as pd
from datetime import timedelta

Orbit_Propagation(
    orbit_altitude_km = 650,
    eccentricity      = 0,
    inclination_deg   = 97.5,
    raan_deg          = 190,
    argp_deg          = 45,
    mean_anomaly_deg  = 0,
    step_seconds      = 10,
    simulated_days    = 1,
    initial_latitude_deg = -30,
    initial_longitude_deg = -75
)

'''
if latitude == None or longitude == None:

  import pandas as pd

  df = pd.read_csv('ground_track.csv')

  time_index = df[df['timestamp'] == current_time].index[0]

  latitude    = df['latitude_deg'][time_index]
  longitude   = df['longitude_deg'][time_index]

  Orbit_Propagation(
    latitude,
    longitude,
    orbit_altitude_km = 650,
    eccentricity      = 0,
    inclination_deg   = 97.5,
    raan_deg          = 190,
    argp_deg          = 45,
    mean_anomaly_deg  = 0,
    step_seconds      = 1,
    simulated_days    = 1
  )
'''