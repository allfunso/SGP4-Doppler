import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skyfield.api import load, EarthSatellite, Topos
from datetime import datetime, timedelta, timezone
import cartopy

def Orbit_Propagation(orbit_altitude_km, eccentricity, inclination_deg, raan_deg, argp_deg, mean_anomaly_deg,
                      step_seconds, simulated_days, initial_latitude_deg, initial_longitude_deg):

    # Orbital Parameters ================================================================================================================ #
    earth_radius_m = 6371 * 1000
    semi_major_axis = (earth_radius_m + orbit_altitude_km * 1000)
    GM = 3.986004418e14

    orbital_period = np.sqrt(4 * np.pi**2 * semi_major_axis**3 / GM)
    orbits_per_day = (24 * 3600) / orbital_period

    epoch = datetime.now(timezone.utc).strftime("%y%j.%f")[:14]
    tle_line1 = f"1 99999U 20001A   {epoch}  .00000000  00000-0  00000 0  9999"
    tle_line2 = f"2 99999 {inclination_deg:8.4f} {raan_deg:8.4f} {int(eccentricity * 1e7):07d} {argp_deg:8.4f} {mean_anomaly_deg:8.4f} {orbits_per_day}"

    ts = load.timescale()
    satellite = EarthSatellite(tle_line1, tle_line2, "DemoSat", ts)
    # =================================================================================================================================== #


    # Find Starting Point =============================================================================================================== #
    earth_radius_m = 6371 * 1000
    objetivo = Topos(latitude_degrees=initial_latitude_deg, longitude_degrees=initial_longitude_deg)
    t0 = ts.now()
    t1 = ts.utc((datetime.utcnow() + timedelta(hours=24)).replace(tzinfo=timezone.utc))
    times, events = satellite.find_events(objetivo, t0, t1, altitude_degrees=10)

    max_elev_index = np.where(events == 1)[0][0] if 1 in events else None

    if max_elev_index is None: print("⚠️ No hay pase sobre el punto objetivo en las próximas 24h."); return

    pass_time = times[max_elev_index]
    start_time = pass_time.utc_datetime().replace(tzinfo=timezone.utc)

    total_minutes = simulated_days * 24 * 60
    datetimes = [(start_time + timedelta(seconds=dt_seconds)) for dt_seconds in range(0, total_minutes * 60, step_seconds)]
    future_times = ts.utc(datetimes)


    # Propagate Orbit =================================================================================================================== #
    timestamps, latitudes, longitudes, altitudes, speeds, photo_flags, comm_flags = [], [], [], [], [], [], []

    ground_station_lat = 25.67507
    ground_station_lon = -100.31847
    ground_station = Topos(latitude_degrees=ground_station_lat, longitude_degrees=ground_station_lon)

    for t in future_times:
        geocentric = satellite.at(t)
        subpoint = geocentric.subpoint()
        timestamps.append(t.utc_datetime())
        latitudes.append(subpoint.latitude.degrees)
        longitudes.append(subpoint.longitude.degrees)
        altitudes.append(subpoint.elevation.km)
        photo_flags.append(1 if -60 <= subpoint.latitude.degrees <= -40 else 0)
        speed = np.linalg.norm(geocentric.velocity.km_per_s)
        speeds.append(speed)

        # Comunicación con estación terrestre
        difference = satellite - ground_station
        alt, az, distance = difference.at(t).altaz()
        comm_flags.append(1 if alt.degrees >= 10 else 0)


    # Save Data ========================================================================================================================= #
    df = pd.DataFrame({
        'timestamp': timestamps,
        'latitude_deg': latitudes,
        'longitude_deg': longitudes,
        'altitude_km': altitudes,
        'speed_km_s': speeds,
        'photo': photo_flags,
        'comm_window': comm_flags
    })

    df.to_csv('ground_track.csv', index=False)

# ====================================================================================================================================== #

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

def Deliver_Data(current_time):

  import pandas as pd

  df = pd.read_csv(f'ground_track.csv')

  time_index = df[df['timestamp'] == current_time].index[0]

  timestamp   = df['timestamp'][time_index]
  latitude    = df['latitude_deg'][time_index]
  longitude   = df['longitude_deg'][time_index]
  altitude    = df['altitude_km'][time_index]
  speed       = df['speed_km_s'][time_index]
  photo       = df['photo'][time_index]
  comm_window = df['comm_window'][time_index]

  return timestamp, latitude, longitude, altitude, speed, photo, comm_window


if latitude or longitude == None:

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
