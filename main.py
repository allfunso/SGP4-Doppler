from propagator import Orbit_Propagation

# Station location
gs_latitude = 26
gs_longitude = -100
gs_altitude = 6378137.0 + 2100 # meters

# Latitudes of interest
latitude_range = (-60, -40)

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
    initial_longitude_deg = -75,
    gs_position = (gs_latitude, gs_longitude),
    latitude_range = latitude_range
)

