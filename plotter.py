import pandas as pd

def Plot2D(ground_lons, ground_lats, station_lon, station_lat, df, fotos_tomadas):

    import numpy as np
    import matplotlib.pyplot as plt
    from cartopy import crs as ccrs

    # Mercator Projection ==========================================================================================================
    plt.figure(figsize=(10, 6))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.stock_img()
    ax.coastlines()
    ax.gridlines(draw_labels=True)
    ax.plot(ground_lons, ground_lats, 'r-', transform=ccrs.Geodetic(), label='Ground track', zorder=1) # Orbital path
    ax.scatter(station_lon, station_lat, color='blue', s=50, label='Ground station', zorder=10)

    # Marcar el punto inicial (MAX sobre la estación)
    lat_inicio = df.iloc[0]['latitude_deg']
    lon_inicio = df.iloc[0]['longitude_deg']
    ax.scatter(lon_inicio, lat_inicio, color='black', s=60, marker='X', transform=ccrs.PlateCarree(), label='Inicio de tracking', zorder=10)

    # Marcar puntos donde se toma foto con círculos verdes
    foto_lons = fotos_tomadas['longitude_deg'].values
    foto_lats = fotos_tomadas['latitude_deg'].values

    ax.scatter(foto_lons, foto_lats, color='limegreen', s=30, marker='o',transform=ccrs.PlateCarree(), label='Foto tomada', zorder=10)

    comm_lons = df[df['comm_window'] == 1]['longitude_deg'].values
    comm_lats = df[df['comm_window'] == 1]['latitude_deg'].values

    ax.scatter(comm_lons, comm_lats, color='purple', s=30, marker='^',
           transform=ccrs.PlateCarree(), label='Ventana de comunicación', zorder=10)

    plt.legend(loc='upper right')
    plt.title("Ground Track del Satélite")
    plt.show()

    plt.savefig('ground_track.png', dpi = 400)

    # South Pole Projection ==========================================================================================================
    plt.figure(figsize=(10, 10))

    ax = plt.axes(projection=ccrs.SouthPolarStereo())

    # Agregar detalles del mapa
    ax.coastlines()
    ax.gridlines(draw_labels=True)

    # Convertir listas a arrays
    ground_lats = np.array(ground_lats)
    ground_lons = np.array(ground_lons)

    # Corregir saltos de ±180° (ajuste para evitar líneas rectas cruzando el mapa)
    ground_lons_unwrapped = np.unwrap(np.radians(ground_lons))
    ground_lons_unwrapped = np.degrees(ground_lons_unwrapped)

    ax.plot(ground_lons_unwrapped, ground_lats, 'r-', transform=ccrs.PlateCarree(), label='Ground track', zorder=1)     # Dibujar ground track
    ax.set_extent([-180, 180, -90, -45], crs=ccrs.PlateCarree())                                              # Limitar el mapa (opcional)

    # Marcar puntos donde se toma foto con círculos verdes
    foto_lons = fotos_tomadas['longitude_deg'].values
    foto_lats = fotos_tomadas['latitude_deg'].values

    ax.scatter(foto_lons, foto_lats, color='limegreen', s=30, marker='o', transform=ccrs.PlateCarree(), label='Foto tomada', zorder=10) # Plot

    plt.legend()
    plt.title("Ground track desde perspectiva del Polo Sur")
    plt.show()

    plt.savefig('pole.png', dpi = 400)

# Station location
gs_latitude = 26
gs_longitude = -100
gs_altitude = 6378137.0 + 2100

df = pd.read_csv('ground_track.csv')
Plot2D(df['longitude_deg'], df['latitude_deg'], gs_longitude, gs_latitude, df, df[df['photo'] == 1])