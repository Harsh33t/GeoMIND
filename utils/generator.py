import os
import numpy as np
import scipy.ndimage as ndimage
import rasterio
from rasterio.transform import from_origin

def generate_noise_layer(shape, scale_sigmas, weights):
    """
    Generates complex procedural noise by blending multiple octaves of Gaussian-filtered random noise.
    """
    noise = np.zeros(shape)
    for sigma, weight in zip(scale_sigmas, weights):
        raw = np.random.normal(0, 1, shape)
        blurred = ndimage.gaussian_filter(raw, sigma=sigma)
        # Normalize to [-1, 1] range
        if blurred.std() > 0:
            blurred = blurred / blurred.std()
        noise += blurred * weight
    return noise

def generate_district_raster(district, output_path):
    """
    Generates a highly realistic 6-band Sentinel-2 synthetic GeoTIFF for mineral prospecting.
    Bands:
      Band 1: B02 (Blue, 490nm)
      Band 2: B03 (Green, 560nm)
      Band 3: B04 (Red, 665nm)
      Band 4: B08 (NIR, 842nm)
      Band 5: B11 (SWIR 1, 1610nm)
      Band 6: B12 (SWIR 2, 2190nm)
    
    Reflectance values are scaled between 0 and 10000 (Sentinel-2 L2A format).
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 500x500 pixels (equivalent to 5km x 5km at 10m resolution)
    height, width = 500, 500
    shape = (height, width)
    
    # Define District Geolocation Settings (Centers)
    if district == "Pilbara District, Australia":
        lat_c, lon_c = -21.15, 119.15
        description = "Banded Iron Formation (BIF) ridge with extensive iron oxides and basaltic structures."
    elif district == "Escondida Mine, Chile":
        lat_c, lon_c = -24.27, -69.07
        description = "Large open-pit porphyry copper deposit showing distinct circular clay alteration halo."
    elif district == "Carlin Trend, Nevada":
        lat_c, lon_c = 40.95, -116.33
        description = "Fault-controlled Carlin-type gold trend with hydrothermal clay and iron alteration zones."
    else:
        # Default
        lat_c, lon_c = 0.0, 0.0
        description = "Generic exploration site."

    # Define coordinate transform (10m resolution in WGS84 degree equivalent ~ 0.00009 degrees)
    pixel_size = 0.00009
    lon_origin = lon_c - (width * pixel_size) / 2
    lat_origin = lat_c + (height * pixel_size) / 2
    transform = from_origin(lon_origin, lat_origin, pixel_size, pixel_size)
    
    # Generate background and geological structures
    background_noise = generate_noise_layer(shape, [50, 20, 5], [0.6, 0.3, 0.1])
    # Normalize background noise to [0, 1]
    background_noise = (background_noise - background_noise.min()) / (background_noise.max() - background_noise.min())

    # Initialize 6 spectral bands with typical dry rock/soil reflectance
    # Base reflectances: B2=1400, B3=1900, B4=2400, B8=2800, B11=3800, B12=3300
    bands = {
        'B02': 1400 + background_noise * 300,
        'B03': 1900 + background_noise * 400,
        'B04': 2400 + background_noise * 500,
        'B08': 2800 + background_noise * 400,
        'B11': 3800 + background_noise * 600,
        'B12': 3300 + background_noise * 500,
    }

    if district == "Pilbara District, Australia":
        # 1. Banded Iron Ridge (high Iron Oxide B04/B02, running NW to SE)
        x_indices = np.arange(width)
        y_indices = np.arange(height)
        X, Y = np.meshgrid(x_indices, y_indices)
        
        # Diagonal ridge line: y = 0.8 * x + 50
        ridge_dist = np.abs(Y - (0.8 * X + 50))
        # Create a smooth ridge profile
        ridge_mask = np.exp(-(ridge_dist ** 2) / (2 * (40 ** 2))) # 40 pixels wide
        # Add high-frequency geological variations along the ridge
        ridge_noise = generate_noise_layer(shape, [15, 5], [0.8, 0.2])
        ridge_noise = (ridge_noise - ridge_noise.min()) / (ridge_noise.max() - ridge_noise.min())
        ridge_intensity = ridge_mask * (0.6 + 0.4 * ridge_noise)
        
        # Modify bands for Iron Ridge: high Iron Oxide (high B4, low B2)
        bands['B02'] = bands['B02'] * (1.0 - ridge_intensity * 0.4) # absorbs blue
        bands['B04'] = bands['B04'] + ridge_intensity * 1800        # reflects red
        bands['B11'] = bands['B11'] + ridge_intensity * 800
        bands['B12'] = bands['B12'] + ridge_intensity * 500
        
        # 2. Basaltic Dykes (Ferrous Iron rich: high B12/B8)
        dyke_noise = generate_noise_layer(shape, [30, 8], [0.7, 0.3])
        # Isolate some narrow winding veins
        dyke_mask = (dyke_noise > 1.2).astype(float)
        dyke_mask = ndimage.gaussian_filter(dyke_mask, sigma=3)
        
        bands['B12'] = bands['B12'] + dyke_mask * 1500 # High SWIR2
        bands['B08'] = bands['B08'] * (1.0 - dyke_mask * 0.25) # Absorbs NIR

        # 3. Winding Creek with vegetation (high NIR B08, high Green B03)
        creek_path = 150 + 100 * np.sin(X / 80.0) + generate_noise_layer(shape, [25], [1.0]) * 30
        creek_dist = np.abs(Y - creek_path)
        creek_mask = np.exp(-(creek_dist ** 2) / (2 * (8 ** 2))) # Winding narrow creek
        vegetation_mask = np.exp(-(creek_dist ** 2) / (2 * (25 ** 2))) * (1.0 - creek_mask) # Vegetation alongside
        
        # Add vegetation spectral signature
        bands['B02'] = bands['B02'] * (1.0 - vegetation_mask * 0.5)
        bands['B03'] = bands['B03'] + vegetation_mask * 800
        bands['B04'] = bands['B04'] * (1.0 - vegetation_mask * 0.6)
        bands['B08'] = bands['B08'] + vegetation_mask * 3200
        bands['B11'] = bands['B11'] * (1.0 - vegetation_mask * 0.4)
        bands['B12'] = bands['B12'] * (1.0 - vegetation_mask * 0.5)
        
        # Add water spectral signature in the creek bed
        bands['B02'] = bands['B02'] * (1.0 - creek_mask) + creek_mask * 1000
        bands['B03'] = bands['B03'] * (1.0 - creek_mask) + creek_mask * 900
        bands['B04'] = bands['B04'] * (1.0 - creek_mask) + creek_mask * 600
        bands['B08'] = bands['B08'] * (1.0 - creek_mask) + creek_mask * 200
        bands['B11'] = bands['B11'] * (1.0 - creek_mask) + creek_mask * 50
        bands['B12'] = bands['B12'] * (1.0 - creek_mask) + creek_mask * 20

    elif district == "Escondida Mine, Chile":
        # Atacama Desert - extremely dry. No vegetation, no water.
        x_indices = np.arange(width)
        y_indices = np.arange(height)
        X, Y = np.meshgrid(x_indices, y_indices)
        
        # Center of the mine open-pit
        cx, cy = 250, 250
        dist_from_center = np.sqrt((X - cx)**2 + (Y - cy)**2)
        
        # 1. Circular Pit (concentric steps)
        pit_mask = (dist_from_center < 110).astype(float)
        # Create steps
        steps = np.floor(dist_from_center / 15) % 2
        pit_profile = pit_mask * (0.8 + steps * 0.15)
        
        # Pit spectral alteration (exposed deep mineralization: iron oxides and clay alteration)
        bands['B02'] = bands['B02'] * (1.0 - pit_profile * 0.3)
        bands['B04'] = bands['B04'] + pit_profile * 2200
        bands['B08'] = bands['B08'] + pit_profile * 800
        bands['B11'] = bands['B11'] + pit_profile * 2500
        bands['B12'] = bands['B12'] + pit_profile * 1200
        
        # 2. Hydrothermal Clay Alteration Halo (high B11/B12)
        # Intense clay zone surrounding the pit (annulus between 110 and 220 pixels)
        clay_halo = np.exp(-((dist_from_center - 165) ** 2) / (2 * (35 ** 2)))
        halo_noise = generate_noise_layer(shape, [20, 8], [0.8, 0.2])
        halo_noise = (halo_noise - halo_noise.min()) / (halo_noise.max() - halo_noise.min())
        clay_intensity = clay_halo * (0.5 + 0.5 * halo_noise)
        
        bands['B11'] = bands['B11'] + clay_intensity * 3800 # Massive SWIR1 reflection
        bands['B12'] = bands['B12'] * (1.0 - clay_intensity * 0.35) # SWIR2 absorption
        bands['B02'] = bands['B02'] + clay_intensity * 400
        bands['B03'] = bands['B03'] + clay_intensity * 600

        # 3. Flanking Iron Oxide Deposits (High B4/B2)
        iron_noise = generate_noise_layer(shape, [40, 10], [0.7, 0.3])
        # Two large patches to the east and southwest
        patch1 = np.exp(-((X - 400)**2 + (Y - 150)**2) / (2 * (50 ** 2)))
        patch2 = np.exp(-((X - 100)**2 + (Y - 380)**2) / (2 * (60 ** 2)))
        iron_mask = (patch1 + patch2) * (0.6 + 0.4 * iron_noise)
        
        bands['B02'] = bands['B02'] * (1.0 - iron_mask * 0.45)
        bands['B04'] = bands['B04'] + iron_mask * 2500
        bands['B08'] = bands['B08'] + iron_mask * 800

    elif district == "Carlin Trend, Nevada":
        x_indices = np.arange(width)
        y_indices = np.arange(height)
        X, Y = np.meshgrid(x_indices, y_indices)
        
        # 1. Main Fault Zone (runs N-S with a slight bend, high clay B11/B12 & moderate iron B4/B2)
        fault_path = 200 + 40 * np.sin(Y / 60.0) + generate_noise_layer(shape, [20], [1.0]) * 15
        fault_dist = np.abs(X - fault_path)
        fault_mask = np.exp(-(fault_dist ** 2) / (2 * (25 ** 2))) # alteration zone width
        fault_noise = generate_noise_layer(shape, [10, 3], [0.8, 0.2])
        fault_noise = (fault_noise - fault_noise.min()) / (fault_noise.max() - fault_noise.min())
        fault_intensity = fault_mask * (0.4 + 0.6 * fault_noise)
        
        bands['B11'] = bands['B11'] + fault_intensity * 2500
        bands['B12'] = bands['B12'] * (1.0 - fault_intensity * 0.25)
        bands['B04'] = bands['B04'] + fault_intensity * 1200
        bands['B02'] = bands['B02'] * (1.0 - fault_intensity * 0.2)
        
        # 2. Sedimentary rock bands (stratigraphy running east-west)
        strat_wave = Y + 10 * np.sin(X / 40.0)
        strat_band = (np.sin(strat_wave / 20.0) > 0.4).astype(float)
        strat_band = ndimage.gaussian_filter(strat_band, sigma=2)
        
        bands['B02'] = bands['B02'] + strat_band * 150
        bands['B03'] = bands['B03'] + strat_band * 200
        bands['B08'] = bands['B08'] + strat_band * 300
        bands['B11'] = bands['B11'] + strat_band * 400
        
        # 3. Scattered scrub vegetation (sagebrush) - high-frequency speckling
        veg_noise = np.random.normal(0, 1, shape)
        veg_mask = ndimage.gaussian_filter(veg_noise, sigma=1.5) > 1.4
        # Apply filter to make it look clumped
        veg_mask = ndimage.binary_dilation(veg_mask, iterations=1).astype(float)
        veg_mask = ndimage.gaussian_filter(veg_mask, sigma=1.0)
        
        # Sagebrush has moderate NIR reflection
        bands['B02'] = bands['B02'] * (1.0 - veg_mask * 0.2)
        bands['B03'] = bands['B03'] + veg_mask * 300
        bands['B04'] = bands['B04'] * (1.0 - veg_mask * 0.25)
        bands['B08'] = bands['B08'] + veg_mask * 1500
        bands['B11'] = bands['B11'] * (1.0 - veg_mask * 0.15)
        bands['B12'] = bands['B12'] * (1.0 - veg_mask * 0.2)

    # 4. Final Processing: Add spatial sensor noise, scale & clip values to [1, 10000]
    for band_name in bands:
        sensor_noise = np.random.normal(0, 40, shape)
        bands[band_name] = np.clip(bands[band_name] + sensor_noise, 1.0, 10000.0).astype(np.float32)

    # Write as a multi-band GeoTIFF using rasterio
    meta = {
        'driver': 'GTiff',
        'dtype': 'float32',
        'nodata': 0.0,
        'width': width,
        'height': height,
        'count': 6,
        'crs': 'EPSG:4326',
        'transform': transform
    }
    
    with rasterio.open(output_path, 'w', **meta) as dst:
        # Write bands in order: B02, B03, B04, B08, B11, B12
        dst.write(bands['B02'], 1)
        dst.write(bands['B03'], 2)
        dst.write(bands['B04'], 3)
        dst.write(bands['B08'], 4)
        dst.write(bands['B11'], 5)
        dst.write(bands['B12'], 6)
        
        # Tag bands with descriptions
        dst.set_band_description(1, 'B02_Blue')
        dst.set_band_description(2, 'B03_Green')
        dst.set_band_description(3, 'B04_Red')
        dst.set_band_description(4, 'B08_NIR')
        dst.set_band_description(5, 'B11_SWIR1')
        dst.set_band_description(6, 'B12_SWIR2')

    return {
        'district': district,
        'filepath': output_path,
        'description': description,
        'center_lat': lat_c,
        'center_lon': lon_c,
        'width': width,
        'height': height,
        'crs': 'EPSG:4326',
        'transform': [transform.a, transform.b, transform.c, transform.d, transform.e, transform.f]
    }

def get_preseeded_points(district):
    """
    Returns high-confidence geological training ground-truth coordinates (latitude, longitude)
    for Random Forest classification in each district.
    Each point has format: (lat, lon, class_label, class_name)
    Class mapping:
       0: Vegetation
       1: Water
       2: Barren Soil / Bedrock
       3: Iron Oxide Rich
       4: Clay Altered Zone
       5: Ferrous Iron Rich
    """
    if district == "Pilbara District, Australia":
        return [
            # Water (riverbed waterhole)
            (-21.154, 119.145, 1, "Water"),
            (-21.155, 119.148, 1, "Water"),
            # Vegetation (dense green river valley canopy)
            (-21.151, 119.135, 0, "Vegetation"),
            (-21.149, 119.140, 0, "Vegetation"),
            (-21.158, 119.158, 0, "Vegetation"),
            # Barren soil / plain
            (-21.162, 119.135, 2, "Barren Soil"),
            (-21.138, 119.162, 2, "Barren Soil"),
            # Iron Oxide Ridge (Red BIF outcrop)
            (-21.148, 119.148, 3, "Iron Oxide Zone"),
            (-21.144, 119.144, 3, "Iron Oxide Zone"),
            (-21.152, 119.152, 3, "Iron Oxide Zone"),
            # Basaltic dykes (Ferrous-rich volcanic outcrops)
            (-21.136, 119.138, 5, "Ferrous Silicates"),
            (-21.160, 119.142, 5, "Ferrous Silicates"),
        ]
    elif district == "Escondida Mine, Chile":
        # Extreme desert: No water, no vegetation.
        return [
            # Open Pit floor (highly altered mine cuts)
            (-24.270, -69.070, 2, "Pit Bedrock"),
            (-24.271, -69.071, 2, "Pit Bedrock"),
            # Background desert soils
            (-24.250, -69.090, 2, "Desert Soil"),
            (-24.290, -69.050, 2, "Desert Soil"),
            # Hydrothermal Clay Alteration Halo (Alunite/Kaolinite ring)
            (-24.270, -69.052, 4, "Clay Altered Halo"),
            (-24.270, -69.088, 4, "Clay Altered Halo"),
            (-24.254, -69.070, 4, "Clay Altered Halo"),
            (-24.286, -69.070, 4, "Clay Altered Halo"),
            # Peripheral Iron Oxide veins (hematite outcrops)
            (-24.258, -69.040, 3, "Iron Oxide Patch"),
            (-24.288, -69.095, 3, "Iron Oxide Patch"),
        ]
    elif district == "Carlin Trend, Nevada":
        return [
            # Scattered scrub vegetation (sagebrush)
            (40.940, -116.315, 0, "Vegetation"),
            (40.962, -116.348, 0, "Vegetation"),
            # Sedimentary bedrock (Limestone layer)
            (40.938, -116.342, 2, "Sedimentary Bedrock"),
            (40.958, -116.312, 2, "Sedimentary Bedrock"),
            # Fault-controlled altered zone (Clay rich)
            (40.952, -116.331, 4, "Hydrothermal Clay"),
            (40.946, -116.332, 4, "Hydrothermal Clay"),
            (40.956, -116.330, 4, "Hydrothermal Clay"),
            # Oxide weathering (Iron Oxides)
            (40.948, -116.326, 3, "Iron Oxide weathering"),
            (40.954, -116.334, 3, "Iron Oxide weathering"),
        ]
    return []
