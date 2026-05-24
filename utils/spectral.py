import numpy as np
import rasterio
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image

def load_sentinel2_bands(filepath):
    """
    Loads Sentinel-2 bands from a multi-band GeoTIFF file.
    Assumes bands are stored in the order: B02, B03, B04, B08, B11, B12.
    Scales the DN (Digital Numbers) by dividing by 10000.0 to obtain reflectance [0, 1].
    """
    with rasterio.open(filepath) as src:
        # Read bands (1-indexed in rasterio)
        b02 = src.read(1).astype(np.float32) / 10000.0
        b03 = src.read(2).astype(np.float32) / 10000.0
        b04 = src.read(3).astype(np.float32) / 10000.0
        b08 = src.read(4).astype(np.float32) / 10000.0
        b11 = src.read(5).astype(np.float32) / 10000.0
        b12 = src.read(6).astype(np.float32) / 10000.0
        
        # Get metadata
        meta = src.meta.copy()
        bounds = src.bounds
        transform = src.transform
        crs = src.crs

    bands = {
        'B02': b02,
        'B03': b03,
        'B04': b04,
        'B08': b08,
        'B11': b11,
        'B12': b12
    }
    
    metadata = {
        'bounds': bounds,
        'transform': transform,
        'crs': crs,
        'width': meta['width'],
        'height': meta['height']
    }
    
    return bands, metadata

def compute_spectral_indices(bands):
    """
    Computes geological spectral indices from reflectance bands:
      1. Iron Oxide Index = B04 / B02
      2. Clay Minerals Index = B11 / B12
      3. Ferrous Iron Index = B12 / B08
    """
    eps = 1e-6
    
    # Red / Blue ratio
    iron_oxide = bands['B04'] / (bands['B02'] + eps)
    
    # SWIR1 / SWIR2 ratio
    clay_minerals = bands['B11'] / (bands['B12'] + eps)
    
    # SWIR2 / NIR ratio
    ferrous_iron = bands['B12'] / (bands['B08'] + eps)
    
    # Clip extreme outlier ratios (typically caused by clouds or shadows) to keep visualizations sleek
    iron_oxide = np.clip(iron_oxide, 0.0, 5.0)
    clay_minerals = np.clip(clay_minerals, 0.0, 5.0)
    ferrous_iron = np.clip(ferrous_iron, 0.0, 5.0)
    
    return {
        'iron_oxide': iron_oxide,
        'clay_minerals': clay_minerals,
        'ferrous_iron': ferrous_iron
    }

def stretch_band(band, min_pct=2, max_pct=98):
    """
    Applies a linear contrast stretch to a band using the specified percentiles.
    Normalizes the result to [0, 1].
    """
    # Exclude NoData/Zero values if present
    valid_pixels = band[band > 0.0001]
    if len(valid_pixels) == 0:
        valid_pixels = band
        
    vmin = np.percentile(valid_pixels, min_pct)
    vmax = np.percentile(valid_pixels, max_pct)
    
    if vmax <= vmin:
        return np.zeros_like(band)
        
    stretched = np.clip((band - vmin) / (vmax - vmin), 0.0, 1.0)
    return stretched

def generate_rgb_composite(bands, composite_type='True Color'):
    """
    Generates a 3-channel (RGB) stretched and contrast-enhanced image.
    Types:
      - 'True Color': R=B04, G=B03, B=B02
      - 'False Color (Vegetation)': R=B08, G=B04, B=B03 (Highlights foliage in brilliant red)
      - 'SWIR Alteration': R=B12, G=B11, B=B08 (Geology/Alteration highlighting)
    """
    if composite_type == 'True Color':
        r = stretch_band(bands['B04'])
        g = stretch_band(bands['B03'])
        b = stretch_band(bands['B02'])
    elif composite_type == 'False Color (Vegetation)':
        r = stretch_band(bands['B08'])
        g = stretch_band(bands['B04'])
        b = stretch_band(bands['B03'])
    elif composite_type == 'SWIR Alteration':
        r = stretch_band(bands['B12'])
        g = stretch_band(bands['B11'])
        b = stretch_band(bands['B08'])
    else:
        # Default fallback
        r = stretch_band(bands['B04'])
        g = stretch_band(bands['B03'])
        b = stretch_band(bands['B02'])
        
    # Stack to shape (H, W, 3)
    rgb = np.dstack((r, g, b))
    # Convert to 8-bit integer [0, 255]
    rgb_uint8 = (rgb * 255.0).astype(np.uint8)
    return rgb_uint8

def export_overlay_to_png(data, cmap_name=None, is_rgb=False):
    """
    Converts a 2D single-band heatmap or a 3D RGB array into PNG bytes.
    Useful for overlaying rasters directly on Folium Leaflet maps.
    """
    if is_rgb:
        img = Image.fromarray(data)
    else:
        # Apply colormap to 2D band
        # First normalize single band to [0, 1]
        vmin, vmax = data.min(), data.max()
        if vmax > vmin:
            normalized = (data - vmin) / (vmax - vmin)
        else:
            normalized = np.zeros_like(data)
            
        # Get matplotlib colormap
        cmap = plt.get_cmap(cmap_name or 'viridis')
        rgba = cmap(normalized) # shape (H, W, 4)
        rgba_uint8 = (rgba * 255.0).astype(np.uint8)
        img = Image.fromarray(rgba_uint8)
        
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return buffered.getvalue()
