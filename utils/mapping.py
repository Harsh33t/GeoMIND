import base64
import folium
from folium.raster_layers import ImageOverlay
from utils.spectral import export_overlay_to_png

def get_base64_png_url(png_bytes):
    """
    Converts raw PNG bytes into a Base64 Data URL, allowing Folium to embed the image
    directly inside the leaflet HTML without requiring an external file server.
    """
    encoded = base64.b64encode(png_bytes).decode('ascii')
    return f"data:image/png;base64,{encoded}"

def get_folium_bounds(metadata):
    """
    Converts raster bounds (left, bottom, right, top) to Folium's expected format:
    [[lat_min, lon_min], [lat_max, lon_max]]
    """
    bounds = metadata['bounds']
    return [[bounds.bottom, bounds.left], [bounds.top, bounds.right]]

def create_prospector_map(lat_c, lon_c, bands, indices, metadata, active_layer_type='True Color', opacity=0.7, training_points=None, rf_probability_map=None):
    """
    Creates an interactive Folium Map centered on the geological exploration block.
    Integrates custom composite, index index, or ML classification overlays.
    """
    # 1. Initialize Map centered on target area
    m = folium.Map(
        location=[lat_c, lon_c],
        zoom_start=13,
        tiles=None # We will add custom basemaps below
    )

    # 2. Add Premium Basemaps
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Imagery",
        name="Satellite Imagery (Esri)",
        overlay=False,
        control=True
    ).add_to(m)

    folium.TileLayer(
        tiles="https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png",
        attr="OpenTopoMap",
        name="Topographic Map (OpenTopo)",
        overlay=False,
        control=True
    ).add_to(m)

    folium.TileLayer(
        tiles="cartodbpositron",
        name="Light Base (CartoDB)",
        overlay=False,
        control=True
    ).add_to(m)

    # Get WGS84 bounding box for the raster overlay
    overlay_bounds = get_folium_bounds(metadata)

    # 3. Add Custom Raster Overlays depending on selected mode
    # For active layers, we convert our numpy array to an embedded PNG
    
    if active_layer_type in ['True Color', 'False Color (Vegetation)', 'SWIR Alteration']:
        # These are RGB composites (3D uint8 array already contrast-stretched)
        from utils.spectral import generate_rgb_composite
        rgb_data = generate_rgb_composite(bands, composite_type=active_layer_type)
        png_bytes = export_overlay_to_png(rgb_data, is_rgb=True)
        
        ImageOverlay(
            image=get_base64_png_url(png_bytes),
            bounds=overlay_bounds,
            opacity=opacity,
            name=f"Spectral Composite: {active_layer_type}",
            show=True
        ).add_to(m)
        
    elif active_layer_type == 'Iron Oxide (B04/B02)':
        png_bytes = export_overlay_to_png(indices['iron_oxide'], cmap_name='YlOrRd')
        ImageOverlay(
            image=get_base64_png_url(png_bytes),
            bounds=overlay_bounds,
            opacity=opacity,
            name="Iron Oxide Index Heatmap",
            show=True
        ).add_to(m)
        
    elif active_layer_type == 'Clay Minerals (B11/B12)':
        png_bytes = export_overlay_to_png(indices['clay_minerals'], cmap_name='Purples')
        ImageOverlay(
            image=get_base64_png_url(png_bytes),
            bounds=overlay_bounds,
            opacity=opacity,
            name="Clay Minerals Index Heatmap",
            show=True
        ).add_to(m)
        
    elif active_layer_type == 'Ferrous Iron (B12/B08)':
        png_bytes = export_overlay_to_png(indices['ferrous_iron'], cmap_name='Oranges')
        ImageOverlay(
            image=get_base64_png_url(png_bytes),
            bounds=overlay_bounds,
            opacity=opacity,
            name="Ferrous Iron Index Heatmap",
            show=True
        ).add_to(m)

    elif active_layer_type == 'ML Mineral Probability' and rf_probability_map is not None:
        # ML Heatmap
        png_bytes = export_overlay_to_png(rf_probability_map, cmap_name='RdYlBu_r')
        ImageOverlay(
            image=get_base64_png_url(png_bytes),
            bounds=overlay_bounds,
            opacity=opacity,
            name="Random Forest Prospect Probability",
            show=True
        ).add_to(m)

    # 4. Add Training Points Markers
    # Class colors:
    # 0: Vegetation -> Green
    # 1: Water -> Blue
    # 2: Barren -> Gray/Light Orange
    # 3: Iron Oxide -> Red
    # 4: Clay -> Purple
    # 5: Ferrous -> Brown
    class_colors = {
        0: '#2ecc71', # Green
        1: '#3498db', # Blue
        2: '#95a5a6', # Gray
        3: '#e74c3c', # Red
        4: '#9b59b6', # Purple
        5: '#d35400'  # Brown
    }

    if training_points:
        for lat, lon, label, class_name in training_points:
            color = class_colors.get(label, '#7f8c8d')
            folium.CircleMarker(
                location=[lat, lon],
                radius=6,
                color='#ffffff',
                fill=True,
                fill_color=color,
                fill_opacity=0.9,
                weight=1,
                popup=folium.Popup(f"<b>{class_name}</b><br>Lat: {lat:.5f}<br>Lon: {lon:.5f}", max_width=200)
            ).add_to(m)

    # Add standard Layer Controls
    folium.LayerControl().add_to(m)

    return m
