import os
import streamlit as st
import numpy as np
import pandas as pd
import folium
from streamlit_folium import st_folium

from utils.generator import generate_district_raster, get_preseeded_points
from utils.spectral import load_sentinel2_bands, compute_spectral_indices, generate_rgb_composite
from utils.classifier import extract_features_at_points, train_mineral_classifier, predict_mineral_probabilities
from utils.llm import run_groq_geological_analysis, generate_geological_expert_report
from utils.mapping import create_prospector_map

# ---------------------------------------------------------
# Page Configuration & Rich CSS Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="GeoMind — AI Satellite Mineral Prospector",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom dark-theme glassmorphism CSS
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=Space+Grotesk:wght@300;500;700&display=swap" rel="stylesheet">

<style>
    /* Base configuration */
    .stApp {
        background-color: #0b0e14;
        color: #e2e8f0;
        font-family: 'Outfit', sans-serif;
    }
    
    /* Sleek scrollbars */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    ::-webkit-scrollbar-track {
        background: #0b0e14;
    }
    ::-webkit-scrollbar-thumb {
        background: #1e293b;
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #334155;
    }

    /* Titles & Headers */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 700;
        letter-spacing: -0.02em;
    }
    
    .main-title {
        background: linear-gradient(135deg, #00f2fe 0%, #4facfe 50%, #a18cd1 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
        text-shadow: 0 10px 30px rgba(79, 172, 254, 0.15);
    }
    
    .subtitle {
        color: #94a3b8;
        font-size: 1.15rem;
        margin-bottom: 2rem;
        font-weight: 300;
    }

    /* Glassmorphism containers */
    .glass-card {
        background: rgba(20, 26, 38, 0.65);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    }
    
    .glowing-border-red {
        border-left: 5px solid #e74c3c;
    }
    .glowing-border-purple {
        border-left: 5px solid #9b59b6;
    }
    .glowing-border-orange {
        border-left: 5px solid #e67e22;
    }

    /* Metric styles */
    .metric-value {
        font-size: 2.2rem;
        font-weight: 800;
        font-family: 'Space Grotesk', sans-serif;
        color: #00f2fe;
        text-shadow: 0 0 10px rgba(0, 242, 254, 0.2);
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 600;
    }

    /* Custom buttons */
    div.stButton > button {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        border: none;
        color: #0b0e14;
        font-weight: 600;
        padding: 0.5rem 1.5rem;
        border-radius: 8px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0, 242, 254, 0.2);
    }
    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0, 242, 254, 0.4);
        color: #0b0e14;
    }
    
    /* Sidebar customization */
    [data-testid="stSidebar"] {
        background-color: #07090d;
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    /* Styled alert/info blocks */
    .alert-banner {
        background: rgba(231, 76, 60, 0.15);
        border: 1px solid rgba(231, 76, 60, 0.3);
        padding: 1rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Define Cache paths
CACHE_DIR = os.path.join("d:\\GeoMIND", "data")
os.makedirs(CACHE_DIR, exist_ok=True)

# ---------------------------------------------------------
# Sidebar Configuration
# ---------------------------------------------------------
st.sidebar.markdown("<h2 style='text-align: center; color: #00f2fe; font-family: Space Grotesk;'>🛰️ Control Panel</h2>", unsafe_allow_html=True)
st.sidebar.markdown("---")

# 1. Groq API Configuration
st.sidebar.markdown("### 🔑 AI Geological Agent")
groq_key = st.sidebar.text_input(
    "Groq API Key",
    type="password",
    help="Insert your Groq API Key to generate real-time exploration reports. If omitted, the app defaults to a highly detailed, pre-seeded expert system.",
    value=os.getenv("GROQ_API_KEY", "")
)

st.sidebar.markdown("---")

# 2. Select Prospecting Location
st.sidebar.markdown("### 🗺️ Geological Targets")
target_option = st.sidebar.selectbox(
    "Prospecting District",
    [
        "Pilbara District, Australia",
        "Escondida Mine, Chile",
        "Carlin Trend, Nevada",
        "📂 Upload Custom Sentinel-2 GeoTIFF"
    ]
)

# 3. Layer Controls
st.sidebar.markdown("### 🎨 Display Layers")
layer_options = [
    "True Color",
    "False Color (Vegetation)",
    "SWIR Alteration",
    "Iron Oxide (B04/B02)",
    "Clay Minerals (B11/B12)",
    "Ferrous Iron (B12/B08)"
]

# Check if model has been trained to unlock classification layer
if 'rf_trained' in st.session_state and st.session_state.rf_trained:
    layer_options.append("ML Mineral Probability")

selected_layer = st.sidebar.selectbox("Active Map Layer", layer_options)
overlay_opacity = st.sidebar.slider("Layer Opacity", 0.0, 1.0, 0.75, 0.05)

st.sidebar.markdown("---")
st.sidebar.info(
    "💡 **Quick Tip:** Click anywhere on the map to inspect coordinates. "
    "Use the Machine Learning section below to label sites and run mineral mapping."
)

# ---------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------
if 'current_target' not in st.session_state or st.session_state.current_target != target_option:
    st.session_state.current_target = target_option
    st.session_state.rf_trained = False
    st.session_state.rf_model = None
    st.session_state.probability_maps = None
    st.session_state.classification_map = None
    st.session_state.rf_metrics = None
    st.session_state.ai_report = None
    st.session_state.last_prediction_class = None
    
    # Initialize training points
    if target_option != "📂 Upload Custom Sentinel-2 GeoTIFF":
        st.session_state.training_points = get_preseeded_points(target_option)
    else:
        st.session_state.training_points = []

# ---------------------------------------------------------
# Main App Header
# ---------------------------------------------------------
st.markdown("<h1 class='main-title'>GeoMind</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>AI Satellite Mineral Prospector — Multispectral Alteration & Machine Learning Platform</p>", unsafe_allow_html=True)

# ---------------------------------------------------------
# Data Loading / Processing Section
# ---------------------------------------------------------
bands, metadata = None, None
lat_c, lon_c = 0.0, 0.0
geotiff_path = ""

if target_option != "📂 Upload Custom Sentinel-2 GeoTIFF":
    # Load Preset Districts
    filename = target_option.replace(", ", "_").replace(" ", "_").lower() + ".tif"
    geotiff_path = os.path.join(CACHE_DIR, filename)
    
    # Generate raster if it doesn't exist yet
    if not os.path.exists(geotiff_path):
        with st.spinner(f"Generating realistic multispectral Sentinel-2 scene for {target_option}..."):
            generate_district_raster(target_option, geotiff_path)
            
    # Load bands
    bands, metadata = load_sentinel2_bands(geotiff_path)
    
    # Get center coords
    if "Pilbara" in target_option:
        lat_c, lon_c = -21.15, 119.15
    elif "Escondida" in target_option:
        lat_c, lon_c = -24.27, -69.07
    elif "Carlin" in target_option:
        lat_c, lon_c = 40.95, -116.33

else:
    # Custom File Upload Portal
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.subheader("📂 Custom Sentinel-2 Raster Upload")
    st.write("Upload a geo-referenced multispectral GeoTIFF. The file MUST contain exactly 6 bands corresponding to Sentinel-2 bands in the following order: B02 (Blue), B03 (Green), B04 (Red), B08 (NIR), B11 (SWIR1), B12 (SWIR2). Values should be reflectance scaled between 0 and 10000.")
    
    uploaded_file = st.file_uploader("Upload Multi-band GeoTIFF", type=["tif", "tiff"])
    
    if uploaded_file is not None:
        custom_filepath = os.path.join(CACHE_DIR, "custom_uploaded_scene.tif")
        with open(custom_filepath, "wb") as f:
            f.write(uploaded_file.getbuffer())
            
        try:
            bands, metadata = load_sentinel2_bands(custom_filepath)
            bounds = metadata['bounds']
            lat_c = (bounds.bottom + bounds.top) / 2.0
            lon_c = (bounds.left + bounds.right) / 2.0
            st.success("✅ Multi-spectral GeoTIFF loaded successfully!")
        except Exception as e:
            st.error(f"❌ Error loading GeoTIFF: {str(e)}. Please check that the file is a valid geo-referenced multi-band raster.")
            
    else:
        st.info("ℹ️ Upload a Sentinel-2 GeoTIFF or click the button below to generate a mock Nevada-style custom GeoTIFF to test this feature.")
        
        if st.button("🔧 Generate & Cache a Mock Nevada-style GeoTIFF"):
            mock_path = os.path.join(CACHE_DIR, "custom_uploaded_scene.tif")
            generate_district_raster("Carlin Trend, Nevada", mock_path)
            st.success("✅ Mock raster generated and saved to cache! Refreshing page...")
            st.rerun()
            
    st.markdown("</div>", unsafe_allow_html=True)

# Halt if bands are not loaded
if bands is None:
    st.warning("⚠️ Please select a target district or upload a valid GeoTIFF file to begin spectral prospecting.")
    st.stop()

# ---------------------------------------------------------
# Mathematical Spectral Indices Calculations
# ---------------------------------------------------------
indices = compute_spectral_indices(bands)

# Calculate statistics for each index (used for geological reports and metrics)
index_stats = {}
for idx_name, data in indices.items():
    index_stats[idx_name] = {
        'mean': float(data.mean()),
        'max': float(data.max()),
        'p90': float(np.percentile(data, 90))
    }

# ---------------------------------------------------------
# Main App Layout (Map & Analytical Tabs)
# ---------------------------------------------------------
col_left, col_right = st.columns([3, 2])

with col_left:
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.subheader(f"🌐 Interactive Prospecting Map: {target_option}")
    
    # Render active layer on Folium
    active_prob_map = None
    if selected_layer == "ML Mineral Probability" and st.session_state.rf_trained:
        # Determine class coordinate label to overlay probability for
        # Target classes: 3 (Iron Oxide), 4 (Clay Altered), 5 (Ferrous Silicates)
        if target_option == "Pilbara District, Australia":
            target_class_id = 3 # Iron Oxide
            prob_label = "Hematite/Goethite Outcrops"
        elif target_option == "Escondida Mine, Chile":
            target_class_id = 4 # Hydrothermal Clays
            prob_label = "Phyllic/Argillic Alteration Halo"
        elif target_option == "Carlin Trend, Nevada":
            target_class_id = 4 # Clays
            prob_label = "Fault-controlled Hydrothermal Clays"
        else:
            # For custom files, select class that is present in probability maps
            available_classes = list(st.session_state.probability_maps.keys())
            # Prefer 3 or 4 if available
            if 4 in available_classes:
                target_class_id = 4
                prob_label = "Clay Alteration"
            elif 3 in available_classes:
                target_class_id = 3
                prob_label = "Iron Oxide"
            else:
                target_class_id = available_classes[0]
                prob_label = f"Class {target_class_id}"
                
        active_prob_map = st.session_state.probability_maps.get(target_class_id)
        st.write(f"🗺️ *Displaying classification probability layer for:* **{prob_label}**")
        
    m = create_prospector_map(
        lat_c, lon_c,
        bands, indices, metadata,
        active_layer_type=selected_layer,
        opacity=overlay_opacity,
        training_points=st.session_state.training_points,
        rf_probability_map=active_prob_map
    )
    
    # Display the interactive map using streamlit-folium
    map_data = st_folium(m, width="100%", height=600, key="prospector_map")
    
    # Coordinate Click Inspector
    if map_data and map_data.get('last_clicked'):
        clicked_coords = map_data['last_clicked']
        clicked_lat = clicked_coords['lat']
        clicked_lon = clicked_coords['lng']
        
        st.markdown(f"""
        <div style='background: rgba(0, 242, 254, 0.08); border: 1px solid rgba(0, 242, 254, 0.2); padding: 0.8rem; border-radius: 8px; margin-top: 0.5rem;'>
            🔍 <b>Pixel Coordinate Inspector:</b><br>
            Latitude: <code>{clicked_lat:.6f}</code> | Longitude: <code>{clicked_lon:.6f}</code>
        </div>
        """, unsafe_allow_html=True)
        
        # Add new point input form
        st.markdown("##### 📌 Add Clicked Point to Random Forest Training Set")
        c1, c2 = st.columns(2)
        with c1:
            point_class_name = st.selectbox(
                "Geological Class",
                ["Vegetation", "Water", "Barren Soil", "Iron Oxide Zone", "Clay Altered Zone", "Ferrous Silicates"]
            )
        with c2:
            class_mapping = {
                "Vegetation": 0,
                "Water": 1,
                "Barren Soil": 2,
                "Iron Oxide Zone": 3,
                "Clay Altered Zone": 4,
                "Ferrous Silicates": 5
            }
            if st.button("➕ Register Training Pixel"):
                class_id = class_mapping[point_class_name]
                st.session_state.training_points.append(
                    (clicked_lat, clicked_lon, class_id, point_class_name)
                )
                st.success(f"Registered ({clicked_lat:.4f}, {clicked_lon:.4f}) as {point_class_name}!")
                st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

with col_right:
    # ---------------------------------------------------------
    # TAB PANEL: Spectral Indices, Machine Learning, AI Insights
    # ---------------------------------------------------------
    tab_spectral, tab_ml, tab_ai = st.tabs(["📊 Spectral Analytics", "🤖 Machine Learning", "💡 AI Prospector"])
    
    with tab_spectral:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.subheader("Alteration Statistics")
        st.write("Statistical distribution of mineral indices. Higher ratios suggest stronger alteration anomalies.")
        
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"""
            <div class='glass-card glowing-border-red' style='padding: 1rem; text-align: center;'>
                <div class='metric-label'>Iron Oxide (B4/B2)</div>
                <div class='metric-value'>{index_stats['iron_oxide']['p90']:.2f}</div>
                <div style='font-size: 0.75rem; color: #94a3b8;'>90th Percentile</div>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class='glass-card glowing-border-purple' style='padding: 1rem; text-align: center;'>
                <div class='metric-label'>Clay Minerals (B11/B12)</div>
                <div class='metric-value'>{index_stats['clay_minerals']['p90']:.2f}</div>
                <div style='font-size: 0.75rem; color: #94a3b8;'>90th Percentile</div>
            </div>
            """, unsafe_allow_html=True)
        with c3:
            st.markdown(f"""
            <div class='glass-card glowing-border-orange' style='padding: 1rem; text-align: center;'>
                <div class='metric-label'>Ferrous Iron (B12/B8)</div>
                <div class='metric-value'>{index_stats['ferrous_iron']['p90']:.2f}</div>
                <div style='font-size: 0.75rem; color: #94a3b8;'>90th Percentile</div>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("#### Geological Band Ratios Explained")
        st.write(
            "• **Iron Oxide (Red / Blue):** Highly weathered rock beds rich in hematite, jarosite, or limonite show "
            "extreme reflectance peaks in Red (B04) and high absorption valleys in Blue (B02). An index value > 2.0 "
            "strongly indicates oxidized weathered caprocks (gossans).\n\n"
            "• **Clay Minerals (SWIR1 / SWIR2):** Hydroxylated clay minerals (kaolinite, alunite, illite, sericite) "
            "exhibit deep spectral absorption at 2.2 μm (B12) and highly reflective profiles near 1.6 μm (B11). "
            "Spikes in this ratio index map hydrothermal alteration halos.\n\n"
            "• **Ferrous Iron (SWIR2 / NIR):** Divalent iron bonds in pyroxenes, olivines, and chloritic silicates "
            "produce broad absorption features in NIR (B08) and high SWIR2 (B12) signatures. Used to delineate basic volcanics."
        )
        st.markdown("</div>", unsafe_allow_html=True)
        
    with tab_ml:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.subheader("Random Forest Pixel Classifier")
        st.write(
            "Train an interactive supervised Random Forest classifier to generate mineral probability maps. "
            "The model extracts pixel multi-spectral signatures across all 6 Sentinel-2 bands and 3 calculated indices."
        )
        
        # Display Current Training points
        st.markdown("##### Current Training Set")
        df_pts = pd.DataFrame(st.session_state.training_points, columns=['Latitude', 'Longitude', 'Class ID', 'Label'])
        st.dataframe(df_pts, height=180, use_container_width=True)
        
        # Clear points or reset
        c1, c2 = st.columns(2)
        with c1:
            if st.button("🗑️ Clear Training Set"):
                st.session_state.training_points = []
                st.session_state.rf_trained = False
                st.rerun()
        with c2:
            if st.button("🔄 Reset Pre-seeded Points"):
                st.session_state.training_points = get_preseeded_points(target_option)
                st.session_state.rf_trained = False
                st.rerun()
                
        st.markdown("---")
        
        # Train trigger
        if st.button("🚀 Train Random Forest Classifier"):
            if len(st.session_state.training_points) < 5:
                st.error("❌ Need at least 5 training points. Add them via map clicks or click 'Reset Pre-seeded Points'.")
            else:
                with st.spinner("Extracting multi-spectral raster features and fitting Random Forest..."):
                    try:
                        # Extract features
                        X, y, classes, feat_names = extract_features_at_points(
                            bands, indices, st.session_state.training_points,
                            metadata['transform'], metadata['width'], metadata['height']
                        )
                        
                        # Train
                        clf, oob, importances, report = train_mineral_classifier(X, y, feat_names, classes)
                        
                        # Predict
                        class_map, prob_maps = predict_mineral_probabilities(clf, bands, indices, metadata['width'], metadata['height'])
                        
                        # Cache in state
                        st.session_state.rf_model = clf
                        st.session_state.rf_trained = True
                        st.session_state.probability_maps = prob_maps
                        st.session_state.classification_map = class_map
                        st.session_state.rf_metrics = {
                            'oob_score': oob,
                            'importances': importances,
                            'report': report
                        }
                        
                        # Auto-select ML layer in sidebar for instant viewing
                        st.success(f"✅ Random Forest trained! Out-Of-Bag accuracy: **{oob*100:.2f}%**")
                        st.session_state.ai_report = None # Reset AI report to trigger regeneration
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error during training: {str(e)}")
                        
        # Display results if trained
        if st.session_state.rf_trained:
            st.markdown("---")
            st.markdown("##### 📈 Model Evaluation Metrics")
            
            # Progress bar for OOB Accuracy
            oob = st.session_state.rf_metrics['oob_score']
            st.write(f"**Out-Of-Bag Classification Score:** {oob*100:.1f}%")
            st.progress(oob)
            
            # Bar chart for feature importances
            st.markdown("##### 🏷️ Feature Importances")
            importances_df = st.session_state.rf_metrics['importances']
            st.bar_chart(importances_df.set_index('Feature'), height=200)
            
            # Target Prospect Export
            st.markdown("##### 📥 Export Prospect Coordinates")
            st.write("Filter and download coordinates showing high mineral probability based on the classifier:")
            
            # Let geologist set threshold
            p_threshold = st.slider("Min Probability Threshold", 0.5, 1.0, 0.8, 0.05)
            
            # Extract coordinates where prob exceeds threshold
            if st.button("🔍 Delineate Drill Targets"):
                # Determine target class id
                if "Pilbara" in target_option:
                    t_class = 3 # Iron Oxide
                    p_name = "Iron Oxide"
                elif "Escondida" in target_option:
                    t_class = 4 # Clay
                    p_name = "Clay alteration"
                elif "Carlin" in target_option:
                    t_class = 4 # Clay
                    p_name = "Clay alteration"
                else:
                    t_class = list(st.session_state.probability_maps.keys())[0]
                    p_name = f"Class_{t_class}"
                    
                target_prob_raster = st.session_state.probability_maps[t_class]
                
                # Find pixel coordinates where prob > threshold
                rows, cols = np.where(target_prob_raster >= p_threshold)
                
                if len(rows) == 0:
                    st.warning(f"No pixels found with {p_name} probability >= {p_threshold:.0%}.")
                else:
                    # Convert pixel back to lat/lon
                    lons, lats = rasterio.transform.xy(metadata['transform'], rows, cols)
                    probs = target_prob_raster[rows, cols]
                    
                    df_exports = pd.DataFrame({
                        'Latitude': lats,
                        'Longitude': lons,
                        'Probability': probs
                    }).sort_values('Probability', ascending=False)
                    
                    # Cap CSV export size to avoid performance bottlenecks
                    export_limit = 2000
                    if len(df_exports) > export_limit:
                        st.info(f"Showing top {export_limit} out of {len(df_exports)} targets.")
                        df_exports_capped = df_exports.head(export_limit)
                    else:
                        df_exports_capped = df_exports
                        
                    st.dataframe(df_exports_capped, height=180)
                    
                    # CSV Download button
                    csv_bytes = df_exports_capped.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="💾 Download Target Coordinates (CSV)",
                        data=csv_bytes,
                        file_name=f"geomind_{target_option.replace(' ', '_').lower()}_targets.csv",
                        mime="text/csv"
                    )
        st.markdown("</div>", unsafe_allow_html=True)

    with tab_ai:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.subheader("💬 AI Geological Agent Report")
        
        # Trigger report generation if not cached
        if st.session_state.ai_report is None:
            with st.spinner("Generating expert geological alteration report..."):
                st.session_state.ai_report = run_groq_geological_analysis(
                    groq_key,
                    target_option,
                    index_stats,
                    st.session_state.rf_metrics
                )
                
        # Render Markdown report
        st.markdown(st.session_state.ai_report)
        
        # Interactive chat if Groq API is configured
        if groq_key:
            st.markdown("---")
            st.markdown("##### 💬 Chat with AI Geologist")
            user_question = st.text_input("Ask a follow-up geological question about this district:")
            if user_question:
                with st.spinner("AI Geologist is analyzing..."):
                    followup_response = run_groq_geological_analysis(
                        groq_key,
                        target_option,
                        index_stats,
                        st.session_state.rf_metrics,
                        custom_prompt=user_question
                    )
                    st.markdown("##### 🧑‍🔬 AI Geologist Response:")
                    st.info(followup_response)
        st.markdown("</div>", unsafe_allow_html=True)
