# GeoMind — AI Satellite Mineral Prospector

GeoMind is a high-performance Python application designed to assist exploration geologists in identifying mineral targets using Sentinel-2 multispectral satellite imagery, physical band ratios, machine learning pixel classifiers, and AI-driven geological reporting.

---

## 🚀 Key Features

*   **Multispectral Sentinel-2 Processing:** Reads spatial bands with `rasterio` and calculates high-contrast ratios for **Iron Oxide** ($Fe^{3+}$), **Clay Alteration** (hydroxyl absorption), and **Ferrous Iron** ($Fe^{2+}$).
*   **Stunning Interactive Mapping:** Overlay spectral index heatmaps and RGB composites (True Color, False Color, SWIR Alteration) directly over **Esri Satellite** and **OpenTopo** layers with opacity sliders.
*   **Interactive Random Forest ML:** Select preset exploration districts, register map clicks as customized training pixels, and train a supervised `scikit-learn` Random Forest to build mineral probability heatmaps.
*   **Groq LLM Exploration Reports:** Inject a secure Groq API key in the sidebar to generate comprehensive, context-aware prospecting summaries or chat directly with an AI geologist. Falls back to a highly specialized local Expert System if no key is supplied.
*   **Target Coordinate Export:** Filter high-probability exploration spots by thresholding classifier probability, review drill vectors, and export coordinates as CSV targets.

---

## 🛠️ Installation & Setup

1.  **Clone or navigate to the workspace directory:**
    ```bash
    cd d:\GeoMIND
    ```

2.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Streamlit application:**
    ```bash
    streamlit run app.py
    ```

4.  Open `http://localhost:8501` in your browser.

---

## 📂 Project Structure

```
d:\GeoMIND\
│
├── app.py                      # Streamlit Application frontend & custom CSS
├── requirements.txt            # Python dependencies list
├── README.md                   # This project guide
│
├── utils/                      # Modular backend packages
│   ├── __init__.py
│   ├── generator.py            # Realistic synthetic Sentinel-2 GeoTIFF generator
│   ├── spectral.py             # Ratio indices and RGB contrast-stretching
│   ├── classifier.py           # Pixel Random Forest ML model fitting & predictions
│   ├── llm.py                  # Groq API LLM integrations and local expert fallback
│   └── mapping.py              # Folium Leaflet layers & spatial marker configurations
│
└── data/                       # Cached geotiff datasets and exports
```

---

## 🛰️ Sentinel-2 Spectral Band Ratios

| Mineral Class | Index Formula | Description |
| :--- | :--- | :--- |
| **Iron Oxide Index** | $\frac{\text{B04}}{\text{B02}}$ (Red / Blue) | Detects oxidized iron weathering caps (hematite/goethite/jarosite). |
| **Clay Minerals Index** | $\frac{\text{B11}}{\text{B12}}$ (SWIR1 / SWIR2) | Detects hydroxyl absorption typical of argillic/phyllic porphyry alteration. |
| **Ferrous Iron Index** | $\frac{\text{B12}}{\text{B08}}$ (SWIR2 / NIR) | Delineates divalent iron ($Fe^{2+}$) in silicates, basic volcanics, and chlorite. |
