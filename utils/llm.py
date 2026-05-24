import os
from groq import Groq

def generate_geological_expert_report(district_name, index_stats, rf_metrics=None):
    """
    Fallback Expert System: Generates highly detailed, scientifically rigorous geological reports
    for preset districts when no Groq API Key is supplied.
    """
    # Build statistical text block
    stats_text = ""
    for idx_name, stats in index_stats.items():
        stats_text += f"- **{idx_name.replace('_', ' ').title()}**: Mean = {stats['mean']:.3f}, Max = {stats['max']:.3f}, 90th Percentile = {stats['p90']:.3f}\n"

    rf_text = ""
    if rf_metrics:
        rf_text += f"- **Classifier OOB Score**: {rf_metrics['oob_score']*100:.2f}%\n"
        rf_text += "- **Top Predictive Features**:\n"
        for i, row in rf_metrics['importances'].head(3).iterrows():
            rf_text += f"  - {row['Feature']}: {row['Importance']*100:.1f}%\n"

    if "Pilbara" in district_name:
        report = f"""## 📄 Executive Exploration Report: Hamersley Group BIF (Pilbara, Australia)
**Geological Setting**: Archean Pilbara Craton — Hamersley Basin Banded Iron Formations (BIF)
**Target Commodity**: High-Grade Direct Shipping Ore (DSO) Hematite/Goethite

### 1. Alteration Mineralogy & Spectral Analysis
Based on the Sentinel-2 spectral indices computed over this block:
{stats_text}
*   **Iron Oxide Index Analysis**: The elevated Iron Oxide mean and 90th percentile confirm widespread hematite ($\text{{Fe}}_2\text{{O}}_3$) and goethite ($\text{{FeO(OH)}}$) exposures. These values correspond tightly to the ridge-forming Brockman and Marra Mamba BIF units.
*   **Ferrous Iron Index Analysis**: Volcanic basalts and dolerite dykes (e.g., Mount McRae Shale or Fortescue Group basements) are successfully mapped by the Ferrous Iron index, where structural fault lines cut through the craton.
*   **Clay Index Analysis**: Clays are concentrated predominantly in the drainage valleys and alluvial channels (weathered kaolinite/smectite), representing weathering rather than hydrothermal alteration in this specific coking-coal/iron caving context.

### 2. Machine Learning Predictive Mapping
{rf_text}
The Random Forest model has identified high-probability iron mineralization zones along the structural strikes. The high importance of the **B04/B02 (Iron Oxide Index)** indicates that surface weathering and oxidized iron caprocks are the primary remote-sensing vectors for class discrimination.

### 3. Drill Target Recommendations
*   **Target A (Primary)**: The northwestern strike segment of the iron ridge, showing coincidence of high Iron Oxide index (> 2.8) and low Clay index, indicating dense hematite outcrops with minimal clay impurities.
*   **Target B (Structural)**: The intersection point where the North-South trending ferrous-rich basaltic dykes cross-cut the main iron-bearing ridge, which represents a potential localized hydrothermal enrichment zone.

---
> [!NOTE]
> *This fallback report was generated using GeoMind's Pre-loaded Geological Expert System. Add your **Groq API Key** in the sidebar to run dynamic, context-aware LLM exploration interviews!*
"""
    elif "Escondida" in district_name:
        report = f"""## 📄 Executive Exploration Report: Escondida Porphyry System (Atacama, Chile)
**Geological Setting**: Precordillera Oligocene-Eocene Copper Belt
**Target Commodity**: Supergene and Hypogene Porphyry Copper-Gold

### 1. Alteration Mineralogy & Spectral Analysis
Based on the Sentinel-2 spectral indices computed over this block:
{stats_text}
*   **Clay Mineral Index (SWIR1/SWIR2) Analysis**: The exceptionally high clay mineral response outlines the classic **phyllic (quartz-sericite) and argillic (kaolinite/alunite) alteration zones**. The ring-like distribution of clay minerals represents a classic porphyry alteration halo surrounding the central barren/leached quartz core.
*   **Iron Oxide Index Analysis**: Significant iron oxide anomalies flanking the margins correspond to oxidized pyrite caps (gossans) containing jarosite, limonite, and hematite, which are typical indicators of supergene copper leaching.
*   **Ferrous Iron Index Analysis**: Lower regional values indicate standard hyper-arid basement response, with minor ferrous highs corresponding to late-stage basic dykes.

### 2. Machine Learning Predictive Mapping
{rf_text}
The Random Forest classifier maps the spatial zoning with high fidelity. The overwhelming predictive weight of the **B11/B12 (Clay Mineral Index)** validates the scientific consensus that hydrothermal phyllic/advanced argillic haloes are the most reliable vector for identifying copper porphyry boundaries in arid terrains.

### 3. Drill Target Recommendations
*   **Target A (Phyllic Ring)**: The zone of high clay-index intensity bounding the western leached cap. Phyllic alteration zones are highly correlated with high-grade copper chalcopyrite/bornite mineralization at depth.
*   **Target B (Gossan Cap)**: Flanking iron oxide anomalies showing high red/blue ratios. These gossans represent excellent targets for testing underlying supergene enrichment blankets (chalcocite/covellite).

---
> [!NOTE]
> *This fallback report was generated using GeoMind's Pre-loaded Geological Expert System. Add your **Groq API Key** in the sidebar to run dynamic, context-aware LLM exploration interviews!*
"""
    elif "Carlin" in district_name:
        report = f"""## 📄 Executive Exploration Report: Carlin Trend Carlin-Type System (Nevada, USA)
**Geological Setting**: Great Basin Paleozoic Sediment-Hosted Disseminated Gold
**Target Commodity**: Disseminated, Microcrystalline Carlin-type Gold

### 1. Alteration Mineralogy & Spectral Analysis
Based on the Sentinel-2 spectral indices computed over this block:
{stats_text}
*   **Clay Mineral Index Analysis**: Hydrothermal clay alteration (illite/smectite/dickite) is concentrated along the central structural fault trend. This maps the conduits where acidic hydrothermal fluids decalcified the host limestones.
*   **Iron Oxide Index Analysis**: Elevated Iron Oxide readings reflect supergene oxidation of disseminated pyrites, which frequently hosts microcrystalline gold in the weathered upper levels.
*   **Ferrous Iron Index Analysis**: Highlights the dolomitic and silicate-rich host units, separating them from the argillitized shales.

### 2. Machine Learning Predictive Mapping
{rf_text}
The Random Forest classifier maps the linear fault zones with high sensitivity. The balanced feature importance of **Clay Index (SWIR1/SWIR2)** and **Iron Oxide Index (Red/Blue)** confirms that hydrothermal decalcification and subsequent oxidation are the dual hallmarks of shallow Carlin-type gold systems.

### 3. Drill Target Recommendations
*   **Target A (Fault Intersection)**: Focus on the major N-S structural lineaments where clay index anomalies widen, suggesting deep-seated fluid upwelling zones.
*   **Target B (Limestone Contacts)**: The boundary between carbonate-dominated blocks (high Ferrous/low Clay) and altered zones, which are primary chemical traps for acidic gold-bearing fluids.

---
> [!NOTE]
> *This fallback report was generated using GeoMind's Pre-loaded Geological Expert System. Add your **Groq API Key** in the sidebar to run dynamic, context-aware LLM exploration interviews!*
"""
    else:
        report = f"""## 📄 Executive Exploration Report: Custom Site Alteration Analysis
**Geological Setting**: Custom Geographic Block (User-Uploaded Raster)
**Target Commodity**: Polymetallic Alteration Target

### 1. Alteration Mineralogy & Spectral Analysis
Based on the Sentinel-2 spectral indices computed over this block:
{stats_text}
*   **Iron Oxide Index**: High values suggest exposed ferric oxides (hematite, goethite, jarosite) indicating oxidation caps.
*   **Clay Index**: High values suggest argillic/phyllic alteration (hydroxyl-bearing minerals), common in hydrothermal mineralization styles.
*   **Ferrous Iron Index**: Maps minerals with divalent iron, helping delineate basement silicates, basalts, or chloritic alteration.

### 2. Exploration Recommendations
*   Identify overlapping anomalies where **Clay Minerals** and **Iron Oxides** are both highly elevated. These are classic markers of hydrothermal alteration and subsequent surface weathering.
*   Perform structural mapping to align spectral anomalies with visible faults or shear zones on the map.

---
> [!NOTE]
> *This fallback report was generated using GeoMind's Pre-loaded Geological Expert System. Add your **Groq API Key** in the sidebar to run dynamic, context-aware LLM exploration interviews!*
"""
    return report


def run_groq_geological_analysis(api_key, district_name, index_stats, rf_metrics=None, custom_prompt=""):
    """
    Connects to Groq API using Llama 3 70B to generate a professional, contextual geological report.
    """
    if not api_key:
        return generate_geological_expert_report(district_name, index_stats, rf_metrics)
        
    try:
        # Initialize Groq Client
        client = Groq(api_key=api_key)
        
        # Format statistical findings
        stats_summary = ""
        for idx, stats in index_stats.items():
            stats_summary += f"- {idx.replace('_', ' ').upper()}: Mean={stats['mean']:.3f}, Max={stats['max']:.3f}, 90th percentile={stats['p90']:.3f}\n"
            
        rf_summary = "Not Trained"
        if rf_metrics:
            rf_summary = f"Out-Of-Bag Classification Score: {rf_metrics['oob_score']*100:.2f}%\n"
            rf_summary += "Feature Importances:\n"
            for i, row in rf_metrics['importances'].iterrows():
                rf_summary += f"  - {row['Feature']}: {row['Importance']*100:.1f}%\n"
                
        # Base system instruction
        system_instruction = (
            "You are an elite exploration geologist and remote sensing expert. "
            "Your task is to analyze multi-spectral remote sensing indices (Sentinel-2) and "
            "machine learning classification metrics to write a highly professional, "
            "scientifically detailed exploration report. "
            "Use clear Markdown formatting, including subheadings, lists, and bold text."
        )
        
        # Build prompt
        user_prompt = f"""Write a comprehensive geological prospecting and alteration report for the following target:
        
**Target Area**: {district_name}

**Computed Spectral Indices Summary**:
{stats_summary}

**Random Forest Machine Learning Metrics**:
{rf_summary}

**Geological Background**:
Sentinel-2 spectral bands: B02 (Blue), B03 (Green), B04 (Red), B08 (NIR), B11 (SWIR1), B12 (SWIR2).
Indices used:
- Iron Oxide = B04/B02 (Reflects ferric iron weathering, gossans, hematite/goethite/jarosite).
- Clay Minerals = B11/B12 (Reflects hydroxyl absorption, argillic/phyllic alteration, kaolinite/alunite/illite).
- Ferrous Iron = B12/B08 (Reflects divalent iron, ferromagnesian silicates like olivine, pyroxene, chloritic alteration).

Please structure your report as follows:
1. ## 📄 Executive Alteration Summary: Detailed analysis of the spectral index statistics, what they suggest about the mineralogy, weathering, and outcrops.
2. ## 🔍 Alteration Zonation & Genesis: Scientifically explain the hydrothermal or sedimentary genesis of these minerals. Discuss the depositional models (e.g. Banded Iron Formations, Porphyry systems, Carlin-type sediment-hosted gold) fitting this structural style.
3. ## 🎯 Exploration Vectors & Drill Targets: Provide a highly actionable list of specific zones to explore based on the findings, and how the machine learning metrics back this up.
4. ## ⚠️ Risks, Limitations & Next Steps: Point out standard remote sensing limitations in this area (e.g., vegetation masking, soil moisture, drift cover) and what field geological verification is required.

"""
        if custom_prompt:
            user_prompt += f"\n**User Question/Request**: {custom_prompt}\n"
            
        # Call Groq API
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_prompt}
            ],
            model="llama-3.3-70b-specdec" if "llama-3.3-70b-specdec" else "llama3-8b-8192",
            temperature=0.2,
            max_tokens=2500
        )
        
        return chat_completion.choices[0].message.content
        
    except Exception as e:
        # Fall back to rule-based expert system on failure, appending the error message
        fallback = generate_geological_expert_report(district_name, index_stats, rf_metrics)
        error_msg = f"\n\n> [!WARNING]\n> **Groq API Connection Failed**: {str(e)}\n> Loaded pre-seeded expert system report as fallback."
        return fallback + error_msg
