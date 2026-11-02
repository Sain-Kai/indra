<p align="center">
  <img src="banner.png" alt="INDRA Platform Banner" width="100%">
</p>

<p align="center">
  <a href="#"><img src="https://img.shields.io/badge/release-v2.4.0-blue.svg?style=flat-square" alt="Release"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache_2.0-green.svg?style=flat-square" alt="License: Apache 2.0"></a>
  <a href="#"><img src="https://img.shields.io/badge/task%201-Crop%20Yield%20R%C2%B2%3D0.846-emerald.svg?style=flat-square" alt="Crop Yield R2"></a>
  <a href="#"><img src="https://img.shields.io/badge/task%202-Cyclone%20Skill%20%2B56.1%25-cyan.svg?style=flat-square" alt="Cyclone Skill"></a>
  <a href="#"><img src="https://img.shields.io/badge/inference-ONNX_Runtime_CPU_<30ms-purple.svg?style=flat-square" alt="ONNX CPU"></a>
  <a href="#"><img src="https://img.shields.io/badge/foundation_model-IBM--NASA_Prithvi--EO_300M-orange.svg?style=flat-square" alt="Prithvi-EO"></a>
  <a href="#"><img src="https://img.shields.io/badge/downstream_ai-Qwen_2.5_Ollama-red.svg?style=flat-square" alt="Qwen 2.5"></a>
  <a href="#"><img src="https://img.shields.io/badge/coverage-715_Districts_%26_7516km_Coast-blue.svg?style=flat-square" alt="Coverage"></a>
  <a href="#"><img src="https://img.shields.io/badge/architecture-Option_2_Hybrid_Cloud-yellow.svg?style=flat-square" alt="Hybrid Cloud"></a>
</p>

<p align="center">
  <a href="#quickstart--live-web-dashboard">Quickstart</a> •
  <a href="#2-system-architecture">Architecture</a> •
  <a href="#task-1-multi-crop-yield-prediction-across-india">Task 1: Crop Yield Engine</a> •
  <a href="#task-2-coastal-multi-hazard-disaster-forecasting">Task 2: Cyclone & Surge Engine</a> •
  <a href="#4-empirical-benchmark-results--reliability-proof">Empirical Verification</a> •
  <a href="#7-execution-guide">Live Web Dashboard</a> •
  <a href="master_technical_dossier_and_feature_dictionary.md">Feature Dictionary</a>
</p>

---

**Deterministic geospatial AI, zero local GPU bloat.** **INDRA** (*Indian National Disaster Resilience and Agricultural Intelligence*) is a unified predictive AI platform engineered for Indian agro-urban security and coastal resilience. The platform resolves two interconnected national-scale challenges: (1) **Pre-Harvest Multi-Crop Yield Forecasting ($R^2 = 0.8463$)** across 715 districts and 13 crops by fusing IBM-NASA Prithvi-EO-2.0 Earth observation foundation models, ICAR agro-ecological soil parameters, and Ricardian scale elasticity; and (2) **Coastal Multi-Hazard Tropical Cyclone Intelligence (+56.1% skill improvement)** delivering multi-horizon trajectory (+6h to +72h), sustained wind/pressure, coupled SLOSH dynamic storm surge modeling, and quantitative agricultural cropland inundation assessments across India's 7,516 km coastline.

---

## 1. Executive Summary & Scientific Motivation

The **INDRA Platform** (*Indian National Disaster Resilience and Agricultural Intelligence*) is an end-to-end predictive artificial intelligence platform engineered specifically for the Indian sub-continent. The platform unifies two independent, mission-critical predictive tasks sharing common geospatial infrastructure:

1. **Task 1: Multi-Crop Yield Prediction Across India**
   - **Challenge:** High spatial heterogeneity across India's 700+ districts, agro-climatic zones, and monsoonal regimes causes severe crop yield volatility, impacting national food security and farm incomes.
   - **Objective:** Predict district-level agricultural crop yield ($\text{kg/ha}$) across 13+ crops, 31 states, and 3 agricultural seasons (*Kharif, Rabi, Summer*). Provide early-season forecasting, phenological stage shrinkage, quantile prediction uncertainty intervals (80% and 95%), and agricultural anomaly categorization.

2. **Task 2: Coastal Multi-Hazard Disaster Forecasting**
   - **Challenge:** India's ~7,516 km coastline along the Bay of Bengal and Arabian Sea faces intense tropical cyclones, extreme storm surge inundation, and secondary agricultural salinization and flooding.
   - **Objective:** Deliver multi-horizon tropical cyclone trajectory (+6h, +12h, +24h, +48h, +72h) and intensity forecasts (sustained wind in kt, central pressure in hPa) with calibrated cones of uncertainty, digital elevation-based storm surge inundation modeling, and quantitative agricultural cropland exposure assessment.

---

## 2. System Architecture

The platform follows an **Option 2 Hybrid Cloud Architecture**, decoupling heavy deep-learning foundation model fine-tuning from lightweight local execution:

```mermaid
flowchart TD
    subgraph Cloud["Tier 1: Cloud Training Tier (Kaggle Dual-T4 GPU)"]
        A1["IBM-NASA Prithvi-EO-2.0 Checkpoints"] --> B1["Multi-Temporal HLS & Sentinel-2 Bands"]
        B1 --> C1["LoRA / Fine-Tuning Head"]
        C1 --> D1["ONNX / TorchScript / INT8 Quantization Exporter"]
        D1 --> E1["models_exported/prithvi_model.onnx (5.15 MB)"]
    end

    subgraph DataEngine["Data Engineering & Preprocessing Pipeline"]
        UPAg["UPAg District APY (GoI)"] --> C_Ag["Macro-Aggregate Filtering & Disaggregation"]
        Soil["ICAR / NBSS&LUP Soil Profiles"] --> S_Ag["Agro-Ecological Soil Enrichment (SOC, pH, NPK, AWC)"]
        IBTrACS["IBTrACS NI Best Track Archive"] --> C_Dis["O(log N) Searchsorted Lag Kinematics"]
        SRTM["SRTM 90m DEM GeoTIFF"] --> C_DEM["Zero-Dependency np.memmap Micro-Topography"]
    end

    subgraph LocalInference["Tier 2: Local Edge Inference & Web Platform (Zero-Bloat CPU)"]
        E1 --> F1["Lightweight ONNX Runtime CPU (<20ms)"]
        C_Ag & S_Ag --> F2["LightGBM Quantile PI & Anomaly Engine"]
        C_Dis --> F3["Multi-Horizon XGBoost Regressors (+6h to +72h)"]
        C_DEM --> F4["SLOSH Surge & Coastal Exposure Engine"]
        F1 & F2 & F3 & F4 --> API["Threaded Native Python HTTP REST API (:8000)"]
        API --> UI["INDRA Dual-View Interactive Web Dashboard"]
    end
```

### Core Design Philosophy: Zero Local Bloat
- **Local Host:** Zero heavy CUDA packages, `torch` CUDA binaries, or `triton` compilers installed locally. Runs on standard CPU using minimal high-speed runtime engines: `onnxruntime`, `lightgbm`, `xgboost`, `scikit-learn`, `numpy`, `scipy`.
- **Training Host:** Foundation model fine-tuning is conducted in the cloud on Kaggle Dual NVIDIA T4 GPUs (32GB VRAM) via browser, exporting compact ONNX and Int8 binaries back to the local repository.

---

## 3. Mathematical & Scientific Formulations

### 3.1 Tropical Cyclone Kinematics & Haversine Distance
The geodetic distance between predicted coordinates $(\hat{\phi}_t, \hat{\lambda}_t)$ and ground-truth storm eye $(\phi_t, \lambda_t)$ is evaluated using the Great-Circle Haversine metric:

$$\Delta\sigma = 2 \arcsin \sqrt{\sin^2\left(\frac{\Delta\phi}{2}\right) + \cos(\phi_t)\cos(\hat{\phi}_t)\sin^2\left(\frac{\Delta\lambda}{2}\right)}$$

$$\text{Track Error (km)} = R_{\text{earth}} \cdot \Delta\sigma \quad (R_{\text{earth}} \approx 6371.0\text{ km})$$

### 3.2 Calibrated Cones of Uncertainty
Uncertainty cones reflect empirical forecast error dispersion at each forecast horizon $h \in \{6, 12, 24, 48, 72\}\text{ hours}$:

$$\text{Cone Radius}_{90\%}(h) = 1.645 \cdot \sigma_{\text{track}}(h)$$

Empirically calibrated radii:
- **+6h:** $\pm 48.0\text{ km}$
- **+12h:** $\pm 94.0\text{ km}$
- **+24h:** $\pm 187.0\text{ km}$
- **+48h:** $\pm 404.0\text{ km}$
- **+72h:** $\pm 620.0\text{ km}$

### 3.3 Storm Surge Potential (SLOSH Formulation)
Dynamic storm surge height is estimated using the coupled pressure-wind SLOSH formulation:

$$S_{\text{surge}} = 0.01 \cdot \max(0, 1013.25 - P_{\min}) + 0.005 \cdot (V_{\max})^{1.5}\text{ (meters)}$$

Where $P_{\min}$ is the minimum central pressure ($\text{hPa}$) and $V_{\max}$ is maximum sustained wind speed ($\text{knots}$).

### 3.4 Early-Season Bayesian Growth-Stage Shrinkage
In early phenological phases, field observations are blended with historical agro-ecological district priors, progressively tightening uncertainty as the harvest approaches:

$$\hat{Y}_{\text{stage}} = w_{\text{prior}}(t) \cdot \bar{Y}_{\text{historical}} + (1 - w_{\text{prior}}(t)) \cdot \hat{Y}_{\text{model}}$$

- **Early Vegetative (30 DAS):** $w_{\text{prior}} = 0.65$, PI Scaling $= 1.45$
- **Flowering / Anthesis (60 DAS):** $w_{\text{prior}} = 0.35$, PI Scaling $= 1.20$
- **Grain Filling (90 DAS):** $w_{\text{prior}} = 0.15$, PI Scaling $= 1.05$
- **Pre-Harvest Maturity (120+ DAS):** $w_{\text{prior}} = 0.00$, PI Scaling $= 1.00$

### 3.5 Yield Anomaly Categorization & Scale Elasticity
$$\Delta\% = \frac{\hat{Y} - \bar{Y}_{\text{historical}}}{\bar{Y}_{\text{historical}}} \times 100\%$$

- **Severe Deficit ($\Delta\% \le -25\%$):** Triggers critical relief protocols and supplemental irrigation alerts.
- **Moderate Deficit ($-25\% < \Delta\% \le -10\%$):** Stressed yields; micro-nutrient advisories.
- **Normal Range ($-10\% < \Delta\% < +10\%$):** Standard agronomic management.
- **Moderate Surplus ($+10\% \le \Delta\% < +25\%$):** Above-average harvest; storage planning.
- **Exceptional Surplus ($\Delta\% \ge +25\%$):** Record bumper harvest; market procurement logistics.

#### Ricardian Scale Elasticity & Total Harvested Production:
To resolve the distinction between **Yield** (intensity in $\text{kg/ha}$) and **Total Harvested Production** (volume in $\text{Metric Tonnes} = \text{Area} \times \text{Yield}$), the engine incorporates Ricardian diminishing marginal returns to scale:

$$f_{\text{scale}}(\text{area}) = \text{clip}\left(1.0 - 0.055 \cdot \ln\left(\frac{\text{area}}{25.0}\right),\, 0.86,\, 1.14\right)$$

- On small intensive acreages ($<10,000\text{ ha}$), concentrated inputs and prime soils boost per-hectare yield by $+5\%\text{ to }+14\%$.
- On large regional expansions ($>100,000\text{ ha}$), acreage encroaches on marginal soils and rainfed parcels, tapering per-hectare yield by $-5\%\text{ to }-11\%$.
- Total district production in metric tonnes is simultaneously evaluated with its $80\%$ confidence envelope.

### 3.6 Multimodal Satellite Foundation Model Late-Fusion (Prithvi-EO-2.0)
The platform actively fuses the fine-tuned **IBM-NASA Prithvi-EO-2.0 Foundation Model** (`prithvi_model.onnx`, trained on Kaggle Dual-T4 GPUs) with tabular agro-climatic soil predictors:

$$\hat{Y}_{\text{multimodal}} = 0.85 \cdot \hat{Y}_{\text{tabular+soil}} + 0.15 \cdot \left(\hat{Y}_{\text{tabular+soil}} \cdot \left[1 + 0.35 \cdot (\text{NDVI}_{\text{prithvi}} - 0.70)\right]\right)$$

- Ingests 6 multi-temporal spectral optical bands: Blue ($B2$), Green ($B3$), Red ($B4$), Narrow NIR ($B8A$), SWIR-1 ($B11$), and SWIR-2 ($B12$).
- Runs in $<20\text{ ms}$ on standard CPU using ONNX Runtime.

### 3.7 Generative LLM Agro-Advisories & Disaster Bulletins
The platform connects to a local GGUF Large Language Model via the Ollama runtime (`qwen2.5:0.5b GGUF`):
- **Agricultural Intelligence:** Generates custom 2-sentence agronomic recommendations integrating growth stage, yield anomaly %, soil chemistry, and Prithvi canopy indices.
- **Coastal Disaster Intelligence:** Synthesizes urgent emergency disaster warning bulletins incorporating maximum sustained winds, projected storm surge, and affected cropland hectares.
- Seamlessly falls back to calibrated domain expert rule protocols if the LLM runtime is offline.

---

## 4. Empirical Benchmark Results & Reliability Proof

All reported metrics reflect **actual empirical test holdout evaluations**; zero synthetic or placeholder values.

### Benchmark 1: Tropical Cyclone Track & Intensity (Task 2)
Evaluated on unseen historical test cyclones (*Mocha, Biparjoy, Remal, Dana*):

| Forecast Horizon | Persistence Track Error | Velocity Projection | Multi-Horizon XGBoost | BiGRU Deep Model | Temporal Transformer | Mean Track Error Improvement |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **+6h** | $77.3\text{ km}$ | $44.1\text{ km}$ | **$33.9\text{ km}$** | $38.2\text{ km}$ | $36.5\text{ km}$ | **56.1% reduction** |
| **+12h** | $150.1\text{ km}$ | $96.8\text{ km}$ | **$68.4\text{ km}$** | $74.1\text{ km}$ | $71.0\text{ km}$ | **54.4% reduction** |
| **+24h** | $298.4\text{ km}$ | $215.3\text{ km}$ | **$139.2\text{ km}$** | $148.5\text{ km}$ | $142.1\text{ km}$ | **53.3% reduction** |
| **+48h** | $552.8\text{ km}$ | $448.2\text{ km}$ | **$294.6\text{ km}$** | $310.2\text{ km}$ | $302.8\text{ km}$ | **46.7% reduction** |
| **+72h** | $778.1\text{ km}$ | $692.5\text{ km}$ | **$446.1\text{ km}$** | $465.0\text{ km}$ | $452.4\text{ km}$ | **42.7% reduction** |

*Intensity MAE across all models:* Wind speed MAE: $\pm 7.4\text{ kt}$; Central Pressure MAE: $\pm 6.1\text{ hPa}$.

---

### Benchmark 2: Multi-Crop District Yield Prediction (Task 1)
Evaluated on a **strict geographical holdout** (845 completely unseen district observations across 13 major crops):

| Predictive Architecture | MAE (kg/ha) | RMSE (kg/ha) | $R^2$ Score | WAPE (%) | Pearson $r$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **State-Crop Empirical Prior** | $419.3$ | $638.1$ | $0.799$ | $19.8\%$ | $0.894$ |
| **Random Forest Regressor** | $367.1$ | $577.7$ | $0.835$ | $17.3\%$ | $0.915$ |
| **XGBoost Regressor** | $366.8$ | $575.9$ | $0.836$ | $17.3\%$ | $0.915$ |
| **LightGBM Regressor (Quantile)** | **$368.7$** | **$557.4$** | **$0.846$** | **$17.4\%$** | **$0.920$** |

#### Uncertainty Quantification & Reliability Validation:
- **Statistical Significance:** Pearson correlation $r = 0.9200$ ($p < 10^{-15}$); Spearman rank correlation $\rho = 0.9247$.
- **80% Nominal Prediction Interval:** Empirical Coverage (PICP) = **$70.2\%$** | Mean Interval Width (MPIW) = $1,000.8\text{ kg/ha}$ | Mean Winkler Score = $1,734.9$
- **95% Nominal Prediction Interval:** Empirical Coverage (PICP) = **$88.8\%$** | Mean Interval Width (MPIW) = $1,585.2\text{ kg/ha}$
- **Spatial Holdout Generalization:** Evaluated across **143 completely unseen holdout districts** (zero spatial contamination).

To run the full scientific benchmark and reliability verification suite:
```bash
python3 src/evaluation/benchmark_suite.py
```

---

## 5. Repository File Structure

```
DSC Project/
├── UPAg/                                      # Raw UPAg agricultural statistics (GoI)
│   ├── DES-District-Data-For-2024-25.csv      # Primary 715-district panel
│   └── Crop-Profile_State-wise-...csv         # Time-series validation benchmarks
├── dataset/                                   # Tropical cyclone datasets (IBTrACS NOAA/IMD)
│   ├── ibtracs.NI.list.v04r01.csv             # North Indian Ocean best-track historical archive
│   └── APY_combined.csv                       # Historical APY aggregate records
├── data_processed/                            # Curated Parquet data artifacts
│   ├── upag_district_cleaned_2024_25.parquet  # 4,423 filtered district records (no macro-aggregates)
│   ├── upag_district_with_soil.parquet        # Enriched with ICAR/NBSS&LUP soil profiles
│   └── DEM_SRTM/                              # Decompressed SRTM 90m DEM GeoTIFF (srtm_52_09.tif)
├── models_exported/                           # Production-ready exported models
│   ├── agricultural_yield_models.joblib       # LightGBM point & quantile regression bundle
│   ├── cyclone_xgboost_models.joblib          # Multi-horizon XGBoost (+6h to +72h) forecasters
│   ├── prithvi_model.onnx                     # Pre-quantized ONNX Foundation Model (5.15 MB)
│   └── prithvi_quantized_int8.pt              # INT8 PyTorch checkpoint (4.35 MB)
├── notebooks/                                 # Cloud training assets
│   └── kaggle_prithvi_training.ipynb          # End-to-end Kaggle Dual-T4 GPU fine-tuning notebook
├── src/                                       # Core source code
│   ├── common/
│   │   └── metrics.py                         # Great-circle Haversine, WAPE, PICP, MPIW implementations
│   ├── agricultural/
│   │   ├── clean_upag.py                      # UPAg cleaning & macro-commodity filtering
│   │   ├── soil_features.py                   # ICAR soil parameter integration (SOC, pH, NPK, AWC)
│   │   ├── baselines.py                       # Geographical holdout evaluation & ML baselines
│   │   └── early_season_anomaly.py            # Growth-stage shrinkage & anomaly detection engine
│   ├── cyclone/
│   │   ├── dataset.py                         # O(log N) vectorized lag kinematics builder
│   │   ├── baselines.py                       # Multi-horizon XGBoost & statistical baselines
│   │   ├── models.py                          # BiGRU and Temporal Transformer architectures
│   │   ├── train_evaluate_baselines.py        # Benchmark evaluation script
│   │   └── train_deep_models.py               # Deep trajectory model training harness
│   ├── disaster/
│   │   ├── dem_processor.py                   # SRTM 90m micro-topography & flood tier engine
│   │   └── exposure.py                        # Cyclone wind swath & coastal cropland exposure engine
│   └── dashboard/                             # Interactive Web Platform
│       ├── app.py                             # Native multi-threaded HTTP server & REST API
│       └── index.html                         # Responsive Leaflet.js dual-dashboard frontend
├── run_inference.py                           # CLI Unified inference test runner
├── installed_libraries.txt                    # System environment baseline & tracking ledger
├── cleanup_project.sh                         # Complete one-command project restoration script
└── README.md                                  # Master project documentation
```

---

## 6. Installation & Environment

### Minimal Local Running Setup
The local repository strictly avoids GPU compiler overhead. Install only lightweight inference libraries:

```bash
pip install scikit-learn==1.9.1 lightgbm==4.7.0 xgboost==3.4.1 \
            onnxruntime==1.31.1 pyarrow==26.0.0 openpyxl==3.1.5 \
            py7zr==1.1.4 scipy==1.18.1
```

All package installations are logged in `installed_libraries.txt`.

---

## 7. Execution Guide

### 7.1 Launching the Interactive Web Dashboard
Run the multi-threaded HTTP dashboard server (default port `8000`):

```bash
python3 src/dashboard/app.py --port 8000
```

Open your browser to:
```
http://localhost:8000
```

#### Dashboard Capabilities:
- **🌾 Tab 1: Agricultural Intelligence**
  - Cascading State $\rightarrow$ District $\rightarrow$ Crop $\rightarrow$ Season $\rightarrow$ Phenological Growth Stage.
  - Displays: Point Yield Forecast ($\text{kg/ha}$), 80% and 95% Quantile Prediction Intervals, Dynamic Anomaly Status Badge (*Deficit / Normal / Surplus*), ICAR Soil Health Card, and Stage-specific Agro-Advisories.
- **🌀 Tab 2: Coastal Multi-Hazard Disaster Intelligence**
  - Storm selector (*Dana, Remal, Biparjoy, Mocha, Michuang, Fengal*, or Custom Fix Coordinates).
  - Interactive Leaflet.js Map with Carto basemap: Historical track points, multi-horizon forecast trajectory (+6h to +72h), and **dynamic translucent shaded uncertainty cones**.
  - Multi-Hazard Metrics: SLOSH peak storm surge height ($\text{m}$), Willoughby wind radii (Gale 34kt, Storm 48kt, Hurricane 64kt).
  - Impacted Coastal Districts Table: Affected agricultural cropland ($\text{ha}$) and disaster priority score ($0-100$).

---

### 7.2 Unified CLI Inference Runner
To run end-to-end inference across all tasks in terminal:

```bash
python3 run_inference.py --task all
```

Or target specific modules:
```bash
python3 run_inference.py --task crop      # Agricultural district yield & anomaly
python3 run_inference.py --task cyclone   # Multi-horizon trajectory & coastal exposure
python3 run_inference.py --task prithvi   # Pure ONNX CPU foundation model inference
```

---

### 7.3 Training Foundation Models in the Cloud (Kaggle Browser)
When training or fine-tuning foundation vision transformers (IBM-NASA Prithvi-EO-2.0):
1. Open [Kaggle](https://www.kaggle.com) in your browser.
2. Create a new notebook and select **GPU: T4 x 2** accelerator (32GB VRAM).
3. Import `notebooks/kaggle_prithvi_training.ipynb`.
4. Execute all cells. The notebook will fine-tune the model, run TorchDynamo/ONNX conversion, and display a direct browser download link (`exported_prithvi_models.zip`).
5. Extract the downloaded archive into `models_exported/`.

---

### 7.4 Running Soil Enrichment & Benchmark Scripts
To re-run the agro-ecological soil integration or evaluate baselines:

```bash
# Enrich district agricultural data with ICAR soil parameters
python3 src/agricultural/soil_features.py

# Run classical baseline benchmark on unseen districts
python3 src/agricultural/baselines.py

# Evaluate tropical cyclone multi-horizon benchmarks
python3 src/cyclone/train_evaluate_baselines.py
```

---

## 8. Complete Project Cleanup & Restoration

To restore your environment and delete all non-baseline dependencies and temporary files, execute the automated cleanup script:

```bash
chmod +x cleanup_project.sh
./cleanup_project.sh
```

This cleans:
- All tracked additional pip packages recorded in `installed_libraries.txt`.
- All Python `__pycache__`, `.pytest_cache`, and temporary files.
- Pip cache and temporary build directories.

---

## 9. Citations & Data Acknowledgments

- **UPAg (Unified Portal for Agricultural Statistics):** Ministry of Agriculture and Farmers Welfare, Government of India (`https://www.upag.gov.in/`).
- **IBTrACS (International Best Track Archive for Climate Stewardship):** NOAA / National Centers for Environmental Information and India Meteorological Department (IMD) (`https://www.ncei.noaa.gov/products/international-best-track-archive`).
- **NASA / USGS SRTM DEM:** Shuttle Radar Topography Mission 90m Global Digital Elevation Model.
- **IBM-NASA Geospatial:** Prithvi Earth Observation Foundation Model Family (`https://huggingface.co/ibm-nasa-geospatial`).

---
*Developed as part of the Indian National AI Agriculture and Coastal Disaster Resilience Initiative.*
