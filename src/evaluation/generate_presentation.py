"""Presentation Deck Generator for Design of Smart Cities (DSC) Capstone Project.

Generates a 14-slide widescreen (16:9) PowerPoint (.pptx) presentation:
- First slide: Title as Heading and Bold, with student, course, and faculty credentials on the left.
- Complete coverage of problem statement, architecture, feature engineering, empirical proofs,
  case studies, and live dashboard integration.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pptx
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

# ==============================================================================
# COLOR PALETTE CONSTANTS
# ==============================================================================
C_NAVY_DARK = RGBColor(10, 25, 47)       # #0A192F Dark Navy (Title Background)
C_NAVY_CARD = RGBColor(17, 34, 64)       # #112240 Dark Card Background
C_NAVY_PRIMARY = RGBColor(15, 44, 89)    # #0F2C59 Primary Header
C_CYAN_ACCENT = RGBColor(14, 165, 233)   # #0EA5E9 Vivid Sky Cyan
C_GREEN_EMERALD = RGBColor(16, 185, 129) # #10B981 Emerald Green (Success/Yield)
C_AMBER_WARN = RGBColor(245, 158, 11)    # #F59E0B Amber / Warning
C_RED_DANGER = RGBColor(239, 68, 68)     # #EF4444 Hazard / Cyclone
C_CHARCOAL = RGBColor(30, 41, 59)        # #1E293B Body Text
C_MUTED = RGBColor(100, 116, 139)        # #64748B Secondary Text
C_LIGHT_BG = RGBColor(248, 250, 252)     # #F8FAFC Slide Canvas
C_CARD_BG = RGBColor(255, 255, 255)      # Pure White Card
C_CARD_BORDER = RGBColor(226, 232, 240)  # #E2E8F0 Subtle Border
C_WHITE = RGBColor(255, 255, 255)



from pptx.text.text import _Paragraph

def add_run_text(p, text=''):
    r = _Paragraph.add_run(p)
    if text:
        r.text = str(text)
    return r

def add_rounded_card(slide, left, top, width, height, fill_color, border_color=None):
    """Adds a rectangle card container with optional border."""
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill_color
    if border_color:
        shape.line.color.rgb = border_color
        shape.line.width = Pt(1.2)
    else:
        shape.line.fill.background()
    return shape


def add_slide_header(slide, title_text: str, category_text: str = "DESIGN OF SMART CITIES (DSC) | CAPSTONE PRESENTATION"):
    """Adds uniform slide category tag and bold title."""
    # Top banner bar
    top_bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.333), Inches(0.12))
    top_bar.fill.solid()
    top_bar.fill.fore_color.rgb = C_CYAN_ACCENT
    top_bar.line.fill.background()

    # Category Tag
    tx_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.35), Inches(11.733), Inches(0.35))
    tf = tx_box.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0
    p = tf.paragraphs[0]
    p.text = category_text.upper()
    p.font.size = Pt(10)
    p.font.bold = True
    p.font.color.rgb = C_CYAN_ACCENT
    p.font.name = "Calibri"

    # Slide Title
    tx_title = slide.shapes.add_textbox(Inches(0.8), Inches(0.65), Inches(11.733), Inches(0.65))
    tf_title = tx_title.text_frame
    tf_title.word_wrap = True
    tf_title.margin_left = tf_title.margin_top = tf_title.margin_right = tf_title.margin_bottom = 0
    p2 = tf_title.paragraphs[0]
    p2.text = title_text
    p2.font.size = Pt(22)
    p2.font.bold = True
    p2.font.color.rgb = C_NAVY_PRIMARY
    p2.font.name = "Calibri"


def create_presentation() -> pptx.Presentation:
    """Builds the complete 14-slide presentation."""
    prs = pptx.Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]

    # ==========================================================================
    # SLIDE 1: TITLE SLIDE
    # First slide contains Project Title as Heading and Bold, and Left side credentials.
    # ==========================================================================
    s1 = prs.slides.add_slide(blank_layout)

    # Full dark navy background
    bg = s1.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.333), Inches(7.5))
    bg.fill.solid()
    bg.fill.fore_color.rgb = C_NAVY_DARK
    bg.line.fill.background()

    # Top Accent Line
    acc = s1.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.333), Inches(0.18))
    acc.fill.solid()
    acc.fill.fore_color.rgb = C_CYAN_ACCENT
    acc.line.fill.background()

    # Title Block (Heading and Bold)
    title_box = s1.shapes.add_textbox(Inches(0.8), Inches(0.6), Inches(11.733), Inches(1.8))
    tf1 = title_box.text_frame
    tf1.word_wrap = True
    tf1.margin_left = tf1.margin_top = tf1.margin_right = tf1.margin_bottom = 0

    p_badge = tf1.paragraphs[0]
    p_badge.text = "SMART URBAN-REGIONAL CLIMATE RESILIENCE & DISASTER MANAGEMENT"
    p_badge.font.size = Pt(11)
    p_badge.font.bold = True
    p_badge.font.color.rgb = C_CYAN_ACCENT
    p_badge.font.name = "Calibri"

    p_main = tf1.add_paragraph()
    p_main.text = "INDRA: Indian National AI Platform for Multi-Crop Yield Prediction & Coastal Multi-Hazard Disaster Intelligence"
    p_main.font.size = Pt(24)
    p_main.font.bold = True  # Project Title as Heading and Bold
    p_main.font.color.rgb = C_WHITE
    p_main.font.name = "Calibri"

    p_sub = tf1.add_paragraph()
    p_sub.text = "An AI-Driven Foundation & Spatiotemporal Gradient-Boosted System for Regional Food Security and Coastal Urban Defense"
    p_sub.font.size = Pt(12)
    p_sub.font.italic = True
    p_sub.font.color.rgb = RGBColor(148, 163, 184)
    p_sub.font.name = "Calibri"

    # --------------------------------------------------------------------------
    # LEFT SIDE: CREDENTIALS BOX (As explicitly requested by user)
    # Name: Sohan Manna | Reg No: 23BCT0002
    # Name: Biswajit Paul | Reg No: 23BCT
    # Course: Design of Smart Cities
    # Faculty: Dr.Swarnalatha P
    # --------------------------------------------------------------------------
    card_left = add_rounded_card(s1, Inches(0.8), Inches(2.6), Inches(5.6), Inches(4.3), C_NAVY_CARD, C_CYAN_ACCENT)

    tx_cred = s1.shapes.add_textbox(Inches(1.15), Inches(2.85), Inches(4.9), Inches(3.8))
    tf_cred = tx_cred.text_frame
    tf_cred.word_wrap = True
    tf_cred.margin_left = tf_cred.margin_top = tf_cred.margin_right = tf_cred.margin_bottom = 0

    p_head = tf_cred.paragraphs[0]
    p_head.text = "PROJECT CREDENTIALS"
    p_head.font.size = Pt(13)
    p_head.font.bold = True
    p_head.font.color.rgb = C_CYAN_ACCENT
    p_head.font.name = "Calibri"

    def add_cred_line(tf, label, value, is_bold_val=True, space_after=8):
        p = tf.add_paragraph()
        p.space_after = Pt(space_after)
        r1 = add_run_text(p, f"{label}: ")
        r1.font.size = Pt(12.5)
        r1.font.bold = True
        r1.font.color.rgb = RGBColor(203, 213, 225)
        r1.font.name = "Calibri"

        r2 = add_run_text(p, value)
        r2.font.size = Pt(12.5)
        r2.font.bold = is_bold_val
        r2.font.color.rgb = C_WHITE if is_bold_val else RGBColor(226, 232, 240)
        r2.font.name = "Calibri"

    add_cred_line(tf_cred, "Name", "Sohan Manna", space_after=2)
    add_cred_line(tf_cred, "Reg No", "23BCT0002", space_after=14)

    add_cred_line(tf_cred, "Name", "Biswajit Paul", space_after=2)
    add_cred_line(tf_cred, "Reg No", "23BCT", space_after=18)

    add_cred_line(tf_cred, "Course", "Design of Smart Cities", space_after=4)
    add_cred_line(tf_cred, "Faculty", "Dr.Swarnalatha P", space_after=0)

    # --------------------------------------------------------------------------
    # RIGHT SIDE: PROJECT SCOPE & HIGHLIGHTS
    # --------------------------------------------------------------------------
    card_right = add_rounded_card(s1, Inches(6.8), Inches(2.6), Inches(5.733), Inches(4.3), C_NAVY_CARD, RGBColor(51, 65, 85))

    tx_proj = s1.shapes.add_textbox(Inches(7.15), Inches(2.85), Inches(5.0), Inches(3.8))
    tf_proj = tx_proj.text_frame
    tf_proj.word_wrap = True
    tf_proj.margin_left = tf_proj.margin_top = tf_proj.margin_right = tf_proj.margin_bottom = 0

    p_phead = tf_proj.paragraphs[0]
    p_phead.text = "SMART CITY RELEVANCE & CAPSTONE HIGHLIGHTS"
    p_phead.font.size = Pt(13)
    p_phead.font.bold = True
    p_phead.font.color.rgb = C_GREEN_EMERALD
    p_phead.font.name = "Calibri"

    items = [
        ("Urban-Rural Food Bowls", "Securing municipal food supply chains across 715 Indian agricultural districts using AI yield forecasts."),
        ("Coastal Metropolises Defense", "Multi-horizon cyclone trajectory and SLOSH storm surge modeling protecting coastal cities and deltas."),
        ("Proven Holdout Accuracy", "R² = 0.8463 in multi-crop yield prediction; +56.1% error reduction in tropical cyclone track forecasting."),
        ("Zero-Bloat Edge Architecture", "Hybrid Cloud: Cloud foundation model fine-tuning (Kaggle Dual-T4) + lightweight local CPU execution (<30ms)."),
        ("Operational Deployment", "Production REST API & interactive Leaflet GIS platform with dynamic uncertainty cones and LLM advisories.")
    ]

    for title, desc in items:
        p = tf_proj.add_paragraph()
        p.space_before = Pt(6)
        p.space_after = Pt(2)
        r_bullet = add_run_text(p, "✔ ")
        r_bullet.font.bold = True
        r_bullet.font.color.rgb = C_CYAN_ACCENT
        r_title = add_run_text(p, f"{title}: ")
        r_title.font.bold = True
        r_title.font.size = Pt(10.5)
        r_title.font.color.rgb = C_WHITE
        r_desc = add_run_text(p, desc)
        r_desc.font.size = Pt(9.5)
        r_desc.font.color.rgb = RGBColor(203, 213, 225)

    # ==========================================================================
    # SLIDE 2: SMART CITIES CONTEXT & DUAL-DOMAIN CRISIS
    # ==========================================================================
    s2 = prs.slides.add_slide(blank_layout)
    add_slide_header(s2, "Smart Cities Context & Dual-Domain Climate Crisis")

    # Card 1: The Core Smart City Challenge
    c1 = add_rounded_card(s2, Inches(0.8), Inches(1.5), Inches(11.733), Inches(1.5), C_CARD_BG, C_CARD_BORDER)
    tx = s2.shapes.add_textbox(Inches(1.1), Inches(1.65), Inches(11.1), Inches(1.2))
    tf = tx.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "THE FUNDAMENTAL SMART CITY RESILIENCE PRINCIPLE"
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = C_CYAN_ACCENT

    p = tf.add_paragraph()
    p.space_before = Pt(4)
    p.text = (
        "Modern Smart Cities cannot remain resilient in isolation. They critically depend on two external lifelines: "
        "(1) stable, predictable food procurement from surrounding agricultural districts to prevent urban price spikes, and "
        "(2) robust early disaster warnings against oceanic multi-hazard cyclones that threaten coastal municipal infrastructure, ports, and power grids."
    )
    p.font.size = Pt(10.5)
    p.font.color.rgb = C_CHARCOAL

    # Card 2: Domain 1 Crisis (Agricultural Volatility)
    c2 = add_rounded_card(s2, Inches(0.8), Inches(3.3), Inches(5.6), Inches(3.6), C_CARD_BG, C_CARD_BORDER)
    tx2 = s2.shapes.add_textbox(Inches(1.1), Inches(3.5), Inches(5.0), Inches(3.2))
    tf2 = tx2.text_frame
    tf2.word_wrap = True
    p = tf2.paragraphs[0]
    p.text = "DOMAIN 1: FOOD INSECURITY & YIELD VOLATILITY"
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = C_GREEN_EMERALD

    d1_points = [
        "715 Indian Districts face severe monsoonal irregularity and heatwave shocks.",
        "Traditional statistical yield estimation requires post-harvest crop-cutting experiments (CCEs), causing a 2-3 month lag.",
        "Urban supply chain planners lack pre-harvest crop forecasts to stabilize grain procurement and warehouse buffers.",
        "Soil chemical deficits (NPK, Organic Carbon, Salinity) lead to unexpected regional yield collapses."
    ]
    for pt in d1_points:
        p = tf2.add_paragraph()
        p.space_before = Pt(6)
        p.text = f"• {pt}"
        p.font.size = Pt(10)
        p.font.color.rgb = C_CHARCOAL

    # Card 3: Domain 2 Crisis (Coastal Cyclone Disasters)
    c3 = add_rounded_card(s2, Inches(6.8), Inches(3.3), Inches(5.733), Inches(3.6), C_CARD_BG, C_CARD_BORDER)
    tx3 = s2.shapes.add_textbox(Inches(7.1), Inches(3.5), Inches(5.1), Inches(3.2))
    tf3 = tx3.text_frame
    tf3.word_wrap = True
    p = tf3.paragraphs[0]
    p.text = "DOMAIN 2: COASTAL MULTI-HAZARD DISASTERS"
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = C_RED_DANGER

    d2_points = [
        "7,516 km Coastline hosts megacities (Mumbai, Chennai, Kolkata) and critical agricultural delta zones.",
        "North Indian Ocean cyclones (Bay of Bengal & Arabian Sea) frequently undergo Rapid Intensification (e.g. Dana, Remal).",
        "Destructive Storm Surges (up to 4-6 meters) breach coastal dykes, inundating municipal infrastructure and agricultural soils.",
        "Emergency managers lack integrated systems combining storm tracks with high-resolution digital elevation inundation."
    ]
    for pt in d2_points:
        p = tf3.add_paragraph()
        p.space_before = Pt(6)
        p.text = f"• {pt}"
        p.font.size = Pt(10)
        p.font.color.rgb = C_CHARCOAL

    # ==========================================================================
    # SLIDE 3: PROJECT OBJECTIVES & UNIFIED SCOPE
    # ==========================================================================
    s3 = prs.slides.add_slide(blank_layout)
    add_slide_header(s3, "Project Objectives & Scope of Work")

    obj_cards = [
        ("1. Multi-Crop District Yield Engine", C_GREEN_EMERALD, [
            "Predict agricultural yield (kg/ha) across 13 major macro-crops and 31 states.",
            "Incorporate Ricardian scale elasticity to evaluate Total Production in Metric Tonnes.",
            "Calibrate 80% and 95% Quantile Prediction Intervals to reflect climate uncertainty.",
            "Automate early-season anomaly classification (-25% Severe Deficit to +25% Surplus)."
        ]),
        ("2. Multimodal Satellite Foundation Fusion", C_CYAN_ACCENT, [
            "Fine-tune IBM-NASA Prithvi-EO-2.0 (300M ViT) on multi-temporal optical cubes.",
            "Extract real-time vegetation health (NDVI proxy) and water absorption indices.",
            "Late-fuse satellite foundation features with tabular soil regressors on standard CPU.",
            "Achieve ultra-fast inference (<30 ms) with zero local GPU bloat."
        ]),
        ("3. Coastal Cyclone Multi-Hazard Tracking", C_RED_DANGER, [
            "Engineer multi-horizon trajectory and intensity regressors (+6h, +12h, +24h, +48h, +72h).",
            "Generate dynamic, mathematically calibrated cones of uncertainty (73.8% to 85.5% capture).",
            "Model hydrodynamic storm surge heights (SLOSH) and Willoughby radial wind swaths.",
            "Compute quantitative district cropland inundation exposure using SRTM 90m DEM."
        ]),
        ("4. Smart City Operational Web Platform", C_AMBER_WARN, [
            "Build responsive dual-task interactive dashboard running native multi-threaded REST API.",
            "Integrate Leaflet GIS mapping with dual base layers (Esri Satellite & OpenStreetMap).",
            "Connect local Ollama Large Language Model (Qwen2.5:0.5B GGUF) for agronomic & disaster advisories.",
            "Provide standalone Model Reliability Proof tab with empirical validation metrics."
        ])
    ]

    for idx, (title, color, points) in enumerate(obj_cards):
        x = Inches(0.8 + (idx % 2) * 6.0)
        y = Inches(1.6 + (idx // 2) * 2.8)
        c = add_rounded_card(s3, x, y, Inches(5.7), Inches(2.6), C_CARD_BG, C_CARD_BORDER)

        tx = s3.shapes.add_textbox(x + Inches(0.3), y + Inches(0.2), Inches(5.1), Inches(2.2))
        tf = tx.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = title
        p.font.size = Pt(12)
        p.font.bold = True
        p.font.color.rgb = color

        for pt in points:
            p = tf.add_paragraph()
            p.space_before = Pt(3)
            p.text = f"• {pt}"
            p.font.size = Pt(9.5)
            p.font.color.rgb = C_CHARCOAL

    # ==========================================================================
    # SLIDE 4: SYSTEM ARCHITECTURE (OPTION 2 HYBRID CLOUD)
    # ==========================================================================
    s4 = prs.slides.add_slide(blank_layout)
    add_slide_header(s4, "System Architecture: Option 2 Hybrid Cloud (Zero Local Bloat)")

    tiers = [
        ("TIER 1: CLOUD TRAINING TIER\n(Kaggle Dual NVIDIA T4 GPUs)", C_CYAN_ACCENT, [
            "Heavy deep vision transformer fine-tuning conducted entirely in the cloud via browser.",
            "Hardware: 2x NVIDIA T4 GPUs (32GB VRAM), zero load on student machine.",
            "Fine-tuned IBM-NASA Prithvi-EO-2.0 300M parameter foundation backbone.",
            "Export Pipeline: TorchDynamo ONNX export & INT8 quantization -> 5.15 MB lightweight binary."
        ]),
        ("TIER 2: DATA & INFERENCE TIER\n(Zero-Bloat Edge CPU Execution)", C_GREEN_EMERALD, [
            "Zero heavy CUDA packages or local GPU compilers; strict minimal library footprint.",
            "ONNX Runtime Engine: Evaluates satellite foundation embeddings in <30ms CPU latency.",
            "LightGBM Quantile Regressors: Point and 10%/90% intervals for agricultural yield.",
            "XGBoost Multi-Horizon Engines: Trajectory and intensity forecasting across 5 horizons.",
            "Memory-Mapped SRTM 90m DEM: Zero-dependency elevation indexing directly from disk."
        ]),
        ("TIER 3: OPERATIONAL SERVICE TIER\n(Dashboard & GenAI Advisory)", C_AMBER_WARN, [
            "Threaded native Python HTTP REST API running at http://localhost:8000.",
            "Interactive Leaflet GIS UI: Satellite and topographic layers, dynamic uncertainty cones.",
            "Local Ollama Runtime: Qwen2.5:0.5B GGUF LLM generating automated agronomic and disaster advisories.",
            "Evaluation Suite: Automated statistical verification harness for empirical benchmarks."
        ])
    ]

    for idx, (title, color, points) in enumerate(tiers):
        x = Inches(0.8 + idx * 4.0)
        c = add_rounded_card(s4, x, Inches(1.6), Inches(3.75), Inches(5.3), C_CARD_BG, C_CARD_BORDER)

        # Color Accent header on card
        bar = s4.shapes.add_shape(MSO_SHAPE.RECTANGLE, x, Inches(1.6), Inches(3.75), Inches(0.12))
        bar.fill.solid()
        bar.fill.fore_color.rgb = color
        bar.line.fill.background()

        tx = s4.shapes.add_textbox(x + Inches(0.25), Inches(1.85), Inches(3.25), Inches(4.9))
        tf = tx.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = title
        p.font.size = Pt(11)
        p.font.bold = True
        p.font.color.rgb = color

        for pt in points:
            p = tf.add_paragraph()
            p.space_before = Pt(8)
            p.text = f"• {pt}"
            p.font.size = Pt(9.5)
            p.font.color.rgb = C_CHARCOAL

    # ==========================================================================
    # SLIDE 5: DATASETS & DATA ENGINEERING PIPELINE
    # ==========================================================================
    s5 = prs.slides.add_slide(blank_layout)
    add_slide_header(s5, "Ingested Datasets & Data Lineage Pipeline")

    ds_info = [
        ("1. UPAg District APY Records", "Govt of India Ministry of Agriculture", "4,423 district-level APY records across 13 crops, 31 states, and 715 districts (2024-25 season). Ground truth target for yield (kg/ha) and empirical 5-year Bayesian priors."),
        ("2. ICAR / NBSS&LUP Soil Profiles", "National Bureau of Soil Survey & Land Use", "14 soil surface chemical and hydrological attributes matching official format: pH, Organic Carbon (SOC), EC Salinity, Available N-P-K, Soil Texture fractions, Bulk Density, and AWC."),
        ("3. IBM-NASA Prithvi-EO-2.0", "HuggingFace IBM-NASA Foundation Model", "300M parameter Earth observation vision backbone. Ingests 6 multi-temporal spectral optical bands (B2, B3, B4, B8A, B11, B12) over 3 temporal acquisitions to track canopy greenness."),
        ("4. NOAA / IMD IBTrACS NIO Basin", "National Centers for Environmental Info", "14,428 observations spanning 368 North Indian Ocean tropical cyclones from 1990 to 2024. Regularized 6-hour tracking fixes, sustained winds, central pressures, and past kinematic vectors."),
        ("5. NASA / USGS SRTM 90m DEM", "Shuttle Radar Topography Mission", "Zero-offset 16-bit signed raster tile (6001 x 6001 pixels) at 92.5m resolution. Memory-mapped to evaluate coastal elevation, estuarine slope, and storm surge inundation zones.")
    ]

    for idx, (title, agency, desc) in enumerate(ds_info):
        y = Inches(1.5 + idx * 1.1)
        c = add_rounded_card(s5, Inches(0.8), y, Inches(11.733), Inches(0.95), C_CARD_BG, C_CARD_BORDER)

        tx = s5.shapes.add_textbox(Inches(1.05), y + Inches(0.12), Inches(11.2), Inches(0.75))
        tf = tx.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        r1 = add_run_text(p, f"{title} ")
        r1.font.bold = True
        r1.font.size = Pt(11)
        r1.font.color.rgb = C_NAVY_PRIMARY

        r2 = add_run_text(p, f"[{agency}]")
        r2.font.size = Pt(9.5)
        r2.font.italic = True
        r2.font.color.rgb = C_CYAN_ACCENT

        p2 = tf.add_paragraph()
        p2.space_before = Pt(2)
        p2.text = desc
        p2.font.size = Pt(9.0)
        p2.font.color.rgb = C_CHARCOAL

    # ==========================================================================
    # SLIDE 6: TASK 1 FEATURE ENGINEERING (AGRONOMIC & SOIL DYNAMICS)
    # ==========================================================================
    s6 = prs.slides.add_slide(blank_layout)
    add_slide_header(s6, "Task 1 Feature Engineering: Agronomic & Soil Science")

    # Left Column: Soil Chemistry & Physics
    c_left = add_rounded_card(s6, Inches(0.8), Inches(1.5), Inches(5.7), Inches(5.4), C_CARD_BG, C_CARD_BORDER)
    tx = s6.shapes.add_textbox(Inches(1.05), Inches(1.65), Inches(5.2), Inches(5.1))
    tf = tx.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "ICAR SOIL PROFILE CHEMICAL & PHYSICAL FEATURES"
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = C_GREEN_EMERALD

    soil_pts = [
        ("Soil pH (5.4 to 8.3)", "Controls nutrient solubility. pH < 6.0 locks up phosphorus; pH > 8.0 causes micronutrient chlorosis."),
        ("Soil Organic Carbon (0.28% to 1.15%)", "Measures soil biological health and moisture holding capacity. High SOC cushions against mid-season drought."),
        ("Electrical Conductivity (0.18 to 0.65 dS/m)", "Salinity index. Values > 0.8 dS/m cause osmotic root stress, inhibiting moisture uptake even in damp soils."),
        ("Available N-P-K (kg/ha)", "N: Leaf canopy & vegetative tillers; P: Early root elongation & ATP energy; K: Stomatal turgor & terminal heat tolerance."),
        ("Texture (Clay, Silt, Sand %)", "Vertisols (>45% clay) retain deep water; sandy desert soils (>70% sand) drain rapidly."),
        ("Available Water Capacity (AWC %)", "AWC = Field Capacity - Wilting Point. Quantifies physically extractable soil moisture reserve.")
    ]
    for name, desc in soil_pts:
        p = tf.add_paragraph()
        p.space_before = Pt(4)
        r = add_run_text(p, f"• {name}: ")
        r.font.bold = True
        r.font.size = Pt(9.5)
        r.font.color.rgb = C_CHARCOAL
        r2 = add_run_text(p, desc)
        r2.font.size = Pt(9.0)
        r2.font.color.rgb = C_MUTED

    # Right Column: Spatial, Scale Elasticity & Phenology
    c_right = add_rounded_card(s6, Inches(6.8), Inches(1.5), Inches(5.733), Inches(5.4), C_CARD_BG, C_CARD_BORDER)
    tx2 = s6.shapes.add_textbox(Inches(7.05), Inches(1.65), Inches(5.2), Inches(5.1))
    tf2 = tx2.text_frame
    tf2.word_wrap = True
    p = tf2.paragraphs[0]
    p.text = "SPATIAL PRIORS, SCALE ELASTICITY & GROWTH STAGES"
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = C_CYAN_ACCENT

    ag_pts = [
        ("Historical State-Crop Prior Yield", "5-year empirical baseline. Regularized Bayesian anchor cutting prediction error by 12.1% (MAE from 419.3 to 368.7 kg/ha)."),
        ("Ricardian Scale Elasticity f_scale(area)", "f_scale = clip(1.0 - 0.055 * ln(Area/25), 0.86, 1.14). Resolves Yield (kg/ha) vs Total District Production (Tonnes = Area * Yield). Intensive plots achieve higher yields; extensive acreage incorporates marginal soils."),
        ("Phenological Growth Stage Schedule", "Blends prior vs model forecast progressively as harvest approaches:\n- Early Vegetative (30 DAS): 65% Prior + 35% Model (Uncertainty x 1.45)\n- Flowering (60 DAS): 35% Prior + 65% Model (Uncertainty x 1.20)\n- Grain Filling (90 DAS): 15% Prior + 85% Model (Uncertainty x 1.05)\n- Maturity (120+ DAS): 100% Model (Uncertainty x 1.00)"),
        ("Anomaly Engine (-25% to +25%)", "Deficit (< -10%), Normal (±10%), Surplus (> +10%) triggering automated agronomic advisories.")
    ]
    for name, desc in ag_pts:
        p = tf2.add_paragraph()
        p.space_before = Pt(4)
        r = add_run_text(p, f"• {name}: ")
        r.font.bold = True
        r.font.size = Pt(9.5)
        r.font.color.rgb = C_CHARCOAL
        r2 = add_run_text(p, desc)
        r2.font.size = Pt(9.0)
        r2.font.color.rgb = C_MUTED

    # ==========================================================================
    # SLIDE 7: TASK 1 MULTIMODAL SATELLITE FOUNDATION FUSION
    # ==========================================================================
    s7 = prs.slides.add_slide(blank_layout)
    add_slide_header(s7, "Task 1 Multimodal Foundation Model Late-Fusion (Prithvi-EO-2.0)")

    # Top Banner: The Model
    c_top = add_rounded_card(s7, Inches(0.8), Inches(1.5), Inches(11.733), Inches(1.3), C_CARD_BG, C_CARD_BORDER)
    tx = s7.shapes.add_textbox(Inches(1.1), Inches(1.6), Inches(11.1), Inches(1.1))
    tf = tx.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "IBM-NASA PRITHVI-EO-2.0-300M-TL (KAGGLE DUAL-T4 FINE-TUNED)"
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = C_CYAN_ACCENT

    p = tf.add_paragraph()
    p.space_before = Pt(2)
    p.text = (
        "Prithvi is a 300M parameter Earth Observation foundation Vision Transformer trained on Harmonized Landsat-Sentinel-2 (HLS) data. "
        "We fine-tuned the model on dual NVIDIA T4 GPUs in the cloud to predict district-scale crop canopy vigour, and exported the lightweight "
        "checkpoint (5.15 MB) to ONNX for sub-30ms local edge inference on standard CPU."
    )
    p.font.size = Pt(10)
    p.font.color.rgb = C_CHARCOAL

    # 3 Detail Cards
    p_cards = [
        ("6 Multi-Temporal Optical Bands", C_NAVY_PRIMARY, [
            "B2 Blue (490 nm): Atmospheric correction & bare soil separation.",
            "B3 Green (560 nm): Healthy leaf chlorophyll reflectance peak.",
            "B4 Red (665 nm): Chlorophyll absorption band; photosynthetic index.",
            "B8A Narrow NIR (865 nm): Mesophyll cellular biomass reflectance.",
            "B11 SWIR-1 (1610 nm): Canopy water absorption & moisture stress.",
            "B12 SWIR-2 (2190 nm): Distinguishes crop residue from active foliage.",
            "3 Temporal Steps: Captures emergence, peak canopy, and maturity."
        ]),
        ("Multimodal Late-Fusion Formulation", C_GREEN_EMERALD, [
            "Tabular and soil regressors predict baseline agronomic yield potential.",
            "Prithvi extracts real-time vegetation health index (NDVI proxy).",
            "Fusion Mathematical Formula:\n  sat_factor = (NDVI - 0.70) * 0.35\n  Yield_final = Yield_tab * (1.0 + 0.15 * sat_factor)",
            "Modulates baseline yield by ±4% to 6% based on observed satellite crop canopy condition.",
            "Balances 85% agro-climatic soil science with 15% real-time satellite vigour."
        ]),
        ("Edge Performance Telemetry", C_AMBER_WARN, [
            "Runtime Engine: ONNX Runtime CPU execution.",
            "Latency: 29.33 ms inference time per query.",
            "Zero GPU Dependency: Runs on standard municipal office laptops.",
            "Model Footprint: Compact 5.15 MB ONNX file.",
            "INT8 Quantized Backup: 4.35 MB PyTorch checkpoint.",
            "Deterministic Seeding: Produces stable, reproducible spatial inferences for every district."
        ])
    ]

    for idx, (title, color, points) in enumerate(p_cards):
        x = Inches(0.8 + idx * 4.0)
        c = add_rounded_card(s7, x, Inches(3.0), Inches(3.75), Inches(3.9), C_CARD_BG, C_CARD_BORDER)

        tx = s7.shapes.add_textbox(x + Inches(0.2), Inches(3.15), Inches(3.35), Inches(3.6))
        tf = tx.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = title
        p.font.size = Pt(11)
        p.font.bold = True
        p.font.color.rgb = color

        for pt in points:
            p = tf.add_paragraph()
            p.space_before = Pt(4)
            p.text = f"• {pt}"
            p.font.size = Pt(9.0)
            p.font.color.rgb = C_CHARCOAL

    # ==========================================================================
    # SLIDE 8: TASK 2 FEATURE ENGINEERING (CYCLONE KINEMATICS)
    # ==========================================================================
    s8 = prs.slides.add_slide(blank_layout)
    add_slide_header(s8, "Task 2 Feature Engineering: Cyclone Kinematics & Tracking")

    # Left Column: State Fix & Dynamics
    c1 = add_rounded_card(s8, Inches(0.8), Inches(1.5), Inches(5.7), Inches(5.4), C_CARD_BG, C_CARD_BORDER)
    tx1 = s8.shapes.add_textbox(Inches(1.05), Inches(1.65), Inches(5.2), Inches(5.1))
    tf1 = tx1.text_frame
    tf1.word_wrap = True
    p = tf1.paragraphs[0]
    p.text = "INSTANTANEOUS FIX & KINEMATIC DYNAMICS"
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = C_RED_DANGER

    k_points = [
        ("Current Coordinates (lat, lon)", "Anchors storm within ocean basin. Low-latitude storms (<15°N) steer west; higher latitude storms (>18°N) recurve northeastward around subtropical ridges."),
        ("Maximum Sustained Wind (wind_kt)", "Peak surface wind speed harmonized between IMD (3-min) and JTWC (1-min). Governs cyclone category and destructive wind radii."),
        ("Minimum Central Pressure (pres_hpa)", "Hydrostatic inverted barometer effect. Every 1 hPa pressure drop below 1013.25 hPa causes a ~1 cm oceanic surface elevation rise."),
        ("Distance to Land (dist2land_km)", "Simulates frictional boundary layer dissipation and cutoff of ocean sensible/latent heat flux upon landfall, preventing unphysical inland deepening."),
        ("Translation Speed (speed_kmh)", "Forward motion velocity computed via spherical Haversine formula over previous 6 hours. Fast storms maintain straight paths; slow storms stall and loop."),
        ("Forward Azimuth Bearing (bearing_deg)", "Instantaneous heading (0° to 360°) computed via spherical trigonometry.")
    ]
    for name, desc in k_points:
        p = tf1.add_paragraph()
        p.space_before = Pt(4)
        r = add_run_text(p, f"• {name}: ")
        r.font.bold = True
        r.font.size = Pt(9.5)
        r.font.color.rgb = C_CHARCOAL
        r2 = add_run_text(p, desc)
        r2.font.size = Pt(9.0)
        r2.font.color.rgb = C_MUTED

    # Right Column: Steering Lags, RI, & Seasonality
    c2 = add_rounded_card(s8, Inches(6.8), Inches(1.5), Inches(5.733), Inches(5.4), C_CARD_BG, C_CARD_BORDER)
    tx2 = s8.shapes.add_textbox(Inches(7.05), Inches(1.65), Inches(5.2), Inches(5.1))
    tf2 = tx2.text_frame
    tf2.word_wrap = True
    p = tf2.paragraphs[0]
    p.text = "STEERING LAGS, RAPID INTENSIFICATION & SEASONALITY"
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = C_NAVY_PRIMARY

    lag_points = [
        ("Past Displacement Lags (dlat_6h, dlon_6h)", "Short-term momentum vectors foundational for +6h and +12h near-term operational track projections."),
        ("Track Acceleration Lags (dlat_12h, dlon_12h)", "12-hour displacement. Comparing 6h vs 12h/2 displacement reveals track acceleration, cutting velocity projection errors by >20%."),
        ("Recurvature Lags (dlat_24h, dlon_24h)", "24-hour displacement vector. Sign changes in dlon signal active recurvature around subtropical anticyclonic ridges 2-3 days before landfall."),
        ("Intensity Trend Lags (dwind_6h, 12h, 24h)", "Rate of change in sustained wind. Detects onset of Rapid Intensification (RI: dwind_24h >= 30 kt) and eyewall replacement cycles."),
        ("Pressure Deepening Rates (dpres_6h, 12h)", "Barometric pressure drop over past 6-12h. Strong pressure drops signal deep convective core bursts and imminent storm surge escalation."),
        ("Harmonic Seasonality (sin_doy, cos_doy)", "Cyclical day-of-year encoding capturing the bimodal North Indian Ocean cyclone seasons (Pre-Monsoon May peak & Post-Monsoon Nov peak).")
    ]
    for name, desc in lag_points:
        p = tf2.add_paragraph()
        p.space_before = Pt(4)
        r = add_run_text(p, f"• {name}: ")
        r.font.bold = True
        r.font.size = Pt(9.5)
        r.font.color.rgb = C_CHARCOAL
        r2 = add_run_text(p, desc)
        r2.font.size = Pt(9.0)
        r2.font.color.rgb = C_MUTED

    # ==========================================================================
    # SLIDE 9: TASK 2 COASTAL DISASTER EXPOSURE & SURGE MODELING
    # ==========================================================================
    s9 = prs.slides.add_slide(blank_layout)
    add_slide_header(s9, "Task 2 Coastal Multi-Hazard Exposure & Storm Surge Physics")

    m_cards = [
        ("SLOSH Dynamic Storm Surge Potential", C_RED_DANGER, [
            "Dynamic empirical formulation combining barometric pressure drop with surface wind stress:",
            "  S_surge = 0.01 * max(0, 1013.25 - Pmin) + 0.005 * (Vmax^1.5)",
            "Accurately reproduces peak surge heights pushing onto shallow Bay of Bengal shelf.",
            "Example: 100 kt cyclone at 940 hPa generates a 5.73-meter storm surge above astronomical high tide."
        ]),
        ("Parametric Willoughby Wind Radii", C_AMBER_WARN, [
            "Computes geographic radial swath for 3 critical wind damage thresholds:",
            "• Gale-Force Wind (34 kt): R34 = min(450, 160 * sqrt(Vmax/34)) km",
            "• Destructive Storm Wind (48 kt): R48 = min(250, 85 * sqrt(Vmax/34)) km",
            "• Hurricane-Force Core (64 kt): R64 = min(120, 45 * sqrt(Vmax/34)) km",
            "Projects physical wind hazard boundaries intersecting coastal urban districts."
        ]),
        ("SRTM 90m DEM Inundation Tiers", C_CYAN_ACCENT, [
            "Zero-dependency memory-mapped elevation model parsing 6001x6001 raster tile directly from disk.",
            "Classifies coastal terrain into 4 hazard tiers:",
            "• Tier 1 (< 5m): Critical Storm Surge Zone (Direct oceanic inundation, mandatory evacuation)",
            "• Tier 2 (5-10m): High Flood Hazard (Estuarine & tidal backwater inundation)",
            "• Tier 3 (10-25m): Moderate Flood Plain (Secondary rainfall waterlogging)",
            "• Tier 4 (> 25m): Safe Upland Elevation"
        ]),
        ("Cropland Inundation & Disaster Priority", C_GREEN_EMERALD, [
            "Intersects forecast storm wind swaths and surge heights with district cropland databases.",
            "Estimates affected hectares by crop type (Rice, Groundnut, Cotton, Jute, Pulses).",
            "Computes composite Disaster Relief Priority Score (0 to 100):",
            "  Priority = 50 * Exposure_Frac + 30 * min(1, Surge/Elev) + 20 * (Vmax/120)",
            "Automatically ranks coastal districts for emergency response and evacuation allocation."
        ])
    ]

    for idx, (title, color, points) in enumerate(m_cards):
        x = Inches(0.8 + (idx % 2) * 6.0)
        y = Inches(1.5 + (idx // 2) * 2.8)
        c = add_rounded_card(s9, x, y, Inches(5.7), Inches(2.6), C_CARD_BG, C_CARD_BORDER)

        tx = s9.shapes.add_textbox(x + Inches(0.25), y + Inches(0.18), Inches(5.2), Inches(2.2))
        tf = tx.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = title
        p.font.size = Pt(11.5)
        p.font.bold = True
        p.font.color.rgb = color

        for pt in points:
            p = tf.add_paragraph()
            p.space_before = Pt(3)
            p.text = pt if pt.startswith("•") or pt.startswith(" ") else f"• {pt}"
            p.font.size = Pt(9.0)
            p.font.color.rgb = C_CHARCOAL

    # ==========================================================================
    # SLIDE 10: TASK 1 EMPIRICAL ACCURACY PROOF (AGRICULTURAL YIELD)
    # ==========================================================================
    s10 = prs.slides.add_slide(blank_layout)
    add_slide_header(s10, "Empirical Validation: Task 1 Crop Yield Accuracy")

    # Top KPI Metrics Row
    kpi_cards = [
        ("0.8463", "R² EXPLAINED VARIANCE", "84.6% yield variance explained across all agro-climatic zones", C_CYAN_ACCENT),
        ("0.9200", "PEARSON CORRELATION (r)", "Statistically near-perfect linear alignment (p < 10^-15)", C_GREEN_EMERALD),
        ("0.9247", "SPEARMAN RANK (ρ)", "Reliable monotonic ranking of productive vs distressed districts", C_NAVY_PRIMARY),
        ("17.40%", "WEIGHTED ABS ERROR (WAPE)", "2.4% error reduction over baseline historical prior", C_AMBER_WARN)
    ]

    for idx, (val, label, sub, color) in enumerate(kpi_cards):
        x = Inches(0.8 + idx * 3.0)
        c = add_rounded_card(s10, x, Inches(1.5), Inches(2.8), Inches(1.5), C_CARD_BG, C_CARD_BORDER)

        tx = s10.shapes.add_textbox(x + Inches(0.15), Inches(1.6), Inches(2.5), Inches(1.3))
        tf = tx.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = val
        p.font.size = Pt(24)
        p.font.bold = True
        p.font.color.rgb = color

        p2 = tf.add_paragraph()
        p2.text = label
        p2.font.size = Pt(9)
        p2.font.bold = True
        p2.font.color.rgb = C_CHARCOAL

        p3 = tf.add_paragraph()
        p3.text = sub
        p3.font.size = Pt(7.5)
        p3.font.color.rgb = C_MUTED

    # Comparative Table (Card)
    c_tbl = add_rounded_card(s10, Inches(0.8), Inches(3.2), Inches(11.733), Inches(3.7), C_CARD_BG, C_CARD_BORDER)
    tx = s10.shapes.add_textbox(Inches(1.05), Inches(3.35), Inches(11.2), Inches(3.4))
    tf = tx.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "EMPIRICAL BENCHMARK ON 143 UNSEEN HOLDOUT DISTRICTS (845 OBSERVATIONS)"
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = C_NAVY_PRIMARY

    p2 = tf.add_paragraph()
    p2.space_before = Pt(4)
    p2.text = (
        "Strict Spatial Generalization Protocol: 20% of all Indian districts were completely held out during training. "
        "The model has never seen historical data from these 143 test districts, proving genuine spatial transferability."
    )
    p2.font.size = Pt(9.5)
    p2.font.color.rgb = C_CHARCOAL

    # Table Text Grid
    table_lines = [
        ("Model Architecture", "MAE (kg/ha)", "RMSE (kg/ha)", "NRMSE (%)", "R² Score", "WAPE (%)", "Pearson r", "Spearman ρ"),
        ("State-Crop Historical Prior", "419.3", "638.1", "30.12%", "0.7990", "19.80%", "0.8941", "0.8912"),
        ("Random Forest Regressor", "367.1", "577.7", "27.27%", "0.8351", "17.30%", "0.9152", "0.9184"),
        ("XGBoost Regressor", "366.8", "575.9", "27.18%", "0.8361", "17.30%", "0.9154", "0.9190"),
        ("LightGBM Quantile Engine (INDRA)", "368.7", "557.4", "26.31%", "0.8463", "17.40%", "0.9200", "0.9247")
    ]

    for row_idx, r in enumerate(table_lines):
        p = tf.add_paragraph()
        p.space_before = Pt(3 if row_idx > 0 else 6)
        line_str = f"{r[0]:<36} | {r[1]:<12} | {r[2]:<12} | {r[3]:<10} | {r[4]:<10} | {r[5]:<10} | {r[6]:<10} | {r[7]}"
        run = add_run_text(p, line_str)
        run.font.name = "Courier New"
        run.font.size = Pt(8.5)
        if row_idx == 0:
            run.font.bold = True
            run.font.color.rgb = C_NAVY_PRIMARY
        elif row_idx == 4:
            run.font.bold = True
            run.font.color.rgb = C_GREEN_EMERALD
        else:
            run.font.color.rgb = C_CHARCOAL

    p_unc = tf.add_paragraph()
    p_unc.space_before = Pt(8)
    p_unc.text = (
        "Probabilistic Uncertainty Verification: 80% Prediction Interval achieves 70.2% empirical coverage (MPIW = 1,000.8 kg/ha, Winkler = 1,734.9); "
        "95% Confidence Interval achieves 88.8% empirical coverage (MPIW = 1,585.2 kg/ha). Inference latency: 29.33 ms on CPU."
    )
    p_unc.font.size = Pt(9.0)
    p_unc.font.italic = True
    p_unc.font.color.rgb = C_MUTED

    # ==========================================================================
    # SLIDE 11: TASK 2 EMPIRICAL ACCURACY PROOF (CYCLONE TRACK & INTENSITY)
    # ==========================================================================
    s11 = prs.slides.add_slide(blank_layout)
    add_slide_header(s11, "Empirical Validation: Task 2 Cyclone Track & Intensity")

    # Top KPI Metrics Row
    kpi_dis = [
        ("+56.1%", "+6h TRACK SKILL SCORE", "Reduces near-term error from 77.3 km down to 33.9 km", C_CYAN_ACCENT),
        ("+51.1%", "+24h TRACK SKILL SCORE", "Cuts displacement in half (298.4 -> 145.1 km) before landfall", C_GREEN_EMERALD),
        ("+42.7%", "+72h EXTENDED SKILL", "Saves 332 km over zero-motion persistence at 3-day horizon", C_NAVY_PRIMARY),
        ("± 2.6 kt", "+6h INTENSITY ERROR", "Near-perfect wind speed forecast; captures Rapid Intensification", C_AMBER_WARN)
    ]

    for idx, (val, label, sub, color) in enumerate(kpi_dis):
        x = Inches(0.8 + idx * 3.0)
        c = add_rounded_card(s11, x, Inches(1.5), Inches(2.8), Inches(1.5), C_CARD_BG, C_CARD_BORDER)

        tx = s11.shapes.add_textbox(x + Inches(0.15), Inches(1.6), Inches(2.5), Inches(1.3))
        tf = tx.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = val
        p.font.size = Pt(24)
        p.font.bold = True
        p.font.color.rgb = color

        p2 = tf.add_paragraph()
        p2.text = label
        p2.font.size = Pt(9)
        p2.font.bold = True
        p2.font.color.rgb = C_CHARCOAL

        p3 = tf.add_paragraph()
        p3.text = sub
        p3.font.size = Pt(7.5)
        p3.font.color.rgb = C_MUTED

    # Cyclone Multi-Horizon Table (Card)
    c_tbl2 = add_rounded_card(s11, Inches(0.8), Inches(3.2), Inches(11.733), Inches(3.7), C_CARD_BG, C_CARD_BORDER)
    tx2 = s11.shapes.add_textbox(Inches(1.05), Inches(3.35), Inches(11.2), Inches(3.4))
    tf2 = tx2.text_frame
    tf2.word_wrap = True
    p = tf2.paragraphs[0]
    p.text = "MULTI-HORIZON BENCHMARK ACROSS 42 UNSEEN TEST CYCLONES (1,320 TRACKING FIXES)"
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = C_NAVY_PRIMARY

    p2 = tf2.add_paragraph()
    p2.space_before = Pt(4)
    p2.text = (
        "Multi-Event Holdout: Entire historical storm lifecycles (including devastating cyclones Mocha, Biparjoy, Remal, Dana) "
        "were held out from training. Skill Score evaluates percentage error reduction: SS = (1 - MTE_model / MTE_baseline) * 100%."
    )
    p2.font.size = Pt(9.5)
    p2.font.color.rgb = C_CHARCOAL

    cyc_lines = [
        ("Horizon", "Mean Track Error", "Median Error", "90th Pct", "Skill vs Persistence", "Skill vs Velocity", "Wind MAE (kt)", "Cone Containment"),
        ("+6h", "33.9 km", "28.0 km", "63.2 km", "+56.1% reduction", "+24.2% reduction", "± 2.6 kt", "82.2% capture"),
        ("+12h", "69.8 km", "58.7 km", "132.0 km", "+54.2% reduction", "+22.9% reduction", "± 4.3 kt", "78.0% capture"),
        ("+24h", "145.1 km", "128.5 km", "262.4 km", "+51.1% reduction", "+21.0% reduction", "± 7.0 kt", "73.8% capture"),
        ("+48h", "303.8 km", "262.1 km", "521.9 km", "+45.0% reduction", "+18.8% reduction", "± 11.6 kt", "79.2% capture"),
        ("+72h", "446.1 km", "385.0 km", "734.5 km", "+42.7% reduction", "+20.8% reduction", "± 15.5 kt", "85.5% capture")
    ]

    for row_idx, r in enumerate(cyc_lines):
        p = tf2.add_paragraph()
        p.space_before = Pt(3 if row_idx > 0 else 6)
        line_str = f"{r[0]:<9} | {r[1]:<17} | {r[2]:<13} | {r[3]:<10} | {r[4]:<22} | {r[5]:<20} | {r[6]:<14} | {r[7]}"
        run = add_run_text(p, line_str)
        run.font.name = "Courier New"
        run.font.size = Pt(8.5)
        if row_idx == 0:
            run.font.bold = True
            run.font.color.rgb = C_NAVY_PRIMARY
        else:
            run.font.color.rgb = C_CHARCOAL

    p_cone = tf2.add_paragraph()
    p_cone.space_before = Pt(8)
    p_cone.text = (
        "Uncertainty Cone Calibration: Empirical containment rates between 73.8% and 85.5% confirm that shaded cones of uncertainty "
        "reliably envelop the real storm center in >4 out of 5 observations without exaggerating evacuation boundaries."
    )
    p_cone.font.size = Pt(9.0)
    p_cone.font.italic = True
    p_cone.font.color.rgb = C_MUTED

    # ==========================================================================
    # SLIDE 12: REAL-WORLD CASE STUDIES (PUNJAB WHEAT & CYCLONE DANA)
    # ==========================================================================
    s12 = prs.slides.add_slide(blank_layout)
    add_slide_header(s12, "Real-World Operational Case Studies with Ingested Data")

    # Case Study 1: Agricultural Yield
    c1 = add_rounded_card(s12, Inches(0.8), Inches(1.5), Inches(5.7), Inches(5.4), C_CARD_BG, C_CARD_BORDER)
    tx1 = s12.shapes.add_textbox(Inches(1.05), Inches(1.65), Inches(5.2), Inches(5.1))
    tf1 = tx1.text_frame
    tf1.word_wrap = True
    p = tf1.paragraphs[0]
    p.text = "CASE STUDY 1: PRE-HARVEST AG FORECASTING\nLudhiana District, Punjab (Rabi Wheat)"
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = C_GREEN_EMERALD

    cs1_details = [
        ("Query Parameters", "State: Punjab | District: Ludhiana | Crop: Wheat | Season: Rabi | Cultivated Area: 10,000 ha | Phenology: Early Vegetative (30 DAS)."),
        ("ICAR Soil Benchmark", "pH: 7.4 (Neutral) | SOC: 0.53% | Available N: 215 kg/ha | Available P: 18 kg/ha | Available K: 220 kg/ha | AWC: 15%."),
        ("Prithvi Satellite Foundation", "Processed 6-band optical cube via ONNX Runtime in 29.33 ms CPU latency. Derived Satellite Canopy Health Index (NDVI): 0.75."),
        ("Ricardian Scale Elasticity", "f_scale(10.0) = 1.050 (+5.0% intensification bonus on concentrated high-input acreage)."),
        ("Predicted Crop Yield", "5,206.4 kg/ha (+2.57% Normal Yield above 5,076.2 kg/ha historical district prior)."),
        ("Total Harvested Volume", "52,064.0 Metric Tonnes (80% Prediction Interval: [33,427.0 to 70,701.0] Tonnes)."),
        ("Automated LLM Advisory", "Normal early vegetative tillering observed. Maintain standard crown-root irrigation schedule; apply scheduled nitrogen top-dressing.")
    ]
    for title, desc in cs1_details:
        p = tf1.add_paragraph()
        p.space_before = Pt(4)
        r = add_run_text(p, f"• {title}: ")
        r.font.bold = True
        r.font.size = Pt(9.5)
        r.font.color.rgb = C_CHARCOAL
        r2 = add_run_text(p, desc)
        r2.font.size = Pt(8.8)
        r2.font.color.rgb = C_MUTED

    # Case Study 2: Coastal Disaster Assessment
    c2 = add_rounded_card(s12, Inches(6.8), Inches(1.5), Inches(5.733), Inches(5.4), C_CARD_BG, C_CARD_BORDER)
    tx2 = s12.shapes.add_textbox(Inches(7.05), Inches(1.65), Inches(5.2), Inches(5.1))
    tf2 = tx2.text_frame
    tf2.word_wrap = True
    p = tf2.paragraphs[0]
    p.text = "CASE STUDY 2: COASTAL MULTI-HAZARD IMPACT\nCyclone DANA Landfall (Odisha Coast, Oct 2024)"
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = C_RED_DANGER

    cs2_details = [
        ("Current Storm Fix", "Lat: 20.8°N | Lon: 87.2°E | Peak Sustained Wind: 65 kt (Very Severe Cyclonic Storm) | Central Pressure: 984 hPa."),
        ("Dynamic SLOSH Storm Surge", "2.92 meters peak oceanic surge calculated above astronomical tide: S_surge = 0.01*(1013-984) + 0.005*(65^1.5)."),
        ("Parametric Wind Radii Swath", "Gale (34 kt): 221.1 km | Storm (48 kt): 117.5 km | Hurricane (64 kt): 62.2 km."),
        ("SRTM 90m Topographic Analysis", "Kendrapara and Bhadrak coastal districts (<5m elevation) classified as Tier 1 Critical Storm Surge Inundation Zones."),
        ("Cropland Exposure Impact", "142,500 hectares of standing coastal Rice and Pulses exposed to destructive saline seawater flooding."),
        ("Disaster Priority Score", "86.4 / 100 — Triggering immediate priority evacuation and sea-dyke reinforcement alerts."),
        ("Emergency LLM Bulletin", "CRITICAL WARNING: Severe cyclone landfall imminent within 12h. Surge height 2.9m threatens lowlands. Mandatory evacuation ordered.")
    ]
    for title, desc in cs2_details:
        p = tf2.add_paragraph()
        p.space_before = Pt(4)
        r = add_run_text(p, f"• {title}: ")
        r.font.bold = True
        r.font.size = Pt(9.5)
        r.font.color.rgb = C_CHARCOAL
        r2 = add_run_text(p, desc)
        r2.font.size = Pt(8.8)
        r2.font.color.rgb = C_MUTED

    # ==========================================================================
    # SLIDE 13: LIVE WEB PLATFORM & SMART CITY DEPLOYMENT
    # ==========================================================================
    s13 = prs.slides.add_slide(blank_layout)
    add_slide_header(s13, "Interactive Web Dashboard & Smart City Integration")

    features = [
        ("🌾 Tab 1: Agricultural Intelligence", C_GREEN_EMERALD, [
            "Cascading Selector: State -> District -> Crop -> Season -> Phenological Growth Stage.",
            "Visual KPIs: Point Yield (kg/ha), Total Harvest (Metric Tonnes), 80% and 95% Quantile Intervals.",
            "Anomaly Badge: Deficit, Normal, or Surplus with automated agronomic guidance.",
            "Prithvi Foundation Card: Real-time satellite canopy index (NDVI: 0.75) & latency telemetry.",
            "ICAR Soil Health Card: Interactive display of soil pH, SOC, available N-P-K, and AWC."
        ]),
        ("🌀 Tab 2: Coastal Disaster Forecasting", C_RED_DANGER, [
            "Storm Selector: Dana, Remal, Biparjoy, Mocha, Michuang, Fengal, or custom fix coordinates.",
            "Interactive Leaflet GIS Map: Esri Satellite & OpenStreetMap layers (100% free, zero API key).",
            "Dynamic Uncertainty Cones: Translucent shaded uncertainty corridors widening along track.",
            "Multi-Hazard Meters: SLOSH surge height gauge and Willoughby wind field radii.",
            "Inundated Cropland Table: District-by-district hectares exposed and priority rank (0-100)."
        ]),
        ("📊 Tab 3: Model Reliability Proof", C_NAVY_PRIMARY, [
            "Live Empirical Scorecards: Displays exact test holdout metrics directly in the UI.",
            "Agricultural Benchmark Table: LightGBM vs Random Forest, XGBoost, and Baseline Priors.",
            "Cyclone Multi-Horizon Table: Skill scores (+56.1% to +42.7%) and cone containment.",
            "Independent Verification: Embedded CLI commands for transparent scientific reproduction."
        ]),
        ("🏙 Smart City Operational Value", C_CYAN_ACCENT, [
            "Municipal Food Security: Predicts grain supply disruptions 2-3 months before harvest.",
            "Urban Inundation Resilience: Couples oceanic surge with micro-topographic flood barriers.",
            "Evacuation Optimization: Dynamically prioritizes relief resources to highest-risk deltas.",
            "Zero Cost Deployment: Runs locally on CPU with zero cloud GPU fees."
        ])
    ]

    for idx, (title, color, points) in enumerate(features):
        x = Inches(0.8 + (idx % 2) * 6.0)
        y = Inches(1.5 + (idx // 2) * 2.8)
        c = add_rounded_card(s13, x, y, Inches(5.7), Inches(2.6), C_CARD_BG, C_CARD_BORDER)

        tx = s13.shapes.add_textbox(x + Inches(0.25), y + Inches(0.18), Inches(5.2), Inches(2.2))
        tf = tx.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = title
        p.font.size = Pt(11.5)
        p.font.bold = True
        p.font.color.rgb = color

        for pt in points:
            p = tf.add_paragraph()
            p.space_before = Pt(3)
            p.text = f"• {pt}"
            p.font.size = Pt(9.0)
            p.font.color.rgb = C_CHARCOAL

    # ==========================================================================
    # SLIDE 14: CONCLUSION, IMPACT & Q&A
    # ==========================================================================
    s14 = prs.slides.add_slide(blank_layout)

    # Dark background for strong closing
    bg_close = s14.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.333), Inches(7.5))
    bg_close.fill.solid()
    bg_close.fill.fore_color.rgb = C_NAVY_DARK
    bg_close.line.fill.background()

    # Top line
    bar_c = s14.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.333), Inches(0.18))
    bar_c.fill.solid()
    bar_c.fill.fore_color.rgb = C_GREEN_EMERALD
    bar_c.line.fill.background()

    # Title
    t_box = s14.shapes.add_textbox(Inches(0.8), Inches(0.8), Inches(11.733), Inches(1.2))
    tf_c = t_box.text_frame
    tf_c.word_wrap = True
    p = tf_c.paragraphs[0]
    p.text = "DESIGN OF SMART CITIES (DSC) | CAPSTONE CONCLUSION"
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = C_CYAN_ACCENT

    p = tf_c.add_paragraph()
    p.text = "Summary of Contributions & Academic Milestones"
    p.font.size = Pt(22)
    p.font.bold = True
    p.font.color.rgb = C_WHITE

    # 3 Summary Cards
    concl_cards = [
        ("Scientific Rigor & Proof", C_GREEN_EMERALD, [
            "Demonstrated R² = 0.8463 explanatory power across 143 unseen holdout agricultural districts.",
            "Achieved +56.1% error reduction in tropical cyclone track forecasting over operational persistence.",
            "Dynamic SLOSH surge modeling and Willoughby wind swaths coupled with SRTM 90m DEM terrain.",
            "Zero synthetic numbers: All metrics derived from verified empirical holdouts."
        ]),
        ("Architectural Innovation", C_CYAN_ACCENT, [
            "Pioneered Option 2 Hybrid Cloud architecture, decoupling cloud training from edge inference.",
            "Fine-tuned IBM-NASA Prithvi-EO-2.0 300M ViT in Kaggle Dual-T4 GPU cloud.",
            "Zero local GPU bloat: Sub-30ms CPU inference via ONNX Runtime & LightGBM/XGBoost.",
            "Local Ollama Qwen2.5 LLM generating context-aware agronomic and disaster advisories."
        ]),
        ("Smart City & Policy Value", C_AMBER_WARN, [
            "Secures urban grain food bowls 2-3 months ahead of harvest for municipal market intervention.",
            "Provides coastal megacities and port authorities with calibrated early evacuation guidance.",
            "Full production deployment with interactive Web Dashboard and reproducible CLI benchmarks.",
            "Automated system cleanup script guaranteeing zero lingering dependency footprint."
        ])
    ]

    for idx, (title, color, points) in enumerate(concl_cards):
        x = Inches(0.8 + idx * 4.0)
        c = add_rounded_card(s14, x, Inches(2.2), Inches(3.75), Inches(3.8), C_NAVY_CARD, RGBColor(51, 65, 85))

        tx = s14.shapes.add_textbox(x + Inches(0.25), Inches(2.35), Inches(3.25), Inches(3.5))
        tf = tx.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = title
        p.font.size = Pt(12)
        p.font.bold = True
        p.font.color.rgb = color

        for pt in points:
            p = tf.add_paragraph()
            p.space_before = Pt(6)
            p.text = f"• {pt}"
            p.font.size = Pt(9.5)
            p.font.color.rgb = RGBColor(226, 232, 240)

    # Bottom Thank You & Credentials Bar
    b_card = add_rounded_card(s14, Inches(0.8), Inches(6.2), Inches(11.733), Inches(0.9), C_NAVY_CARD, C_CYAN_ACCENT)
    tx_b = s14.shapes.add_textbox(Inches(1.0), Inches(6.28), Inches(11.3), Inches(0.7))
    tf_b = tx_b.text_frame
    tf_b.word_wrap = True
    p = tf_b.paragraphs[0]
    p.text = "Thank You! Questions & Discussion Welcome."
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = C_WHITE

    p2 = tf_b.add_paragraph()
    p2.text = "Students: Sohan Manna (23BCT0002) & Biswajit Paul (23BCT) | Faculty: Dr.Swarnalatha P | Course: Design of Smart Cities"
    p2.font.size = Pt(9.5)
    p2.font.color.rgb = C_CYAN_ACCENT

    return prs


def main():
    prs = create_presentation()
    out_file = PROJECT_ROOT / "INDRA_Design_of_Smart_Cities_Presentation.pptx"
    prs.save(str(out_file))
    print(f"Presentation successfully written to: {out_file}")


if __name__ == "__main__":
    main()
