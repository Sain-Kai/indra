"""Generates a modern, publication-grade project banner for INDRA.
Uses 2x supersampling with Lanczos antialiasing to produce crisp vector-like graphics,
emblem, typography, and badges matching the design inspiration from AgIsoStack, AgriStack, and MaqAgr.
"""
from __future__ import annotations

import math
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageFilter

PROJECT_ROOT = Path(__file__).resolve().parent

# Target Dimensions
WIDTH = 1600
HEIGHT = 560
SCALE = 2  # Render at 3200 x 1120 for ultra-crisp antialiased edges
W = WIDTH * SCALE
H = HEIGHT * SCALE

# Colors (RGBA)
BG_WHITE = (255, 255, 255, 255)
BG_TINT = (248, 250, 253, 255)
BORDER_COLOR = (226, 232, 240, 255)
NAVY_DEEP = (15, 30, 60, 255)
NAVY_SLATE = (30, 58, 100, 255)
TEXT_MUTED = (100, 116, 139, 255)
GREEN_EMERALD = (16, 185, 129, 255)
GREEN_DARK = (5, 150, 105, 255)
GREEN_LIGHT = (52, 211, 153, 255)
BLUE_CYAN = (14, 165, 233, 255)
BLUE_OCEAN = (2, 132, 199, 255)
AMBER_GOLD = (245, 158, 11, 255)

# Fonts
FONT_BOLD = "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"
FONT_REG = "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"
if not Path(FONT_BOLD).exists():
    FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    FONT_REG = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def draw_emblem(draw: ImageDraw.ImageDraw, cx: float, cy: float, radius: float):
    """Draws a circular tech-nature emblem combining cyclone swirl, crop furrow leaves, and radar rings."""
    # Outer decorative tech rings
    for r_offset, width, color in [
        (40, 3, (226, 232, 240, 255)),
        (25, 4, (203, 213, 225, 255)),
        (10, 6, (186, 230, 253, 255)),
    ]:
        r = radius + r_offset
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=color, width=width)

    # Compass tick marks around outer circle
    outer_r = radius + 32
    for deg in range(0, 360, 15):
        rad = math.radians(deg)
        tick_len = 16 if deg % 45 == 0 else 8
        x1 = cx + (outer_r - tick_len) * math.cos(rad)
        y1 = cy + (outer_r - tick_len) * math.sin(rad)
        x2 = cx + outer_r * math.cos(rad)
        y2 = cy + outer_r * math.sin(rad)
        draw.line([x1, y1, x2, y2], fill=(148, 163, 184, 255), width=3 if deg % 45 == 0 else 2)

    # Circular base background for emblem
    draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], fill=(240, 249, 255, 255), outline=BLUE_CYAN, width=8)

    # Agricultural green furrow arcs (bottom half of circle)
    # 4 sweeping curved fields representing crop rows
    for i, (arc_y, thickness, col) in enumerate([
        (cy + radius * 0.70, 26, GREEN_DARK),
        (cy + radius * 0.45, 24, GREEN_EMERALD),
        (cy + radius * 0.20, 20, GREEN_LIGHT),
        (cy - radius * 0.05, 16, (110, 231, 183, 255)),
    ]):
        bbox = [cx - radius * 0.92, arc_y - radius * 0.6, cx + radius * 0.92, arc_y + radius * 0.7]
        draw.arc(bbox, start=25, end=155, fill=col, width=thickness)

    # Cyclone Spiral Vortex (Aerodynamic counter-clockwise storm arm in cyan/blue)
    # Upper left to center spiral
    points_cyclone_1 = []
    points_cyclone_2 = []
    for t in range(0, 320, 4):
        rad = math.radians(t + 40)
        dist = radius * 0.85 * (1.0 - t / 360.0 * 0.75)
        px = cx + dist * math.cos(rad)
        py = cy - dist * math.sin(rad) * 0.85
        points_cyclone_1.append((px, py))

    for t in range(0, 280, 4):
        rad = math.radians(t + 220)
        dist = radius * 0.75 * (1.0 - t / 320.0 * 0.70)
        px = cx + dist * math.cos(rad)
        py = cy - dist * math.sin(rad) * 0.85
        points_cyclone_2.append((px, py))

    if len(points_cyclone_1) > 1:
        draw.line(points_cyclone_1, fill=BLUE_OCEAN, width=18, joint="curve")
    if len(points_cyclone_2) > 1:
        draw.line(points_cyclone_2, fill=BLUE_CYAN, width=14, joint="curve")

    # Central Botanical Leaf / Sprout (Rising from storm center toward top-right)
    # Elegant stylized leaf outline and vein
    leaf_pts = [
        (cx - 20, cy + 20),
        (cx - 50, cy - 60),
        (cx - 30, cy - 150),
        (cx + 30, cy - 220),
        (cx + 50, cy - 140),
        (cx + 30, cy - 50),
        (cx - 20, cy + 20),
    ]
    draw.polygon(leaf_pts, fill=GREEN_EMERALD, outline=GREEN_DARK)

    # Leaf central rib / vein
    draw.line([(cx - 15, cy + 15), (cx + 30, cy - 215)], fill=(255, 255, 255, 220), width=6)

    # Second smaller companion leaf
    leaf2_pts = [
        (cx + 10, cy - 30),
        (cx + 80, cy - 80),
        (cx + 120, cy - 130),
        (cx + 70, cy - 160),
        (cx + 30, cy - 100),
        (cx + 10, cy - 30),
    ]
    draw.polygon(leaf2_pts, fill=GREEN_LIGHT, outline=GREEN_EMERALD)

    # Cyclone Eye / Core Radar Center
    draw.ellipse([cx - 28, cy - 28, cx + 28, cy + 28], fill=NAVY_DEEP, outline=(255, 255, 255, 255), width=6)
    draw.ellipse([cx - 12, cy - 12, cx + 12, cy + 12], fill=BLUE_CYAN)

    # Digital telemetry circuit lines emanating to the right
    circ_y1 = cy - radius * 0.4
    circ_y2 = cy + radius * 0.4
    draw.line([(cx + radius + 10, circ_y1), (cx + radius + 70, circ_y1)], fill=BLUE_CYAN, width=5)
    draw.ellipse([cx + radius + 65, circ_y1 - 8, cx + radius + 81, circ_y1 + 8], fill=BLUE_CYAN)

    draw.line([(cx + radius + 10, circ_y2), (cx + radius + 90, circ_y2)], fill=GREEN_EMERALD, width=5)
    draw.ellipse([cx + radius + 85, circ_y2 - 8, cx + radius + 101, circ_y2 + 8], fill=GREEN_EMERALD)


def draw_badge(draw: ImageDraw.ImageDraw, x: float, y: float, text: str, font: ImageFont.FreeTypeFont,
               bg_col, text_col, border_col=None, pad_x=24, pad_y=12):
    """Draws a stylish rounded pill badge."""
    bbox = font.getbbox(text)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]

    w = tw + pad_x * 2
    h = th + pad_y * 2
    r = h // 2

    draw.rounded_rectangle([x, y, x + w, y + h], radius=r, fill=bg_col, outline=border_col, width=2 if border_col else 0)
    draw.text((x + pad_x, y + pad_y - bbox[1]), text, font=font, fill=text_col)
    return w


def generate_banner(output_path: Path):
    """Generates the master banner image."""
    img = Image.new("RGBA", (W, H), BG_WHITE)
    draw = ImageDraw.Draw(img)

    # Subtle horizontal top border accent
    draw.rectangle([0, 0, W, 14], fill=BLUE_CYAN)

    # Subtle right decorative pattern (faint data grid)
    for gx in range(int(W * 0.55), W, 80):
        draw.line([(gx, 0), (gx, H)], fill=(241, 245, 249, 255), width=2)
    for gy in range(40, H, 80):
        draw.line([(int(W * 0.55), gy), (W, gy)], fill=(241, 245, 249, 255), width=2)

    # Load typography fonts
    font_brand = ImageFont.truetype(FONT_BOLD, 170)
    font_tagline = ImageFont.truetype(FONT_BOLD, 46)
    font_sub = ImageFont.truetype(FONT_REG, 38)
    font_badge = ImageFont.truetype(FONT_BOLD, 28)
    font_badge_small = ImageFont.truetype(FONT_REG, 26)

    # Left: Draw the high-res emblem
    emblem_cx = int(320 * SCALE)
    emblem_cy = int(280 * SCALE)
    emblem_r = int(195 * SCALE)
    draw_emblem(draw, emblem_cx, emblem_cy, emblem_r)

    # Right: Text Content Anchor
    text_x = int(580 * SCALE)

    # 1. Category Pill Badge above title
    pill_text = "GEOSPATIAL AI FOR SMART CITIES & REGIONAL RESILIENCE"
    draw_badge(draw, text_x, 100, pill_text, font_badge_small, (238, 242, 255, 255), (79, 70, 229, 255), (199, 210, 254, 255), pad_x=28, pad_y=12)

    # 2. Main Brand Title: INDRA
    # Split color for branding: IND in Navy, RA in Emerald Green
    brand_y = 175
    draw.text((text_x, brand_y), "IND", font=font_brand, fill=NAVY_DEEP)
    bbox_ind = font_brand.getbbox("IND")
    ind_w = bbox_ind[2] - bbox_ind[0]

    draw.text((text_x + ind_w + 10, brand_y), "RA", font=font_brand, fill=GREEN_DARK)

    # Little leaf sprout accent above the letter "I" in INDRA
    leaf_accent = [
        (text_x + 12, brand_y + 10),
        (text_x + 35, brand_y - 30),
        (text_x + 75, brand_y - 25),
        (text_x + 50, brand_y + 5),
    ]
    draw.polygon(leaf_accent, fill=GREEN_EMERALD)

    # 3. Main Descriptive Name
    tag_y = 390
    draw.text((text_x, tag_y), "Indian National Disaster Resilience & Agricultural Intelligence", font=font_tagline, fill=NAVY_SLATE)

    # 4. Explanatory Scope
    sub_y = 465
    draw.text(
        (text_x, sub_y),
        "AI-Powered Pre-Harvest Crop Yield Forecasting & Coastal Multi-Hazard Cyclone Early Warning",
        font=font_sub,
        fill=TEXT_MUTED
    )

    # Thin Divider Line
    draw.line([(text_x, 545), (W - 80, 545)], fill=BORDER_COLOR, width=3)

    # 5. Row 1: KPI & Accuracy Highlights (Inspired by sample README)
    badges_y = 575
    bx = text_x
    kpi_items = [
        ("🌾 Crop Yield R² = 0.85 (MAE 368 kg/ha)", (236, 253, 245, 255), (6, 95, 70, 255), GREEN_EMERALD),
        ("🌀 Cyclone Track Skill +56.1% vs Persist", (240, 249, 255, 255), (7, 89, 133, 255), BLUE_CYAN),
        ("🌊 SLOSH Surge & 90m DEM", (241, 245, 249, 255), (30, 41, 59, 255), (148, 163, 184, 255)),
    ]
    for btext, bg, fg, border in kpi_items:
        bw = draw_badge(draw, bx, badges_y, btext, font_badge, bg, fg, border, pad_x=26, pad_y=14)
        bx += bw + 20

    # 6. Row 2: Architecture & Open Source Creds
    badges_y2 = 665
    bx2 = text_x
    tech_items = [
        ("IBM-NASA Prithvi-EO 300M ViT", (245, 243, 255, 255), (91, 33, 182, 255), (196, 181, 253, 255)),
        ("ONNX Runtime CPU <30ms", (254, 243, 199, 255), (146, 64, 14, 255), AMBER_GOLD),
        ("715 Districts & 7,516 km Coast", (240, 253, 244, 255), (22, 101, 52, 255), (134, 239, 172, 255)),
        ("Apache 2.0 Open Source", (248, 250, 252, 255), (51, 65, 85, 255), (203, 213, 225, 255))
    ]
    for btext, bg, fg, border in tech_items:
        bw = draw_badge(draw, bx2, badges_y2, btext, font_badge, bg, fg, border, pad_x=24, pad_y=14)
        bx2 += bw + 18

    # Subtle bottom border
    draw.rectangle([0, H - 12, W, H], fill=GREEN_DARK)

    # Downsample using high-quality Lanczos antialiasing
    final_img = img.resize((WIDTH, HEIGHT), resample=Image.Resampling.LANCZOS)

    # Save to disk
    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_img.save(str(output_path), "PNG", optimize=True)
    print(f"Master banner generated successfully: {output_path} ({WIDTH}x{HEIGHT})")
    return output_path


if __name__ == "__main__":
    out = PROJECT_ROOT / "assets" / "indra_banner.png"
    generate_banner(out)
    # Also save banner.png at root so it can be referenced simply
    generate_banner(PROJECT_ROOT / "banner.png")
