"""
Firestarter - UI theme: fonts, palette and vector icon drawing helpers (PIL).
"""

import math
import os
from functools import lru_cache
from typing import Tuple, List

from PIL import ImageFont, ImageDraw

FONT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "fonts")

PAL = {
    "panel": (14, 16, 15, 200),
    "panel_hi": (26, 29, 27, 225),
    "border": (255, 255, 255, 38),
    "text": (236, 232, 222, 255),
    "dim": (170, 168, 160, 255),
    "faint": (120, 120, 115, 255),
    "amber": (255, 176, 72, 255),
    "ember": (255, 112, 44, 255),
    "gold": (246, 206, 110, 255),
    "green": (120, 214, 140, 255),
    "blue": (128, 190, 255, 255),
    "red": (255, 96, 80, 255),
    "cyan": (110, 220, 230, 255),
}


@lru_cache(maxsize=64)
def font(name: str, size: int):
    files = {
        "regular": "Poppins-Regular.ttf", "medium": "Poppins-Medium.ttf", "bold": "Poppins-Bold.ttf",
        "light": "Poppins-Light.ttf", "serif": "Lora-Variable.ttf", "serif_italic": "Lora-Italic-Variable.ttf",
        "serif_bold": "Lora-Variable.ttf", "mono": "DejaVuSansMono.ttf",
    }
    path = os.path.join(FONT_DIR, files.get(name, "Poppins-Regular.ttf"))
    try:
        f = ImageFont.truetype(path, max(6, int(size)))
        if name == "serif_bold":
            try:
                f.set_variation_by_name("Bold")
            except Exception:
                pass
        return f
    except Exception:
        try:
            return ImageFont.load_default(size=max(6, int(size)))
        except TypeError:
            return ImageFont.load_default()


def text_w(draw: ImageDraw.ImageDraw, s: str, f) -> float:
    return draw.textlength(s, font=f)


def wrap(draw: ImageDraw.ImageDraw, text: str, f, width: float) -> List[str]:
    out = []
    for para in text.split("\n"):
        words = para.split(" ")
        line = ""
        for w in words:
            t = (line + " " + w).strip()
            if text_w(draw, t, f) <= width or not line:
                line = t
            else:
                out.append(line)
                line = w
        out.append(line)
    return out


def panel(draw, box, radius=10, fill=None, outline=None, width=1):
    draw.rounded_rectangle(box, radius=radius, fill=fill or PAL["panel"], outline=outline or PAL["border"], width=width)


def comp_shape(img, box, fill, radius=0, outline=None, width=1, kind="rounded"):
    """Alpha-composite a translucent shape over existing (possibly opaque) UI pixels.
    (PIL's ImageDraw overwrites alpha instead of blending.)"""
    from PIL import Image
    x0, y0, x1, y1 = [int(round(v)) for v in box]
    w, h = max(1, x1 - x0 + 1), max(1, y1 - y0 + 1)
    layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    ld = ImageDraw.Draw(layer)
    if kind == "ellipse":
        ld.ellipse([0, 0, w - 1, h - 1], fill=fill, outline=outline, width=width)
    elif kind == "rect" or radius <= 0:
        ld.rectangle([0, 0, w - 1, h - 1], fill=fill, outline=outline, width=width)
    else:
        ld.rounded_rectangle([0, 0, w - 1, h - 1], radius=radius, fill=fill, outline=outline, width=width)
    img.alpha_composite(layer, (x0, y0))


def key_cap(draw, x, y, label, f, fg=(20, 20, 20, 255), bg=(236, 232, 222, 235)) -> float:
    w = text_w(draw, label, f) + 10
    h = f.size + 6
    draw.rounded_rectangle([x, y, x + w, y + h], radius=4, fill=bg)
    draw.text((x + 5, y + 2), label, font=f, fill=fg)
    return w


# ------------------------------------------------------------------------------------ icons
def icon(draw: ImageDraw.ImageDraw, name: str, cx: float, cy: float, s: float, col=(236, 232, 222, 255)):
    """Minimal vector icons centred at (cx, cy), size s."""
    w = max(1, int(s * 0.09))
    if name == "lighter":
        draw.rounded_rectangle([cx - s * 0.18, cy - s * 0.15, cx + s * 0.18, cy + s * 0.42], radius=int(s * 0.06), fill=(220, 70, 50, 255))
        draw.rectangle([cx - s * 0.17, cy - s * 0.32, cx + s * 0.17, cy - s * 0.15], fill=(190, 190, 195, 255))
        draw.polygon([(cx, cy - s * 0.62), (cx - s * 0.1, cy - s * 0.38), (cx + s * 0.1, cy - s * 0.38)], fill=(255, 190, 80, 255))
    elif name == "knife":
        draw.polygon([(cx - s * 0.05, cy - s * 0.45), (cx + s * 0.12, cy + s * 0.02), (cx - s * 0.05, cy + s * 0.02)], fill=(210, 215, 222, 255))
        draw.rounded_rectangle([cx - s * 0.07, cy + s * 0.02, cx + s * 0.07, cy + s * 0.45], radius=int(s * 0.04), fill=(120, 80, 50, 255))
    elif name == "axe":
        draw.line([(cx - s * 0.3, cy + s * 0.42), (cx + s * 0.15, cy - s * 0.38)], fill=(160, 115, 70, 255), width=max(2, int(s * 0.1)))
        draw.polygon([(cx + s * 0.05, cy - s * 0.45), (cx + s * 0.42, cy - s * 0.3), (cx + s * 0.32, cy + s * 0.02), (cx + s * 0.02, cy - s * 0.18)], fill=(170, 175, 182, 255))
    elif name == "bow":
        draw.arc([cx - s * 0.45, cy - s * 0.25, cx + s * 0.45, cy + s * 0.55], 200, 340, fill=(170, 125, 80, 255), width=max(2, int(s * 0.08)))
        draw.line([(cx - s * 0.41, cy + s * 0.08), (cx + s * 0.41, cy + s * 0.08)], fill=(230, 210, 170, 255), width=1)
        draw.line([(cx, cy - s * 0.45), (cx, cy + s * 0.42)], fill=(215, 180, 130, 255), width=max(2, int(s * 0.09)))
    elif name == "gas":
        draw.rounded_rectangle([cx - s * 0.3, cy - s * 0.2, cx + s * 0.3, cy + s * 0.45], radius=int(s * 0.06), fill=(210, 50, 40, 255))
        draw.line([(cx + s * 0.15, cy - s * 0.2), (cx + s * 0.35, cy - s * 0.45)], fill=(40, 40, 40, 255), width=max(2, int(s * 0.08)))
        draw.rectangle([cx - s * 0.22, cy - s * 0.3, cx + s * 0.05, cy - s * 0.2], fill=(170, 40, 35, 255))
    elif name == "tinder":
        for i in range(7):
            a = i / 7 * math.pi
            draw.arc([cx - s * 0.42, cy - s * 0.2 + i % 2 * 3, cx + s * 0.42, cy + s * 0.42], 180 + i * 4, 360 - i * 4, fill=(220, 170, 110, 255), width=max(1, int(s * 0.05)))
        draw.ellipse([cx - s * 0.12, cy, cx + s * 0.12, cy + s * 0.16], fill=(255, 120, 50, 255))
    elif name == "kindling":
        for i, off in enumerate((-0.2, 0.0, 0.2)):
            draw.line([(cx - s * 0.42, cy + s * (0.3 + off * 0.3)), (cx + s * 0.42, cy + s * (-0.3 + off * 0.5))], fill=(190, 145, 95, 255), width=max(2, int(s * 0.07)))
    elif name == "firewood":
        draw.pieslice([cx - s * 0.4, cy - s * 0.35, cx + s * 0.4, cy + s * 0.45], 180, 360, fill=(150, 110, 70, 255))
        draw.pieslice([cx - s * 0.32, cy - s * 0.27, cx + s * 0.32, cy + s * 0.37], 180, 360, fill=(225, 185, 130, 255))
    elif name == "log":
        draw.ellipse([cx - s * 0.38, cy - s * 0.38, cx + s * 0.38, cy + s * 0.38], fill=(120, 85, 55, 255))
        draw.ellipse([cx - s * 0.3, cy - s * 0.3, cx + s * 0.3, cy + s * 0.3], fill=(220, 180, 125, 255))
        for r in (0.2, 0.11):
            draw.ellipse([cx - s * r, cy - s * r, cx + s * r, cy + s * r], outline=(170, 125, 80, 255), width=1)
    elif name == "fungus":
        draw.pieslice([cx - s * 0.42, cy - s * 0.3, cx + s * 0.42, cy + s * 0.3], 180, 360, fill=(170, 135, 95, 255))
        draw.rectangle([cx - s * 0.42, cy - s * 0.02, cx + s * 0.42, cy + s * 0.06], fill=(225, 210, 180, 255))
    elif name == "sun":
        draw.ellipse([cx - s * 0.22, cy - s * 0.22, cx + s * 0.22, cy + s * 0.22], fill=(255, 206, 90, 255))
        for i in range(8):
            a = i * math.pi / 4
            draw.line([(cx + math.cos(a) * s * 0.32, cy + math.sin(a) * s * 0.32), (cx + math.cos(a) * s * 0.45, cy + math.sin(a) * s * 0.45)], fill=(255, 206, 90, 255), width=max(1, w))
    elif name == "moon":
        draw.ellipse([cx - s * 0.3, cy - s * 0.3, cx + s * 0.3, cy + s * 0.3], fill=(230, 232, 240, 255))
        draw.ellipse([cx - s * 0.15, cy - s * 0.38, cx + s * 0.45, cy + s * 0.22], fill=(22, 26, 30, 255))
    elif name == "cloud":
        c = (205, 210, 218, 255)
        draw.ellipse([cx - s * 0.42, cy - s * 0.05, cx - s * 0.02, cy + s * 0.3], fill=c)
        draw.ellipse([cx - s * 0.2, cy - s * 0.28, cx + s * 0.24, cy + s * 0.2], fill=c)
        draw.ellipse([cx + s * 0.02, cy - s * 0.08, cx + s * 0.42, cy + s * 0.3], fill=c)
    elif name == "rain":
        icon(draw, "cloud", cx, cy - s * 0.12, s * 0.9)
        for i in range(3):
            x = cx - s * 0.2 + i * s * 0.2
            draw.line([(x, cy + s * 0.22), (x - s * 0.06, cy + s * 0.42)], fill=(120, 180, 255, 255), width=max(1, w))
    elif name == "snow":
        icon(draw, "cloud", cx, cy - s * 0.12, s * 0.9)
        for i in range(3):
            x = cx - s * 0.2 + i * s * 0.2
            draw.ellipse([x - 2, cy + s * 0.28, x + 2, cy + s * 0.32 + 2], fill=(240, 245, 255, 255))
    elif name == "credits":
        draw.polygon([(cx, cy - s * 0.42), (cx + s * 0.35, cy), (cx, cy + s * 0.42), (cx - s * 0.35, cy)], fill=(246, 206, 110, 255))
        draw.polygon([(cx, cy - s * 0.2), (cx + s * 0.16, cy), (cx, cy + s * 0.2), (cx - s * 0.16, cy)], fill=(255, 236, 170, 255))
    elif name == "flame":
        draw.polygon([(cx, cy - s * 0.45), (cx + s * 0.3, cy + s * 0.1), (cx + s * 0.18, cy + s * 0.4), (cx - s * 0.18, cy + s * 0.4), (cx - s * 0.3, cy + s * 0.1)], fill=(255, 130, 40, 255))
        draw.polygon([(cx, cy - s * 0.1), (cx + s * 0.14, cy + s * 0.2), (cx, cy + s * 0.38), (cx - s * 0.14, cy + s * 0.2)], fill=(255, 220, 120, 255))
    elif name == "store":
        draw.rounded_rectangle([cx - s * 0.36, cy - s * 0.15, cx + s * 0.36, cy + s * 0.4], radius=int(s * 0.08), fill=col)
        draw.arc([cx - s * 0.18, cy - s * 0.42, cx + s * 0.18, cy - s * 0.02], 180, 360, fill=col, width=max(2, int(s * 0.08)))
    elif name == "map":
        draw.polygon([(cx - s * 0.4, cy - s * 0.3), (cx - s * 0.13, cy - s * 0.4), (cx + s * 0.13, cy - s * 0.3), (cx + s * 0.4, cy - s * 0.4),
                      (cx + s * 0.4, cy + s * 0.3), (cx + s * 0.13, cy + s * 0.4), (cx - s * 0.13, cy + s * 0.3), (cx - s * 0.4, cy + s * 0.4)], fill=col)
        draw.line([(cx - s * 0.13, cy - s * 0.4), (cx - s * 0.13, cy + s * 0.3)], fill=(0, 0, 0, 90), width=1)
        draw.line([(cx + s * 0.13, cy - s * 0.3), (cx + s * 0.13, cy + s * 0.4)], fill=(0, 0, 0, 90), width=1)
    elif name == "tarp":
        draw.polygon([(cx - s * 0.42, cy + s * 0.3), (cx, cy - s * 0.32), (cx + s * 0.42, cy + s * 0.3)], fill=(60, 120, 210, 255))
    elif name == "cord":
        for i in range(4):
            draw.arc([cx - s * 0.35 + i * s * 0.15, cy - s * 0.2, cx - s * 0.05 + i * s * 0.15, cy + s * 0.2], 0, 360, fill=(225, 200, 150, 255), width=max(1, w))
    elif name == "book":
        draw.rounded_rectangle([cx - s * 0.34, cy - s * 0.4, cx + s * 0.34, cy + s * 0.4], radius=int(s * 0.05), fill=(140, 80, 45, 255))
        draw.rectangle([cx - s * 0.26, cy - s * 0.32, cx + s * 0.3, cy + s * 0.32], fill=(235, 222, 190, 255))
        draw.line([(cx - s * 0.18, cy - s * 0.12), (cx + s * 0.2, cy - s * 0.12)], fill=(120, 90, 60, 255), width=1)
        draw.line([(cx - s * 0.18, cy + s * 0.02), (cx + s * 0.2, cy + s * 0.02)], fill=(120, 90, 60, 255), width=1)
    else:  # box
        draw.rectangle([cx - s * 0.3, cy - s * 0.3, cx + s * 0.3, cy + s * 0.3], outline=col, width=max(1, w))


TOOL_ICONS = {"pocket_lighter": "lighter", "pocket_knife": "knife", "felling_axe": "axe",
              "bow_drill_spindle": "bow", "accelerant": "gas"}


def lerp_col(a, b, t):
    t = max(0.0, min(1.0, t))
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(len(a)))


def temp_color(t_c: float):
    """Thermometer gradient: cold blue -> steam white -> smoke yellow -> flame orange -> glowing red/white."""
    stops = [(0, (90, 150, 255, 255)), (100, (200, 220, 235, 255)), (250, (240, 200, 90, 255)),
             (450, (255, 140, 40, 255)), (700, (255, 80, 30, 255)), (1000, (255, 235, 190, 255))]
    for i in range(len(stops) - 1):
        a, b = stops[i], stops[i + 1]
        if t_c <= b[0]:
            return lerp_col(a[1], b[1], (t_c - a[0]) / max(1e-6, b[0] - a[0]))
    return stops[-1][1]
