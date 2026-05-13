#!/usr/bin/env python3
"""Render the deterministic GitHub social preview image for Demian."""

from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = ROOT / "docs" / "assets" / "social_preview.png"
WIDTH = 1280
HEIGHT = 640


def font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def text_width(draw: ImageDraw.ImageDraw, text: str, text_font: ImageFont.ImageFont) -> int:
    left, _top, right, _bottom = draw.textbbox((0, 0), text, font=text_font)
    return right - left


def draw_line(
    draw: ImageDraw.ImageDraw,
    points: list[tuple[float, float]],
    *,
    fill: tuple[int, int, int],
    width: int,
) -> None:
    draw.line([(round(x), round(y)) for x, y in points], fill=fill, width=width, joint="curve")


def draw_anatomy_motif(draw: ImageDraw.ImageDraw) -> None:
    origin_x = 790
    origin_y = 122
    row_gap = 72
    box_w = 250
    box_h = 40
    channels = [
        ("fast", (20, 118, 112)),
        ("slow", (54, 92, 148)),
        ("control", (130, 91, 146)),
        ("message", (178, 94, 84)),
        ("carrier", (156, 132, 63)),
    ]
    release_x = 1076
    release_y = origin_y + 2 * row_gap

    for index, (name, color) in enumerate(channels):
        y = origin_y + index * row_gap
        draw.rounded_rectangle(
            (origin_x, y, origin_x + box_w, y + box_h),
            radius=6,
            outline=color,
            width=3,
            fill=(241, 243, 239),
        )
        draw.text((origin_x + 18, y + 8), name, fill=(35, 43, 45), font=font(22, bold=True))
        draw.line((origin_x - 54, y + box_h // 2, origin_x, y + box_h // 2), fill=(89, 107, 111), width=2)
        if index in (3, 4):
            draw.line(
                (origin_x + box_w, y + box_h // 2, release_x, release_y + 24),
                fill=(178, 94, 84),
                width=2,
            )

    draw.ellipse((release_x - 18, release_y + 6, release_x + 30, release_y + 54), outline=(178, 94, 84), width=4)
    draw.text((release_x + 44, release_y + 15), "rare release", fill=(80, 75, 70), font=font(18))


def draw_trace_panel(draw: ImageDraw.ImageDraw) -> None:
    panel = (750, 455, 1160, 548)
    draw.rounded_rectangle(panel, radius=6, outline=(180, 188, 184), width=1, fill=(248, 249, 246))
    colors = [(20, 118, 112), (54, 92, 148), (178, 94, 84)]
    for line_index, color in enumerate(colors):
        points = []
        for step in range(90):
            x = panel[0] + 20 + step * 4.35
            wave = math.sin(step / (7.5 + line_index * 1.7)) + 0.35 * math.sin(step / (2.7 + line_index))
            y = panel[1] + 47 + wave * (16 - line_index * 2) + line_index * 4
            points.append((x, y))
        draw_line(draw, points, fill=color, width=3)
    for x in range(panel[0] + 20, panel[2] - 18, 52):
        draw.line((x, panel[1] + 12, x, panel[3] - 10), fill=(226, 229, 224), width=1)


def render() -> None:
    image = Image.new("RGB", (WIDTH, HEIGHT), (236, 238, 234))
    draw = ImageDraw.Draw(image)

    draw.rectangle((0, 0, WIDTH, HEIGHT), fill=(236, 238, 234))
    draw.rectangle((0, 0, WIDTH, 640), outline=(207, 213, 207), width=18)
    draw.rectangle((0, 0, 42, HEIGHT), fill=(30, 67, 73))
    draw.rectangle((42, 0, 54, HEIGHT), fill=(178, 94, 84))

    label_font = font(24, bold=True)
    title_font = font(92, bold=True)
    subtitle_font = font(36)
    body_font = font(27)
    small_font = font(20)
    meta_font = font(18)

    draw.text((112, 110), "DEMIAN", fill=(29, 43, 45), font=title_font)
    draw.text((116, 220), "Gate-State Causal Propagation", fill=(30, 67, 73), font=subtitle_font)
    draw.text((116, 292), "Native mechanisms in structured", fill=(68, 77, 79), font=body_font)
    draw.text((116, 332), "recurrent substrates.", fill=(68, 77, 79), font=body_font)

    meta = "V9 5CH  /  CAPSULE CONTINUITY  /  DIAGNOSTICS"
    draw.text((116, 407), meta, fill=(85, 99, 101), font=small_font)
    draw.line((116, 442, 612, 442), fill=(188, 198, 194), width=2)

    draw.text((116, 492), "fixed-point surface behavior can hide", fill=(29, 43, 45), font=label_font)
    draw.text((116, 528), "structured internal channel dynamics", fill=(29, 43, 45), font=label_font)

    draw_anatomy_motif(draw)
    draw_trace_panel(draw)

    draw.line((688, 82, 688, 568), fill=(197, 203, 198), width=1)
    draw.text((750, 585), "deterministic preview | docs/assets/social_preview.png", fill=(91, 100, 101), font=meta_font)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    image.save(OUT_PATH, optimize=True)


if __name__ == "__main__":
    render()
