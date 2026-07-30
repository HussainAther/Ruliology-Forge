#!/usr/bin/env python3
"""Generate the animated README demonstration for Ruliology Forge.

Run from the repository root after installing the package and Pillow:

    python scripts/generate_readme_demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ruliology_forge import run_perturbation_experiment  # noqa: E402

OUTPUT = REPO_ROOT / "docs" / "assets" / "ruliology-forge-demo.gif"
POSTER = REPO_ROOT / "docs" / "assets" / "ruliology-forge-demo.png"

CANVAS_WIDTH = 1200
CANVAS_HEIGHT = 675
PANEL_WIDTH = 344
PANEL_HEIGHT = 430
PANEL_Y = 142
PANEL_XS = (48, 428, 808)

BACKGROUND = (12, 15, 24)
PANEL_BACKGROUND = (20, 25, 38)
BORDER = (63, 72, 92)
TEXT = (240, 243, 249)
MUTED = (165, 174, 194)
ACCENT = (81, 195, 255)
CONTROL_ON = (231, 236, 245)
PERTURBED_ON = (117, 214, 255)
DIFFERENCE_ON = (255, 104, 104)
INJURY = (255, 198, 73)


def font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
    ]
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


TITLE_FONT = font(34, bold=True)
SUBTITLE_FONT = font(18)
PANEL_TITLE_FONT = font(20, bold=True)
LABEL_FONT = font(15)
METRIC_FONT = font(18, bold=True)
SMALL_FONT = font(13)


def trajectory_image(array: np.ndarray, on_color: tuple[int, int, int]) -> Image.Image:
    """Convert a binary spacetime array into a nearest-neighbor RGB image."""

    rgb = np.empty((*array.shape, 3), dtype=np.uint8)
    rgb[:] = PANEL_BACKGROUND
    rgb[array.astype(bool)] = on_color
    return Image.fromarray(rgb, mode="RGB")


def draw_panel(
    canvas: Image.Image,
    *,
    x: int,
    title: str,
    subtitle: str,
    array: np.ndarray,
    on_color: tuple[int, int, int],
    current_step: int,
    perturb_time: int,
    perturb_center: int,
    perturb_radius: int,
) -> None:
    draw = ImageDraw.Draw(canvas)
    draw.rounded_rectangle(
        (x, PANEL_Y, x + PANEL_WIDTH, PANEL_Y + PANEL_HEIGHT),
        radius=18,
        fill=PANEL_BACKGROUND,
        outline=BORDER,
        width=2,
    )
    draw.text((x + 18, PANEL_Y + 15), title, font=PANEL_TITLE_FONT, fill=TEXT)
    draw.text((x + 18, PANEL_Y + 44), subtitle, font=SMALL_FONT, fill=MUTED)

    plot_x = x + 18
    plot_y = PANEL_Y + 74
    plot_w = PANEL_WIDTH - 36
    plot_h = PANEL_HEIGHT - 94

    visible = array[: current_step + 1]
    padded = np.zeros_like(array)
    padded[: visible.shape[0]] = visible
    rendered = trajectory_image(padded, on_color).resize((plot_w, plot_h), Image.Resampling.NEAREST)
    canvas.paste(rendered, (plot_x, plot_y))

    draw.rectangle((plot_x, plot_y, plot_x + plot_w, plot_y + plot_h), outline=BORDER, width=1)

    if current_step >= perturb_time:
        marker_y = plot_y + int((perturb_time / max(1, array.shape[0] - 1)) * (plot_h - 1))
        draw.line((plot_x, marker_y, plot_x + plot_w, marker_y), fill=INJURY, width=2)
        center_x = plot_x + int((perturb_center / max(1, array.shape[1] - 1)) * (plot_w - 1))
        radius_px = max(3, int((perturb_radius / array.shape[1]) * plot_w))
        draw.line((center_x - radius_px, marker_y, center_x + radius_px, marker_y), fill=(255, 255, 255), width=3)

    cursor_y = plot_y + int((current_step / max(1, array.shape[0] - 1)) * (plot_h - 1))
    draw.line((plot_x, cursor_y, plot_x + plot_w, cursor_y), fill=ACCENT, width=1)


def build_frame(result, current_step: int) -> Image.Image:
    canvas = Image.new("RGB", (CANVAS_WIDTH, CANVAS_HEIGHT), BACKGROUND)
    draw = ImageDraw.Draw(canvas)

    draw.text((48, 34), "Ruliology Forge", font=TITLE_FONT, fill=TEXT)
    draw.text(
        (48, 80),
        "Perturb a cellular automaton. Measure whether its computation heals, scars, or diverges.",
        font=SUBTITLE_FONT,
        fill=MUTED,
    )

    draw_panel(
        canvas,
        x=PANEL_XS[0],
        title="Control",
        subtitle="Rule 110 baseline evolution",
        array=result.control,
        on_color=CONTROL_ON,
        current_step=current_step,
        perturb_time=result.perturb_time,
        perturb_center=result.perturb_center,
        perturb_radius=result.perturb_radius,
    )
    draw_panel(
        canvas,
        x=PANEL_XS[1],
        title="Perturbed",
        subtitle="Localized bit flip at t = 55",
        array=result.perturbed,
        on_color=PERTURBED_ON,
        current_step=current_step,
        perturb_time=result.perturb_time,
        perturb_center=result.perturb_center,
        perturb_radius=result.perturb_radius,
    )
    draw_panel(
        canvas,
        x=PANEL_XS[2],
        title="Computational scar",
        subtitle="XOR difference from the control",
        array=result.difference,
        on_color=DIFFERENCE_ON,
        current_step=current_step,
        perturb_time=result.perturb_time,
        perturb_center=result.perturb_center,
        perturb_radius=result.perturb_radius,
    )

    status = "Baseline evolution"
    status_color = MUTED
    if current_step >= result.perturb_time:
        status = "Perturbation introduced"
        status_color = INJURY
    if current_step > result.perturb_time + 18:
        status = "Tracking recovery and scar propagation"
        status_color = ACCENT

    footer_y = 604
    draw.text((48, footer_y), status, font=METRIC_FONT, fill=status_color)
    draw.text((48, footer_y + 31), f"Generation {current_step:03d} / {result.control.shape[0] - 1}", font=LABEL_FONT, fill=MUTED)

    metric_x = 780
    draw.text((metric_x, footer_y), "Restoration coefficient", font=LABEL_FONT, fill=MUTED)
    draw.text(
        (metric_x, footer_y + 24),
        f"R = {result.restoration_coefficient:.3f}",
        font=METRIC_FONT,
        fill=TEXT,
    )
    draw.text((1010, footer_y), "Rule", font=LABEL_FONT, fill=MUTED)
    draw.text((1010, footer_y + 24), str(result.rule), font=METRIC_FONT, fill=TEXT)

    return canvas


def main() -> None:
    result = run_perturbation_experiment(
        110,
        width=201,
        steps=130,
        perturb_time=55,
        perturb_radius=5,
        perturbation="bit_flip",
        initial_condition="single",
        boundary="periodic",
        seed=110,
    )

    frame_steps = list(range(0, result.control.shape[0], 2))
    if frame_steps[-1] != result.control.shape[0] - 1:
        frame_steps.append(result.control.shape[0] - 1)
    frames = [build_frame(result, step) for step in frame_steps]

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    frames[-1].save(POSTER, optimize=True)
    frames[0].save(
        OUTPUT,
        save_all=True,
        append_images=frames[1:],
        duration=75,
        loop=0,
        optimize=True,
        disposal=2,
    )
    print(f"Wrote {OUTPUT.relative_to(REPO_ROOT)}")
    print(f"Wrote {POSTER.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
