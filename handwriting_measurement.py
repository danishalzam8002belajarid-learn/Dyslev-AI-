"""Classical image-processing baseline for measuring connected handwriting regions."""

from typing import Any

import cv2
import numpy as np
import pandas as pd
from PIL import Image


def measure_connected_regions(
    image: Image.Image,
    min_area: int = 12,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Return a foreground mask, a box overlay, and pixel measurements.

    Connected ink regions are not guaranteed to correspond to individual
    characters: touching letters may merge, while dotted letters may split.
    """
    rgb = np.asarray(image.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    _, mask = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )
    if np.mean(mask) > 127:
        mask = cv2.bitwise_not(mask)

    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    regions: list[dict[str, int | float]] = []
    for component_id in range(1, count):
        x, y, width, height, area = map(int, stats[component_id])
        if area < min_area:
            continue
        regions.append(
            {
                "x (px)": x,
                "y (px)": y,
                "lebar (px)": width,
                "tinggi (px)": height,
                "luas tinta (px)": area,
                "rasio tinggi/lebar": round(height / width, 3),
            }
        )

    regions.sort(
        key=lambda region: (
            region["y (px)"] + region["tinggi (px)"],
            region["x (px)"],
        )
    )
    baseline_tolerance = (
        max(
            3.0,
            float(np.median([region["tinggi (px)"] for region in regions])) * 0.45,
        )
        if regions
        else 3.0
    )
    line_groups: list[dict[str, Any]] = []
    for region in regions:
        baseline = float(region["y (px)"] + region["tinggi (px)"])
        matching_lines = [
            line
            for line in line_groups
            if abs(float(np.median(line["baselines"])) - baseline)
            <= baseline_tolerance
        ]
        if matching_lines:
            closest_line = min(
                matching_lines,
                key=lambda line: abs(float(np.median(line["baselines"])) - baseline),
            )
            closest_line["baselines"].append(baseline)
            closest_line["regions"].append(region)
        else:
            line_groups.append({"baselines": [baseline], "regions": [region]})

    line_groups.sort(key=lambda line: float(np.median(line["baselines"])))
    regions = []
    for line_number, line in enumerate(line_groups, start=1):
        for region in line["regions"]:
            region["Baris"] = line_number
            regions.append(region)
    regions.sort(key=lambda region: (region["Baris"], region["x (px)"]))

    rows = []
    overlay = rgb.copy()
    line_width = max(1, min(rgb.shape[:2]) // 400)
    for index, region in enumerate(regions, start=1):
        x = int(region["x (px)"])
        y = int(region["y (px)"])
        width = int(region["lebar (px)"])
        height = int(region["tinggi (px)"])
        cv2.rectangle(
            overlay,
            (x, y),
            (x + width - 1, y + height - 1),
            (0, 190, 90),
            line_width,
        )
        cv2.putText(
            overlay,
            str(index),
            (x, max(12, y - 3)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (220, 30, 30),
            1,
            cv2.LINE_AA,
        )
        rows.append({"Komponen": index, **region})

    return mask, overlay, pd.DataFrame(rows)


def summarize_line_variability(measurements: pd.DataFrame) -> pd.DataFrame:
    """Summarize width/height coefficient of variation within each text line.

    A line needs at least two connected components for its variability to be
    defined. The combined index is the arithmetic mean of width and height CV;
    it describes image variation and is not a dysgraphia probability.
    """
    columns = [
        "Baris",
        "Jumlah komponen",
        "Variasi tinggi (%)",
        "Variasi lebar (%)",
        "Indeks variasi ukuran (%)",
    ]
    if measurements.empty:
        return pd.DataFrame(columns=columns)

    line_summaries = []
    for line_number, line in measurements.groupby("Baris", sort=True):
        heights = line["tinggi (px)"].to_numpy(dtype=float)
        widths = line["lebar (px)"].to_numpy(dtype=float)
        if len(line) < 2:
            height_cv = None
            width_cv = None
            size_index = None
        else:
            mean_height = float(np.mean(heights))
            mean_width = float(np.mean(widths))
            height_cv = (
                float(np.std(heights) / mean_height * 100)
                if mean_height
                else None
            )
            width_cv = (
                float(np.std(widths) / mean_width * 100)
                if mean_width
                else None
            )
            valid_cvs = [value for value in (height_cv, width_cv) if value is not None]
            size_index = float(np.mean(valid_cvs)) if valid_cvs else None

        line_summaries.append(
            {
                "Baris": int(line_number),
                "Jumlah komponen": int(len(line)),
                "Variasi tinggi (%)": height_cv,
                "Variasi lebar (%)": width_cv,
                "Indeks variasi ukuran (%)": size_index,
            }
        )
    return pd.DataFrame(line_summaries, columns=columns)


def crop_connected_regions(
    image: Image.Image,
    measurements: pd.DataFrame,
    padding_ratio: float = 0.15,
) -> list[dict[str, Any]]:
    """Crop measured ink components with a small margin for experimental inference."""
    rgb_image = image.convert("RGB")
    image_width, image_height = rgb_image.size
    crops = []

    for _, component in measurements.iterrows():
        x = int(component["x (px)"])
        y = int(component["y (px)"])
        width = int(component["lebar (px)"])
        height = int(component["tinggi (px)"])
        padding_x = max(2, int(round(width * padding_ratio)))
        padding_y = max(2, int(round(height * padding_ratio)))
        left = max(0, x - padding_x)
        top = max(0, y - padding_y)
        right = min(image_width, x + width + padding_x)
        bottom = min(image_height, y + height + padding_y)

        crops.append(
            {
                "Komponen": int(component["Komponen"]),
                "Baris": int(component["Baris"]),
                "crop": rgb_image.crop((left, top, right, bottom)),
            }
        )

    return crops
