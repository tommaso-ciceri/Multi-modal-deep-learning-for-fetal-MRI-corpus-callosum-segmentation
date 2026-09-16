#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 24 20:43:14 2026

@author: marina.distefano

Stage 2. Characterization

Usage:
    python cc_morphometry.py --masks MASK_DIR --fa FA_DIR --output results
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from matplotlib.patches import Polygon
from scipy import ndimage as ndi
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import cKDTree
from skimage import measure
from skimage.measure import label, regionprops
from skimage.morphology import disk

logger = logging.getLogger("cc_morphometry")

GENU, BODY, SPLENIUM = 1, 2, 3
REGION_COLORS = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
EPS = 1e-8


@dataclass
class CCVolume:
    path: Path
    image: nib.Nifti1Image
    data: np.ndarray
    slices: list[np.ndarray]
    center: int

    @property
    def subject_id(self) -> str:
        return self.path.name.split(".nii")[0]

    @property
    def zooms(self) -> tuple[float, float, float]:
        return tuple(float(z) for z in self.image.header.get_zooms()[:3])


@dataclass
class Morphometry:
    """Scalar measures for one subject."""

    subject_id: str
    filename: str

    area_mm2: float
    genu_area_mm2: float
    body_area_mm2: float
    splenium_area_mm2: float

    perimeter_length_mm: float
    perimeter_curvature_mean: float
    perimeter_curvature_max: float

    skeleton_length_mm: float
    skeleton_curvature_mean: float
    rostrum_splenium_distance_mm: float

    thickness_mean_mm: float
    thickness_std_mm: float
    thickness_min_mm: float
    thickness_max_mm: float
    thickness_genu_mm: float
    thickness_body_mm: float
    thickness_splenium_mm: float

    cc_volume_mm3: float
    genu_volume_mm3: float
    body_volume_mm3: float
    splenium_volume_mm3: float
    length_mm: float
    height_mm: float

    fa_mean: float | None = None
    fa_genu: float | None = None
    fa_body: float | None = None
    fa_splenium: float | None = None

    def as_row(self) -> dict[str, float | str]:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class AnalysisResult:
    """Scalar measures plus the arrays for profiles, QC."""

    morphometry: Morphometry
    thickness_profile: np.ndarray
    fa_profile: np.ndarray | None
    split3d: np.ndarray
    mask: np.ndarray = field(repr=False)
    perimeter: np.ndarray = field(repr=False)
    superior: np.ndarray = field(repr=False)
    inferior: np.ndarray = field(repr=False)
    midline: np.ndarray = field(repr=False)
    rostrum_idx: int = field(repr=False)
    splenium_idx: int = field(repr=False)
    split2d: np.ndarray = field(repr=False)
    filled_mask: np.ndarray = field(repr=False)
    box: np.ndarray | None = field(repr=False)

def find_midsagittal_slice(volume: np.ndarray) -> int:
    occupied = np.flatnonzero(volume.sum(axis=(1, 2)))
    if occupied.size == 0:
        raise ValueError("No CC voxels found in volume")
    return int(occupied[occupied.size // 2])


def largest_component(mask: np.ndarray) -> np.ndarray:
    labelled = label(mask, connectivity=2)
    regions = regionprops(labelled)
    if not regions:
        raise RuntimeError("Slice contains no connected component")
    return labelled == max(regions, key=lambda r: r.area).label


def load_cc_slices(path: str | Path, search_radius: int = 3, half_width: int = 2) -> CCVolume:
    path = Path(path)
    image = nib.load(str(path))
    data = np.asarray(image.dataobj) > 0
    mid = find_midsagittal_slice(data)

    lo = max(mid - search_radius, 0)
    hi = min(mid + search_radius + 1, data.shape[0])
    center = next(
        (x for x in range(lo, hi) if data[x].any() and label(data[x], connectivity=2).max() <= 1),
        None,)

    if center is None:
        data[mid] = largest_component(data[mid])
        center = mid

    lo = max(center - half_width, 0)
    hi = min(center + half_width + 1, data.shape[0])
    return CCVolume(
        path=path,
        image=image,
        data=data,
        slices=[data[x] for x in range(lo, hi)],
        center=center,)


def average_slices(slices: list[np.ndarray], threshold: float = 0.5) -> np.ndarray:
    return np.mean(np.stack(slices, axis=0), axis=0) > threshold

# ------------------
# Contour, landmarks
# ------------------


def extract_contour(mask: np.ndarray) -> np.ndarray:
    if not mask.any():
        raise ValueError("Cannot extract a contour from an empty mask")
    contours = measure.find_contours(mask.astype(float), 0.5)
    if not contours:
        raise ValueError("No contour found")
    longest = max(contours, key=len)
    return np.column_stack((longest[:, 1], longest[:, 0]))


def contour_length(contour: np.ndarray) -> float:
    closed = np.vstack([contour, contour[0]])
    return float(np.linalg.norm(np.diff(closed, axis=0), axis=1).sum())


def curvature(curve: np.ndarray, sigma: float | None = None) -> np.ndarray:
    if len(curve) < 3:
        return np.zeros(len(curve))
    if sigma is None:
        sigma = max(len(curve) / 50.0, 1.0)

    dx = gaussian_filter1d(np.gradient(curve[:, 0]), sigma)
    dy = gaussian_filter1d(np.gradient(curve[:, 1]), sigma)
    ddx = gaussian_filter1d(np.gradient(dx), sigma)
    ddy = gaussian_filter1d(np.gradient(dy), sigma)

    return np.abs(dx * ddy - dy * ddx) / ((dx**2 + dy**2) ** 1.5 + EPS)


def find_landmarks(
    perimeter: np.ndarray, anterior_frac: float = 0.25, posterior_frac: float = 0.25) -> tuple[int, int]:
    """Indices of the max curvature in the anterior and posterior extremities."""
    y = perimeter[:, 1]
    y_min, y_max = y.min(), y.max()
    span = y_max - y_min
    if span == 0:
        return 0, len(perimeter) // 2

    k = curvature(perimeter)
    anterior = y < y_min + anterior_frac * span
    posterior = y > y_max - posterior_frac * span

    if not anterior.any() or not posterior.any():
        return int(np.argmin(y)), int(np.argmax(y))

    rostrum = int(np.flatnonzero(anterior)[np.argmax(k[anterior])])
    splenium = int(np.flatnonzero(posterior)[np.argmax(k[posterior])])
    return rostrum, splenium


def split_perimeter(
    perimeter: np.ndarray, rostrum_idx: int, splenium_idx: int) -> tuple[np.ndarray, np.ndarray]:
    """Cut the perimeter at the two landmarks into superior and inferior arcs."""
    first, second = sorted((rostrum_idx, splenium_idx))
    arc_a = perimeter[first : second + 1]
    arc_b = np.vstack([perimeter[second:], perimeter[: first + 1]])
    return (arc_a, arc_b) if arc_a[:, 1].mean() < arc_b[:, 1].mean() else (arc_b, arc_a)


def compute_midline(
    superior: np.ndarray, inferior: np.ndarray, n_points: int = 50) -> tuple[np.ndarray, np.ndarray]:
    if len(superior) == 0 or len(inferior) == 0:
        return np.zeros((n_points, 2)), np.zeros(n_points)

    _, idx = cKDTree(inferior).query(superior)
    partners = inferior[idx]
    midline = (superior + partners) / 2.0
    thickness = np.linalg.norm(superior - partners, axis=1)
    return resample_midline(midline, thickness, n_points)


def resample_midline(
    midline: np.ndarray, thickness: np.ndarray, n_points: int) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate midline and thickness onto n equally spaced arc-length positions."""
    if len(midline) < 2:
        point = midline[0] if len(midline) else np.zeros(2)
        value = thickness[0] if len(thickness) else 0.0
        return np.tile(point, (n_points, 1)), np.full(n_points, value)

    steps = np.linalg.norm(np.diff(midline, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(steps)])
    if arc[-1] == 0:
        return np.tile(midline[0], (n_points, 1)), np.full(n_points, thickness[0])

    arc /= arc[-1]
    targets = np.linspace(0.0, 1.0, n_points)
    resampled = np.column_stack([np.interp(targets, arc, midline[:, 0]), np.interp(targets, arc, midline[:, 1])])
    return resampled, np.interp(targets, arc, thickness)


def oriented_bounding_box(mask: np.ndarray) -> tuple[float, float, np.ndarray | None, np.ndarray]:
    filled = ndi.binary_closing(ndi.binary_fill_holes(mask), structure=disk(2))
    coords = cv2.findNonZero(filled.astype(np.uint8))
    if coords is None:
        return 0.0, 0.0, None, filled

    rect = cv2.minAreaRect(coords)
    side_a, side_b = rect[1]
    return max(side_a, side_b), min(side_a, side_b), cv2.boxPoints(rect).astype(int), filled


# ------------------
# Parcellation
# ------------------

def split_into_thirds(volume: np.ndarray) -> np.ndarray:
    """Label each voxel 1 (genu, anterior) / 2 (body) / 3 (splenium, posterior)."""
    split = np.zeros(volume.shape, dtype=np.uint8)
    occupied = np.flatnonzero(np.any(volume > 0, axis=(0, 2)))
    if occupied.size == 0:
        return split

    start, end = int(occupied[0]), int(occupied[-1])
    extent = end - start + 1
    positions = np.arange(extent) / (extent - 1) if extent > 1 else np.zeros(1)
    labels = np.digitize(positions, [1 / 3, 2 / 3]) + 1
    labels = 4 - labels  # low y index is posterior: 1 = genu, 3 = splenium

    for offset, value in enumerate(labels):
        index = start + offset
        split[:, index, :] = np.where(volume[:, index, :] > 0, value, 0)
    return split


def regional_areas(mask: np.ndarray, split2d: np.ndarray) -> tuple[int, int, int]:
    if not mask.any():
        return 0, 0, 0
    return tuple(int(np.sum((split2d == r) & mask)) for r in (GENU, BODY, SPLENIUM))


def label_points(points: np.ndarray, split2d: np.ndarray) -> np.ndarray:
    x = np.clip(np.round(points[:, 0]).astype(int), 0, split2d.shape[1] - 1)
    y = np.clip(np.round(points[:, 1]).astype(int), 0, split2d.shape[0] - 1)
    return split2d[y, x]


def region_mean(values: np.ndarray, labels: np.ndarray, region: int) -> float:
    mask = labels == region
    return float(values[mask].mean()) if mask.any() else 0.0


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


def sample_fa(fa_volume: np.ndarray, midline: np.ndarray, sagittal_index: int) -> np.ndarray:
    x = np.clip(np.round(midline[:, 0]).astype(int), 0, fa_volume.shape[2] - 1)
    y = np.clip(np.round(midline[:, 1]).astype(int), 0, fa_volume.shape[1] - 1)
    return fa_volume[sagittal_index, y, x]


def load_fa_profile(fa_path: str | Path | None, midline: np.ndarray, sagittal_index: int) -> np.ndarray | None:
    if fa_path is None:
        return None
    fa_path = Path(fa_path)
    if not fa_path.exists():
        logger.warning("  FA map not found: %s (FA metrics skipped)", fa_path)
        return None
    try:
        fa_volume = np.asarray(nib.load(str(fa_path)).dataobj)
        return sample_fa(fa_volume, midline, sagittal_index).astype(float)
    except Exception as error:
        logger.warning("  could not sample FA from %s: %s", fa_path, error)
        return None


def analyze(
    cc: CCVolume,
    fa_path: str | Path | None = None,
    n_points: int = 50,
    threshold: float = 0.5,) -> AnalysisResult:

    """Run the full morphometry pipeline on one loaded CC volume."""
    mask = average_slices(cc.slices, threshold)
    if not mask.any():
        raise ValueError("Averaged mask is empty")

    sx, sy, sz = cc.zooms
    in_plane_mm = np.mean([sy, sz])
    voxel_area = sy * sz
    voxel_volume = sx * sy * sz

    perimeter = extract_contour(mask)
    perimeter_curvature = curvature(perimeter)
    rostrum_idx, splenium_idx = find_landmarks(perimeter)

    superior, inferior = split_perimeter(perimeter, rostrum_idx, splenium_idx)
    midline, thickness = compute_midline(superior, inferior, n_points)
    thickness_mm = thickness * in_plane_mm
    midline_curvature = curvature(midline)
    skeleton_length = np.linalg.norm(np.diff(midline, axis=0), axis=1).sum() * in_plane_mm

    midsagittal = find_midsagittal_slice(cc.data)
    split3d = split_into_thirds(cc.data)
    split2d = split3d[midsagittal]
    genu_area, body_area, splenium_area = regional_areas(mask, split2d)

    point_labels = label_points(midline, split2d)
    length_px, height_px, box, filled = oriented_bounding_box(mask)
    fa_profile = load_fa_profile(fa_path, midline, midsagittal)

    fa_metrics: dict[str, float] = {}
    if fa_profile is not None:
        fa_metrics = dict(
            fa_mean=float(fa_profile.mean()),
            fa_genu=region_mean(fa_profile, point_labels, GENU),
            fa_body=region_mean(fa_profile, point_labels, BODY),
            fa_splenium=region_mean(fa_profile, point_labels, SPLENIUM),
        )

    morphometry = Morphometry(
        subject_id=cc.subject_id,
        filename=cc.path.name,
        area_mm2=float(mask.sum() * voxel_area),
        genu_area_mm2=genu_area * voxel_area,
        body_area_mm2=body_area * voxel_area,
        splenium_area_mm2=splenium_area * voxel_area,
        perimeter_length_mm=contour_length(perimeter) * in_plane_mm,
        perimeter_curvature_mean=float(perimeter_curvature.mean()),
        perimeter_curvature_max=float(perimeter_curvature.max()),
        skeleton_length_mm=float(skeleton_length),
        skeleton_curvature_mean=float(midline_curvature.mean()),
        rostrum_splenium_distance_mm=float(
            np.linalg.norm(perimeter[rostrum_idx] - perimeter[splenium_idx]) * in_plane_mm),
        thickness_mean_mm=float(thickness_mm.mean()),
        thickness_std_mm=float(thickness_mm.std()),
        thickness_min_mm=float(thickness_mm.min()),
        thickness_max_mm=float(thickness_mm.max()),
        thickness_genu_mm=region_mean(thickness_mm, point_labels, GENU),
        thickness_body_mm=region_mean(thickness_mm, point_labels, BODY),
        thickness_splenium_mm=region_mean(thickness_mm, point_labels, SPLENIUM),
        cc_volume_mm3=float(np.count_nonzero(cc.data) * voxel_volume),
        genu_volume_mm3=float(np.count_nonzero(split3d == GENU) * voxel_volume),
        body_volume_mm3=float(np.count_nonzero(split3d == BODY) * voxel_volume),
        splenium_volume_mm3=float(np.count_nonzero(split3d == SPLENIUM) * voxel_volume),
        length_mm=length_px * in_plane_mm,
        height_mm=height_px * in_plane_mm,
        **fa_metrics,)

    return AnalysisResult(
        morphometry=morphometry,
        thickness_profile=thickness_mm,
        fa_profile=fa_profile,
        split3d=split3d,
        mask=mask,
        perimeter=perimeter,
        superior=superior,
        inferior=inferior,
        midline=midline,
        rostrum_idx=rostrum_idx,
        splenium_idx=splenium_idx,
        split2d=split2d,
        filled_mask=filled,
        box=box,)

# ------------------
# QC
# ------------------

def save_qc_figure(result: AnalysisResult, output_dir: str | Path, title: str) -> Path:
    """QC overview"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mask, perimeter, midline = result.mask, result.perimeter, result.midline
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    axes[0, 0].imshow(mask, cmap="gray")
    axes[0, 0].plot(perimeter[:, 0], perimeter[:, 1], "r", lw=2, label="Perimeter")
    axes[0, 0].set_title("Averaged binary mask")
    axes[0, 0].legend()

    axes[0, 1].imshow(mask, cmap="gray")
    axes[0, 1].plot(perimeter[:, 0], perimeter[:, 1], "r", lw=1)
    for idx, color, name in (
        (result.rostrum_idx, "cyan", "Splenium"),
        (result.splenium_idx, "lime", "Rostrum"),
    ):
        axes[0, 1].scatter(
            perimeter[idx, 0], perimeter[idx, 1], c=color, s=100,
            edgecolors="black", label=name, zorder=5,
        )
    axes[0, 1].plot(
        perimeter[[result.rostrum_idx, result.splenium_idx], 0],
        perimeter[[result.rostrum_idx, result.splenium_idx], 1],
        "y--", lw=2,
    )
    axes[0, 1].set_title("Landmarks")
    axes[0, 1].legend()

    axes[0, 2].imshow(mask, cmap="gray")
    axes[0, 2].plot(perimeter[:, 0], perimeter[:, 1], "r", lw=1, alpha=0.5)
    axes[0, 2].plot(result.superior[:, 0], result.superior[:, 1], "b", lw=1.5, label="Inferior")
    axes[0, 2].plot(result.inferior[:, 0], result.inferior[:, 1], "g", lw=1.5, label="Superior")
    axes[0, 2].plot(midline[:, 0], midline[:, 1], "y.-", lw=2, label="Midline")
    axes[0, 2].set_title("Boundaries and midline")
    axes[0, 2].legend()

    axes[1, 0].imshow(mask, cmap="gray")
    scatter = axes[1, 0].scatter(
        midline[:, 0], midline[:, 1], c=result.thickness_profile,
        cmap="coolwarm", s=50, edgecolors="black", linewidths=0.5,
    )
    plt.colorbar(scatter, ax=axes[1, 0], label="Thickness (mm)")
    axes[1, 0].set_title("Thickness map")

    if result.box is not None:
        axes[1, 1].imshow(result.filled_mask, cmap="gray")
        axes[1, 1].add_patch(
            Polygon(result.box, closed=True, edgecolor="red", facecolor="none", lw=2)
        )
        overlay = np.zeros(result.split2d.shape + (3,))
        for value, color in enumerate(REGION_COLORS, start=1):
            overlay[result.split2d == value] = color
        axes[1, 1].imshow(overlay, alpha=0.4)
        axes[1, 1].set_title("Regional split and bounding box")
    else:
        axes[1, 1].text(0.5, 0.5, "Bounding box failed", ha="center", va="center")

    for ax in axes.flat[:5]:
        ax.axis("off")

    axes[1, 2].plot(result.thickness_profile, "b-", lw=2)
    axes[1, 2].set_xlabel("Midline position")
    axes[1, 2].set_ylabel("Thickness (mm)", color="b")
    axes[1, 2].tick_params(axis="y", labelcolor="b")
    axes[1, 2].grid(alpha=0.3)
    if result.fa_profile is not None:
        twin = axes[1, 2].twinx()
        twin.plot(result.fa_profile, "r-", lw=2)
        twin.set_ylabel("FA", color="r")
        twin.tick_params(axis="y", labelcolor="r")
        axes[1, 2].set_title("Thickness and FA profiles")
    else:
        axes[1, 2].set_title("Thickness profile")

    fig.suptitle(title, fontsize=14, y=0.995)
    fig.tight_layout()

    path = output_dir / f"{title.replace(' ', '_')}.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


# ------------------
# CLI
# ------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cc_morphometry.py",
        description="Morphometric characterization of the corpus callosum from CC masks.",
    )
    parser.add_argument("--masks", required=True, type=Path, help="directory of CC mask NIfTIs")
    parser.add_argument("--output", required=True, type=Path, help="output directory")
    parser.add_argument("--fa", type=Path, help="directory of FA maps (optional)")
    parser.add_argument(
        "--fa-suffix", default="_0001.nii.gz", help="suffix appended to the subject ID for FA maps"
    )
    parser.add_argument("--points", type=int, default=50, help="samples along the midline")
    parser.add_argument("--half-width", type=int, default=2, help="slices on each side of centre")
    parser.add_argument("--search-radius", type=int, default=3, help="search range for the centre")
    parser.add_argument("--threshold", type=float, default=0.5, help="binarisation threshold")
    parser.add_argument("--prefix", default="cc", help="prefix for output CSV files")
    parser.add_argument("--no-qc", action="store_true", help="skip QC figures")
    parser.add_argument("--no-split", action="store_true", help="skip writing split label maps")
    return parser

def profile_table(profiles: dict[str, np.ndarray], prefix: str) -> pd.DataFrame:
    table = pd.DataFrame(profiles).T
    table.columns = [f"{prefix}_p{i}" for i in range(table.shape[1])]
    table.index.name = "subject_id"
    return table

def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if not args.masks.is_dir():
        logger.error("Mask directory not found: %s", args.masks)
        return 1

    if args.fa is not None and not args.fa.is_dir():
        logger.error("FA directory not found: %s", args.fa)
        return 1
    if args.fa is None:
        logger.info("No FA directory given: computing morphometry only")

    files = sorted(p for p in args.masks.iterdir() if p.name.endswith((".nii", ".nii.gz")))
    if not files:
        logger.error("No NIfTI files found in %s", args.masks)
        return 1

    args.output.mkdir(parents=True, exist_ok=True)
    rows, thickness_profiles, fa_profiles = [], {}, {}

    for index, path in enumerate(files, start=1):
        logger.info("[%d/%d] %s", index, len(files), path.name)
        try:
            cc = load_cc_slices(path, args.search_radius, args.half_width)
            fa_path = args.fa / f"{cc.subject_id}{args.fa_suffix}" if args.fa else None
            result = analyze(cc, fa_path, args.points, args.threshold)
        except Exception as error:
            logger.warning("  failed: %s", error)
            continue

        rows.append(result.morphometry.as_row())
        thickness_profiles[cc.subject_id] = result.thickness_profile
        if result.fa_profile is not None:
            fa_profiles[cc.subject_id] = result.fa_profile

        if not args.no_split:
            nib.save(
                nib.Nifti1Image(result.split3d, cc.image.affine),
                args.output / f"{cc.subject_id}_split.nii.gz",
            )
        if not args.no_qc:
            save_qc_figure(result, args.output / "qc", cc.subject_id)

    if not rows:
        logger.error("All files failed; nothing written")
        return 1

    tables = {
        f"{args.prefix}_morphometry.csv": (pd.DataFrame(rows), False),
        f"{args.prefix}_thickness_profiles.csv": (profile_table(thickness_profiles, "thickness"), True),
    }
    if fa_profiles:
        tables[f"{args.prefix}_fa_profiles.csv"] = (profile_table(fa_profiles, "fa"), True)
    for name, (table, with_index) in tables.items():
        table.to_csv(args.output / name, index=with_index)
        logger.info("Wrote %s", args.output / name)

    logger.info("Done: %d/%d subjects processed", len(rows), len(files))
    return 0


if __name__ == "__main__":
    sys.exit(main())