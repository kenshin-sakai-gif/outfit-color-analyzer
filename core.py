# core.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import numpy as np
import cv2

from sklearn.cluster import KMeans
from skimage import color as skcolor
from skimage.color import deltaE_ciede2000

import matplotlib.pyplot as plt


# =========================
# Config
# =========================
@dataclass(frozen=True)
class AnalyzerConfig:
    random_state: int = 42

    # 黄金比 7:2.5:0.5
    target_ratios: Tuple[float, float, float] = (0.70, 0.25, 0.05)

    # KMeans
    k_global: int = 10
    kmeans_n_init: str | int = "auto"
    sample_max_fit: int = 60000

    # 集約
    agg_min_presence: float = 0.03    # 3% 未満はノイズ扱い
    deltae_merge_thresh: float = 9.0  # ΔE < 9 を同系色として統合
    max_top_colors: int = 3

    # ROI（人物抽出なしの暫定策）
    roi_top: float = 0.02
    roi_bottom: float = 0.995
    roi_left: float = 0.10
    roi_right: float = 0.90


# =========================
# Label palette (expanded)
# =========================
CATEGORY_RGB: Dict[str, Tuple[int, int, int]] = {
    # neutrals
    "Black": (30, 30, 30),
    "Charcoal": (55, 55, 60),
    "Slate Gray": (95, 100, 110),
    "Ash Gray": (145, 150, 155),
    "Silver Gray": (170, 175, 180),
    "Gray": (150, 150, 150),
    "Stone": (190, 190, 180),
    "Greige": (210, 205, 195),
    "Off White": (240, 240, 235),
    "Ivory": (246, 240, 220),
    "Snow White": (252, 252, 248),
    "White": (255, 255, 255),
    "Dark Navy": (25, 35, 70),

    # beige / brown
    "Light Beige": (240, 225, 205),
    "Beige": (230, 215, 190),
    "Pink Beige": (235, 210, 200),
    "Camel": (190, 150, 100),
    "Taupe": (150, 135, 120),
    "Mocha": (130, 95, 80),
    "Brown": (140, 90, 60),
    "Dark Brown": (90, 60, 40),

    # greens
    "Khaki": (160, 150, 100),
    "Olive": (120, 120, 60),
    "Sage": (160, 185, 160),
    "Mint": (180, 230, 190),
    "Green": (70, 170, 90),
    "Dark Green": (40, 100, 70),

    # blues
    "Navy": (30, 50, 100),
    "Denim": (95, 120, 175),
    "Dusty Blue": (120, 150, 180),
    "Light Blue": (160, 190, 230),
    "Sky Blue": (120, 180, 240),
    "Blue": (70, 120, 230),
    "Blue Green": (80, 140, 150),
    "Teal": (60, 120, 135),

    # yellows / oranges
    "Yellow": (245, 210, 60),
    "Mustard": (200, 170, 50),
    "Orange": (240, 140, 50),
    "Coral": (255, 127, 80),
    "Terracotta": (200, 110, 70),
    "Brick": (190, 80, 60),

    # reds / pinks
    "Red": (220, 50, 50),
    "Wine Red": (130, 30, 50),
    "Dusty Pink": (220, 170, 175),
    "Rose": (210, 120, 140),

    # purples
    "Purple": (150, 80, 200),
    "Lavender": (200, 160, 220),
}

CAT_NAMES = list(CATEGORY_RGB.keys())
_CAT_RGB_ARR = np.array([CATEGORY_RGB[n] for n in CAT_NAMES], dtype=np.uint8)
CAT_LAB_ARR = skcolor.rgb2lab(_CAT_RGB_ARR[None, :, :] / 255.0)[0]


# =========================
# Utils
# =========================
def torso_crop(img: np.ndarray, top: float, bottom: float, left: float, right: float) -> np.ndarray:
    """構図依存を少し下げるための暫定ROI（人物抽出なしPhase用）．"""
    h, w = img.shape[:2]
    y1 = int(h * top)
    y2 = int(h * bottom)
    x1 = int(w * left)
    x2 = int(w * right)
    y1 = np.clip(y1, 0, h - 1)
    y2 = np.clip(y2, y1 + 1, h)
    x1 = np.clip(x1, 0, w - 1)
    x2 = np.clip(x2, x1 + 1, w)
    return img[y1:y2, x1:x2]


def label_consistent(rgb255: np.ndarray) -> str:
    """カテゴリ全体とのΔE2000最小を色名として採用する．"""
    rgb1 = np.uint8([[rgb255]])
    lab1 = skcolor.rgb2lab(rgb1 / 255.0)[0, 0]
    dE = deltaE_ciede2000(CAT_LAB_ARR, np.repeat(lab1[None, :], len(CAT_LAB_ARR), axis=0))
    return CAT_NAMES[int(np.argmin(dE))]


# =========================
# Core: KMeans (Lab) + predict ratio
# =========================
def extract_dominant_colors_lab_kmeans(
    rgb_img: np.ndarray,
    *,
    k: int,
    random_state: int,
    n_init: str | int,
    sample_max_fit: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Lab KMeans（fitはサンプル）＋predictで全画素割合を集計する．
    戻り値: (centers_rgb_sorted, ratios_sorted)
    """
    H, W = rgb_img.shape[:2]
    pixels_rgb = rgb_img.reshape(-1, 3).astype(np.float32) / 255.0

    if pixels_rgb.size == 0:
        raise ValueError("入力画像が空である．")

    rng = np.random.default_rng(random_state)

    # (1) fit（サンプル）
    n_fit = min(sample_max_fit, len(pixels_rgb))
    idx_fit = rng.choice(len(pixels_rgb), size=n_fit, replace=False)
    fit_rgb = pixels_rgb[idx_fit]
    fit_lab = skcolor.rgb2lab(fit_rgb.reshape(-1, 1, 3)).reshape(-1, 3)

    km = KMeans(n_clusters=k, random_state=random_state, n_init=n_init)
    km.fit(fit_lab)

    # (2) ratio（全画素でpredict）
    all_lab = skcolor.rgb2lab(pixels_rgb.reshape(-1, 1, 3)).reshape(-1, 3)
    labels_all = km.predict(all_lab)
    counts = np.bincount(labels_all, minlength=k).astype(np.float32)
    ratios = counts / max(1.0, counts.sum())

    # (3) rep（fitサンプルから「中心に近い実画素」を代表色にする）
    centers_lab = km.cluster_centers_.astype(np.float32)
    labels_fit = km.predict(fit_lab)

    rep_rgbs = []
    for ci in range(k):
        m = (labels_fit == ci)
        if not np.any(m):
            rgb = skcolor.lab2rgb(centers_lab[ci][None, None, :])[0, 0] * 255.0
            rep_rgbs.append(np.clip(rgb, 0, 255).astype(np.uint8))
            continue

        lab_ci = fit_lab[m]
        j = int(np.argmin(np.sum((lab_ci - centers_lab[ci][None, :]) ** 2, axis=1)))
        rep_rgbs.append(np.clip(fit_rgb[m][j] * 255.0, 0, 255).astype(np.uint8))

    rep_rgbs = np.stack(rep_rgbs, axis=0)

    order = np.argsort(ratios)[::-1]
    return rep_rgbs[order], ratios[order]


# =========================
# Core: ΔE merge + label re-aggregate
# =========================
def aggregate_top_colors_by_deltaE(
    centers_rgb: np.ndarray,
    ratios: np.ndarray,
    *,
    max_colors: int,
    de_thresh: float,
    min_presence: float,
) -> Tuple[List[np.ndarray], np.ndarray, List[str]]:
    """
    1) 小比率を除外．
    2) 比率降順に，ΔE < de_thresh なら同一グループに統合．
    3) 各グループを色名に変換し，同じ色名は比率合算．
    4) 上位max_colorsを返す（比率は再正規化）．
    """
    centers_rgb = np.asarray(centers_rgb, dtype=np.uint8)
    ratios = np.asarray(ratios, dtype=np.float32)

    if centers_rgb.size == 0 or ratios.size == 0:
        return [], np.array([], dtype=np.float32), []

    valid = ratios >= min_presence
    if not np.any(valid):
        return [], np.array([], dtype=np.float32), []

    centers_rgb = centers_rgb[valid]
    ratios = ratios[valid]

    labs = skcolor.rgb2lab(centers_rgb.reshape(-1, 1, 3) / 255.0).reshape(-1, 3)
    order = np.argsort(ratios)[::-1]
    labs, centers_rgb, ratios = labs[order], centers_rgb[order], ratios[order]

    # ΔEでグループ化
    groups = []  # {"rep_lab","rep_rgb","ratio_sum","max_member_ratio"}
    for r, lab_i, rgb_i in zip(ratios, labs, centers_rgb):
        if not groups:
            groups.append(
                {"rep_lab": lab_i, "rep_rgb": rgb_i, "ratio_sum": float(r), "max_member_ratio": float(r)}
            )
            continue

        dists = [float(deltaE_ciede2000(np.array([g["rep_lab"]]), np.array([lab_i]))[0]) for g in groups]
        j = int(np.argmin(dists))
        if dists[j] < de_thresh:
            g = groups[j]
            g["ratio_sum"] += float(r)
            # 代表色は「グループ内で最大比率の要素」を採用する
            if float(r) > g["max_member_ratio"]:
                g["max_member_ratio"] = float(r)
                g["rep_lab"], g["rep_rgb"] = lab_i, rgb_i
        else:
            groups.append(
                {"rep_lab": lab_i, "rep_rgb": rgb_i, "ratio_sum": float(r), "max_member_ratio": float(r)}
            )

    # 色名ごとに再集約
    label_data = {}  # name -> {"ratio_sum","rep_rgb","rep_strength"}
    for g in groups:
        name = label_consistent(g["rep_rgb"])
        if name not in label_data:
            label_data[name] = {
                "ratio_sum": g["ratio_sum"],
                "rep_rgb": g["rep_rgb"],
                "rep_strength": g["max_member_ratio"],
            }
        else:
            label_data[name]["ratio_sum"] += g["ratio_sum"]
            if g["max_member_ratio"] > label_data[name]["rep_strength"]:
                label_data[name]["rep_strength"] = g["max_member_ratio"]
                label_data[name]["rep_rgb"] = g["rep_rgb"]

    items = list(label_data.items())
    items.sort(key=lambda kv: kv[1]["ratio_sum"], reverse=True)
    items = items[:max_colors]

    total = sum(info["ratio_sum"] for _, info in items)
    rep_ratios = np.array([info["ratio_sum"] for _, info in items], dtype=np.float32)
    if total > 0:
        rep_ratios /= total

    rep_colors = [info["rep_rgb"] for _, info in items]
    labels = [name for name, _ in items]
    return rep_colors, rep_ratios, labels


# =========================
# Score / Visualization
# =========================
def ratio_score(ratios: np.ndarray, target: Tuple[float, float, float]) -> int:
    r = np.array(ratios, dtype=np.float32)
    if r.size < 3:
        r = np.pad(r, (0, 3 - r.size), constant_values=0)
    r = np.sort(r)[::-1]
    target_arr = np.array(target, dtype=np.float32)
    diff = float(np.abs(r - target_arr).sum())
    return int(round(max(0.0, 1.0 - diff) * 100))


def compute_name_gap_deltaE(rep_colors: List[np.ndarray], labels: List[str]) -> np.ndarray:
    if rep_colors is None or len(rep_colors) == 0 or len(rep_colors) != len(labels):
        return np.zeros(len(labels), dtype=np.float32)

    name_rgbs = np.array([CATEGORY_RGB.get(lab, (0, 0, 0)) for lab in labels], dtype=np.uint8)
    lab_actual = skcolor.rgb2lab(np.array(rep_colors, dtype=np.uint8)[None, :, :] / 255.0)[0]
    lab_named = skcolor.rgb2lab(name_rgbs[None, :, :] / 255.0)[0]
    return deltaE_ciede2000(lab_actual, lab_named)


def make_result_figure(rep_colors: List[np.ndarray], ratios: np.ndarray, labels: List[str]):
    dE = compute_name_gap_deltaE(rep_colors, labels)

    fig, ax = plt.subplots(1, 2, figsize=(10, 3))

    # swatch
    n = max(1, len(rep_colors))
    swatch = np.ones((80, 60 * n, 3), dtype=np.uint8) * 255
    for i in range(n):
        c = rep_colors[i] if i < len(rep_colors) else np.array([255, 255, 255], dtype=np.uint8)
        swatch[:, 60 * i : 60 * (i + 1)] = c
    ax[0].imshow(swatch)
    ax[0].axis("off")
    ax[0].set_title("Top colors (Actual)", fontsize=11)

    # pie
    pie_labels = [f"{lab} ({r*100:.1f}%)\nΔE≈{de:.1f}" for lab, r, de in zip(labels, ratios, dE)]
    ax[1].pie(ratios, labels=pie_labels, textprops={"fontsize": 10})
    ax[1].set_title("Color Ratios / Name Gap", fontsize=11)

    plt.tight_layout()
    return fig


# =========================
# Public API: analyze_rgb (Phase 1)
# =========================
def analyze_rgb(rgb: np.ndarray, cfg: Optional[AnalyzerConfig] = None):
    """
    Phase 1: 人物抽出なしで解析コアだけ動かす．
      - 入力RGB画像からROIを切り出し
      - Lab KMeans → ΔE統合 → 色名再集約 → Top3 → スコア
      - 可視化figも返す
    """
    if cfg is None:
        cfg = AnalyzerConfig()

    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("入力はRGB画像（H×W×3）である必要がある．")

    rgb_roi = torso_crop(
        rgb, top=cfg.roi_top, bottom=cfg.roi_bottom, left=cfg.roi_left, right=cfg.roi_right
    )

    centers_rgb, ratios = extract_dominant_colors_lab_kmeans(
        rgb_roi,
        k=cfg.k_global,
        random_state=cfg.random_state,
        n_init=cfg.kmeans_n_init,
        sample_max_fit=cfg.sample_max_fit,
    )

    rep_colors, rep_ratios, labels = aggregate_top_colors_by_deltaE(
        centers_rgb,
        ratios,
        max_colors=cfg.max_top_colors,
        de_thresh=cfg.deltae_merge_thresh,
        min_presence=cfg.agg_min_presence,
    )

    score = ratio_score(rep_ratios, cfg.target_ratios)

    # 図
    fig = make_result_figure(rep_colors, rep_ratios, labels)

    # 返却（Streamlit側で表示する）
    return {
        "score": score,
        "labels": labels,
        "ratios": rep_ratios,
        "rep_colors": rep_colors,
        "rgb_roi": rgb_roi,
        "fig": fig,
    }
