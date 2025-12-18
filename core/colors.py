from __future__ import annotations
import numpy as np
from sklearn.cluster import KMeans
from skimage import color as skcolor
from skimage.color import deltaE_ciede2000

RANDOM_STATE = 42
KMEANS_N_INIT = "auto"

def extract_dominant_colors(rgb: np.ndarray, mask01: np.ndarray, k: int = 10,
                            sample_max_fit: int = 60000) -> tuple[np.ndarray, np.ndarray]:
    """
    Lab KMeans (fit=sample) + predictで全画素割合を集計．
    戻り値: (centers_rgb_uint8[k,3], ratios[k])
    """
    H, W = rgb.shape[:2]
    valid = (mask01 > 0)
    yy, xx = np.nonzero(valid)
    if len(yy) == 0:
        raise ValueError("有効画素が無い（マスクが空）．")

    pixels_rgb = rgb[yy, xx].astype(np.float32) / 255.0
    rng = np.random.default_rng(RANDOM_STATE)

    n_fit = min(sample_max_fit, len(pixels_rgb))
    idx_fit = rng.choice(len(pixels_rgb), size=n_fit, replace=False)
    fit_rgb = pixels_rgb[idx_fit]
    fit_lab = skcolor.rgb2lab(fit_rgb.reshape(-1, 1, 3)).reshape(-1, 3)

    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=KMEANS_N_INIT)
    km.fit(fit_lab)

    all_lab = skcolor.rgb2lab(pixels_rgb.reshape(-1, 1, 3)).reshape(-1, 3)
    labels = km.predict(all_lab)
    counts = np.bincount(labels, minlength=k).astype(np.float32)
    ratios = counts / max(1.0, counts.sum())

    # 代表色（cluster centerをRGBへ戻す）
    centers_lab = km.cluster_centers_.astype(np.float32)
    centers_rgb = (skcolor.lab2rgb(centers_lab[None, :, :])[0] * 255.0).clip(0, 255).astype(np.uint8)

    order = np.argsort(ratios)[::-1]
    return centers_rgb[order], ratios[order]

# ------- 色名ラベリング用パレット（必要なら増やす） -------
CATEGORY_RGB = {
    "Black": (30, 30, 30),
    "Gray": (150, 150, 150),
    "White": (255, 255, 255),
    "Navy": (30, 50, 100),
    "Blue": (70, 120, 230),
    "Green": (70, 170, 90),
    "Brown": (140, 90, 60),
    "Beige": (230, 215, 190),
    "Red": (220, 50, 50),
    "Yellow": (245, 210, 60),
    "Purple": (150, 80, 200),
}
CAT_NAMES = list(CATEGORY_RGB.keys())
_CAT_RGB_ARR = np.array([CATEGORY_RGB[n] for n in CAT_NAMES], dtype=np.uint8)
CAT_LAB_ARR  = skcolor.rgb2lab(_CAT_RGB_ARR[None, :, :] / 255.0)[0]

def label_consistent(rgb255: np.ndarray) -> str:
    rgb1 = np.uint8([[rgb255]])
    lab1 = skcolor.rgb2lab(rgb1 / 255.0)[0, 0]
    dE = deltaE_ciede2000(CAT_LAB_ARR, np.repeat(lab1[None, :], len(CAT_LAB_ARR), axis=0))
    return CAT_NAMES[int(np.argmin(dE))]

def aggregate_by_deltaE(centers_rgb: np.ndarray, ratios: np.ndarray,
                        max_colors: int = 3, de_thresh: float = 9.0, min_presence: float = 0.03):
    """
    ΔEで同系色統合 → 色名で再集約 → Top3返却
    """
    centers_rgb = np.asarray(centers_rgb, dtype=np.uint8)
    ratios = np.asarray(ratios, dtype=np.float32)

    valid = ratios >= float(min_presence)
    if not np.any(valid):
        return [], np.array([], dtype=np.float32), []

    centers_rgb = centers_rgb[valid]
    ratios = ratios[valid]

    labs = skcolor.rgb2lab(centers_rgb.reshape(-1, 1, 3) / 255.0).reshape(-1, 3)
    order = np.argsort(ratios)[::-1]
    labs, centers_rgb, ratios = labs[order], centers_rgb[order], ratios[order]

    groups = []
    for r, lab_i, rgb_i in zip(ratios, labs, centers_rgb):
        if not groups:
            groups.append({"rep_lab": lab_i, "rep_rgb": rgb_i, "ratio_sum": float(r), "max_r": float(r)})
            continue

        dists = [float(deltaE_ciede2000(np.array([g["rep_lab"]]), np.array([lab_i]))[0]) for g in groups]
        j = int(np.argmin(dists))
        if dists[j] < float(de_thresh):
            g = groups[j]
            g["ratio_sum"] += float(r)
            if float(r) > g["max_r"]:
                g["max_r"] = float(r)
                g["rep_lab"], g["rep_rgb"] = lab_i, rgb_i
        else:
            groups.append({"rep_lab": lab_i, "rep_rgb": rgb_i, "ratio_sum": float(r), "max_r": float(r)})

    label_map = {}
    for g in groups:
        name = label_consistent(g["rep_rgb"])
        if name not in label_map:
            label_map[name] = {"ratio_sum": g["ratio_sum"], "rep_rgb": g["rep_rgb"], "strength": g["max_r"]}
        else:
            label_map[name]["ratio_sum"] += g["ratio_sum"]
            if g["max_r"] > label_map[name]["strength"]:
                label_map[name]["strength"] = g["max_r"]
                label_map[name]["rep_rgb"] = g["rep_rgb"]

    items = sorted(label_map.items(), key=lambda kv: kv[1]["ratio_sum"], reverse=True)[:max_colors]
    total = sum(v["ratio_sum"] for _, v in items) or 1.0
    rep_colors = [v["rep_rgb"] for _, v in items]
    rep_ratios = np.array([v["ratio_sum"] / total for _, v in items], dtype=np.float32)
    labels = [k for k, _ in items]
    return rep_colors, rep_ratios, labels

