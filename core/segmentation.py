# core/segmentation.py
from __future__ import annotations
import numpy as np
import cv2

# -------------------------
# utilities
# -------------------------
def _keep_largest_component(mask01: np.ndarray) -> np.ndarray:
    mask01 = (mask01 > 0).astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask01, connectivity=8)
    if num <= 2:
        return mask01
    areas = stats[1:, cv2.CC_STAT_AREA]
    i = 1 + int(np.argmax(areas))
    return (labels == i).astype(np.uint8)

def _morph_clean(mask01: np.ndarray, ksize: int = 5, it: int = 1) -> np.ndarray:
    k = np.ones((ksize, ksize), np.uint8)
    m = cv2.morphologyEx(mask01.astype(np.uint8), cv2.MORPH_CLOSE, k, iterations=it)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN,  k, iterations=it)
    return (m > 0).astype(np.uint8)

def _torso_roi_mask(h: int, w: int,
                    top: float = 0.02, bottom: float = 0.995,
                    left: float = 0.10, right: float = 0.90) -> np.ndarray:
    m = np.zeros((h, w), dtype=np.uint8)
    y1, y2 = int(h*top), int(h*bottom)
    x1, x2 = int(w*left), int(w*right)
    m[y1:y2, x1:x2] = 1
    return m

def _resize_keep_aspect(rgb: np.ndarray, max_side: int = 1024) -> tuple[np.ndarray, float]:
    """長辺 max_side に収めて縮小。scale は (new / old)."""
    h, w = rgb.shape[:2]
    scale = min(1.0, max_side / float(max(h, w)))
    if scale >= 1.0:
        return rgb, 1.0
    new_w, new_h = int(w * scale), int(h * scale)
    small = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return small, scale

def _resize_mask_to(mask01_small: np.ndarray, h: int, w: int) -> np.ndarray:
    m = cv2.resize(mask01_small.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    return (m > 0).astype(np.uint8)

# -------------------------
# person mask (rembg) with fallback
# -------------------------
def person_mask_rembg(rgb: np.ndarray) -> np.ndarray:
    """
    入力: RGB uint8 (H,W,3)
    出力: person mask 0/1 (H,W)
    失敗時は例外を投げる（呼び出し側でfallbackする）
    """
    from rembg import remove  # ここでimport（失敗時に検知できるように）

    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("rgbは(H,W,3)のRGB配列である必要がある．")

    # rembg を軽くするため縮小して処理
    rgb_small, scale = _resize_keep_aspect(rgb, max_side=1024)

    ok, buf = cv2.imencode(".png", cv2.cvtColor(rgb_small, cv2.COLOR_RGB2BGR))
    if not ok:
        raise ValueError("画像のエンコードに失敗した．")

    out_bytes = remove(buf.tobytes())  # RGBA PNG bytes
    out = cv2.imdecode(np.frombuffer(out_bytes, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if out is None or out.ndim != 3 or out.shape[2] != 4:
        raise ValueError("rembgの出力が想定外である（RGBA PNGでない）．")

    alpha = out[..., 3]
    mask01_small = (alpha > 0).astype(np.uint8)
    mask01_small = _keep_largest_component(mask01_small)
    mask01_small = _morph_clean(mask01_small, ksize=5, it=1)

    # 元サイズに戻す
    h, w = rgb.shape[:2]
    mask01 = _resize_mask_to(mask01_small, h, w)
    return mask01

# -------------------------
# skin / hair (lightweight)
# -------------------------
def skin_mask_ycrcb(rgb: np.ndarray, base_mask01: np.ndarray) -> np.ndarray:
    ycrcb = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)
    Y, Cr, Cb = cv2.split(ycrcb)

    ratio = Cr.astype(np.float32) / (Cb.astype(np.float32) + 1e-6)
    skin = (
        (Y > 60) &
        (Cr > 130) & (Cr < 175) &
        (Cb > 77)  & (Cb < 135) &
        (ratio > 1.2) & (ratio < 1.6)
    ).astype(np.uint8)

    skin = (skin & (base_mask01 > 0).astype(np.uint8)).astype(np.uint8)
    skin = _morph_clean(skin, ksize=3, it=1)
    return skin

def head_roi_from_person_bbox(person_mask01: np.ndarray, top_ratio: float = 0.35) -> np.ndarray:
    ys, xs = np.where(person_mask01 > 0)
    m = np.zeros_like(person_mask01, dtype=np.uint8)
    if len(ys) == 0:
        return m

    y1, y2 = int(ys.min()), int(ys.max())
    x1, x2 = int(xs.min()), int(xs.max())
    h = max(1, y2 - y1 + 1)

    top_end = y1 + int(h * top_ratio)
    m[y1:top_end, x1:x2+1] = 1
    return (m & (person_mask01 > 0).astype(np.uint8)).astype(np.uint8)

def hair_mask_lab(rgb: np.ndarray, person_mask01: np.ndarray,
                 hair_l_max: float = 65.0, hair_chroma_max: float = 32.0) -> np.ndarray:
    head_roi = head_roi_from_person_bbox(person_mask01, top_ratio=0.40)
    if head_roi.sum() == 0:
        return np.zeros_like(person_mask01, dtype=np.uint8)

    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    L = lab[..., 0] * (100.0 / 255.0)
    a = lab[..., 1] - 128.0
    b = lab[..., 2] - 128.0
    chroma = np.sqrt(a*a + b*b)

    hair = ((L < hair_l_max) & (chroma < hair_chroma_max) & (head_roi > 0)).astype(np.uint8)
    hair = _morph_clean(hair, ksize=3, it=1)
    return hair

# -------------------------
# main api
# -------------------------
def clothes_mask(rgb: np.ndarray) -> dict[str, np.ndarray]:
    """
    戻り値: {"person","skin","hair","clothes"} (各0/1)
    rembgが落ちても clothes は必ず返す（ROIフォールバック）
    """
    h, w = rgb.shape[:2]
    zeros = np.zeros((h, w), dtype=np.uint8)

    try:
        person = person_mask_rembg(rgb)
    except Exception:
        # 最低限のフォールバック（止めない）
        person = _torso_roi_mask(h, w)

    skin = skin_mask_ycrcb(rgb, person) if person.sum() > 0 else zeros.copy()
    hair = hair_mask_lab(rgb, person)   if person.sum() > 0 else zeros.copy()

    clothes = person.astype(np.int16) - skin.astype(np.int16) - hair.astype(np.int16)
    clothes = np.clip(clothes, 0, 1).astype(np.uint8)
    clothes = _morph_clean(clothes, ksize=5, it=1)

    return {"person": person, "skin": skin, "hair": hair, "clothes": clothes}
