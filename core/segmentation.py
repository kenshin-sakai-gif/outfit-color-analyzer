from __future__ import annotations
import io
import numpy as np
import cv2
from rembg import remove

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

# -------------------------
# person mask (rembg)
# -------------------------
def person_mask_rembg(rgb: np.ndarray) -> np.ndarray:
    """
    入力: RGB uint8 (H,W,3)
    出力: person mask 0/1 (H,W)
    """
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("rgbは(H,W,3)のRGB配列である必要がある．")

    # PNG bytes化 → rembg → RGBA bytes → OpenCV decode
    ok, buf = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    if not ok:
        raise ValueError("画像のエンコードに失敗した．")

    out_bytes = remove(buf.tobytes())  # 背景除去（RGBA PNGが返る）
    out = cv2.imdecode(np.frombuffer(out_bytes, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if out is None or out.ndim != 3 or out.shape[2] != 4:
        raise ValueError("rembgの出力が想定外である（RGBA PNGでない）．")

    alpha = out[..., 3]
    mask01 = (alpha > 0).astype(np.uint8)
    mask01 = _keep_largest_component(mask01)
    mask01 = _morph_clean(mask01, ksize=5, it=1)
    return mask01

# -------------------------
# skin / hair (lightweight, no mediapipe)
# -------------------------
def skin_mask_ycrcb(rgb: np.ndarray, base_mask01: np.ndarray) -> np.ndarray:
    """
    YCrCbで肌っぽい画素を推定（軽量）．人物マスク内部のみ返す．
    """
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
    """
    顔検出なしで頭部ROIを近似（人物bbox上部を頭部とみなす）．
    """
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
    """
    頭部ROI内で「暗い＋低彩度」を髪として推定（軽量）．
    """
    head_roi = head_roi_from_person_bbox(person_mask01, top_ratio=0.40)
    if head_roi.sum() == 0:
        return np.zeros_like(person_mask01, dtype=np.uint8)

    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    L = lab[..., 0] * (100.0 / 255.0)  # OpenCV L(0-255)を概ね0-100へ正規化
    a = lab[..., 1] - 128.0
    b = lab[..., 2] - 128.0
    chroma = np.sqrt(a*a + b*b)

    hair = ((L < hair_l_max) & (chroma < hair_chroma_max) & (head_roi > 0)).astype(np.uint8)
    hair = _morph_clean(hair, ksize=3, it=1)
    return hair

def clothes_mask(rgb: np.ndarray) -> dict:
    """
    服マスクを作る．
    戻り値: {"person","skin","hair","clothes"} (各0/1)
    """
    person = person_mask_rembg(rgb)
    skin = skin_mask_ycrcb(rgb, person)
    hair = hair_mask_lab(rgb, person)

    clothes = person.astype(np.int16) - skin.astype(np.int16) - hair.astype(np.int16)
    clothes = np.clip(clothes, 0, 1).astype(np.uint8)
    clothes = _morph_clean(clothes, ksize=5, it=1)

    return {"person": person, "skin": skin, "hair": hair, "clothes": clothes}

