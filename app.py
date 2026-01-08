# app.py
from __future__ import annotations

import time
import numpy as np
import cv2
import streamlit as st

from core.segmentation import clothes_mask
from core.colors import extract_dominant_colors, aggregate_by_deltaE
from core.scoring import ratio_score, explain_text
from core.visualize import fig_masks, fig_swatches


# -------------------------
# Streamlit page
# -------------------------
st.set_page_config(page_title="Outfit Color Analyzer", layout="centered")
st.title("Outfit Color Analyzer")
st.write("人物画像1枚からTop色・比率・スコアを算出する（Web版）．")

uploaded = st.file_uploader("人物画像をアップロード（JPEG/PNG）", type=["jpg", "jpeg", "png"])


# -------------------------
# image utils
# -------------------------
def bytes_to_rgb(file_bytes: bytes) -> np.ndarray:
    """
    入力: アップロードされた画像bytes
    出力: RGB uint8 (H,W,3)
    """
    arr = np.frombuffer(file_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("画像の読み込みに失敗した．")

    # RGBA -> 白背景合成
    if img.ndim == 3 and img.shape[2] == 4:
        alpha = img[..., 3].astype(np.float32) / 255.0
        bg = np.full_like(img[..., :3], 255)
        img = (img[..., :3].astype(np.float32) * alpha[..., None] + bg * (1.0 - alpha[..., None])).astype(np.uint8)
    else:
        img = img[..., :3]

    # BGR -> RGB
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return rgb


# -------------------------
# analysis core (web)
# -------------------------
def analyze_rgb(rgb: np.ndarray):
    """
    入力: RGB uint8
    出力:
      score: int
      labels: list[str]
      rep_ratios: np.ndarray (<=3,)
      rep_colors: list[np.ndarray]  # RGB uint8
      masks: dict (person/skin/hair/clothes + used_fallback)
    """
    masks = clothes_mask(rgb)
    cm = masks["clothes"]

    centers_rgb, ratios = extract_dominant_colors(rgb, cm, k=10)
    rep_colors, rep_ratios, labels = aggregate_by_deltaE(
        centers_rgb, ratios, max_colors=3, de_thresh=9.0, min_presence=0.03
    )
    score = ratio_score(rep_ratios)
    return score, labels, rep_ratios, rep_colors, masks


# -------------------------
# UI
# -------------------------
if uploaded is not None:
    rgb = bytes_to_rgb(uploaded.read())
    st.image(rgb, caption="入力画像", use_container_width=True)

    try:
        with st.spinner("解析中..."):
            t0 = time.time()
            score, labels, ratios, rep_colors, masks = analyze_rgb(rgb)
            dt = time.time() - t0

        # rembg fallback 判定（segmentation.py 側で used_fallback を返す前提）
        used_fallback = bool(int(masks.get("used_fallback", np.array([0], dtype=np.uint8))[0]))
        if used_fallback:
            st.warning("背景除去(rembg)が不安定だったため、簡易ROIで解析しました（精度が低下する可能性があります）。")
        else:
            st.success("背景除去(rembg)で人物領域を推定して解析しました。")

        # metrics
        st.metric("スコア", f"{score} / 100")
        st.metric("処理時間", f"{dt:.2f} 秒")

        # ratios text
        if len(labels) == 0:
            st.error("服領域が取得できませんでした（clothes mask が空）。画像を変えて再試行してください。")
        else:
            for lab, r in zip(labels, ratios):
                st.write(f"- {lab}: {float(r) * 100:.1f}%")

        st.code(explain_text(score, labels, ratios), language="text")

        # figures
        st.pyplot(fig_swatches(rep_colors, ratios, labels))
        st.pyplot(fig_masks(rgb, masks["person"], masks["clothes"], masks["skin"], masks["hair"]))

    except Exception as e:
        st.error(f"処理に失敗した．理由: {e}")
