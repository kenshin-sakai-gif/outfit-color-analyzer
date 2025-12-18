import streamlit as st
import numpy as np
import cv2

from core.segmentation import clothes_mask
from core.colors import extract_dominant_colors, aggregate_by_deltaE
from core.scoring import ratio_score, explain_text
from core.visualize import fig_masks, fig_swatches

st.set_page_config(page_title="Outfit Color Analyzer", layout="centered")
st.title("Outfit Color Analyzer")
st.write("人物画像1枚からTop色・比率・スコアを算出する（Web版）．")

uploaded = st.file_uploader("人物画像をアップロード（JPEG/PNG）", type=["jpg", "jpeg", "png"])

def bytes_to_rgb(file_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(file_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("画像の読み込みに失敗した．")

    if img.ndim == 3 and img.shape[2] == 4:
        alpha = img[..., 3].astype(np.float32) / 255.0
        bg = np.full_like(img[..., :3], 255)
        img = (img[..., :3].astype(np.float32) * alpha[..., None] + bg * (1.0 - alpha[..., None])).astype(np.uint8)
    else:
        img = img[..., :3]

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return rgb

def analyze_rgb(rgb: np.ndarray):
    masks = clothes_mask(rgb)
    cm = masks["clothes"]

    centers_rgb, ratios = extract_dominant_colors(rgb, cm, k=10)
    rep_colors, rep_ratios, labels = aggregate_by_deltaE(
        centers_rgb, ratios, max_colors=3, de_thresh=9.0, min_presence=0.03
    )
    score = ratio_score(rep_ratios)
    return score, labels, rep_ratios, rep_colors, masks

if uploaded is not None:
    rgb = bytes_to_rgb(uploaded.read())
    st.image(rgb, caption="入力画像", use_container_width=True)

    try:
        with st.spinner("解析中..."):
            score, labels, ratios, rep_colors, masks = analyze_rgb(rgb)

        st.metric("スコア", f"{score} / 100")
        for lab, r in zip(labels, ratios):
            st.write(f"- {lab}: {float(r)*100:.1f}%")

        st.code(explain_text(score, labels, ratios), language="text")

        st.pyplot(fig_swatches(rep_colors, ratios, labels))
        st.pyplot(fig_masks(rgb, masks["person"], masks["clothes"], masks["skin"], masks["hair"]))

    except Exception as e:
        st.error(f"処理に失敗した．理由: {e}")
