import streamlit as st
import numpy as np
import cv2

st.set_page_config(page_title="Outfit Color Analyzer", layout="centered")
st.title("Outfit Color Analyzer")
st.write("人物画像1枚からTop色・比率・スコアを算出する（Web版）．")

uploaded = st.file_uploader("人物画像をアップロード（JPEG/PNG）", type=["jpg", "jpeg", "png"])

def bytes_to_rgb(file_bytes: bytes) -> np.ndarray:
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

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return rgb

def analyze_rgb(rgb: np.ndarray):
    # いったんダミー（Web導線確認用）
    score = 0
    labels = ["—", "—", "—"]
    ratios = [0.0, 0.0, 0.0]
    return score, labels, ratios

if uploaded is not None:
    rgb = bytes_to_rgb(uploaded.read())
    st.image(rgb, caption="入力画像", use_container_width=True)

    try:
        score, labels, ratios = analyze_rgb(rgb)
        st.metric("スコア", f"{score} / 100")
        for lab, r in zip(labels, ratios):
            st.write(f"- {lab}: {r*100:.1f}%")
        st.info("次：Colabの解析コアを analyze_rgb に移植する．")
    except Exception as e:
        st.error(f"処理に失敗した．理由: {e}")
