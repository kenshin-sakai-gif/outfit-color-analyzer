# app.py
import streamlit as st
import numpy as np
import cv2

from core import analyze_rgb, AnalyzerConfig

st.set_page_config(page_title="Outfit Color Analyzer", layout="centered")
st.title("Outfit Color Analyzer（Web版 / Phase 1）")
st.write("人物抽出なしで，解析コア（Lab KMeans→ΔE統合→Top3→スコア）をWebで動作確認する段階である．")

uploaded = st.file_uploader("画像をアップロード（JPEG/PNG）", type=["jpg", "jpeg", "png"])


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


with st.expander("設定（Phase 1）", expanded=True):
    k_global = st.slider("KMeansクラスタ数 k", min_value=5, max_value=15, value=10, step=1)
    de_thresh = st.slider("ΔE統合しきい値", min_value=3.0, max_value=20.0, value=9.0, step=0.5)
    min_presence = st.slider("ノイズ除外（最小比率）", min_value=0.0, max_value=0.10, value=0.03, step=0.01)
    show_roi = st.checkbox("ROI（解析対象領域）も表示する", value=True)

    cfg = AnalyzerConfig(
        k_global=int(k_global),
        deltae_merge_thresh=float(de_thresh),
        agg_min_presence=float(min_presence),
    )

if uploaded is not None:
    try:
        rgb = bytes_to_rgb(uploaded.read())
        st.image(rgb, caption="入力画像（RGB表示）", use_container_width=True)

        if st.button("解析する"):
            out = analyze_rgb(rgb, cfg)

            st.subheader("結果")
            st.metric("スコア", f"{out['score']} / 100")

            labels = out["labels"]
            ratios = out["ratios"]

            if len(labels) == 0:
                st.warning("有効な色が抽出できなかった（ノイズ除外の閾値が高すぎる等）．")
            else:
                st.write("Top色・割合（再正規化）")
                for lab, r in zip(labels, ratios):
                    st.write(f"- {lab}: {float(r)*100:.1f}%")

            st.pyplot(out["fig"], clear_figure=True)

            if show_roi:
                st.subheader("ROI（Phase 1 の暫定解析対象）")
                st.image(out["rgb_roi"], caption="ROI", use_container_width=True)

            st.info("次のPhaseでは，人物抽出（セグメンテーション）を追加して『服領域のみ』で同じ解析コアを回す．")

    except Exception as e:
        st.error(f"処理に失敗した．理由: {e}")

