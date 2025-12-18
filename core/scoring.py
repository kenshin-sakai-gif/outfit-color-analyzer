from __future__ import annotations
import numpy as np

TARGET_RATIOS = np.array([0.70, 0.25, 0.05], dtype=np.float32)
RATIO_TITLE = "黄金比(7:2.5:0.5)"

def ratio_score(top3_ratios: np.ndarray) -> int:
    r = np.array(top3_ratios, dtype=np.float32)
    if r.size < 3:
        r = np.pad(r, (0, 3 - r.size), constant_values=0)
    r = np.sort(r)[::-1]
    diff = float(np.abs(r - TARGET_RATIOS).sum())
    return int(round(max(0.0, 1.0 - diff) * 100))

def explain_text(score: int, labels: list[str], ratios: np.ndarray) -> str:
    parts = [f"{lab}: {float(r)*100:.1f}%" for lab, r in zip(labels, ratios)]
    while len(parts) < 3:
        parts.append("—: 0.0%")

    msg = [f"{RATIO_TITLE}スコア: {score} / 100", "配色内訳: " + " / ".join(parts)]
    if score >= 85:
        msg.append("判定: バランス良好．")
    elif score >= 60:
        msg.append("判定: 概ね良好．")
    else:
        msg.append("判定: 改善余地あり．")
    return "\n".join(msg)

