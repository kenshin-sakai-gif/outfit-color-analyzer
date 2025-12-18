from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt

def fig_masks(rgb: np.ndarray, person: np.ndarray, clothes: np.ndarray, skin: np.ndarray, hair: np.ndarray):
    fig, ax = plt.subplots(1, 5, figsize=(16, 3))
    ax[0].imshow(rgb); ax[0].axis("off"); ax[0].set_title("Input")
    ax[1].imshow(person * 255, cmap="gray"); ax[1].axis("off"); ax[1].set_title("Person")
    ax[2].imshow(clothes * 255, cmap="gray"); ax[2].axis("off"); ax[2].set_title("Clothes")
    ax[3].imshow(skin * 255, cmap="Reds"); ax[3].axis("off"); ax[3].set_title("Skin")
    ax[4].imshow(hair * 255, cmap="magma"); ax[4].axis("off"); ax[4].set_title("Hair")
    plt.tight_layout()
    return fig

def fig_swatches(rep_colors: list[np.ndarray], ratios: np.ndarray, labels: list[str]):
    n = max(1, len(rep_colors))
    swatch = np.ones((80, 70*n, 3), dtype=np.uint8) * 255
    for i in range(n):
        swatch[:, 70*i:70*(i+1)] = rep_colors[i]

    fig, ax = plt.subplots(1, 2, figsize=(10, 3))
    ax[0].imshow(swatch); ax[0].axis("off"); ax[0].set_title("Top colors (Actual)")
    pie_labels = [f"{lab} ({float(r)*100:.1f}%)" for lab, r in zip(labels, ratios)]
    ax[1].pie(ratios, labels=pie_labels, textprops={"fontsize":10})
    ax[1].set_title("Color Ratios")
    plt.tight_layout()
    return fig

