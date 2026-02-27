import matplotlib.pyplot as plt
import pandas as pd

# ===== Data =====
data = {
    "method": [
        "cMLP", "cLSTM", "TCDF", "eSRU", "GVAR",
         "CR-VAE", "CUTS+", "GCD", "GN-GC"
    ],
    "params_M": [6.0, 23.5, 4.0, 6.5, 21.5, 7.5, 8.0, 0.5, 0.8],
    "AUROC": [0.652, 0.633, 0.598, 0.647, 0.662, 0.583, 0.738, 0.782, 0.802],
    "AUPRC": [0.017, 0.035, 0.011, 0.045, 0.103, 0.025, 0.086, 0.123, 0.277]
}
df = pd.DataFrame(data)

# ===== Marker & Color Mapping (STRICTLY AS REQUESTED) =====
marker_map = {
    "cMLP":      ("s", "#7b3fb2"),   # 紫色 正方形
    "cLSTM":     ("o", "#8ecae6"),   # 淡蓝色 圆形
    "TCDF":      ("^", "#2ecc71"),   # 正绿色 上三角
    "eSRU":      ("v", "#f1c40f"),   # 深黄色 倒三角
    "GVAR":      ("D", "#e67e22"),   # 原图橙色菱形（未指定，保持）
    "CR-VAE":    ("h", "#1f4fd8"),   # 中等绿色 六边形
    "CUTS+":     ("*", "#8b4513"),   # 褐色 五角星
    "GCD": ("p", "#636363"),   # 黑色 五边形
    "GN-GC": ("o", "#e74c3c")    # 红色 圆形
}

# ===== Plot =====
fig, axes = plt.subplots(1, 2, figsize=(12,6), dpi=200)
plt.subplots_adjust(wspace=0.35)

# --- AUROC ---
ax = axes[0]
for _, r in df.iterrows():
    m, c = marker_map[r.method]
    ax.scatter(r.params_M, r.AUROC,
               s=160, marker=m, c=c,
               edgecolor="black", linewidth=1)

ax.set_xlim(-1, 27)
ax.set_ylim(0.50, 0.83)
ax.set_xlabel("Number of Tunable Parameters (M)")
ax.set_ylabel("AUROC")
ax.grid(axis="y", linestyle="--", alpha=0.6)

gcd = df[df.method == "GN-GC"].iloc[0]
ax.text(
    gcd.params_M,
    gcd.AUROC + 0.007,   # AUROC 量级更大，偏移量稍大一点
    "Ours",
    color="#e74c3c",
    fontweight="bold",
    ha="center",
    va="bottom"
)


# --- AUPRC ---
ax = axes[1]
for _, r in df.iterrows():
    m, c = marker_map[r.method]
    ax.scatter(r.params_M, r.AUPRC,
               s=160, marker=m, c=c,
               edgecolor="black", linewidth=1)

ax.set_xlim(-1, 27)
ax.set_ylim(0, 0.30)
ax.set_xlabel("Number of Tunable Parameters (M)")
ax.set_ylabel("AUPRC")
ax.grid(axis="y", linestyle="--", alpha=0.6)

# ax.annotate("Ours", xy=(gcd.params_M, gcd.AUPRC),
#             xytext=(gcd.params_M + 1.5, gcd.AUPRC + 0.035),
#             color="#e74c3c", fontweight="bold",
#             arrowprops=dict(arrowstyle="->", color="#e74c3c"))
ax.text(
    gcd.params_M,
    gcd.AUPRC + 0.005,   # 文字放在点的正上方，可微调
    "Ours",
    color="#e74c3c",
    fontweight="bold",
    ha="center",
    va="bottom"
)

# ===== Legend =====
handles, labels = [], []
for k, (m, c) in marker_map.items():
    h = plt.Line2D([0], [0], marker=m, linestyle="None",
                   markerfacecolor=c, markeredgecolor="black",
                   markersize=10)
    handles.append(h)
    labels.append(k)

axes[0].legend(handles, labels, loc="upper right", fontsize=9, frameon=True)
axes[1].legend(handles, labels, loc="upper right", fontsize=9, frameon=True)


plt.suptitle(
    "Performance comparisons on DREAM4 (Gene-1): AUROC, AUPRC, and the number of tunable parameters",
    y=0.02, fontsize=10
)

# plt.savefig("replicated_figure.png", dpi=200, bbox_inches="tight")
plt.savefig(
    r"D:\Aproject\KANGCI-main\res\replicated_figure.png",
    dpi=200,
    bbox_inches="tight"
)

plt.show()
