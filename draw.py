# import numpy as np
# import matplotlib.pyplot as plt
#
# # ========== 1. 构造“水平走势 + 密集大波动” ==========
# np.random.seed(42)
# n = 600
# x = np.arange(n)
#
# # 水平基线（不随时间变化）
# baseline = 0.0
#
# # 更密集的高频抖动（加入一点相关性）
# noise = 0.15 * np.random.randn(n)
# noise = np.convolve(noise, np.ones(3)/3, mode="same")  # 轻度相关，使曲线更连贯
#
# # 合成信号：整体与横轴平行
# y = baseline + noise
# fig, ax = plt.subplots(figsize=(8, 4), dpi=200, facecolor="white")  # ↑ DPI 提高清晰度
# ax.plot(x, y, color="#228B22", linewidth=2.0, antialiased=True)     # ↑ 线条更粗更清晰
# # ========== 2. 画图 ==========
# plt.figure(figsize=(7, 3.5), facecolor="white")
# plt.plot(x, y,color="green", linewidth=1.6)
#
# plt.xlabel("Time Index")
# plt.ylabel("Amplitude")
# plt.title("Dense Fluctuations with Horizontal Trend (Green, White Background)")
#
# plt.tight_layout()
# plt.show()




import numpy as np
import matplotlib.pyplot as plt

# ========== 1. 构造“水平走势 + 密集大波动” ==========
np.random.seed(42)
n = 600
x = np.arange(n)

baseline = 0.0

def gen_signal(seed, scale=0.15):
    rng = np.random.RandomState(seed)
    noise = scale * rng.randn(n)
    noise = np.convolve(noise, np.ones(3)/3, mode="same")  # 轻度相关
    return baseline + noise

y1 = gen_signal(1)
y2 = gen_signal(2)
y3 = gen_signal(3)
y4 = gen_signal(4)

# ========== 2. 画图（同一张图 4 条线） ==========
fig, ax = plt.subplots(figsize=(8, 4), dpi=150, facecolor="white")

ax.plot(x, y1, color="#228B22", linewidth=0.8, antialiased=True, label="Green")      # 绿色
ax.plot(x, y2, color="#FF8C00", linewidth=0.8, antialiased=True, label="Orange")     # 橙色
ax.plot(x, y3, color="#003366", linewidth=0.8, antialiased=True, label="Dark Blue")  # 深蓝色
ax.plot(x, y4, color="#6A0DAD", linewidth=0.8, antialiased=True, label="Purple")     # 紫色

ax.set_facecolor("white")
# ax.grid(True, alpha=0.25, linewidth=0.6)

ax.set_xlabel("Time Index", fontsize=11)
ax.set_ylabel("Amplitude", fontsize=11)
ax.set_title("Dense Fluctuations with Horizontal Trend", fontsize=12)

# ax.legend(frameon=False, fontsize=10)  # 可选：去边框图例
plt.tight_layout()
plt.show()
