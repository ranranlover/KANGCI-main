import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.font_manager as fm
font_list = [f.name for f in fm.fontManager.ttflist]
print("Times New Roman" in font_list)
print("Helvetica" in font_list)
print("Arial" in font_list)
# 1. 数据准备
models = ['GN-GC', 'w/o FDGD', 'GN-GC-mean', 'GN-GC-max', 'w/o reg', 'GN-GC-L1']
datasets = ['Lorenz-96', 'DREAM4', 'Medical']

# 数据 (AUROC)
auroc_val = {
    'Lorenz-96': [0.9993, 0.9982, 0.9982, 0.9982, 0.9962, 0.9982],
    'DREAM4': [0.8021, 0.7996, 0.7996, 0.7975, 0.7003, 0.7899],
    'Medical': [0.9351, 0.9345, 0.9306, 0.9087, 0.7206, 0.8596]
}
# 数据 (AUPRC)
auprc_val = {
    'Lorenz-96': [0.9940, 0.9925, 0.9831, 0.9834, 0.9726, 0.9933],
    'DREAM4': [0.2777, 0.2762, 0.2775, 0.2774, 0.1059, 0.2743],
    'Medical': [0.9298, 0.9292, 0.9247, 0.9052, 0.6820, 0.8508]
}

# 2. 顶会风格配色（更高级、更稳重）
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # 经典且专业的三色
markers = ['o', 's', 'D']

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['STIXGeneral']



sns.set_context("paper", font_scale=1.2)
sns.set_style("ticks")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6), dpi=200)


def plot_fancy_line(ax, data_dict, title, ylabel):
    x = np.arange(len(models))
    for i, dataset in enumerate(datasets):
        y_values = data_dict[dataset]

        # 为避免数字遮挡：同一 x 上的 y 值加一个微小偏移
        # 让数字在同一列中有轻微错开
        offset = (i - 1) * 0.06  # -0.06, 0, +0.06

        ax.plot(x, y_values, color=colors[i], label=dataset,
                marker=markers[i], markersize=9, linewidth=2.5,
                markerfacecolor='white', markeredgewidth=2, alpha=0.95)

        for xi, yi in zip(x, y_values):
            ax.annotate(f'{yi:.3f}',
                        xy=(xi, yi),
                        xytext=(offset * 10, 5.5),  # x偏移 + y偏移
                        textcoords='offset points',
                        ha='center',
                        va='bottom',
                        fontsize=8,
                        fontweight='bold',
                        color=colors[i])

        ax.fill_between(x, y_values, min(y_values) - 0.05,
                        color=colors[i], alpha=0.03)

    ax.set_title(title, fontsize=15, fontweight='bold', pad=20)
    ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=0, ha='center')
    ax.tick_params(axis='both', labelsize=10)
    ax.tick_params(axis='x', labelsize=9)  # 只改横坐标刻度大小

    for label in ax.get_xticklabels():
        label.set_fontweight('bold')
    for label in ax.get_yticklabels():
        label.set_fontweight('bold')
    ax.grid(axis='y', linestyle=':', alpha=0.4)
    sns.despine(ax=ax,  trim=False)



plot_fancy_line(ax1, auroc_val, '', 'AUROC Score')
plot_fancy_line(ax2, auprc_val, '', 'AUPRC Score')
fig.suptitle('Ablation Study', fontsize=18, fontweight='bold', y=0.92)


# 3. 图例美化
handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.87),
           ncol=3, frameon=True, fontsize=12, handletextpad=0.5,
           edgecolor='black', framealpha=0.1)
fig.tight_layout(rect=[0, 0, 1, 0.88])  # 关键：留出上方空间给图例
plt.show()
