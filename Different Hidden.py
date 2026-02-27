import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# 1. 数据准备
H = ['H=32', 'H=64', 'H=128', 'H=256', 'H=512']
datasets = ['Lorenz-96', 'DREAM4', 'Medical']

# AUROC
auroc_val = {
    'Lorenz-96': [0.995, 0.996, 0.997, 0.998, 0.999],
    'DREAM4':    [0.757, 0.758, 0.768, 0.773, 0.802],
    'Medical':   [0.912, 0.918, 0.936, 0.935, 0.936]
}

# AUPRC
auprc_val = {
    'Lorenz-96': [0.960, 0.975, 0.977, 0.985, 0.994],
    'DREAM4':    [0.195, 0.221, 0.228, 0.261, 0.277],
    'Medical':   [0.903, 0.918, 0.934, 0.929, 0.932]
}

# 2. 颜色/标记
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
markers = ['o', 's', 'D']

sns.set_context("paper", font_scale=1.2)
sns.set_style("ticks")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6), dpi=200)

def plot_fancy_line(ax, data_dict, title, ylabel):
    x = np.arange(len(H))
    for i, dataset in enumerate(datasets):
        y = data_dict[dataset]
        offset = (i - 1) * 0.06  # 轻微偏移，避免数值重叠

        ax.plot(x, y, color=colors[i], label=dataset,
                marker=markers[i], markersize=9, linewidth=2.5,
                markerfacecolor='white', markeredgewidth=2, alpha=0.95)

        # 标注数值
        for xi, yi in zip(x, y):
            ax.annotate(f'{yi:.3f}',
                        xy=(xi, yi),
                        xytext=(offset * 10, 5.5),
                        textcoords='offset points',
                        ha='center', va='bottom',
                        fontsize=8, fontweight='bold', color=colors[i])

        ax.fill_between(x, y, min(y) - 0.07, color=colors[i], alpha=0.03)

    ax.set_title(title, fontsize=15, fontweight='bold', pad=20)
    ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.tick_params(axis='both', labelsize=10)

    for label in ax.get_xticklabels():
        label.set_fontweight('bold')
    for label in ax.get_yticklabels():
        label.set_fontweight('bold')

    ax.set_xticklabels(H, rotation=0, ha='center')
    ax.grid(axis='y', linestyle=':', alpha=0.4)
    sns.despine(ax=ax, trim=False)

plot_fancy_line(ax1, auroc_val, '', 'AUROC Score')
plot_fancy_line(ax2, auprc_val, '', 'AUPRC Score')

fig.suptitle('Performance with Different Number of Hidden Layers', fontsize=18, fontweight='bold', y=0.92)

# 图例
handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.87),
           ncol=3, frameon=True, fontsize=12, handletextpad=0.5,
           edgecolor='black', framealpha=0.1)

fig.tight_layout(rect=[0, 0, 1, 0.88])
plt.show()
