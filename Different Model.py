import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 数据准备（均值 ± 标准差）
datasets = ['Lorenz-96', 'DREAM4', 'Medical']
models = ['KAN', 'MLP', 'TCN']

# AUROC (mean, std)
auroc_mean = np.array([
    [1.000, 0.802, 0.936],   # KAN
    [1.000, 0.777, 0.892],   # MLP
    [0.909, 0.721, 0.716]    # TCN
])
auroc_std = np.array([
    [0.000, 0.001, 0.002],
    [0.000, 0.001, 0.002],
    [0.001, 0.001, 0.002]
])

# AUPRC (mean, std)
auprc_mean = np.array([
    [1.000, 0.277, 0.934],
    [1.000, 0.253, 0.894],
    [0.777, 0.128, 0.681]
])
auprc_std = np.array([
    [0.000, 0.001, 0.002],
    [0.000, 0.002, 0.002],
    [0.002, 0.002, 0.002]
])

# 更好看的配色（论文风）
colors = ['#d87636', '#eaa975', '#f4d9bd']

sns.set_context("paper", font_scale=1.2)
sns.set_style("ticks")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6), dpi=200)

def plot_bar(ax, mean_data, std_data, title, ylabel):
    x = np.arange(len(datasets))
    width = 0.25

    for i, model in enumerate(models):
        ax.bar(x + i * width, mean_data[i], width,
               yerr=std_data[i], capsize=5,
               label=model, color=colors[i], alpha=0.9)
        # ax.bar(x + i * width, mean_data[i], width,
        #        label=model, color=colors[i], alpha=0.9)

        ax.errorbar(x + i * width, mean_data[i], yerr=std_data[i],
                    fmt='none', ecolor='black', elinewidth=1.2, capsize=0)

    ax.set_title(title, fontsize=15, fontweight='bold', pad=15)
    ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(datasets, fontsize=10, fontweight='bold')

    # 纵坐标刻度加粗
    ax.tick_params(axis='y', labelsize=10)
    for label in ax.get_yticklabels():
        label.set_fontweight('bold')

    ax.grid(axis='y', linestyle=':', alpha=0.4)
    sns.despine(ax=ax, trim=False)


plot_bar(ax1, auroc_mean, auroc_std, '', 'AUROC Score')
plot_bar(ax2, auprc_mean, auprc_std, '', 'AUPRC Score')

fig.suptitle('Different Prediction Model Comparison on Different Datasets', fontsize=18, fontweight='bold', y=0.92)

# 图例
handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.87),
           ncol=3, frameon=True, fontsize=12, edgecolor='black', framealpha=0.1)

fig.tight_layout(rect=[0, 0, 1, 0.88])
plt.show()
