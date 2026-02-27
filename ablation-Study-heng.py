import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.font_manager as fm

# 1. 数据准备
models = ['GN-GC', 'w/o FDGD', 'GN-GC-mean', 'GN-GC-max', 'w/o reg', 'GN-GC-L1']
datasets = ['Lorenz-96', 'DREAM4', 'Medical']

auroc_val = {
    'Lorenz-96': [0.9993, 0.9982, 0.9982, 0.9982, 0.9962, 0.9982],
    'DREAM4': [0.8021, 0.7996, 0.7996, 0.7975, 0.7003, 0.7899],
    'Medical': [0.9351, 0.9345, 0.9306, 0.9087, 0.7206, 0.8596]
}

auprc_val = {
    'Lorenz-96': [0.9940, 0.9925, 0.9831, 0.9834, 0.9726, 0.9933],
    'DREAM4': [0.2777, 0.2762, 0.2775, 0.2774, 0.1059, 0.2743],
    'Medical': [0.9298, 0.9292, 0.9247, 0.9052, 0.6820, 0.8508]
}

# 2. 风格设置（论文风）
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['STIXGeneral']
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6), dpi=200)

# def plot_horizontal_bar(ax, data_dict, title, xlabel):
#     y = np.arange(len(models))
#     bar_h = 0.22
#
#     for i, dataset in enumerate(datasets):
#         values = data_dict[dataset]
#         ax.barh(y + (i - 1) * bar_h, values,
#                 height=bar_h, color=colors[i], alpha=0.9,
#                 edgecolor='black', linewidth=0.6,
#                 label=dataset)
#
#         # 数值标注
#         for yi, xi in zip(y, values):
#             ax.text(xi + 0.005, yi + (i - 1) * bar_h,
#                     f'{xi:.3f}', va='center',
#                     fontsize=9, fontweight='bold', color=colors[i])
#
#     ax.set_yticks(y)
#     ax.set_yticklabels(models, fontsize=11, fontweight='bold')
#     ax.set_xlabel(xlabel, fontsize=12, fontweight='bold')
#     ax.set_title(title, fontsize=15, fontweight='bold', pad=15)
#     ax.grid(axis='x', linestyle=':', alpha=0.4)
#     ax.invert_yaxis()  # 让 GN-GC 在最上面

# 绘制
def plot_horizontal_bar(ax, data_dict, title, xlabel):
    y = np.arange(len(models))
    bar_h = 0.22

    for i, dataset in enumerate(datasets):
        values = data_dict[dataset]
        ax.barh(y + (i - 1) * bar_h, values,
                height=bar_h, color=colors[i], alpha=0.9,
                edgecolor='black', linewidth=0.6,
                label=dataset)

        for yi, xi in zip(y, values):
            ax.text(xi - 0.01, yi + (i - 1) * bar_h,   # ✅ 往左挪，放进柱子内部
                    f'{xi:.3f}',
                    va='center', ha='right',
                    fontsize=9, fontweight='bold',
                    color='white')                   # ✅ 深色柱子用白字更清楚

    ax.set_yticks(y)
    ax.set_yticklabels(models, fontsize=11, fontweight='bold')
    ax.set_xlabel(xlabel, fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=15, fontweight='bold', pad=15)
    ax.grid(axis='x', linestyle=':', alpha=0.4)
    ax.invert_yaxis()


plot_horizontal_bar(ax1, auroc_val, '', 'AUROC Score')
plot_horizontal_bar(ax2, auprc_val, '', 'AUPRC Score')

fig.suptitle('Ablation Study', fontsize=18, fontweight='bold', y=0.95)

# 图例
handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center',
           bbox_to_anchor=(0.5, 0.925), ncol=3,
           frameon=True, edgecolor='black',
           framealpha=0.1, fontsize=12)

fig.tight_layout(rect=[0, 0, 1, 0.88])
plt.show()
