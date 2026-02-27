import matplotlib.pyplot as plt
import numpy as np

# 数据
models = ['GN-GC', 'w/o GSSR']
training_loss = [0.9763, 0.0153]
testing_loss = [1.0234, 0.9471]

x = np.arange(len(models))
width = 0.5

# 创建子图
fig, axes = plt.subplots(1, 2, figsize=(8, 6), sharey=False)

# 颜色（低饱和度）
train_color = '#6c8ebf'
test_color = '#b7c3d0'

# --- 子图1：Training Loss ---
bars1 = axes[0].bar(x, training_loss, width, color=train_color)
axes[0].set_xticks(x)
axes[0].set_xticklabels(models)
axes[0].set_ylabel('Training Loss')
axes[0].set_title('Training Loss')
axes[0].grid(axis='y', linestyle='--', alpha=0.4)

for bar in bars1:
    h = bar.get_height()
    axes[0].text(bar.get_x() + bar.get_width()/2, h, f'{h:.4f}',
                 ha='center', va='bottom', fontsize=9)

# --- 子图2：Testing Loss ---
bars2 = axes[1].bar(x, testing_loss, width, color=test_color)
axes[1].set_xticks(x)
axes[1].set_xticklabels(models)
axes[1].set_ylabel('Testing Loss')
axes[1].set_title('Testing Loss')
axes[1].grid(axis='y', linestyle='--', alpha=0.4)

for bar in bars2:
    h = bar.get_height()
    axes[1].text(bar.get_x() + bar.get_width()/2, h, f'{h:.4f}',
                 ha='center', va='bottom', fontsize=9)

# 总标题（可选）
fig.suptitle(
    'Overfitting Analysis of GN-GC and Same Model without\n'
    'Gradient Stability-Guided Sparsity Regularization (w/o GSSR)\n'
    'on Lorenz-96 Dataset with F=40',
    fontsize=12,
    y=1.00
)
plt.tight_layout()
plt.show()
