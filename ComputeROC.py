import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics


def compute_roc(gc_label, gc_predict, draw):

    # gc_predict = normalization(gc_predict)
    gc_label = gc_label.flatten().astype(float)
    gc_predict = gc_predict.flatten().astype(float)
    if draw:
        score = draw_roc_curve(gc_label, gc_predict / gc_predict.max())
    else:
        score = metrics.roc_auc_score(gc_label, gc_predict / gc_predict.max())
    return score


def compute_auprc(gc_label, gc_predict, draw):
    """
    计算 AUPRC (Area Under the Precision-Recall Curve)

    输入:
        gc_label: 真实的因果关系矩阵 (二值矩阵, PxP)
        gc_predict: 预测的因果关系分数矩阵 (浮点矩阵, PxP)
        draw: 布尔值，是否绘制 PR 曲线 (如果 draw_pr_curve 函数存在)

    输出:
        score: 计算出的 AUPRC 值
    """

    # 1. 展平矩阵并转换为浮点数
    gc_label = gc_label.flatten().astype(float)
    gc_predict = gc_predict.flatten().astype(float)

    # 2. 对预测分数进行归一化 (可选，但保持与 compute_roc 一致)
    # 归一化并不改变 AUPRC 值，但如果用于绘图或某些特殊需要，可以保留
    # 避免除以零，并确保 max() 至少为 1e-8
    max_score = np.max(gc_predict)
    if max_score < 1e-8:
        # 如果分数太小或为零，可能表明模型未训练，返回 0 或 NaN
        return 0.0

    normalized_predict = gc_predict / max_score

    if draw:
        # 如果需要绘图，调用假设的绘图函数
        # 注意: 如果 draw_pr_curve 没实现，这里会报错
        # score = draw_pr_curve(gc_label, normalized_predict)

        # 暂时只返回计算结果，如果需要绘图，请确保 draw_pr_curve 可用
        score = metrics.average_precision_score(gc_label, normalized_predict)

    else:
        # 使用 sklearn 的 average_precision_score 计算 AUPRC
        score = metrics.average_precision_score(gc_label, normalized_predict)

    return score

def normalization(GC):
    max_values = np.max(GC, axis=1, keepdims=True)
    GC = GC / max_values
    return GC


def draw_roc_curve(label, predict):
    FPR, TPR, P = metrics.roc_curve(label, predict)
    plt.plot(FPR, TPR, 'b*-', label='roc')
    plt.plot([0, 1], [0, 1], 'r--', label="45°")
    plt.legend()
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.show()
    AUC_score = metrics.auc(FPR, TPR)
    return AUC_score
