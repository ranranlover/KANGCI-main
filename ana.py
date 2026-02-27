import numpy as np

# ==================== 参数配置区域 ====================
# 请确保文件路径正确
DATA_FILE_PATH = "/home/user/wcj/KANGCI-main/res-vis-50/FMRI_fusion.txt"
# ====================================================

try:
    # 加载数据（只取前50列）
    data = np.loadtxt(DATA_FILE_PATH)[:, :50]
    print(f"成功加载数据，数据形状: {data.shape}")

    # 计算每一行的平均值
    row_means = np.mean(data, axis=1)

    print("\n=== 每一行的平均值 ===")
    print("行号（i节点）\t平均值\t\t因果强度评估")
    print("-" * 50)

    # 输出每一行的平均值，并添加简单评估
    for i, mean_val in enumerate(row_means):
        # 根据平均值大小简单分类
        if mean_val > 0.7:
            assessment = "强因果影响"
        elif mean_val > 0.4:
            assessment = "中等因果影响"
        elif mean_val > 0.1:
            assessment = "弱因果影响"
        else:
            assessment = "极小因果影响"

        print(f"行 {i+1:2d}\t\t{mean_val:.6f}\t{assessment}")

    # 输出统计摘要
    print("\n=== 统计摘要 ===")
    print(f"总行数: {len(row_means)}")
    print(f"平均值的平均值: {np.mean(row_means):.6f}")
    print(f"平均值的最小值: {np.min(row_means):.6f}")
    print(f"平均值的最大值: {np.max(row_means):.6f}")
    print(f"平均值的标准差: {np.std(row_means):.6f}")

    # 找出最具影响力的行
    max_index = np.argmax(row_means)
    min_index = np.argmin(row_means)
    print(f"\n最具因果影响力的行: 行 {max_index} (平均值 = {row_means[max_index]:.6f})")
    print(f"最弱因果影响力的行: 行 {min_index} (平均值 = {row_means[min_index]:.6f})")

except FileNotFoundError:
    print(f"错误: 文件 {DATA_FILE_PATH} 未找到")
    print("请检查文件路径是否正确，或确认文件已存在")
except Exception as e:
    print(f"处理数据时出错: {e}")
