"""
DeepLabV3+ 裂纹检测 - 画图脚本
================================================
职责：只负责绘制图表
- 读取保存的训练数据
- 生成各种对比分析图表
- 生成过程性图表
不做：训练
"""

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import json
import pickle
import os
from scipy import stats

# 设置matplotlib参数
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.titlesize'] = 20
plt.rcParams['axes.labelsize'] = 20
plt.rcParams['xtick.labelsize'] = 20
plt.rcParams['ytick.labelsize'] = 20
plt.rcParams['legend.fontsize'] = 20

print("="*70)
print("DeepLabV3+ 裂纹检测 - 画图脚本")
print("="*70)
print()

# ===========================
# 检查数据文件
# ===========================
print("检查训练数据文件...")
if not os.path.exists("training_outputs/training_history.pkl"):
    print("❌ 错误: 找不到 training_history.pkl")
    print("请先运行 python training_only.py 进行训练")
    exit(1)

print("✓ 找到训练数据\n")

# ===========================
# 创建输出目录
# ===========================
os.makedirs("training_outputs/figures", exist_ok=True)
os.makedirs("training_outputs/figures/single_metrics", exist_ok=True)
os.makedirs("training_outputs/figures/comparison", exist_ok=True)
os.makedirs("training_outputs/figures/analysis", exist_ok=True)
os.makedirs("training_outputs/figures/process", exist_ok=True)

# ===========================
# 加载数据
# ===========================
print("加载训练数据...")
with open("training_outputs/training_history.pkl", "rb") as f:
    history = pickle.load(f)

# 加载CSV用于验证
df = pd.read_csv("training_outputs/training_history.csv")
num_epochs = len(df)

# 加载统计摘要
with open("training_outputs/training_summary.json", "r") as f:
    summary = json.load(f)

print(f"✓ 加载完成: {num_epochs} 个 Epoch 的数据\n")

# ===========================
# 颜色定义
# ===========================
colors = {
    'loss': '#1f77b4',      # 蓝色
    'acc': '#2ca02c',       # 绿色
    'iou': '#d62728',       # 红色
    'dice': '#ff7f0e',      # 橙色
    'f1': '#9467bd',        # 紫色
    'precision': '#8c564b', # 棕色
    'recall': '#e377c2',    # 粉色
}

# ===========================
# 1. 单指标曲线 - 简单版
# ===========================
def plot_single_metric_simple():
    """绘制单个指标的简单曲线"""
    print("绘制单指标简单曲线...")
    
    epochs = range(1, num_epochs + 1)
    
    # Loss曲线
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(epochs, history['loss'], linewidth=2.5, color=colors['loss'], label='Loss')
    ax.fill_between(epochs, history['loss'], alpha=0.3, color=colors['loss'])
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss Curve', fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('training_outputs/figures/single_metrics/01_loss.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ 01_loss.png")
    
    # Accuracy曲线
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(epochs, history['acc'], linewidth=2.5, color=colors['acc'], label='Accuracy')
    ax.fill_between(epochs, history['acc'], alpha=0.3, color=colors['acc'])
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title('Training Accuracy Curve', fontsize=20, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=20)
    plt.tight_layout()
    plt.savefig('training_outputs/figures/single_metrics/02_accuracy.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ 02_accuracy.png")
    
    # IoU曲线 (最重要)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(epochs, history['iou'], linewidth=2.5, color=colors['iou'], label='IoU')
    ax.fill_between(epochs, history['iou'], alpha=0.3, color=colors['iou'])
    best_epoch = summary['best_epoch']
    best_iou = summary['best_iou']
    ax.axvline(x=best_epoch, color='red', linestyle='--', linewidth=2, alpha=0.7, label=f'Best: {best_iou:.4f}')
    ax.plot(best_epoch, best_iou, 'r*', markersize=15, label=f'Peak at Epoch {best_epoch}')
    ax.set_xlabel('Epoch', fontsize=20)
    ax.set_ylabel('IoU', fontsize=20)
    ax.set_title('Training IoU Curve (Primary Metric)', fontsize=20, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=20)
    plt.tight_layout()
    plt.savefig('training_outputs/figures/single_metrics/03_iou.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ 03_iou.png")
    
    # Dice曲线
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(epochs, history['dice'], linewidth=2.5, color=colors['dice'], label='Dice')
    ax.fill_between(epochs, history['dice'], alpha=0.3, color=colors['dice'])
    ax.set_xlabel('Epoch', fontsize=20)
    ax.set_ylabel('Dice Coefficient', fontsize=20)
    ax.set_title('Training Dice Curve', fontsize=20, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=20)
    plt.tight_layout()
    plt.savefig('training_outputs/figures/single_metrics/04_dice.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ 04_dice.png")
    
    # F1曲线
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(epochs, history['f1'], linewidth=2.5, color=colors['f1'], label='F1 Score')
    ax.fill_between(epochs, history['f1'], alpha=0.3, color=colors['f1'])
    ax.set_xlabel('Epoch', fontsize=20)
    ax.set_ylabel('F1 Score', fontsize=20)
    ax.set_title('Training F1 Score Curve', fontsize=20, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=20)
    plt.tight_layout()
    plt.savefig('training_outputs/figures/single_metrics/05_f1.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ 05_f1.png")


# ===========================
# 2. 多指标对比 - 2x3 网格
# ===========================
def plot_comprehensive_overview():
    """绘制综合概览: 2x3 子图"""
    print("绘制综合概览 (2x3 子图)...")
    
    epochs = range(1, num_epochs + 1)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Loss
    axes[0, 0].plot(epochs, history['loss'], linewidth=2, color=colors['loss'])
    axes[0, 0].fill_between(epochs, history['loss'], alpha=0.3, color=colors['loss'])
    axes[0, 0].set_title('Loss', fontsize=20, fontweight='bold')
    axes[0, 0].set_ylabel('Value', fontsize=20)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[0, 1].plot(epochs, history['acc'], linewidth=2, color=colors['acc'])
    axes[0, 1].fill_between(epochs, history['acc'], alpha=0.3, color=colors['acc'])
    axes[0, 1].set_title('Accuracy', fontsize=20, fontweight='bold')
    axes[0, 1].set_ylabel('Value', fontsize=20)
    axes[0, 1].grid(True, alpha=0.3)
    
    # IoU
    axes[0, 2].plot(epochs, history['iou'], linewidth=2, color=colors['iou'])
    axes[0, 2].fill_between(epochs, history['iou'], alpha=0.3, color=colors['iou'])
    best_epoch = summary['best_epoch']
    best_iou = summary['best_iou']
    axes[0, 2].plot(best_epoch, best_iou, 'r*', markersize=12)
    axes[0, 2].set_title(f'IoU (Best: {best_iou:.4f})', fontsize=20, fontweight='bold')
    axes[0, 2].set_ylabel('Value', fontsize=20)
    axes[0, 2].grid(True, alpha=0.3)
    
    # Dice
    axes[1, 0].plot(epochs, history['dice'], linewidth=2, color=colors['dice'])
    axes[1, 0].fill_between(epochs, history['dice'], alpha=0.3, color=colors['dice'])
    axes[1, 0].set_xlabel('Epoch', fontsize=20)
    axes[1, 0].set_title('Dice', fontsize=20, fontweight='bold')
    axes[1, 0].set_ylabel('Value', fontsize=20)
    axes[1, 0].grid(True, alpha=0.3)
    
    # F1
    axes[1, 1].plot(epochs, history['f1'], linewidth=2, color=colors['f1'])
    axes[1, 1].fill_between(epochs, history['f1'], alpha=0.3, color=colors['f1'])
    axes[1, 1].set_xlabel('Epoch', fontsize=20)
    axes[1, 1].set_title('F1 Score', fontsize=20, fontweight='bold')
    axes[1, 1].set_ylabel('Value', fontsize=20)
    axes[1, 1].grid(True, alpha=0.3)
    
    # Epoch Time
    axes[1, 2].plot(epochs, history['epoch_time'], linewidth=2, color='brown')
    axes[1, 2].fill_between(epochs, history['epoch_time'], alpha=0.3, color='brown')
    mean_time = np.mean(history['epoch_time'])
    axes[1, 2].axhline(y=mean_time, color='red', linestyle='--', linewidth=1.5, label=f'Mean: {mean_time:.2f}s')
    axes[1, 2].set_xlabel('Epoch', fontsize=20)
    axes[1, 2].set_title('Epoch Time', fontsize=20, fontweight='bold')
    axes[1, 2].set_ylabel('Time (seconds)', fontsize=20)
    axes[1, 2].grid(True, alpha=0.3)
    axes[1, 2].legend(fontsize=10)
    
    plt.suptitle('Training Metrics Overview', fontsize=20, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig('training_outputs/figures/comparison/06_comprehensive.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ 06_comprehensive.png")


# ===========================
# 3. Loss vs 其他指标 - 双轴图
# ===========================
def plot_loss_vs_metrics():
    """绘制Loss与性能指标的关联性"""
    print("绘制Loss vs其他指标...")
    
    epochs = range(1, num_epochs + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 左图: Loss vs IoU
    ax1_twin = ax1.twinx()
    line1 = ax1.plot(epochs, history['loss'], linewidth=2.5, color=colors['loss'], label='Loss')
    line2 = ax1_twin.plot(epochs, history['iou'], linewidth=2.5, color=colors['iou'], label='IoU')
    
    ax1.set_xlabel('Epoch', fontsize=20)
    ax1.set_ylabel('Loss', fontsize=20, color=colors['loss'])
    ax1_twin.set_ylabel('IoU', fontsize=20, color=colors['iou'])
    ax1.set_title('Loss vs IoU Correlation', fontsize=20, fontweight='bold')
    ax1.tick_params(axis='y', labelcolor=colors['loss'])
    ax1_twin.tick_params(axis='y', labelcolor=colors['iou'])
    ax1.grid(True, alpha=0.3)
    
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='center left', fontsize=20)
    
    # 右图: Loss vs 多个指标
    ax2_twin = ax2.twinx()
    line1 = ax2.plot(epochs, history['loss'], linewidth=2, color=colors['loss'], label='Loss', linestyle='-', alpha=0.8)
    line2 = ax2_twin.plot(epochs, history['acc'], linewidth=2, color=colors['acc'], label='Accuracy', linestyle='--', alpha=0.8)
    line3 = ax2_twin.plot(epochs, history['dice'], linewidth=2, color=colors['dice'], label='Dice', linestyle='-.', alpha=0.8)
    line4 = ax2_twin.plot(epochs, history['f1'], linewidth=2, color=colors['f1'], label='F1', linestyle=':', alpha=0.8)
    
    ax2.set_xlabel('Epoch', fontsize=20)
    ax2.set_ylabel('Loss', fontsize=20, color=colors['loss'])
    ax2_twin.set_ylabel('Other Metrics', fontsize=20)
    ax2.set_title('Loss vs Multiple Metrics', fontsize=20, fontweight='bold')
    ax2.tick_params(axis='y', labelcolor=colors['loss'])
    ax2.grid(True, alpha=0.3)
    
    lines = line1 + line2 + line3 + line4
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels, loc='center right', fontsize=20)
    
    plt.tight_layout()
    plt.savefig('training_outputs/figures/comparison/07_loss_vs_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ 07_loss_vs_metrics.png")


# ===========================
# 4. Precision vs Recall
# ===========================
def plot_precision_recall():
    """绘制精确率vs召回率演变"""
    print("绘制Precision vs Recall...")
    
    epochs = range(1, num_epochs + 1)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(epochs, history['precision'], linewidth=2.5, color=colors['precision'], 
            label='Precision', marker='o', markersize=2, markevery=max(1, num_epochs//50))
    ax.plot(epochs, history['recall'], linewidth=2.5, color=colors['recall'], 
            label='Recall', marker='s', markersize=2, markevery=max(1, num_epochs//50))
    ax.fill_between(epochs, history['precision'], history['recall'], alpha=0.2, color='purple', 
                    label='Precision-Recall Gap')
    
    ax.set_xlabel('Epoch', fontsize=20)
    ax.set_ylabel('Score', fontsize=20)
    ax.set_title('Precision and Recall Evolution', fontsize=20, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=20, loc='best')
    ax.set_ylim([0, 1.05])
    
    plt.tight_layout()
    plt.savefig('training_outputs/figures/comparison/08_precision_recall.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ 08_precision_recall.png")


# ===========================
# 5. 学习曲线 - 阶段分析
# ===========================
def plot_learning_stages():
    """绘制学习阶段分析"""
    print("绘制学习阶段分析...")
    
    epochs = range(1, num_epochs + 1)
    
    # 划分阶段
    early_epoch = num_epochs // 4
    mid_epoch = num_epochs // 2
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 全局视图
    axes[0, 0].plot(epochs, history['iou'], linewidth=2.5, color=colors['iou'], label='IoU')
    axes[0, 0].axvspan(1, early_epoch, alpha=0.1, color='yellow', label='Early Stage')
    axes[0, 0].axvspan(early_epoch, mid_epoch, alpha=0.1, color='orange', label='Middle Stage')
    axes[0, 0].axvspan(mid_epoch, num_epochs, alpha=0.1, color='green', label='Late Stage')
    axes[0, 0].set_title('Learning Curve - Full View', fontsize=20, fontweight='bold')
    axes[0, 0].set_ylabel('IoU', fontsize=20)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend(fontsize=20)
    
    # Early阶段
    early_iou = history['iou'][:early_epoch]
    axes[0, 1].plot(range(1, len(early_iou)+1), early_iou, linewidth=2.5, color='yellow', marker='o', markersize=3)
    axes[0, 1].fill_between(range(1, len(early_iou)+1), early_iou, alpha=0.3, color='yellow')
    axes[0, 1].set_title(f'Early Stage (Epoch 1-{early_epoch})', fontsize=20, fontweight='bold')
    axes[0, 1].set_ylabel('IoU', fontsize=20)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Middle阶段
    mid_iou = history['iou'][early_epoch:mid_epoch]
    axes[1, 0].plot(range(early_epoch+1, mid_epoch+1), mid_iou, linewidth=2.5, color='orange', marker='s', markersize=3)
    axes[1, 0].fill_between(range(early_epoch+1, mid_epoch+1), mid_iou, alpha=0.3, color='orange')
    axes[1, 0].set_title(f'Middle Stage (Epoch {early_epoch+1}-{mid_epoch})', fontsize=20, fontweight='bold')
    axes[1, 0].set_ylabel('IoU', fontsize=20)
    axes[1, 0].set_xlabel('Epoch', fontsize=20)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Late阶段
    late_iou = history['iou'][mid_epoch:]
    axes[1, 1].plot(range(mid_epoch+1, num_epochs+1), late_iou, linewidth=2.5, color='green', marker='^', markersize=3)
    axes[1, 1].fill_between(range(mid_epoch+1, num_epochs+1), late_iou, alpha=0.3, color='green')
    axes[1, 1].set_title(f'Late Stage (Epoch {mid_epoch+1}-{num_epochs})', fontsize=20, fontweight='bold')
    axes[1, 1].set_ylabel('IoU', fontsize=20)
    axes[1, 1].set_xlabel('Epoch', fontsize=20)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle('Learning Curve Analysis by Training Stages', fontsize=20, fontweight='bold')
    plt.tight_layout()
    plt.savefig('training_outputs/figures/analysis/09_learning_stages.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ 09_learning_stages.png")


# ===========================
# 6. 关键指标箱线图
# ===========================
def plot_boxplots():
    """绘制箱线图"""
    print("绘制箱线图分析...")
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    axes[0, 0].boxplot(history['loss'], vert=True, patch_artist=True, 
                       boxprops=dict(facecolor=colors['loss'], alpha=0.7))
    axes[0, 0].set_title('Loss Box Plot', fontsize=20, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    axes[0, 1].boxplot(history['acc'], vert=True, patch_artist=True, 
                       boxprops=dict(facecolor=colors['acc'], alpha=0.7))
    axes[0, 1].set_title('Accuracy Box Plot', fontsize=20, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    axes[0, 2].boxplot(history['iou'], vert=True, patch_artist=True, 
                       boxprops=dict(facecolor=colors['iou'], alpha=0.7))
    axes[0, 2].set_title('IoU Box Plot', fontsize=20, fontweight='bold')
    axes[0, 2].grid(True, alpha=0.3, axis='y')
    
    axes[1, 0].boxplot(history['dice'], vert=True, patch_artist=True, 
                       boxprops=dict(facecolor=colors['dice'], alpha=0.7))
    axes[1, 0].set_title('Dice Box Plot', fontsize=20, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    axes[1, 1].boxplot(history['f1'], vert=True, patch_artist=True, 
                       boxprops=dict(facecolor=colors['f1'], alpha=0.7))
    axes[1, 1].set_title('F1 Box Plot', fontsize=20, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    axes[1, 2].boxplot(history['epoch_time'], vert=True, patch_artist=True, 
                       boxprops=dict(facecolor='brown', alpha=0.7))
    axes[1, 2].set_title('Epoch Time Box Plot', fontsize=20, fontweight='bold')
    axes[1, 2].grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Metrics Box Plot Analysis', fontsize=20, fontweight='bold')
    plt.tight_layout()
    plt.savefig('training_outputs/figures/analysis/10_boxplots.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ 10_boxplots.png")


# ===========================
# 7. 过程性图表 - 每个阶段的指标分布
# ===========================
def plot_metric_distribution():
    """绘制指标分布直方图"""
    print("绘制指标分布直方图...")
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    axes[0, 0].hist(history['loss'], bins=50, color=colors['loss'], alpha=0.7, edgecolor='black')
    axes[0, 0].axvline(np.mean(history['loss']), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(history["loss"]):.4f}')
    axes[0, 0].set_title('Loss Distribution', fontsize=20, fontweight='bold')
    axes[0, 0].set_xlabel('Loss Value', fontsize=20)
    axes[0, 0].set_ylabel('Frequency', fontsize=20)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    axes[0, 1].hist(history['acc'], bins=50, color=colors['acc'], alpha=0.7, edgecolor='black')
    axes[0, 1].axvline(np.mean(history['acc']), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(history["acc"]):.4f}')
    axes[0, 1].set_title('Accuracy Distribution', fontsize=20, fontweight='bold')
    axes[0, 1].set_xlabel('Accuracy Value', fontsize=20)
    axes[0, 1].set_ylabel('Frequency', fontsize=20)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    axes[0, 2].hist(history['iou'], bins=50, color=colors['iou'], alpha=0.7, edgecolor='black')
    axes[0, 2].axvline(np.mean(history['iou']), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(history["iou"]):.4f}')
    axes[0, 2].set_title('IoU Distribution', fontsize=20, fontweight='bold')
    axes[0, 2].set_xlabel('IoU Value', fontsize=20)
    axes[0, 2].set_ylabel('Frequency', fontsize=20)
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3, axis='y')
    
    axes[1, 0].hist(history['dice'], bins=50, color=colors['dice'], alpha=0.7, edgecolor='black')
    axes[1, 0].axvline(np.mean(history['dice']), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(history["dice"]):.4f}')
    axes[1, 0].set_title('Dice Distribution', fontsize=20, fontweight='bold')
    axes[1, 0].set_xlabel('Dice Value', fontsize=20)
    axes[1, 0].set_ylabel('Frequency', fontsize=20)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    axes[1, 1].hist(history['f1'], bins=50, color=colors['f1'], alpha=0.7, edgecolor='black')
    axes[1, 1].axvline(np.mean(history['f1']), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(history["f1"]):.4f}')
    axes[1, 1].set_title('F1 Distribution', fontsize=20, fontweight='bold')
    axes[1, 1].set_xlabel('F1 Value', fontsize=20)
    axes[1, 1].set_ylabel('Frequency', fontsize=20)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    axes[1, 2].hist(history['epoch_time'], bins=50, color='brown', alpha=0.7, edgecolor='black')
    axes[1, 2].axvline(np.mean(history['epoch_time']), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(history["epoch_time"]):.2f}s')
    axes[1, 2].set_title('Epoch Time Distribution', fontsize=20, fontweight='bold')
    axes[1, 2].set_xlabel('Time (seconds)', fontsize=20)
    axes[1, 2].set_ylabel('Frequency', fontsize=20)
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Metrics Distribution Analysis', fontsize=20, fontweight='bold')
    plt.tight_layout()
    plt.savefig('training_outputs/figures/analysis/11_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ 11_distributions.png")


# ===========================
# 8. 过程性图表 - 指标变化速度
# ===========================
def plot_metric_changes():
    """绘制指标每个epoch的变化"""
    print("绘制指标变化速度...")
    
    epochs = range(1, num_epochs + 1)
    
    # 计算每个epoch的变化
    loss_changes = np.diff(history['loss'], prepend=history['loss'][0])
    acc_changes = np.diff(history['acc'], prepend=history['acc'][0])
    iou_changes = np.diff(history['iou'], prepend=history['iou'][0])
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    axes[0].bar(epochs, loss_changes, color=colors['loss'], alpha=0.6, width=1)
    axes[0].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    axes[0].set_title('Loss Change Per Epoch', fontsize=20, fontweight='bold')
    axes[0].set_xlabel('Epoch', fontsize=20)
    axes[0].set_ylabel('ΔLoss', fontsize=20)
    axes[0].grid(True, alpha=0.3, axis='y')
    
    axes[1].bar(epochs, acc_changes, color=colors['acc'], alpha=0.6, width=1)
    axes[1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    axes[1].set_title('Accuracy Change Per Epoch', fontsize=20, fontweight='bold')
    axes[1].set_xlabel('Epoch', fontsize=20)
    axes[1].set_ylabel('ΔAccuracy', fontsize=20)
    axes[1].grid(True, alpha=0.3, axis='y')
    
    axes[2].bar(epochs, iou_changes, color=colors['iou'], alpha=0.6, width=1)
    axes[2].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    axes[2].set_title('IoU Change Per Epoch', fontsize=20, fontweight='bold')
    axes[2].set_xlabel('Epoch', fontsize=20)
    axes[2].set_ylabel('ΔIoU', fontsize=20)
    axes[2].grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Metric Changes Per Epoch', fontsize=20, fontweight='bold')
    plt.tight_layout()
    plt.savefig('training_outputs/figures/process/12_metric_changes.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ 12_metric_changes.png")


# ===========================
# 9. 过程性图表 - 累积改进
# ===========================
def plot_cumulative_improvement():
    """绘制累积改进图"""
    print("绘制累积改进图...")
    
    epochs = range(1, num_epochs + 1)
    
    # 计算相对改进
    initial_loss = history['loss'][0]
    initial_iou = history['iou'][0]
    initial_dice = history['dice'][0]
    
    loss_improvement = [(initial_loss - x) / initial_loss * 100 for x in history['loss']]
    iou_improvement = [(x - initial_iou) / initial_iou * 100 for x in history['iou']]
    dice_improvement = [(x - initial_dice) / initial_dice * 100 for x in history['dice']]
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    ax.plot(epochs, loss_improvement, linewidth=2.5, label='Loss Improvement (%)', color=colors['loss'], marker='o', markersize=2, markevery=max(1, num_epochs//50))
    ax.plot(epochs, iou_improvement, linewidth=2.5, label='IoU Improvement (%)', color=colors['iou'], marker='s', markersize=2, markevery=max(1, num_epochs//50))
    ax.plot(epochs, dice_improvement, linewidth=2.5, label='Dice Improvement (%)', color=colors['dice'], marker='^', markersize=2, markevery=max(1, num_epochs//50))
    
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xlabel('Epoch', fontsize=20)
    ax.set_ylabel('Improvement (%)', fontsize=20)
    ax.set_title('Cumulative Metric Improvement Over Training', fontsize=20, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=20, loc='best')
    
    plt.tight_layout()
    plt.savefig('training_outputs/figures/process/13_cumulative_improvement.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ 13_cumulative_improvement.png")


# ===========================
# 10. 过程性图表 - 训练速度
# ===========================
def plot_training_speed():
    """绘制训练速度相关的图表"""
    print("绘制训练速度分析...")
    
    epochs = range(1, num_epochs + 1)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Epoch时间
    axes[0].plot(epochs, history['epoch_time'], linewidth=2, color='brown')
    axes[0].fill_between(epochs, history['epoch_time'], alpha=0.3, color='brown')
    axes[0].axhline(y=np.mean(history['epoch_time']), color='red', linestyle='--', linewidth=2, 
                    label=f"Mean: {np.mean(history['epoch_time']):.2f}s")
    axes[0].set_xlabel('Epoch', fontsize=20)
    axes[0].set_ylabel('Time (seconds)', fontsize=20)
    axes[0].set_title('Epoch Training Time', fontsize=20, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=20)
    
    # 累积时间
    cumulative_time = np.cumsum(history['epoch_time'])
    axes[1].plot(epochs, cumulative_time/3600, linewidth=2.5, color='darkgreen')
    axes[1].fill_between(epochs, cumulative_time/3600, alpha=0.3, color='green')
    axes[1].set_xlabel('Epoch', fontsize=20)
    axes[1].set_ylabel('Cumulative Time (hours)', fontsize=20)
    axes[1].set_title('Cumulative Training Time', fontsize=20, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_outputs/figures/process/14_training_speed.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ 14_training_speed.png")


# ===========================
# 11. 过程性图表 - Batch损失分析
# ===========================
def plot_batch_loss_analysis():
    """绘制Batch级损失分析"""
    print("绘制Batch损失分析...")
    
    if len(history['batch_losses']) < 2:
        print("  ⚠ Batch损失数据不足，跳过此图")
        return
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # 所有batch的损失曲线
    axes[0].plot(history['batch_losses'], linewidth=0.5, color='blue', alpha=0.7)
    axes[0].set_xlabel('Batch Number', fontsize=20)
    axes[0].set_ylabel('Loss', fontsize=20)
    axes[0].set_title('Loss for Each Batch (All Epochs)', fontsize=20, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # Batch损失分布
    axes[1].hist(history['batch_losses'], bins=100, color='blue', alpha=0.7, edgecolor='black')
    axes[1].axvline(np.mean(history['batch_losses']), color='red', linestyle='--', linewidth=2, 
                    label=f'Mean: {np.mean(history["batch_losses"]):.4f}')
    axes[1].axvline(np.median(history['batch_losses']), color='green', linestyle='--', linewidth=2, 
                    label=f'Median: {np.median(history["batch_losses"]):.4f}')
    axes[1].set_xlabel('Loss Value', fontsize=20)
    axes[1].set_ylabel('Frequency', fontsize=20)
    axes[1].set_title('Batch Loss Distribution', fontsize=20, fontweight='bold')
    axes[1].legend(fontsize=20)
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('training_outputs/figures/process/15_batch_loss.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ 15_batch_loss.png")


# ===========================
# 12. 过程性图表 - 相关系数热力图
# ===========================
def plot_correlation_heatmap():
    """绘制指标相关系数热力图"""
    print("绘制相关系数热力图...")
    
    correlations_data = {
        'Loss': history['loss'],
        'Accuracy': history['acc'],
        'IoU': history['iou'],
        'Dice': history['dice'],
        'F1': history['f1'],
        'Precision': history['precision'],
        'Recall': history['recall']
    }
    
    corr_matrix = np.corrcoef([correlations_data[key] for key in correlations_data.keys()])
    
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr_matrix, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
    
    ax.set_xticks(range(len(correlations_data)))
    ax.set_yticks(range(len(correlations_data)))
    ax.set_xticklabels(correlations_data.keys(), fontsize=20, rotation=45, ha='right')
    ax.set_yticklabels(correlations_data.keys(), fontsize=20)
    
    # 添加相关系数文本
    for i in range(len(correlations_data)):
        for j in range(len(correlations_data)):
            text = ax.text(j, i, f'{corr_matrix[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=20, fontweight='bold')
    
    ax.set_title('Metrics Correlation Heatmap', fontsize=20, fontweight='bold', pad=20)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Correlation Coefficient', fontsize=20)
    
    plt.tight_layout()
    plt.savefig('training_outputs/figures/analysis/16_correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ 16_correlation_heatmap.png")


# ===========================
# 运行所有绘图函数
# ===========================
def main():
    print("\n开始绘制图表...\n")
    
    print("【单指标曲线】")
    plot_single_metric_simple()
    
    print("\n【多指标对比】")
    plot_comprehensive_overview()
    plot_loss_vs_metrics()
    plot_precision_recall()
    plot_learning_stages()
    
    print("\n【统计分析】")
    plot_boxplots()
    plot_metric_distribution()
    plot_correlation_heatmap()
    
    print("\n【过程性图表】")
    plot_metric_changes()
    plot_cumulative_improvement()
    plot_training_speed()
    plot_batch_loss_analysis()
    
    print()
    print("="*70)
    print("图表绘制完成！")
    print("="*70)
    print()
    print("图表保存位置:")
    print("  training_outputs/figures/single_metrics/    - 单指标曲线 (5张)")
    print("  training_outputs/figures/comparison/         - 多指标对比 (4张)")
    print("  training_outputs/figures/analysis/           - 统计分析 (4张)")
    print("  training_outputs/figures/process/            - 过程性图表 (4张)")
    print()
    print("总共生成: 17 张高分辨率对比分析图表")
    print()


if __name__ == "__main__":
    main()