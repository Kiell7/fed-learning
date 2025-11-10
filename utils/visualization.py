import matplotlib.pyplot as plt
import torch
import numpy as np


# 训练结束后，添加可视化
def plot_training_progress(losses, accuracies, all_lengths=None, save_path='training_progress.png'):
    """
    绘制训练过程的损失、准确率和编码长度曲线

    Args:
        losses: 损失列表
        accuracies: 准确率列表
        all_lengths: 所有轮次的编码长度列表 (每个元素是一个包含3个长度值的列表)
        save_path: 保存路径（默认为当前目录）
    """
    # 确保数据在 CPU 上并且是 numpy 格式
    losses_cpu = [item.cpu().numpy() if isinstance(item, torch.Tensor) else item for item in losses]
    accuracies_cpu = [item.cpu().numpy() if isinstance(item, torch.Tensor) else item for item in accuracies]

    # 判断是否有编码长度数据
    has_lengths = all_lengths is not None and len(all_lengths) > 0
    
    if has_lengths:
        # 创建 2x2 的子图布局
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 绘制损失曲线
        axes[0, 0].plot(losses_cpu, linewidth=2, color='#e74c3c')
        axes[0, 0].set_title('Average Training Loss per Round', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Communication Round', fontsize=12)
        axes[0, 0].set_ylabel('Loss', fontsize=12)
        axes[0, 0].grid(True, alpha=0.3)

        # 绘制准确率曲线
        axes[0, 1].plot(accuracies_cpu, linewidth=2, color='#2ecc71')
        axes[0, 1].set_title('Global Model Accuracy per Round', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Communication Round', fontsize=12)
        axes[0, 1].set_ylabel('Accuracy (%)', fontsize=12)
        axes[0, 1].grid(True, alpha=0.3)

        # 绘制编码长度曲线
        # 提取三个编码长度
        try:
            length_1 = [lengths[0] if len(lengths) > 0 else 0 for lengths in all_lengths]
            length_2 = [lengths[1] if len(lengths) > 1 else 0 for lengths in all_lengths]
            length_3 = [lengths[2] if len(lengths) > 2 else 0 for lengths in all_lengths]
            
            axes[1, 0].plot(length_1, linewidth=2, label='Length 1', color='#3498db', marker='o', markersize=4)
            axes[1, 0].plot(length_2, linewidth=2, label='Length 2', color='#9b59b6', marker='s', markersize=4)
            axes[1, 0].plot(length_3, linewidth=2, label='Length 3', color='#f39c12', marker='^', markersize=4)
            axes[1, 0].set_title('Encoding Lengths per Round', fontsize=14, fontweight='bold')
            axes[1, 0].set_xlabel('Communication Round', fontsize=12)
            axes[1, 0].set_ylabel('Length', fontsize=12)
            axes[1, 0].legend(fontsize=10)
            axes[1, 0].grid(True, alpha=0.3)
            
            # 绘制综合对比图（准确率 + 编码长度）
            ax1 = axes[1, 1]
            ax2 = ax1.twinx()
            
            ax1.plot(accuracies_cpu, linewidth=2, color='#2ecc71', label='Accuracy', marker='o', markersize=3)
            ax1.set_xlabel('Communication Round', fontsize=12)
            ax1.set_ylabel('Accuracy (%)', fontsize=12, color='#2ecc71')
            ax1.tick_params(axis='y', labelcolor='#2ecc71')
            
            ax2.plot(length_1, linewidth=2, color='#3498db', label='Length 1', marker='s', markersize=3, alpha=0.7)
            ax2.set_ylabel('Encoding Length', fontsize=12, color='#3498db')
            ax2.tick_params(axis='y', labelcolor='#3498db')
            
            axes[1, 1].set_title('Accuracy vs Encoding Length', fontsize=14, fontweight='bold')
            
            # 合并图例
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='best', fontsize=10)
            ax1.grid(True, alpha=0.3)
            
        except Exception as e:
            print(f"警告: 绘制编码长度曲线时出错: {e}")
            # 如果出错，在该位置显示错误信息
            axes[1, 0].text(0.5, 0.5, f'Error plotting lengths:\n{str(e)}', 
                           ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 1].text(0.5, 0.5, 'Error plotting combined view', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
    else:
        # 没有编码长度数据，只绘制损失和准确率
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        axes[0].plot(losses_cpu, linewidth=2, color='#e74c3c')
        axes[0].set_title('Average Training Loss per Round', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Communication Round', fontsize=12)
        axes[0].set_ylabel('Loss', fontsize=12)
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(accuracies_cpu, linewidth=2, color='#2ecc71')
        axes[1].set_title('Global Model Accuracy per Round', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Communication Round', fontsize=12)
        axes[1].set_ylabel('Accuracy (%)', fontsize=12)
        axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()  # 关闭图形以释放内存
    print(f"训练过程图像已保存至 {save_path}")