#!/usr/bin/env python3
"""
快速测试训练日志功能
创建模拟数据来测试可视化和日志功能
"""

import os
import sys
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)
from utils.visualization import plot_training_progress


def create_test_data(num_rounds=20):
    """
    创建模拟训练数据
    """
    print(f"📊 生成 {num_rounds} 轮的模拟训练数据...")
    
    # 模拟损失：从高到低递减，带有一些波动
    losses = []
    base_loss = 2.5
    for i in range(num_rounds):
        noise = np.random.normal(0, 0.05)
        loss = base_loss * np.exp(-0.05 * i) + noise
        losses.append(max(0.1, loss))
    
    # 模拟准确率：从低到高递增，带有一些波动
    accuracies = []
    for i in range(num_rounds):
        noise = np.random.normal(0, 1)
        acc = 100 * (1 - np.exp(-0.08 * i)) + noise
        accuracies.append(min(99, max(10, acc)))
    
    # 模拟编码长度：三个不同的曲线
    all_lengths = []
    for i in range(num_rounds):
        length_1 = 7.0 + np.sin(i * 0.3) * 0.5 + np.random.normal(0, 0.1)
        length_2 = 5.5 + np.cos(i * 0.2) * 0.3 + np.random.normal(0, 0.08)
        length_3 = 10.0 - i * 0.01 + np.random.normal(0, 0.15)
        all_lengths.append([length_1, length_2, length_3])
    
    return losses, accuracies, all_lengths


def test_csv_logging(output_dir, num_rounds=20):
    """
    测试CSV日志记录功能
    """
    print("\n" + "="*60)
    print("测试 1: CSV 日志记录")
    print("="*60)
    
    losses, accuracies, all_lengths = create_test_data(num_rounds)
    
    # 创建测试目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建CSV文件
    log_csv_path = os.path.join(output_dir, 'training_log.csv')
    print(f"\n📝 创建CSV日志: {log_csv_path}")
    
    with open(log_csv_path, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['Round', 'Accuracy', 'Loss', 'Length_1', 'Length_2', 'Length_3'])
        
        for i in range(num_rounds):
            csv_writer.writerow([
                i + 1,
                f'{accuracies[i]:.4f}',
                f'{losses[i]:.6f}',
                f'{all_lengths[i][0]:.6f}',
                f'{all_lengths[i][1]:.6f}',
                f'{all_lengths[i][2]:.6f}'
            ])
    
    print(f"✅ CSV文件创建成功！")
    
    # 显示前几行
    print(f"\n前5行数据预览:")
    with open(log_csv_path, 'r') as f:
        for i, line in enumerate(f):
            if i < 6:  # 标题 + 5行数据
                print(f"  {line.strip()}")
    
    return log_csv_path


def test_visualization(output_dir, num_rounds=20):
    """
    测试可视化功能
    """
    print("\n" + "="*60)
    print("测试 2: 可视化曲线生成")
    print("="*60)
    
    losses, accuracies, all_lengths = create_test_data(num_rounds)
    
    # 测试可视化函数
    plot_path = os.path.join(output_dir, 'test_training_progress.png')
    print(f"\n📊 生成可视化图表: {plot_path}")
    
    plot_training_progress(losses, accuracies, all_lengths, save_path=plot_path)
    
    if os.path.exists(plot_path):
        print(f"✅ 可视化图表生成成功！")
        file_size = os.path.getsize(plot_path) / 1024  # KB
        print(f"   文件大小: {file_size:.2f} KB")
    else:
        print(f"❌ 可视化图表生成失败！")
    
    return plot_path


def test_checkpoint_format(output_dir):
    """
    测试 checkpoint 格式
    """
    print("\n" + "="*60)
    print("测试 3: Checkpoint 格式")
    print("="*60)
    
    import torch
    
    # 创建模拟checkpoint
    checkpoint = {
        'round': 50,
        'model_state_dict': {'layer.weight': torch.randn(10, 5)},
        'optimizer_state_dict': {},
        'accuracy': 75.234,
        'loss': 0.456789,
        'lengths': [7.3174, 5.4738, 10.0]
    }
    
    checkpoint_path = os.path.join(output_dir, 'test_checkpoint.pth')
    print(f"\n💾 保存测试checkpoint: {checkpoint_path}")
    
    torch.save(checkpoint, checkpoint_path)
    
    # 读取并验证
    loaded = torch.load(checkpoint_path)
    print(f"\n✅ Checkpoint 保存和加载成功！")
    print(f"\n内容:")
    print(f"  轮次: {loaded['round']}")
    print(f"  准确率: {loaded['accuracy']:.4f}")
    print(f"  损失: {loaded['loss']:.6f}")
    print(f"  编码长度: {loaded['lengths']}")
    
    return checkpoint_path


def run_all_tests():
    """
    运行所有测试
    """
    print("\n" + "="*70)
    print(" "*20 + "训练日志功能测试")
    print("="*70)
    
    # 创建测试输出目录
    test_dir = '/home/student4/njj/fed-learning-feature-dev/use_pytorch/test_output'
    print(f"\n📁 测试输出目录: {test_dir}")
    
    os.makedirs(test_dir, exist_ok=True)
    
    # 测试1: CSV日志
    csv_path = test_csv_logging(test_dir, num_rounds=20)
    
    # 测试2: 可视化
    plot_path = test_visualization(test_dir, num_rounds=20)
    
    # 测试3: Checkpoint格式
    checkpoint_path = test_checkpoint_format(test_dir)
    
    # 总结
    print("\n" + "="*70)
    print("测试总结")
    print("="*70)
    print(f"\n✅ 所有测试完成！")
    print(f"\n生成的文件:")
    print(f"  1. CSV日志: {csv_path}")
    print(f"  2. 可视化图表: {plot_path}")
    print(f"  3. Checkpoint: {checkpoint_path}")
    
    print(f"\n💡 提示: 您可以使用以下命令查看日志:")
    print(f"   python scripts/view_training_log.py {csv_path}")
    print(f"   python scripts/view_training_log.py {csv_path} --report")
    print(f"   python scripts/view_training_log.py {csv_path} --detailed")
    
    print("\n" + "="*70)


if __name__ == '__main__':
    try:
        run_all_tests()
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
