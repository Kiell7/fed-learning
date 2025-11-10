#!/usr/bin/env python3
"""
查看和分析训练日志工具
用于读取 training_log.csv 并生成详细的分析报告
"""

import os
import sys
import csv
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


def load_training_log(log_path):
    """
    加载训练日志CSV文件
    
    Args:
        log_path: CSV文件路径
        
    Returns:
        pandas DataFrame
    """
    try:
        df = pd.read_csv(log_path)
        print(f"✅ 成功加载训练日志: {log_path}")
        print(f"   总轮数: {len(df)}")
        return df
    except Exception as e:
        print(f"❌ 加载日志失败: {e}")
        return None


def print_summary(df):
    """
    打印训练摘要信息
    
    Args:
        df: 训练日志DataFrame
    """
    print("\n" + "="*60)
    print("训练摘要")
    print("="*60)
    
    print(f"\n📊 总轮数: {len(df)}")
    print(f"\n🎯 准确率:")
    print(f"   最高: {df['Accuracy'].max():.4f} (轮次 {df.loc[df['Accuracy'].idxmax(), 'Round']:.0f})")
    print(f"   最低: {df['Accuracy'].min():.4f} (轮次 {df.loc[df['Accuracy'].idxmin(), 'Round']:.0f})")
    print(f"   最终: {df['Accuracy'].iloc[-1]:.4f}")
    print(f"   平均: {df['Accuracy'].mean():.4f}")
    
    print(f"\n📉 损失:")
    print(f"   最高: {df['Loss'].max():.6f} (轮次 {df.loc[df['Loss'].idxmax(), 'Round']:.0f})")
    print(f"   最低: {df['Loss'].min():.6f} (轮次 {df.loc[df['Loss'].idxmin(), 'Round']:.0f})")
    print(f"   最终: {df['Loss'].iloc[-1]:.6f}")
    print(f"   平均: {df['Loss'].mean():.6f}")
    
    # 检查是否有编码长度数据
    if 'Length_1' in df.columns and df['Length_1'].notna().any():
        print(f"\n📏 编码长度 (Length 1):")
        try:
            length_1 = pd.to_numeric(df['Length_1'], errors='coerce')
            print(f"   最大: {length_1.max():.6f} (轮次 {df.loc[length_1.idxmax(), 'Round']:.0f})")
            print(f"   最小: {length_1.min():.6f} (轮次 {df.loc[length_1.idxmin(), 'Round']:.0f})")
            print(f"   最终: {length_1.iloc[-1]:.6f}")
            print(f"   平均: {length_1.mean():.6f}")
        except:
            print("   (数据不可用)")
    
    if 'Length_2' in df.columns and df['Length_2'].notna().any():
        print(f"\n📏 编码长度 (Length 2):")
        try:
            length_2 = pd.to_numeric(df['Length_2'], errors='coerce')
            print(f"   最大: {length_2.max():.6f}")
            print(f"   最小: {length_2.min():.6f}")
            print(f"   最终: {length_2.iloc[-1]:.6f}")
            print(f"   平均: {length_2.mean():.6f}")
        except:
            print("   (数据不可用)")
    
    if 'Length_3' in df.columns and df['Length_3'].notna().any():
        print(f"\n📏 编码长度 (Length 3):")
        try:
            length_3 = pd.to_numeric(df['Length_3'], errors='coerce')
            print(f"   最大: {length_3.max():.6f}")
            print(f"   最小: {length_3.min():.6f}")
            print(f"   最终: {length_3.iloc[-1]:.6f}")
            print(f"   平均: {length_3.mean():.6f}")
        except:
            print("   (数据不可用)")
    
    print("\n" + "="*60)


def print_detailed_log(df, num_rows=10):
    """
    打印详细的训练日志
    
    Args:
        df: 训练日志DataFrame
        num_rows: 显示的行数
    """
    print(f"\n前 {num_rows} 轮详细记录:")
    print(df.head(num_rows).to_string(index=False))
    
    print(f"\n后 {num_rows} 轮详细记录:")
    print(df.tail(num_rows).to_string(index=False))


def generate_report(log_path, output_dir=None):
    """
    生成完整的训练报告，包括图表
    
    Args:
        log_path: CSV文件路径
        output_dir: 输出目录，默认为CSV文件所在目录
    """
    df = load_training_log(log_path)
    if df is None:
        return
    
    if output_dir is None:
        output_dir = os.path.dirname(log_path)
    
    # 打印摘要
    print_summary(df)
    
    # 生成图表
    fig = plt.figure(figsize=(16, 12))
    
    # 1. 准确率曲线
    plt.subplot(3, 2, 1)
    plt.plot(df['Round'], df['Accuracy'], linewidth=2, color='#2ecc71', marker='o', markersize=3)
    plt.title('Accuracy per Round', fontsize=14, fontweight='bold')
    plt.xlabel('Round', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # 2. 损失曲线
    plt.subplot(3, 2, 2)
    plt.plot(df['Round'], df['Loss'], linewidth=2, color='#e74c3c', marker='o', markersize=3)
    plt.title('Loss per Round', fontsize=14, fontweight='bold')
    plt.xlabel('Round', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # 3. 编码长度曲线（如果存在）
    plt.subplot(3, 2, 3)
    has_lengths = False
    try:
        if 'Length_1' in df.columns and df['Length_1'].notna().any():
            length_1 = pd.to_numeric(df['Length_1'], errors='coerce')
            plt.plot(df['Round'], length_1, linewidth=2, label='Length 1', 
                    color='#3498db', marker='o', markersize=3)
            has_lengths = True
        if 'Length_2' in df.columns and df['Length_2'].notna().any():
            length_2 = pd.to_numeric(df['Length_2'], errors='coerce')
            plt.plot(df['Round'], length_2, linewidth=2, label='Length 2', 
                    color='#9b59b6', marker='s', markersize=3)
            has_lengths = True
        if 'Length_3' in df.columns and df['Length_3'].notna().any():
            length_3 = pd.to_numeric(df['Length_3'], errors='coerce')
            plt.plot(df['Round'], length_3, linewidth=2, label='Length 3', 
                    color='#f39c12', marker='^', markersize=3)
            has_lengths = True
    except Exception as e:
        print(f"警告: 绘制编码长度时出错: {e}")
    
    if has_lengths:
        plt.title('Encoding Lengths per Round', fontsize=14, fontweight='bold')
        plt.xlabel('Round', fontsize=12)
        plt.ylabel('Length', fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
    else:
        plt.text(0.5, 0.5, 'No Encoding Length Data', 
                ha='center', va='center', transform=plt.gca().transAxes)
    
    # 4. 准确率变化率
    plt.subplot(3, 2, 4)
    acc_diff = df['Accuracy'].diff()
    plt.plot(df['Round'][1:], acc_diff[1:], linewidth=2, color='#16a085', marker='o', markersize=3)
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    plt.title('Accuracy Change per Round', fontsize=14, fontweight='bold')
    plt.xlabel('Round', fontsize=12)
    plt.ylabel('Accuracy Δ', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # 5. 损失变化率
    plt.subplot(3, 2, 5)
    loss_diff = df['Loss'].diff()
    plt.plot(df['Round'][1:], loss_diff[1:], linewidth=2, color='#c0392b', marker='o', markersize=3)
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    plt.title('Loss Change per Round', fontsize=14, fontweight='bold')
    plt.xlabel('Round', fontsize=12)
    plt.ylabel('Loss Δ', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # 6. 综合对比（准确率 vs 损失）
    plt.subplot(3, 2, 6)
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    
    line1 = ax1.plot(df['Round'], df['Accuracy'], linewidth=2, color='#2ecc71', 
                     label='Accuracy', marker='o', markersize=3)
    ax1.set_xlabel('Round', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12, color='#2ecc71')
    ax1.tick_params(axis='y', labelcolor='#2ecc71')
    
    line2 = ax2.plot(df['Round'], df['Loss'], linewidth=2, color='#e74c3c', 
                     label='Loss', marker='s', markersize=3)
    ax2.set_ylabel('Loss', fontsize=12, color='#e74c3c')
    ax2.tick_params(axis='y', labelcolor='#e74c3c')
    
    plt.title('Accuracy vs Loss', fontsize=14, fontweight='bold')
    
    # 合并图例
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图表
    report_path = os.path.join(output_dir, 'training_report.png')
    plt.savefig(report_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✅ 训练报告已保存到: {report_path}")
    
    # 生成文本报告
    txt_report_path = os.path.join(output_dir, 'training_summary.txt')
    with open(txt_report_path, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("训练摘要报告\n")
        f.write("="*60 + "\n\n")
        
        f.write(f"总轮数: {len(df)}\n\n")
        
        f.write(f"准确率:\n")
        f.write(f"  最高: {df['Accuracy'].max():.4f} (轮次 {df.loc[df['Accuracy'].idxmax(), 'Round']:.0f})\n")
        f.write(f"  最低: {df['Accuracy'].min():.4f} (轮次 {df.loc[df['Accuracy'].idxmin(), 'Round']:.0f})\n")
        f.write(f"  最终: {df['Accuracy'].iloc[-1]:.4f}\n")
        f.write(f"  平均: {df['Accuracy'].mean():.4f}\n\n")
        
        f.write(f"损失:\n")
        f.write(f"  最高: {df['Loss'].max():.6f} (轮次 {df.loc[df['Loss'].idxmax(), 'Round']:.0f})\n")
        f.write(f"  最低: {df['Loss'].min():.6f} (轮次 {df.loc[df['Loss'].idxmin(), 'Round']:.0f})\n")
        f.write(f"  最终: {df['Loss'].iloc[-1]:.6f}\n")
        f.write(f"  平均: {df['Loss'].mean():.6f}\n\n")
        
        if 'Length_1' in df.columns and df['Length_1'].notna().any():
            try:
                length_1 = pd.to_numeric(df['Length_1'], errors='coerce')
                f.write(f"编码长度 (Length 1):\n")
                f.write(f"  最大: {length_1.max():.6f}\n")
                f.write(f"  最小: {length_1.min():.6f}\n")
                f.write(f"  最终: {length_1.iloc[-1]:.6f}\n")
                f.write(f"  平均: {length_1.mean():.6f}\n\n")
            except:
                pass
        
        f.write("="*60 + "\n")
        f.write("详细训练记录 (前10轮)\n")
        f.write("="*60 + "\n\n")
        f.write(df.head(10).to_string(index=False))
        f.write("\n\n")
        
        f.write("="*60 + "\n")
        f.write("详细训练记录 (后10轮)\n")
        f.write("="*60 + "\n\n")
        f.write(df.tail(10).to_string(index=False))
    
    print(f"✅ 文本摘要已保存到: {txt_report_path}")


def main():
    parser = argparse.ArgumentParser(
        description='查看和分析联邦学习训练日志',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 查看最新训练日志
  python view_training_log.py ../checkpoints/bert_squad_nc20_comm100_20251107_013508/training_log.csv
  
  # 生成完整报告
  python view_training_log.py ../checkpoints/bert_squad_nc20_comm100_20251107_013508/training_log.csv --report
  
  # 显示详细日志
  python view_training_log.py ../checkpoints/bert_squad_nc20_comm100_20251107_013508/training_log.csv --detailed
        """
    )
    
    parser.add_argument('log_path', type=str, help='训练日志CSV文件路径')
    parser.add_argument('--report', '-r', action='store_true', help='生成完整报告（包括图表）')
    parser.add_argument('--detailed', '-d', action='store_true', help='显示详细日志')
    parser.add_argument('--rows', '-n', type=int, default=10, help='显示的详细日志行数（默认10）')
    parser.add_argument('--output', '-o', type=str, help='输出目录（默认为CSV所在目录）')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.log_path):
        print(f"❌ 文件不存在: {args.log_path}")
        sys.exit(1)
    
    if args.report:
        generate_report(args.log_path, args.output)
    else:
        df = load_training_log(args.log_path)
        if df is not None:
            print_summary(df)
            if args.detailed:
                print_detailed_log(df, args.rows)


if __name__ == '__main__':
    main()
