#!/usr/bin/env python3
"""
读取实验 checkpoint，导出基本信息并生成训练曲线（非侵入式，只读）。
用法示例：
  python3 extract_checkpoint_info.py --exp_dir ./checkpoints/bert_squad_nc20_comm100_20251107_013508
或
  python3 extract_checkpoint_info.py --checkpoint ./checkpoints/.../checkpoint_round_50.pth

该脚本不会连接训练进程，也不会修改任何正在运行的文件。
"""
import argparse
import os
import glob
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def find_latest_checkpoint(exp_dir):
    patterns = [os.path.join(exp_dir, 'checkpoint_*.pth'), os.path.join(exp_dir, '*.pth')]
    files = []
    for p in patterns:
        files.extend(glob.glob(p))
    if not files:
        return None
    files = sorted(files, key=lambda p: os.path.getmtime(p), reverse=True)
    return files[0]


def load_checkpoint(path):
    print(f"Loading checkpoint: {path}")
    ckpt = torch.load(path, map_location='cpu')
    return ckpt


def save_summary(exp_dir, ckpt, out_name='checkpoint_summary.txt'):
    summary_path = os.path.join(exp_dir, out_name)
    with open(summary_path, 'w') as f:
        f.write('Checkpoint summary\n')
        f.write('==================\n')
        for k in sorted(ckpt.keys()):
            try:
                v = ckpt[k]
                # avoid printing very large tensors
                if isinstance(v, (list, tuple)):
                    f.write(f"{k}: list(len={len(v)})\n")
                else:
                    f.write(f"{k}: {type(v)}\n")
            except Exception:
                f.write(f"{k}: <unprintable>\n")
        # try to extract commonly-stored metrics
        f.write('\nExtracted metrics:\n')
        r = ckpt.get('round', 'N/A')
        acc = ckpt.get('accuracy', None)
        loss = ckpt.get('loss', None)
        f.write(f"round: {r}\n")
        f.write(f"accuracy: {acc}\n")
        f.write(f"loss: {loss}\n")
    print(f"Summary saved to {summary_path}")
    return summary_path


def plot_metrics(exp_dir, ckpt, out_name='training_progress_from_ckpt.png'):
    path = os.path.join(exp_dir, out_name)
    # Prefer lists 'losses' and 'accuracies' if present
    losses = ckpt.get('losses', None)
    accs = ckpt.get('accuracies', None)

    if losses is not None and accs is not None:
        try:
            losses = list(map(float, losses))
            accs = list(map(float, accs))
        except Exception:
            pass
        plt.figure(figsize=(8,4))
        plt.subplot(1,2,1)
        plt.plot(range(1, len(losses)+1), losses, marker='o')
        plt.title('Loss')
        plt.xlabel('round')
        plt.grid(True)
        plt.subplot(1,2,2)
        plt.plot(range(1, len(accs)+1), accs, marker='o')
        plt.title('Accuracy')
        plt.xlabel('round')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        print(f"Saved training plot to {path}")
        return path
    else:
        # fallback: create a single-point plot using available accuracy/loss
        acc = ckpt.get('accuracy', None)
        loss = ckpt.get('loss', None)
        plt.figure(figsize=(6,3))
        if loss is not None:
            plt.subplot(1,2,1)
            plt.bar([1], [float(loss)])
            plt.title('Loss (single point)')
            plt.xticks([])
        if acc is not None:
            plt.subplot(1,2,2)
            plt.bar([1], [float(acc)])
            plt.title('Accuracy (single point)')
            plt.xticks([])
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        print(f"Saved single-point plot to {path}")
        return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_dir', type=str, default=None, help='experiment directory containing checkpoint files')
    parser.add_argument('--checkpoint', type=str, default=None, help='path to specific checkpoint .pth file')
    args = parser.parse_args()

    if args.checkpoint is None:
        if args.exp_dir is None:
            print('错误: 需要 --exp_dir 或 --checkpoint')
            return
        ckpt_path = find_latest_checkpoint(args.exp_dir)
        if ckpt_path is None:
            print(f'在 {args.exp_dir} 中找不到 checkpoint 文件')
            return
    else:
        ckpt_path = args.checkpoint
        if not os.path.exists(ckpt_path):
            print(f'指定的 checkpoint 不存在: {ckpt_path}')
            return

    exp_dir = args.exp_dir if args.exp_dir is not None else os.path.dirname(ckpt_path)

    ckpt = load_checkpoint(ckpt_path)
    save_summary(exp_dir, ckpt)
    plot_metrics(exp_dir, ckpt)
    print('\nDone.')

if __name__ == '__main__':
    main()
