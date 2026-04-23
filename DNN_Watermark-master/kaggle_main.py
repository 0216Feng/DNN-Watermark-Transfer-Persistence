"""
Kaggle 适配版本主脚本
直接在 Kaggle Notebook 中运行此脚本
已优化路径、内存、GPU配置
"""

import torch
import os
import sys
import argparse
import warnings
from pathlib import Path
import shutil
import subprocess
import re
import csv
import json
from datetime import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

TRANSFER_TASK_DATASET_MAP = {
    'mnist_svhn': ('MNIST', 'SVHN'),
    'mnist_fashion': ('MNIST', 'FashionMNIST'),
    'cifar10_cifar100': ('CIFAR10', 'CIFAR100'),
}

# ========== 环境配置 ==========
def get_kaggle_config():
    """自动检测和配置Kaggle环境"""
    
    IS_KAGGLE = os.path.exists('/kaggle/working')
    
    if IS_KAGGLE:
        config = {
            'is_kaggle': True,
            'data_path': '/kaggle/temp/data/',
            'trained_path': '/kaggle/working/trained/',
            'key_chain_path': '/kaggle/working/key_chain/',
            'result_path': '/kaggle/working/result/',
            'batch_size': 32,
            'num_workers': 2,   # 降低进程数
            'cuda_benchmark': True,
        }
        print("✅ Kaggle环境检测成功")
    else:
        config = {
            'is_kaggle': False,
            'data_path': './data/',
            'trained_path': './trained/',
            'key_chain_path': './key_chain/',
            'result_path': './result/',
            'batch_size': 32,
            'num_workers': 4,
            'cuda_benchmark': False,
        }
        print("⚠️  本地环境")
    
    # GPU配置
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = config['cuda_benchmark']
        config['device'] = torch.device('cuda')
        print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    else:
        config['device'] = torch.device('cpu')
        print("⚠️  CPU模式（速度会很慢）")
    
    return config


# ========== 模块导入（容错版本） ==========
def safe_import_modules(is_kaggle):
    """
    安全导入项目模块，支持多种路径配置
    
    支持以下场景：
    1. 直接在项目目录中运行
    2. 从父目录运行
    3. Kaggle环境中运行
    """
    
    import os
    
    current_dir = os.getcwd()
    project_dirs = [
        current_dir,  # 当前目录
        '/kaggle/working/DNN_Watermark-master',  # Kaggle克隆位置
        '/kaggle/working',  # Kaggle工作目录
    ]
    
    # 查找并添加包含必要文件的目录到sys.path
    found_project = False
    for proj_dir in project_dirs:
        if os.path.exists(proj_dir):
            # 检查必要文件是否存在
            required_files = ['utilities.py', 'train_backdoor.py', 'model_resnet.py']
            all_exist = all(os.path.exists(os.path.join(proj_dir, f)) for f in required_files)
            
            if all_exist:
                print(f"✅ 找到项目目录: {proj_dir}")
                if proj_dir not in sys.path:
                    sys.path.insert(0, proj_dir)
                found_project = True
                break
    
    if not found_project:
        print("\n❌ 导入失败：找不到项目文件\n")
        print("需要的文件:")
        print("   ✗ utilities.py")
        print("   ✗ train_backdoor.py")
        print("   ✗ model_resnet.py")
        print("\n解决方案:")
        print("\n方案1: Kaggle笔记本（推荐）")
        print("   在笔记本第一个单元运行：")
        print("   !git clone https://github.com/YOUR_REPO/DNN_Watermark-master.git")
        print("   %cd DNN_Watermark-master")
        print("   然后再运行训练脚本")
        print("\n方案2: 本地运行")
        print("   cd /path/to/DNN_Watermark-master")
        print("   python kaggle_main.py --dataset_index 0")
        print("\n方案3: 上传代码到Kaggle Dataset")
        print("   1. 在Kaggle创建新Dataset")
        print("   2. 上传DNN_Watermark-master文件夹")
        print("   3. 在笔记本中选择该Dataset作为输入")
        sys.exit(1)
    
    # 尝试导入模块
    try:
        print("📦 导入模块...")
        from utilities import train_test_loader, gen_key_chain
        print("   ✅ utilities.py")
        
        from train_backdoor import train
        print("   ✅ train_backdoor.py")
        
        import model_resnet
        print("   ✅ model_resnet.py")
        
        return {
            'train_test_loader': train_test_loader,
            'gen_key_chain': gen_key_chain,
            'train': train,
            'model_resnet': model_resnet,
        }
    except ImportError as e:
        print(f"\n❌ 模块导入出错: {e}")
        print("\n可能原因:")
        print("1. 文件损坏或不完整")
        print("2. Python版本不兼容")
        print("3. 缺少依赖库")
        print("\n修复步骤:")
        print("   1. 检查文件完整性: !ls -la *.py")
        print("   2. 重新克隆代码: !git clone <repo_url>")
        print("   3. 安装依赖: !pip install torch torchvision torchtext")
        
        import traceback
        print("\n详细错误:")
        traceback.print_exc()
        sys.exit(1)


# ========== 关键修复：数据加载器 ==========
def download_dataset_with_retry(dataset_class, data_path, dataset_name, transform, is_train=True, max_retries=3):
    """
    带重试机制的数据下载
    Kaggle网络有时不稳定，所以需要重试
    """
    for attempt in range(max_retries):
        try:
            print(f"   [{attempt+1}/{max_retries}] 尝试下载 {dataset_name} {'(训练集)' if is_train else '(测试集)'}...", end=' ')
            dataset = dataset_class(
                data_path, 
                train=is_train, 
                download=True, 
                transform=transform
            )
            print("✅")
            return dataset
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"失败，重试...")
                import time
                time.sleep(2)  # 等待2秒后重试
            else:
                print(f"❌")
                raise RuntimeError(f"无法下载{dataset_name}: {e}")
    
    return None


def get_train_test_loaders(dataset_name, data_path, batch_size, num_workers):
    """
    改进的数据加载器，自动在线下载，处理Kaggle限制
    
    特点:
    ✅ 自动在线下载（无需手动上传Dataset）
    ✅ 网络失败自动重试
    ✅ 检测已下载的数据（加快速度）
    ✅ 优化Kaggle GPU内存使用
    """
    import torchvision
    from torch.utils.data import DataLoader
    
    # 创建数据目录
    os.makedirs(data_path, exist_ok=True)
    
    print(f"\n📥 数据集: {dataset_name}")
    print(f"📁 存储路径: {data_path}")
    
    # 检查数据是否已存在
    is_kaggle = os.path.exists('/kaggle/working')
    if is_kaggle:
        import subprocess
        result = subprocess.run(['du', '-sh', data_path], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"💾 已有数据: {result.stdout.strip()}")
    
    if dataset_name == 'MNIST':
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,))
        ])
        trainset = download_dataset_with_retry(
            torchvision.datasets.MNIST, data_path, dataset_name, transform, is_train=True
        )
        testset = download_dataset_with_retry(
            torchvision.datasets.MNIST, data_path, dataset_name, transform, is_train=False
        )
        
    elif dataset_name == 'CIFAR10':
        transform_train = torchvision.transforms.Compose([
            torchvision.transforms.RandomCrop(32, padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
            ),
        ])
        transform_test = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
            ),
        ])
        trainset = download_dataset_with_retry(
            torchvision.datasets.CIFAR10, data_path, dataset_name, transform_train, is_train=True
        )
        testset = download_dataset_with_retry(
            torchvision.datasets.CIFAR10, data_path, dataset_name, transform_test, is_train=False
        )
        
    elif dataset_name == 'FashionMNIST':
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.2860,), (0.3530,))
        ])
        trainset = download_dataset_with_retry(
            torchvision.datasets.FashionMNIST, data_path, dataset_name, transform, is_train=True
        )
        testset = download_dataset_with_retry(
            torchvision.datasets.FashionMNIST, data_path, dataset_name, transform, is_train=False
        )

    elif dataset_name == 'SVHN':
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))
        ])
        trainset = torchvision.datasets.SVHN(data_path, split='train', download=True, transform=transform)
        testset = torchvision.datasets.SVHN(data_path, split='test', download=True, transform=transform)
        
    elif dataset_name == 'CIFAR100':
        # CIFAR-100: 100个细粒度类别（用于迁移学习对比实验）
        transform_train = torchvision.transforms.Compose([
            torchvision.transforms.RandomCrop(32, padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
            ),
        ])
        transform_test = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
            ),
        ])
        trainset = download_dataset_with_retry(
            torchvision.datasets.CIFAR100, data_path, dataset_name, transform_train, is_train=True
        )
        testset = download_dataset_with_retry(
            torchvision.datasets.CIFAR100, data_path, dataset_name, transform_test, is_train=False
        )

    elif dataset_name == 'FOOD101':
        transform_train = torchvision.transforms.Compose([
            torchvision.transforms.Resize((256, 256)),
            torchvision.transforms.RandomResizedCrop(224),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5567, 0.4381, 0.3198), (0.2591, 0.2623, 0.2633))
        ])
        transform_test = torchvision.transforms.Compose([
            torchvision.transforms.Resize((256, 256)),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5567, 0.4381, 0.3198), (0.2591, 0.2623, 0.2633))
        ])
        trainset = torchvision.datasets.Food101(data_path, split='train', download=True, transform=transform_train)
        testset = torchvision.datasets.Food101(data_path, split='test', download=True, transform=transform_test)
    else:
        raise ValueError(f"不支持的数据集: {dataset_name}")
    
    # 使用pin_memory加速数据传输到GPU
    pin_memory = torch.cuda.is_available()
    
    print(f"🔄 创建DataLoader (batch_size={batch_size}, num_workers={num_workers})...", end=' ')
    
    trainloader = DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=(num_workers > 0)  # 保持workers活跃避免重复创建
    )
    testloader = DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0)
    )
    
    print("✅")
    print(f"✅ 数据加载完成! 训练集: {len(trainset)}, 测试集: {len(testset)}")
    return trainloader, testloader


def _collect_and_save_training_artifacts(trained_root, result_root, dataset_name, mode_name, run_meta):
    """自动收集训练结果并保存 CSV/PNG/JSON 汇总。"""
    target_dir = Path(trained_root) / dataset_name / mode_name
    result_dir = Path(result_root)
    result_dir.mkdir(parents=True, exist_ok=True)

    if not target_dir.exists():
        print(f"⚠️ 未找到训练输出目录: {target_dir}")
        return None

    ckpts = sorted(target_dir.glob('*.pt'), key=lambda p: p.stat().st_mtime)
    if not ckpts:
        print(f"⚠️ 未找到模型文件: {target_dir}")
        return None

    pattern = re.compile(r"Epoch_(\d+)_test_acc_([0-9.]+)%_trigger_acc_([0-9.]+)%")
    rows = []

    # 优先使用训练阶段写出的逐轮历史数据（仅保存最优checkpoint时也能画完整曲线）
    epoch_history_path = target_dir / 'epoch_history.csv'
    if epoch_history_path.exists():
        with open(epoch_history_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    rows.append({
                        'epoch': int(float(row.get('epoch', 0))),
                        'test_acc': float(row.get('test_acc', 0.0)),
                        'trigger_acc': float(row.get('trigger_acc', 0.0)),
                        'checkpoint': row.get('checkpoint', ''),
                    })
                except Exception:
                    continue
    else:
        for p in ckpts:
            m = pattern.search(p.name)
            if not m:
                continue
            rows.append({
                'epoch': int(m.group(1)),
                'test_acc': float(m.group(2)),
                'trigger_acc': float(m.group(3)),
                'checkpoint': str(p),
            })

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    prefix = f"{dataset_name}_{mode_name}_{timestamp}"

    csv_path = result_dir / f"{prefix}_history.csv"
    png_path = result_dir / f"{prefix}_curve.png"
    json_path = result_dir / f"{prefix}_summary.json"

    if rows:
        rows.sort(key=lambda x: x['epoch'])
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['epoch', 'test_acc', 'trigger_acc', 'checkpoint'])
            writer.writeheader()
            writer.writerows(rows)

        epochs = [r['epoch'] for r in rows]
        test_acc = [r['test_acc'] for r in rows]
        trigger_acc = [r['trigger_acc'] for r in rows]

        plt.figure(figsize=(10, 5))
        plt.plot(epochs, test_acc, marker='o', linewidth=2, label='Test Accuracy (%)')
        plt.plot(epochs, trigger_acc, marker='s', linewidth=2, label='Trigger Accuracy (%)')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.title(f'{dataset_name} - {mode_name} Training Curves')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend()
        plt.tight_layout()
        plt.savefig(png_path, dpi=200)
        plt.close()

        best = max(rows, key=lambda x: x['test_acc'])
        latest = rows[-1]
    else:
        best = None
        latest = {
            'checkpoint': str(ckpts[-1]),
            'epoch': None,
            'test_acc': None,
            'trigger_acc': None,
        }

    summary = {
        'run_meta': run_meta,
        'trained_output_dir': str(target_dir),
        'num_checkpoints': len(ckpts),
        'latest_checkpoint': latest,
        'best_by_test_acc': best,
        'history_csv': str(csv_path) if rows else None,
        'curve_png': str(png_path) if rows else None,
    }
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n📦 自动保存完成:")
    print(f"   - Summary JSON: {json_path}")
    if rows:
        print(f"   - History CSV:  {csv_path}")
        print(f"   - Curve PNG:    {png_path}")
    return summary


def _find_latest_checkpoint(search_dir):
    p = Path(search_dir)
    if not p.exists():
        return None
    files = sorted(p.glob('*.pt'), key=lambda x: x.stat().st_mtime)
    return str(files[-1]) if files else None


def _collect_and_save_transfer_artifacts(transfer_root, result_root, run_meta):
    """收集 transfer 学习日志并自动绘图到 result 目录。"""
    transfer_dir = Path(transfer_root)
    result_dir = Path(result_root)
    result_dir.mkdir(parents=True, exist_ok=True)

    if not transfer_dir.exists():
        print(f"⚠️ 未找到迁移输出目录: {transfer_dir}")
        return None

    csv_files = sorted(transfer_dir.glob('history_*.csv'), key=lambda p: p.stat().st_mtime)
    if not csv_files:
        print(f"⚠️ 未找到迁移历史CSV: {transfer_dir}")
        return None

    latest_csv = csv_files[-1]
    rows = []
    with open(latest_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if not rows:
        print(f"⚠️ 迁移历史CSV为空: {latest_csv}")
        return None

    def _to_float(v, default=0.0):
        try:
            return float(v)
        except Exception:
            return default

    epochs = [int(_to_float(r.get('epoch', 0), 0.0)) for r in rows]
    target_acc = [_to_float(r.get('target_acc', 0.0)) for r in rows]
    trigger_acc = [_to_float(r.get('trigger_acc', 0.0)) for r in rows]
    cosine_shift = [_to_float(r.get('cosine_shift', 0.0)) for r in rows]
    margin_drop = [_to_float(r.get('target_margin_drop', 0.0)) for r in rows]

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    prefix = f"transfer_{run_meta.get('task', 'unknown')}_{timestamp}"
    png_path = result_dir / f"{prefix}_curve.png"
    json_path = result_dir / f"{prefix}_summary.json"

    fig, ax1 = plt.subplots(figsize=(10, 6))
    l1, = ax1.plot(epochs, target_acc, color='#1f77b4', marker='o', linewidth=2, label='Target Acc (%)')
    l2, = ax1.plot(epochs, trigger_acc, color='#d62728', marker='s', linewidth=2, label='Trigger Acc (%)')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy (%)')
    ax1.grid(True, linestyle='--', alpha=0.5)

    ax2 = ax1.twinx()
    l3, = ax2.plot(epochs, cosine_shift, color='#2ca02c', marker='^', linestyle='--', linewidth=1.8,
                   label='Cosine Shift')
    l4, = ax2.plot(epochs, margin_drop, color='#9467bd', marker='d', linestyle='--', linewidth=1.8,
                   label='Margin Drop')
    ax2.set_ylabel('Geometric Metrics')

    lines = [l1, l2, l3, l4]
    ax1.legend(lines, [l.get_label() for l in lines], loc='upper center', bbox_to_anchor=(0.5, 1.16), ncol=2)
    plt.title('Transfer Learning Curves')
    fig.tight_layout()
    plt.savefig(png_path, dpi=200)
    plt.close(fig)

    summary = {
        'metric_contract': {
            'primary_trigger_metric': 'source_head_swap',
            'secondary_trigger_metric': 'target_head',
            'primary_retention_definition': 'accuracy on canonical trigger set after transfer/recovery under source-head-swap divided by accuracy on same trigger set before transfer'
        },
        'run_meta': run_meta,
        'transfer_dir': str(transfer_dir),
        'source_csv': str(latest_csv),
        'transfer_curve_png': str(png_path),
        'num_epochs': len(rows),
        'final_target_acc': target_acc[-1] if target_acc else None,
        'final_trigger_acc': trigger_acc[-1] if trigger_acc else None,
    }
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n📦 迁移阶段自动保存完成:")
    print(f"   - Transfer Summary: {json_path}")
    print(f"   - Transfer Curve:   {png_path}")
    return summary


def _run_class_distribution_plot(project_root, source_model_path, target_model_path, dataset_name, data_root, result_root):
    """调用 draw_pic.py 的 class_distribution 模式，自动生成 10 分类特征分布图。"""
    output_dir = Path(result_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    png_path = output_dir / f'feature_distribution_{dataset_name}_{timestamp}.png'

    cmd = [
        sys.executable,
        str(Path(project_root) / 'draw_pic.py'),
        '--class_distribution',
        '--dataset', dataset_name,
        '--source_model', source_model_path,
        '--target_model', target_model_path,
        '--data_root', data_root,
        '--output', str(png_path),
        '--max_points', '10000',
    ]

    print("\n🎨 开始生成10分类特征分布图...")
    print("   命令:", ' '.join(cmd))
    subprocess.run(cmd, check=True)

    if png_path.exists():
        print("✅ 特征分布图已生成:")
        print(f"   - PNG: {png_path}")
        return str(png_path)

    raise FileNotFoundError(f'特征分布图生成失败: {png_path}')


def _build_transfer_save_path(config, args, run_tag=None):
    transfer_source_dataset, transfer_target_dataset = TRANSFER_TASK_DATASET_MAP[args.transfer_task]
    transfer_save_path = args.transfer_save_path or os.path.join(
        config['trained_path'], 'transfer_runs', f"{transfer_source_dataset}2{transfer_target_dataset}"
    )
    if run_tag:
        transfer_save_path = os.path.join(transfer_save_path, run_tag)
    os.makedirs(transfer_save_path, exist_ok=True)
    return transfer_save_path


def _run_transfer_step(project_root, config, args, source_model_path, replay_ratio=None, run_tag=None):
    transfer_save_path = _build_transfer_save_path(config, args, run_tag=run_tag)
    transfer_replay_ratio = args.transfer_replay_ratio if replay_ratio is None else replay_ratio

    cmd = [
        sys.executable,
        str(Path(project_root) / 'transfer_learning.py'),
        '--task', args.transfer_task,
        '--source_model', source_model_path,
        '--num_epochs', str(args.transfer_num_epochs),
        '--learning_rate', str(args.transfer_learning_rate),
        '--freeze_policy', args.transfer_freeze_policy,
        '--replay_ratio', str(transfer_replay_ratio),
        '--replay_weight', str(args.transfer_replay_weight),
        '--feature_anchor_weight', str(args.transfer_feature_anchor_weight),
        '--replay_ramp_ratio', str(args.transfer_replay_ramp_ratio),
        '--trigger_full_threshold', str(args.transfer_trigger_full_threshold),
        '--trigger_rescue_threshold', str(args.transfer_trigger_rescue_threshold),
        '--replay_sustain_scale_after_full', str(args.transfer_replay_sustain_scale_after_full),
        '--trigger_rescue_hold_ratio', str(args.transfer_trigger_rescue_hold_ratio),
        '--replay_rescue_scale', str(args.transfer_replay_rescue_scale),
        '--replay_start_trigger_threshold', str(args.transfer_replay_start_trigger_threshold),
        '--seed', str(args.transfer_seed),
        '--warmup_ratio', str(args.transfer_warmup_ratio),
        '--lr_milestone1_ratio', str(args.transfer_lr_milestone1_ratio),
        '--lr_milestone2_ratio', str(args.transfer_lr_milestone2_ratio),
        '--adaptive_drop_threshold', str(args.transfer_adaptive_drop_threshold),
        '--adaptive_decay', str(args.transfer_adaptive_decay),
        '--adaptive_growth', str(args.transfer_adaptive_growth),
        '--adaptive_min_factor', str(args.transfer_adaptive_min_factor),
        '--adaptive_target_floor_ratio', str(args.transfer_adaptive_target_floor_ratio),
        '--adaptive_trigger_guard', str(args.transfer_adaptive_trigger_guard),
        '--adaptive_guard_decay', str(args.transfer_adaptive_guard_decay),
        '--replay_ratio_cap', str(args.transfer_replay_ratio_cap),
        '--data_path', args.transfer_data_path or config['data_path'],
        '--save_path', transfer_save_path,
        '--device', str(config['device']),
    ]
    if args.no_transfer_penultimate_align:
        cmd.append('--no_penultimate_align')
    if args.no_transfer_adaptive_replay:
        cmd.append('--no_adaptive_replay')
    if args.transfer_lambda_sp is not None:
        cmd.extend(['--lambda_sp', str(args.transfer_lambda_sp)])
    if args.trigger_file:
        cmd.extend(['--trigger_file', args.trigger_file])

    print("\n🚚 开始迁移学习步骤...")
    print(f"   replay_ratio: {transfer_replay_ratio}")
    if run_tag:
        print(f"   run_tag: {run_tag}")
    print("   命令:", ' '.join(cmd))
    subprocess.run(cmd, check=True)

    target_model_path = _find_latest_checkpoint(transfer_save_path)
    if not target_model_path:
        raise FileNotFoundError(f'迁移学习完成后未找到输出模型: {transfer_save_path}')
    return source_model_path, target_model_path, transfer_save_path, transfer_replay_ratio


def _collect_transfer_compare_summary(result_root, run_meta, run_summaries):
    result_dir = Path(result_root)
    result_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    json_path = result_dir / f"transfer_compare_{timestamp}_summary.json"

    def _latest_or_none(summary):
        latest = summary.get('latest_checkpoint') or {}
        return {
            'checkpoint': latest.get('checkpoint'),
            'epoch': latest.get('epoch'),
            'test_acc': latest.get('test_acc'),
            'trigger_acc': latest.get('trigger_acc'),
        }

    compare_rows = []
    for item in run_summaries:
        summary = item['summary']
        latest = _latest_or_none(summary)
        compare_rows.append({
            'label': item['label'],
            'replay_ratio': item['replay_ratio'],
            'save_path': item['save_path'],
            'latest_checkpoint': latest,
            'best_by_test_acc': summary.get('best_by_test_acc'),
            'history_csv': summary.get('history_csv'),
            'curve_png': summary.get('curve_png'),
        })

    compare_summary = {
        'run_meta': run_meta,
        'runs': compare_rows,
    }

    if len(compare_rows) >= 2:
        first = compare_rows[0]['latest_checkpoint'] or {}
        second = compare_rows[1]['latest_checkpoint'] or {}
        compare_summary['delta_latest_test_acc'] = None
        compare_summary['delta_latest_trigger_acc'] = None
        if first.get('test_acc') is not None and second.get('test_acc') is not None:
            compare_summary['delta_latest_test_acc'] = second['test_acc'] - first['test_acc']
        if first.get('trigger_acc') is not None and second.get('trigger_acc') is not None:
            compare_summary['delta_latest_trigger_acc'] = second['trigger_acc'] - first['trigger_acc']

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(compare_summary, f, ensure_ascii=False, indent=2)

    print("\n📊 Replay 对比完成:")
    print(f"   - Compare Summary: {json_path}")
    for row in compare_rows:
        latest = row['latest_checkpoint'] or {}
        print(
            f"   - {row['label']}: replay_ratio={row['replay_ratio']} | "
            f"test_acc={latest.get('test_acc')} | trigger_acc={latest.get('trigger_acc')}"
        )
    return compare_summary


def _run_verify_step(project_root, config, args, source_model_path, target_model_path):
    output_dir = args.verify_output_dir or config['result_path']
    os.makedirs(output_dir, exist_ok=True)
    cmd = [
        sys.executable,
        str(Path(project_root) / 'verify_watermark.py'),
        '--source_model', source_model_path,
        '--target_model', target_model_path,
        '--n_triggers', str(args.verify_n_triggers),
        '--m_chains', str(args.verify_m_chains),
        '--data_path', args.verify_data_path or config['data_path'],
        '--dataset_name', args.verify_dataset_name,
        '--output_dir', output_dir,
        '--device', str(config['device']),
    ]
    if args.trigger_file:
        cmd.extend(['--trigger_file', args.trigger_file])
    print("\n🔍 开始水印验证步骤...")
    print("   命令:", ' '.join(cmd))
    subprocess.run(cmd, check=True)


# ========== 主函数 ==========
def main():
    """Kaggle优化的主训练函数"""
    
    # 配置环境
    config = get_kaggle_config()
    
    # 创建必要目录
    Path(config['data_path']).mkdir(parents=True, exist_ok=True)
    Path(config['trained_path']).mkdir(parents=True, exist_ok=True)
    Path(config['key_chain_path']).mkdir(parents=True, exist_ok=True)
    Path(config['result_path']).mkdir(parents=True, exist_ok=True)
    
    # 命令行参数
    parser = argparse.ArgumentParser(description='DNN Watermarking - Kaggle版本')
    parser.add_argument('--pipeline_step', default='embed', type=str,
                       choices=['baseline', 'embed', 'transfer', 'verify', 'recovery', 'full', 'full_compare', 'full_from_source'],
                       help='流水线步骤: baseline(仅训练baseline/ref), embed(仅嵌入), transfer(仅迁移), verify(仅验证), recovery(验证+恢复分析), full(从baseline开始全流程), full_compare(先跑replay=0.0再跑默认replay对比), full_from_source(从已有source模型开始)')
    parser.add_argument('--dataset_index', default=0, type=int,
                       help='0: MNIST, 1: CIFAR10, 2: CIFAR100, 3: FOOD101, 4: SVHN (本脚本用于水印训练，迁移学习请用transfer_learning.py)')
    parser.add_argument('--embed_mode', default=0, type=int,
                       help='0: ref, 8: FixLL+PFL (推荐), 具体见源代码')
    parser.add_argument('--n', default=100, type=int, help='触发器样本数')
    parser.add_argument('--m', default=10, type=int, help='触发器链长度')
    parser.add_argument('--batch_size', default=config['batch_size'], type=int)
    parser.add_argument('--mix', default=4, type=int, help='混入的触发器数')
    parser.add_argument('--num_workers', default=config['num_workers'], type=int)
    parser.add_argument('--source_model', default=None, type=str, help='迁移/验证时指定源模型路径')
    parser.add_argument('--target_model', default=None, type=str, help='验证时指定目标模型路径')
    parser.add_argument('--trigger_file', default=None, type=str,
                       help='可选：指定触发器文件(.pt)路径，训练和验证都优先使用该文件')
    parser.add_argument('--ref_model_path', default=None, type=str,
                       help='嵌入训练时指定reference模型checkpoint路径(优先于自动搜索)')
    parser.add_argument('--transfer_task', default='mnist_svhn', type=str,
                       choices=list(TRANSFER_TASK_DATASET_MAP.keys()))
    parser.add_argument('--transfer_num_epochs', default=50, type=int)
    parser.add_argument('--transfer_learning_rate', default=5e-5, type=float)
    parser.add_argument('--transfer_lambda_sp', default=0.5, type=float)
    parser.add_argument('--transfer_replay_ratio', default=0.15, type=float)
    parser.add_argument('--transfer_replay_weight', default=0.25, type=float)
    parser.add_argument('--transfer_feature_anchor_weight', default=0.02, type=float)
    parser.add_argument('--transfer_replay_ramp_ratio', default=0.3, type=float)
    parser.add_argument('--transfer_trigger_full_threshold', default=0.999, type=float)
    parser.add_argument('--transfer_trigger_rescue_threshold', default=0.97, type=float)
    parser.add_argument('--transfer_replay_sustain_scale_after_full', default=0.10, type=float)
    parser.add_argument('--transfer_trigger_rescue_hold_ratio', default=0.06, type=float)
    parser.add_argument('--transfer_replay_rescue_scale', default=0.35, type=float)
    parser.add_argument('--transfer_replay_start_trigger_threshold', default=0.85, type=float)
    parser.add_argument('--transfer_seed', default=42, type=int)
    parser.add_argument('--transfer_warmup_ratio', default=0.3, type=float)
    parser.add_argument('--transfer_lr_milestone1_ratio', default=0.65, type=float)
    parser.add_argument('--transfer_lr_milestone2_ratio', default=0.88, type=float)
    parser.add_argument('--no_transfer_adaptive_replay', action='store_true')
    parser.add_argument('--transfer_adaptive_drop_threshold', default=0.012, type=float)
    parser.add_argument('--transfer_adaptive_decay', default=0.8, type=float)
    parser.add_argument('--transfer_adaptive_growth', default=1.0, type=float)
    parser.add_argument('--transfer_adaptive_min_factor', default=0.05, type=float)
    parser.add_argument('--transfer_adaptive_target_floor_ratio', default=0.95, type=float)
    parser.add_argument('--transfer_adaptive_trigger_guard', default=0.98, type=float)
    parser.add_argument('--transfer_adaptive_guard_decay', default=0.85, type=float)
    parser.add_argument('--transfer_replay_ratio_cap', default=0.06, type=float)
    parser.add_argument('--transfer_freeze_policy', default='train_l4_fc_out', type=str,
                       choices=['train_l4_fc_out', 'train_fc_out'])
    parser.add_argument('--no_transfer_penultimate_align', action='store_true',
                       help='关闭迁移阶段的 penultimate 维度自动对齐，恢复旧行为')
    parser.add_argument('--transfer_data_path', default=None, type=str)
    parser.add_argument('--transfer_save_path', default=None, type=str)
    parser.add_argument('--verify_dataset_name', default=None, type=str, choices=['SVHN', 'FashionMNIST', 'CIFAR100'],
                       help='验证目标数据集；默认跟随 --transfer_task 自动设置')
    parser.add_argument('--verify_n_triggers', default=None, type=int,
                       help='验证触发器样本数；默认跟随训练参数 --n')
    parser.add_argument('--verify_m_chains', default=None, type=int,
                       help='验证触发器链长度；默认跟随训练参数 --m')
    parser.add_argument('--verify_data_path', default=None, type=str)
    parser.add_argument('--verify_output_dir', default=None, type=str)
    
    args = parser.parse_args()
    
    # 数据集名称映射
    dataset_names = ['MNIST', 'CIFAR10', 'CIFAR100', 'FOOD101', 'SVHN']
    embed_modes = {
        0: 'ref', 1: 'scratch', 2: 'FTAL', 3: 'FTLL', 4: 'FTAL+PGR',
        5: 'FTAL+TWL', 6: 'FixLL', 7: 'FixLL+TWL', 8: 'FixLL+PFL', 9: 'FixLL+SPL'
    }
    if args.dataset_index < 0 or args.dataset_index >= len(dataset_names):
        raise ValueError(f"不支持的数据集索引: {args.dataset_index}")
    selected_dataset = dataset_names[args.dataset_index]
    KEY_DIM = (28, 32, 32, 224, 32)
    transfer_source_dataset, transfer_target_dataset = TRANSFER_TASK_DATASET_MAP[args.transfer_task]

    if args.verify_dataset_name is None:
        args.verify_dataset_name = transfer_target_dataset

    # 默认让验证参数对齐训练参数，避免 full 流程中 n/m 错配
    if args.verify_n_triggers is None:
        args.verify_n_triggers = args.n
    if args.verify_m_chains is None:
        args.verify_m_chains = args.m

    # 未显式传入 trigger_file 时，自动锁定同一份触发器文件贯穿全流程
    if not args.trigger_file:
        if args.pipeline_step in ['transfer', 'verify', 'recovery', 'full_from_source']:
            trigger_dim_map = {'MNIST': 28, 'CIFAR10': 32, 'CIFAR100': 32, 'FOOD101': 224, 'SVHN': 32}
            trigger_dim = trigger_dim_map[transfer_source_dataset]
        else:
            trigger_dim = KEY_DIM[args.dataset_index]
        args.trigger_file = str(
            Path(config['key_chain_path']) /
            f'trigger_key_chain_{trigger_dim}_{args.n}_{args.m}.pt'
        )
    
    print("\n" + "="*70)
    print("🚀 DNN Watermarking - Kaggle训练")
    print("="*70)
    print(f"📊 配置:")
    print(f"   数据集: {selected_dataset}")
    print(f"   流水线: {args.pipeline_step}")
    print(f"   迁移任务: {args.transfer_task} ({transfer_source_dataset} -> {transfer_target_dataset})")
    print(f"   模式: {embed_modes[args.embed_mode]}")
    print(f"   训练触发器参数: n={args.n}, m={args.m} (总数={args.n*args.m})")
    print(f"   验证触发器参数: n={args.verify_n_triggers}, m={args.verify_m_chains} (总数={args.verify_n_triggers*args.verify_m_chains})")
    print(f"   触发器文件: {args.trigger_file}")
    print(f"   批次大小: {args.batch_size}")
    if args.ref_model_path:
        print(f"   ref路径: {args.ref_model_path}")
    print(f"   设备: {config['device']}")
    print("="*70 + "\n")
    
    # 导入模块
    modules = safe_import_modules(config['is_kaggle'])
    train_test_loader = modules['train_test_loader']
    gen_key_chain = modules['gen_key_chain']
    train = modules['train']
    
    print("✅ 所有模块导入成功\n")
    
    source_model_path = args.source_model
    target_model_path = args.target_model

    if args.pipeline_step in ['baseline', 'embed', 'full', 'full_compare']:
        # 加载数据（使用改进的加载器）
        try:
            trainloader, testloader = get_train_test_loaders(
                selected_dataset,
                config['data_path'],
                args.batch_size,
                args.num_workers
            )
        except Exception as e:
            print(f"\n❌ 数据加载失败: {e}")
            print("\n💡 Kaggle数据下载故障排查:")
            print("   1. 检查网络连接 (Kaggle有时网卡较慢)")
            print("   2. 清空缓存: !rm -rf /kaggle/temp/data")
            print("   3. 重新运行脚本（会重新下载）")
            print("   4. 如果仍然失败，手动上传数据到/kaggle/input/")
            sys.exit(1)
    
        # 触发器生成
        print("🔑 生成触发器集合...")
        # 确保key_chain目录存在
        os.makedirs(config['key_chain_path'], exist_ok=True)

        trigger_file = Path(args.trigger_file)
    
        if not trigger_file.exists():
            print(f"   生成新的触发器... (可能需要几分钟)")
        
        # 临时改变当前目录以便gen_key_chain能正确保存
            original_dir = os.getcwd()
            os.chdir(config['key_chain_path'])
        
            try:
            # gen_key_chain会自动在./key_chain/下保存，需要创建该目录
                os.makedirs('./key_chain/', exist_ok=True)
            
                gen_key_chain(
                    dim=KEY_DIM[args.dataset_index],
                    n=args.n,
                    m=args.m,
                    save=True
                )
            
            # 将生成的文件移到正确位置
                generated_file = f'./key_chain/trigger_key_chain_{KEY_DIM[args.dataset_index]}_{args.n}_{args.m}.pt'
                if os.path.exists(generated_file):
                    target_path = str(trigger_file)
                    target_parent = str(Path(target_path).parent)
                    os.makedirs(target_parent, exist_ok=True)
                    shutil.move(generated_file, target_path)
                    print(f"   ✅ 触发器生成完成")
                else:
                    raise FileNotFoundError(f"触发器生成失败: {generated_file} 不存在")
            finally:
                os.chdir(original_dir)
        else:
            print(f"   使用已存在的触发器")
    
        # 加载触发器
        trigger_set = torch.load(str(trigger_file))
        trigger_sample = trigger_set['data']
        trigger_label = trigger_set['target']
        print(f"✅ 触发器已加载 (数量: {len(trigger_sample)})")
    
        # 开始训练
        print("\n" + "="*70)
        print("🏋️  开始训练...")
        print("="*70 + "\n")
        baseline_ckpt = None
        train_stages = []
        selected_mode_name = embed_modes[args.embed_mode]

        if args.pipeline_step == 'baseline':
            train_stages = ['ref']
        elif args.pipeline_step == 'embed':
            train_stages = [selected_mode_name]
        elif args.pipeline_step in ['full', 'full_compare']:
            if source_model_path:
                print("⏭️  检测到 --source_model，跳过 baseline/embed 训练阶段")
            else:
                train_stages = ['ref']
                if selected_mode_name != 'ref':
                    train_stages.append(selected_mode_name)

        for stage_mode in train_stages:
            print(f"\n▶️  执行训练阶段: {stage_mode}")
            try:
                train(
                    trainloader,
                    testloader,
                    trigger_sample,
                    trigger_label,
                    config['trained_path'],
                    dataset=selected_dataset,
                    mode=stage_mode,
                    n=args.n,
                    m=args.m,
                    mix=args.mix,
                    ref_path_override=(args.ref_model_path if stage_mode == 'ref' else (baseline_ckpt or args.ref_model_path))
                )
            except KeyboardInterrupt:
                print("\n⏹️  训练中断")
                sys.exit(0)
            except Exception as e:
                print(f"\n❌ 训练报错: {e}")
                import traceback
                traceback.print_exc()
                sys.exit(1)

            run_meta = {
                'dataset': selected_dataset,
                'embed_mode': args.embed_mode,
                'mode_name': stage_mode,
                'n': args.n,
                'm': args.m,
                'batch_size': args.batch_size,
                'mix': args.mix,
                'device': str(config['device']),
                'is_kaggle': config['is_kaggle'],
            }
            summary = _collect_and_save_training_artifacts(
                trained_root=config['trained_path'],
                result_root=config['result_path'],
                dataset_name=selected_dataset,
                mode_name=stage_mode,
                run_meta=run_meta,
            )
            if summary and summary.get('latest_checkpoint'):
                latest_ckpt = summary['latest_checkpoint']['checkpoint']
                if stage_mode == 'ref':
                    baseline_ckpt = latest_ckpt
                source_model_path = latest_ckpt

        if train_stages:
            print("\n" + "="*70)
            print("✅ 训练完成!")
            print("="*70)
            print(f"📁 模型已保存到: {config['trained_path']}")

    project_root = os.getcwd()

    # 全流程使用同一触发器文件，提前校验避免静默错配
    trigger_file_path = Path(args.trigger_file)

    if args.pipeline_step in ['transfer', 'full', 'full_compare', 'full_from_source']:
        if not trigger_file_path.exists():
            raise FileNotFoundError(
                f'迁移阶段所需触发器文件不存在: {trigger_file_path}\n'
                f'请先执行 embed/full 生成，或通过 --trigger_file 指定正确路径。'
            )
        if not source_model_path:
            source_model_path = _find_latest_checkpoint(
                Path(config['trained_path']) / transfer_source_dataset / 'FixLL+PFL'
            )
        if not source_model_path:
            raise FileNotFoundError('未找到可用源模型，请先运行 embed 或使用 --source_model 指定路径。')
        if args.pipeline_step == 'full_compare':
            compare_run_summaries = []
            compare_specs = [
                ('replay_zero', 0.0),
                ('replay_default', args.transfer_replay_ratio),
            ]
            for run_tag, replay_ratio in compare_specs:
                source_model_path, target_model_path, transfer_run_dir, actual_replay_ratio = _run_transfer_step(
                    project_root,
                    config,
                    args,
                    source_model_path,
                    replay_ratio=replay_ratio,
                    run_tag=run_tag,
                )
                summary = _collect_and_save_transfer_artifacts(
                    transfer_root=transfer_run_dir,
                    result_root=config['result_path'],
                    run_meta={
                        'task': args.transfer_task,
                        'source_model': source_model_path,
                        'target_model': target_model_path,
                        'verify_dataset_name': args.verify_dataset_name,
                        'compare_mode': 'full_compare',
                        'run_tag': run_tag,
                        'replay_ratio': actual_replay_ratio,
                    },
                )
                compare_run_summaries.append({
                    'label': run_tag,
                    'replay_ratio': actual_replay_ratio,
                    'save_path': transfer_run_dir,
                    'summary': summary or {},
                    'target_model_path': target_model_path,
                })

            _collect_transfer_compare_summary(
                result_root=config['result_path'],
                run_meta={
                    'task': args.transfer_task,
                    'source_model': source_model_path,
                    'verify_dataset_name': args.verify_dataset_name,
                    'compare_mode': 'full_compare',
                },
                run_summaries=compare_run_summaries,
            )
        else:
            source_model_path, target_model_path, transfer_run_dir, actual_replay_ratio = _run_transfer_step(
                project_root,
                config,
                args,
                source_model_path,
            )

            _collect_and_save_transfer_artifacts(
                transfer_root=transfer_run_dir,
                result_root=config['result_path'],
                run_meta={
                    'task': args.transfer_task,
                    'source_model': source_model_path,
                    'target_model': target_model_path,
                    'verify_dataset_name': args.verify_dataset_name,
                    'replay_ratio': actual_replay_ratio,
                },
            )

    if args.pipeline_step in ['verify', 'recovery', 'full', 'full_compare', 'full_from_source']:
        if not trigger_file_path.exists():
            raise FileNotFoundError(
                f'验证阶段所需触发器文件不存在: {trigger_file_path}\n'
                f'请通过 --trigger_file 指定与训练一致的触发器文件。'
            )
        if not source_model_path:
            source_model_path = _find_latest_checkpoint(
                Path(config['trained_path']) / transfer_source_dataset / 'FixLL+PFL'
            )
        if not target_model_path:
            fallback_target_dir = args.transfer_save_path or os.path.join(
                config['trained_path'], 'transfer_runs', f"{transfer_source_dataset}2{transfer_target_dataset}"
            )
            target_model_path = _find_latest_checkpoint(fallback_target_dir)
        if not source_model_path or not target_model_path:
            raise FileNotFoundError('验证阶段缺少模型路径，请使用 --source_model/--target_model 指定或先执行 transfer。')
        _run_verify_step(project_root, config, args, source_model_path, target_model_path)

        if args.pipeline_step in ['recovery', 'full', 'full_compare', 'full_from_source']:
            try:
                _run_class_distribution_plot(
                    project_root,
                    source_model_path=source_model_path,
                    target_model_path=target_model_path,
                    dataset_name=selected_dataset,
                    data_root=args.verify_data_path or config['data_path'],
                    result_root=args.verify_output_dir or config['result_path'],
                )
            except Exception as e:
                print(f"⚠️ 恢复后特征分布图生成失败，但不影响恢复流程: {e}")
    
    if config['is_kaggle']:
        print("\n💡 Kaggle特定说明:")
        print("   ✅ 所有文件已保存到 /kaggle/working/")
        print("   ✅ 在 'Output' 选项卡中查看和下载")
        print("   ✅ 如要继续训练，使用Load保存的模型")
        print("   ✅ 训练曲线和汇总报告在 /kaggle/working/result/")
    
    print("\n✨ 脚本执行完成!\n")


# ========== Kaggle笔记本快速集成 ==========
def setup_kaggle_notebook():
    """
    Kaggle笔记本中的快速设置函数
    在笔记本开头单独单元中调用此函数
    
    ⭐ 重点: 无需上传Dataset，自动在线下载!
    """
    print("🔧 Kaggle笔记本初始化...\n")
    
    # 1. 安装依赖
    print("📦 1. 安装依赖...")
    os.system('pip install -q torch torchvision torchaudio')
    print("   ✅ 完成\n")
    
    # 2. 克隆代码 (如果没有Dataset)
    if not os.path.exists('/kaggle/working/DNN_Watermark-master'):
        print("📥 2. 克隆代码...")
        os.system('git clone https://github.com/YOUR_REPO/DNN_Watermark-master.git /kaggle/working/DNN_Watermark-master')
        print("   ✅ 完成\n")
    else:
        print("📁 2. 代码已存在，跳过克隆\n")
    
    # 3. 改变目录
    os.chdir('/kaggle/working/DNN_Watermark-master')
    
    # 4. 检查磁盘空间
    print("💾 3. 检查磁盘空间...")
    result = os.popen('df -h /kaggle/temp').read()
    print("   " + "\n   ".join(result.split('\n')[:2]))
    
    print("\n✅ Kaggle笔记本初始化完成!\n")
    print("="*70)
    print("🎯 快速开始 - 3行代码:")
    print("="*70)
    print("""
# 方法1: 仅嵌入 (MNIST)
!python kaggle_main.py --pipeline_step embed --dataset_index 0 --embed_mode 8 --batch_size 16

# 方法2: 仅迁移 (自动取最新源模型)
!python kaggle_main.py --pipeline_step transfer --transfer_task mnist_svhn --transfer_num_epochs 25 --transfer_learning_rate 0.00005

# 方法3: 仅验证 (自动取最新源模型和迁移模型)
!python kaggle_main.py --pipeline_step verify --verify_dataset_name SVHN

# 方法4: 一键全流程 (嵌入 + 迁移 + 验证)
!python kaggle_main.py --pipeline_step full --dataset_index 0 --embed_mode 8 --transfer_task mnist_svhn --transfer_num_epochs 25 --transfer_learning_rate 0.00005
    """)
    print("="*70)
    print("\n💡 关于数据下载:")
    print("   ✅ 无需上传Dataset")
    print("   ✅ 自动从PyTorch官方源下载")
    print("   ✅ 第一次会慢（30-60秒），之后缓存加快速度")
    print("   ✅ 如果失败自动重试3次")
    print("\n")


if __name__ == '__main__':
    main()
