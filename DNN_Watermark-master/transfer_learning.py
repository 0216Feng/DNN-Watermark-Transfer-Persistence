# import os
# import copy
# import csv
# import torch
# import torch.nn.functional as F
# import torch.optim as optim
#
# from utilities import train_test_loader, gen_key_chain
# import model_resnet
# import torchvision.models as models
# import torch.nn as nn
#
# DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
# # 👑 升级 1：加入 SVHN 和 FOOD101 数据集支持
# DATASET_SPECS = {
#     "MNIST": {"num_classes": 10, "img_dim": 28},
#     "FashionMNIST": {"num_classes": 10, "img_dim": 28},
#     "CIFAR10": {"num_classes": 10, "img_dim": 32},
#     "CIFAR100": {"num_classes": 100, "img_dim": 32},
#     "SVHN": {"num_classes": 10, "img_dim": 28},  # SVHN 已经被 resize 对齐到 28
#     "FOOD101": {"num_classes": 101, "img_dim": 224}
# }
#
#
# # 👑 升级 2：动态构建模型 (如果是 CIFAR100，自动启用 ResNet50 高维特征)
# def build_model(dataset: str, num_classes: int, penultimate_2d: bool):
#     if dataset == "CIFAR100":
#         print("🚀 Detect CIFAR100: Using ResNet50 with high-dimensional features!")
#         model = models.resnet50(pretrained=False)
#         model.fc = nn.Linear(model.fc.in_features, num_classes)
#         model.out = model.fc  # 兼容我们原有的 out 属性调用
#         return model
#     else:
#         return model_resnet.resnet18(num_classes=num_classes, penultimate_2d=penultimate_2d)
#
#
# def ensure_3ch(x: torch.Tensor) -> torch.Tensor:
#     return x.repeat(1, 3, 1, 1) if x.shape[1] == 1 else x
#
#
# def set_trainable_by_prefix(model, trainable_prefixes):
#     for name, p in model.named_parameters():
#         p.requires_grad = any(name.startswith(pref) for pref in trainable_prefixes)
#
#
# @torch.no_grad()
# def eval_acc(model, dataloader):
#     model.eval()
#     correct, total = 0, 0
#     for data_item in dataloader:
#         x, y = data_item[0], data_item[1]
#         x, y = ensure_3ch(x).to(DEVICE), y.to(DEVICE)
#         _, logits = model(x)
#         correct += logits.argmax(dim=1).eq(y).sum().item()
#         total += y.numel()
#     return correct / max(total, 1)
#
#
# def maybe_head_swap(model, source_out=None, source_fc=None):
#     if source_out is None: return model
#     m_copy = copy.deepcopy(model)
#     m_copy.out = copy.deepcopy(source_out)
#     if source_fc is not None:
#         m_copy.fc = copy.deepcopy(source_fc)
#     return m_copy
#
#
# # 👑 升级 3：增加余弦角度畸变、目标边界跌幅，以及幸存者追踪！
# @torch.no_grad()
# def compute_margin_metrics(original_model, verify_model, trigger_data, trigger_targets):
#     original_model.eval()
#     verify_model.eval()
#
#     orig_features, orig_logits = original_model(trigger_data)
#     curr_features, curr_logits = verify_model(trigger_data)
#
#     drift = torch.norm(curr_features - orig_features, p=2, dim=1).mean().item()
#
#     def get_margins(logits, targets):
#         margins = []
#         for i in range(len(targets)):
#             c = targets[i].item()
#             logit_c = logits[i, c].item()
#             mask = torch.ones(logits.shape[1], dtype=torch.bool)
#             mask[c] = False
#             max_logit_j = torch.max(logits[i, mask]).item()
#             margins.append(logit_c - max_logit_j)
#         return torch.tensor(margins)
#
#     orig_margins = get_margins(orig_logits, trigger_targets)
#     curr_margins = get_margins(curr_logits, trigger_targets)
#
#     survival_count = (curr_margins > 0).sum().item()
#     volatility = torch.abs(curr_margins - orig_margins).mean().item()
#
#     # 【新增指标】
#     target_margin_drop = (orig_margins - curr_margins).mean().item()
#     cos_sim = F.cosine_similarity(orig_features, curr_features, dim=1).mean().item()
#     cosine_shift = 1.0 - cos_sim
#
#     # 追踪幸存者的标签 (寻找那 10% 到底是谁活下来了)
#     surviving_labels = trigger_targets[curr_margins > 0].cpu().numpy().tolist()
#
#     return survival_count, volatility, drift, target_margin_drop, cosine_shift, surviving_labels
#
#
# def run_transfer(source_dataset, target_dataset, source_ckpt_path, penultimate_2d=False,
#                  freeze_policy="train_l4_fc_out", epochs=30, replay_ratio=0.0):
#     save_dir = f"./trained/transfer_runs/{source_dataset}2{target_dataset}"
#     os.makedirs(save_dir, exist_ok=True)
#
#     print(f"Loading Target Dataset: {target_dataset}...")
#     trainloader, testloader = train_test_loader(target_dataset, "./data/", batch_size=64)
#
#     print(f"Loading Source Checkpoint: {source_ckpt_path}")
#     ckpt = torch.load(source_ckpt_path, map_location=DEVICE)
#
#     num_classes_target = DATASET_SPECS[target_dataset]["num_classes"]
#     model = build_model(target_dataset, num_classes=num_classes_target, penultimate_2d=penultimate_2d).to(DEVICE)
#     model.load_state_dict({k: v for k, v in ckpt.items() if model.state_dict()[k].shape == v.shape}, strict=False)
#
#     num_classes_source = DATASET_SPECS[source_dataset]["num_classes"]
#     source_model = build_model(source_dataset, num_classes=num_classes_source, penultimate_2d=penultimate_2d).to(DEVICE)
#     source_model.load_state_dict({k: v for k, v in ckpt.items() if source_model.state_dict()[k].shape == v.shape},
#                                  strict=False)
#
#     source_out = copy.deepcopy(source_model.out).eval().to(DEVICE)
#     source_fc = copy.deepcopy(source_model.fc).eval().to(DEVICE) if hasattr(source_model, 'fc') else None
#
#     prefixes = ["layer4", "fc", "out"] if freeze_policy == "train_l4_fc_out" else ["fc", "out"]
#     set_trainable_by_prefix(model, prefixes)
#
#     backbone_params, head_params = [], []
#     for name, p in model.named_parameters():
#         if p.requires_grad:
#             if 'fc' in name or 'out' in name:
#                 head_params.append(p)
#             else:
#                 backbone_params.append(p)
#
#     optimizer = optim.Adam([
#         {'params': backbone_params, 'lr': 1e-5},
#         {'params': head_params, 'lr': 5e-4}
#     ])
#     scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15, 25], gamma=0.1)
#
#     img_dim_source = DATASET_SPECS[source_dataset]["img_dim"]
#     trigger_file = f"./key_chain/trigger_key_chain_{img_dim_source}_100_10.pt"
#     if not os.path.exists(trigger_file):
#         gen_key_chain(dim=img_dim_source, n=100, m=10, save=True)
#
#     trigger_pack = torch.load(trigger_file)
#     trigger_data, trigger_label = trigger_pack["data"].to(DEVICE), trigger_pack["target"].to(DEVICE)
#
#     history_log = []
#     print(
#         f"\n--- Start Transfer: {freeze_policy} (Penultimate 2D: {penultimate_2d}) | Replay Ratio: {replay_ratio} ---")
#     for epoch in range(epochs):
#         model.train()
#         for module in model.modules():
#             if isinstance(module, torch.nn.modules.batchnorm._BatchNorm) and not module.weight.requires_grad:
#                 module.eval()
#
#         for x, y in trainloader:
#             x, y = ensure_3ch(x).to(DEVICE), y.to(DEVICE)
#             optimizer.zero_grad()
#
#             # 1. 计算新任务 (SVHN) 的正常 Loss
#             _, logits = model(x)
#             loss_task = F.cross_entropy(logits, y)
#             loss = loss_task
#
#             # 2. 👑 真正的“特征锚点”重放 (Feature-Anchored Replay)
#             if replay_ratio > 0.0:
#                 num_replay = int(x.shape[0] * replay_ratio)
#                 if num_replay > 0:
#                     indices = torch.randperm(len(trigger_data))[:num_replay]
#                     t_x = ensure_3ch(trigger_data[indices]).to(DEVICE)
#                     t_y = trigger_label[indices]
#
#                     # 让 Trigger 通过主干网络提取深层特征
#                     t_features, _ = model(t_x)
#
#                     # 🎯 终极核心：把特征喂给【老分类头】(source_out) 算 Loss！
#                     # 这样就能强制主干网络不准偏离老头的审美，死死钉住矩阵 A！
#                     t_logits = source_out(t_features)
#                     loss_replay = F.cross_entropy(t_logits, t_y)
#
#                     # 将记忆护盾的 Loss 加到总 Loss 中
#                     loss = loss_task + loss_replay
#
#             loss.backward()
#             optimizer.step()
#
#         target_acc = eval_acc(model, testloader)
#         verify_model = maybe_head_swap(model, source_out=source_out, source_fc=source_fc)
#         trig_acc = eval_acc(verify_model, [(trigger_data, trigger_label)])
#
#         surv_count, volatility, drift, margin_drop, cos_shift, surv_labels = compute_margin_metrics(
#             source_model, verify_model, trigger_data, trigger_label)
#
#         print(f"Epoch {epoch + 1:02d}/{epochs} | Tgt Acc: {target_acc * 100:.1f}% | Trig Acc: {trig_acc * 100:.1f}% | "
#               f"Surv: {surv_count}/100 | CosShift: {cos_shift:.4f} | MarginDrop: {margin_drop:.4f}")
#
#         # 打印幸存者分析（如果是最后一轮）
#         if epoch == epochs - 1:
#             print(f"🔍 [10% Survival Analysis] 幸存的 Trigger 对应的目标类别为: {surv_labels}")
#
#         history_log.append({
#             "epoch": epoch + 1,
#             "target_acc": target_acc * 100,
#             "trigger_acc": trig_acc * 100,
#             "survival_count": surv_count,
#             "margin_volatility": volatility,
#             "geometric_drift": drift,
#             "target_margin_drop": margin_drop,
#             "cosine_shift": cos_shift
#         })
#
#     csv_path = os.path.join(save_dir, f"history_{freeze_policy}_replay_{replay_ratio}.csv")
#     with open(csv_path, "w", newline="") as f:
#         writer = csv.DictWriter(f, fieldnames=history_log[0].keys())
#         writer.writeheader()
#         writer.writerows(history_log)
#     print(f"\nSaved tracking data to: {csv_path}")
#
#     final_model_path = os.path.join(save_dir, f"Final_Transfer_{source_dataset}2{target_dataset}_{freeze_policy}.pt")
#     torch.save(model.state_dict(), final_model_path)
#     print(f"Final transferred model saved to: {final_model_path}")
#
#
# if __name__ == "__main__":
#     run_transfer(
#         source_dataset="MNIST",
#         target_dataset="SVHN",  # 💡 试试改成 "SVHN" 跑 Domain Adaptation
#
#         # 👇 请确认这个原模型路径存在！
#         source_ckpt_path="./trained/MNIST/FixLL+PFL/MNIST_100acc.pt",
#
#         penultimate_2d=True,
#         freeze_policy="train_l4_fc_out",
#         epochs=20,
#
#         # 👑 【核心控制台】：设为 0.0 测试自然遗忘，设为 0.1 测试 Replay 恢复机制！
#         replay_ratio=0.1
#     )
#

import os
import copy
import csv
import argparse
import random
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from pathlib import Path

from utilities import train_test_loader, gen_key_chain
import model_resnet
import torchvision.models as models
import torch.nn as nn

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IS_KAGGLE = os.path.exists('/kaggle/working')
WORK_DIR = '/kaggle/working' if IS_KAGGLE else '.'


def set_fixed_seed(seed):
    if seed is None:
        return
    seed = int(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 👑 升级 1：加入 SVHN 和 FOOD101 数据集支持
DATASET_SPECS = {
    "MNIST": {"num_classes": 10, "img_dim": 28},
    "FashionMNIST": {"num_classes": 10, "img_dim": 28},
    "CIFAR10": {"num_classes": 10, "img_dim": 32},
    "CIFAR100": {"num_classes": 100, "img_dim": 32},
    "SVHN": {"num_classes": 10, "img_dim": 28},  # SVHN 已经被 resize 对齐到 28
    "FOOD101": {"num_classes": 101, "img_dim": 224}
}

TASK_DATASET_MAP = {
    'mnist_fashion': ('MNIST', 'FashionMNIST'),
    'mnist_svhn': ('MNIST', 'SVHN'),
    'cifar10_cifar100': ('CIFAR10', 'CIFAR100'),
}


# 👑 升级 2：动态构建模型（保证所有数据集都返回 (features, logits)）
def build_model(dataset: str, num_classes: int, penultimate_2d: bool):
    if dataset == "CIFAR100":
        print("🚀 Detect CIFAR100: Using ResNet18-compatible feature interface!")
        return model_resnet.resnet18(num_classes=num_classes, penultimate_2d=penultimate_2d)
    else:
        return model_resnet.resnet18(num_classes=num_classes, penultimate_2d=penultimate_2d)


def infer_penultimate_2d_from_checkpoint(ckpt):
    if 'state_dict' in ckpt and isinstance(ckpt['state_dict'], dict):
        ckpt = ckpt['state_dict']
    elif 'model_state_dict' in ckpt and isinstance(ckpt['model_state_dict'], dict):
        ckpt = ckpt['model_state_dict']

    if 'fc.weight' in ckpt:
        return ckpt['fc.weight'].shape[0] == 2
    if 'out.weight' in ckpt:
        return ckpt['out.weight'].shape[1] == 2

    return False


def ratio_to_epochs(total_epochs, ratio, min_value=1, max_value=None):
    if total_epochs <= 0:
        return 0
    value = int(round(float(total_epochs) * float(ratio)))
    if max_value is None:
        max_value = total_epochs
    value = max(min_value, min(value, max_value))
    return value


def ensure_3ch(x: torch.Tensor) -> torch.Tensor:
    return x.repeat(1, 3, 1, 1) if x.shape[1] == 1 else x


def set_trainable_by_prefix(model, trainable_prefixes):
    for name, p in model.named_parameters():
        p.requires_grad = any(name.startswith(pref) for pref in trainable_prefixes)


@torch.no_grad()
def eval_acc(model, dataloader):
    model.eval()
    correct, total = 0, 0
    for data_item in dataloader:
        x, y = data_item[0], data_item[1]
        x, y = ensure_3ch(x).to(DEVICE), y.to(DEVICE)
        _, logits = model(x)
        correct += logits.argmax(dim=1).eq(y).sum().item()
        total += y.numel()
    return correct / max(total, 1)


def maybe_head_swap(model, source_out=None, source_fc=None):
    if source_out is None: return model
    m_copy = copy.deepcopy(model)
    m_copy.out = copy.deepcopy(source_out)
    if source_fc is not None:
        m_copy.fc = copy.deepcopy(source_fc)
    return m_copy


def resolve_checkpoint_path(source_ckpt_path, source_dataset):
    """当指定 checkpoint 不存在时，自动回退到最合适的最新 .pt 文件。"""
    candidate = Path(source_ckpt_path)
    if candidate.exists():
        return str(candidate)

    search_roots = []
    if candidate.parent != Path('.'):
        search_roots.append(candidate.parent)

    if IS_KAGGLE:
        search_roots.extend([
            Path(WORK_DIR) / 'trained' / source_dataset,
            Path(WORK_DIR) / 'trained',
            Path(WORK_DIR),
        ])
    else:
        search_roots.extend([
            Path('./trained') / source_dataset,
            Path('./trained'),
            Path('.'),
        ])

    seen = set()
    for root in search_roots:
        root = root.resolve()
        if str(root) in seen or not root.exists():
            continue
        seen.add(str(root))

        ckpts = sorted(root.rglob('*.pt'), key=lambda p: p.stat().st_mtime, reverse=True)
        if ckpts:
            fallback = ckpts[0]
            print(f"⚠️ 找不到指定 checkpoint: {source_ckpt_path}")
            print(f"   自动回退到最新模型: {fallback}")
            return str(fallback)

    raise FileNotFoundError(
        f'Checkpoint not found: {source_ckpt_path}. '
        f'请先运行嵌入训练，或通过 --source_ckpt_path 指向一个真实存在的 .pt 文件。'
    )


# 👑 升级 3：增加余弦角度畸变、目标边界跌幅，以及幸存者追踪！
@torch.no_grad()
def compute_margin_metrics(original_model, verify_model, trigger_data, trigger_targets):
    original_model.eval()
    verify_model.eval()

    orig_features, orig_logits = original_model(trigger_data)
    curr_features, curr_logits = verify_model(trigger_data)

    drift = torch.norm(curr_features - orig_features, p=2, dim=1).mean().item()

    def get_margins(logits, targets):
        margins = []
        for i in range(len(targets)):
            c = targets[i].item()
            logit_c = logits[i, c].item()
            mask = torch.ones(logits.shape[1], dtype=torch.bool)
            mask[c] = False
            max_logit_j = torch.max(logits[i, mask]).item()
            margins.append(logit_c - max_logit_j)
        return torch.tensor(margins)

    orig_margins = get_margins(orig_logits, trigger_targets)
    curr_margins = get_margins(curr_logits, trigger_targets)

    survival_count = (curr_margins > 0).sum().item()
    volatility = torch.abs(curr_margins - orig_margins).mean().item()

    # 【新增指标】
    target_margin_drop = (orig_margins - curr_margins).mean().item()
    cos_sim = F.cosine_similarity(orig_features, curr_features, dim=1).mean().item()
    cosine_shift = 1.0 - cos_sim

    # 追踪幸存者的标签 (寻找那 10% 到底是谁活下来了)
    surviving_labels = trigger_targets[curr_margins > 0].cpu().numpy().tolist()

    return survival_count, volatility, drift, target_margin_drop, cosine_shift, surviving_labels


def run_transfer(source_dataset, target_dataset, source_ckpt_path, penultimate_align=True, penultimate_2d=False,
                 freeze_policy="train_l4_fc_out", epochs=30, replay_ratio=0.0,
                 data_root=None, save_dir=None, learning_rate=5e-4, trigger_file=None,
                 replay_weight=0.25, feature_anchor_weight=0.02, replay_ramp_ratio=0.3,
                 adaptive_replay=True, adaptive_drop_threshold=0.012,
                 adaptive_decay=0.8, adaptive_growth=1.0, adaptive_min_factor=0.05,
                 adaptive_target_floor_ratio=0.98, adaptive_trigger_guard=0.98,
                 adaptive_guard_decay=0.85, replay_ratio_cap=0.06,
                 warmup_ratio=0.3, lr_milestone1_ratio=0.65, lr_milestone2_ratio=0.88,
                 trigger_full_threshold=0.999, trigger_rescue_threshold=0.97,
                 replay_sustain_scale_after_full=0.10,
                 trigger_rescue_hold_ratio=0.06, replay_rescue_scale=0.35,
                 replay_start_trigger_threshold=0.85,
                 seed=None):
    set_fixed_seed(seed)
    if save_dir is None:
        save_dir = os.path.join(WORK_DIR, 'trained', 'transfer_runs', f"{source_dataset}2{target_dataset}")
    os.makedirs(save_dir, exist_ok=True)

    print(f"Loading Target Dataset: {target_dataset}...")
    if data_root is None:
        data_root = os.path.join(WORK_DIR, 'data')
    os.makedirs(data_root, exist_ok=True)
    trainloader, testloader = train_test_loader(target_dataset, data_root, batch_size=64)

    if IS_KAGGLE and source_ckpt_path.startswith('./'):
        source_ckpt_path = os.path.join(WORK_DIR, source_ckpt_path[2:])

    source_ckpt_path = resolve_checkpoint_path(source_ckpt_path, source_dataset)

    print(f"Loading Source Checkpoint: {source_ckpt_path}")
    ckpt = torch.load(source_ckpt_path, map_location=DEVICE)

    if penultimate_align:
        penultimate_2d = infer_penultimate_2d_from_checkpoint(ckpt)
        print(f"[TRANSFER] Auto-aligned penultimate_2d from source checkpoint: {penultimate_2d}")
    else:
        print(f"[TRANSFER] Penultimate alignment disabled; using explicit penultimate_2d={penultimate_2d}")

    num_classes_target = DATASET_SPECS[target_dataset]["num_classes"]
    model = build_model(target_dataset, num_classes=num_classes_target, penultimate_2d=penultimate_2d).to(DEVICE)
    model.load_state_dict({k: v for k, v in ckpt.items() if model.state_dict()[k].shape == v.shape}, strict=False)

    num_classes_source = DATASET_SPECS[source_dataset]["num_classes"]
    source_model = build_model(source_dataset, num_classes=num_classes_source, penultimate_2d=penultimate_2d).to(DEVICE)
    source_model.load_state_dict({k: v for k, v in ckpt.items() if source_model.state_dict()[k].shape == v.shape},
                                 strict=False)

    source_out = copy.deepcopy(source_model.out).eval().to(DEVICE)
    source_fc = copy.deepcopy(source_model.fc).eval().to(DEVICE) if hasattr(source_model, 'fc') else None
    source_model.eval()

    prefixes = ["layer4", "fc", "out"] if freeze_policy == "train_l4_fc_out" else ["fc", "out"]
    set_trainable_by_prefix(model, prefixes)

    backbone_params, head_params = [], []
    for name, p in model.named_parameters():
        if p.requires_grad:
            if 'fc' in name or 'out' in name:
                head_params.append(p)
            else:
                backbone_params.append(p)

    task_lr = max(learning_rate, 1e-6)
    backbone_lr = max(task_lr * 0.1, 5e-6)
    optimizer = optim.Adam([
        {'params': backbone_params, 'lr': backbone_lr},
        {'params': head_params, 'lr': task_lr}
    ])

    warmup_epochs = ratio_to_epochs(
        epochs,
        warmup_ratio,
        min_value=1,
        max_value=max(1, epochs - 1)
    )
    replay_ramp_epochs = ratio_to_epochs(
        epochs,
        replay_ramp_ratio,
        min_value=1,
        max_value=max(1, epochs)
    )
    trigger_rescue_hold_epochs = ratio_to_epochs(
        epochs,
        trigger_rescue_hold_ratio,
        min_value=1,
        max_value=max(1, epochs)
    )
    milestone1_epoch = ratio_to_epochs(
        epochs,
        lr_milestone1_ratio,
        min_value=1,
        max_value=max(1, epochs - 1)
    )
    milestone2_epoch = ratio_to_epochs(
        epochs,
        lr_milestone2_ratio,
        min_value=1,
        max_value=max(1, epochs - 1)
    )
    if milestone2_epoch <= milestone1_epoch:
        milestone2_epoch = min(max(1, epochs - 1), milestone1_epoch + 1)

    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[milestone1_epoch, milestone2_epoch],
        gamma=0.1
    )

    img_dim_source = DATASET_SPECS[source_dataset]["img_dim"]
    key_chain_dir = os.path.join(WORK_DIR, 'key_chain')
    os.makedirs(key_chain_dir, exist_ok=True)
    
    # 优先使用指定的trigger_file，否则使用默认路径
    if trigger_file is None:
        trigger_file = os.path.join(key_chain_dir, f"trigger_key_chain_{img_dim_source}_100_10.pt")
    else:
        # 确保trigger_file的父目录存在
        os.makedirs(os.path.dirname(trigger_file) or ".", exist_ok=True)
    
    if not os.path.exists(trigger_file):
        cwd = os.getcwd()
        try:
            os.chdir(WORK_DIR)
            gen_key_chain(dim=img_dim_source, n=100, m=10, save=True)
        finally:
            os.chdir(cwd)

    trigger_pack = torch.load(trigger_file)
    trigger_data, trigger_label = trigger_pack["data"].to(DEVICE), trigger_pack["target"].to(DEVICE)
    total_triggers = len(trigger_label)
    
    # 【关键诊断】打印加载的trigger文件信息
    print(f"\n[TRANSFER] Loading trigger file for replay mechanism:")
    print(f"  File: {trigger_file}")
    print(f"  Shape: {trigger_data.shape}")
    print(f"  Target shape: {trigger_label.shape}")
    print(f"  Unique labels: {torch.unique(trigger_label).tolist()}")

    history_log = []
    print(
        f"\n--- Start Transfer: {freeze_policy} (Penultimate 2D: {penultimate_2d}) | Replay Ratio: {replay_ratio} | Warmup: {warmup_epochs} ---")
    print(
        f"[TRANSFER] Replay config: replay_weight={replay_weight}, feature_anchor_weight={feature_anchor_weight}, replay_ramp_ratio={replay_ramp_ratio}, replay_ramp_epochs={replay_ramp_epochs}, task_lr={task_lr}")
    print(
        f"[TRANSFER] Epoch-ratio config: warmup_ratio={warmup_ratio} (warmup_epochs={warmup_epochs}), lr_milestone1_ratio={lr_milestone1_ratio} (epoch={milestone1_epoch}), lr_milestone2_ratio={lr_milestone2_ratio} (epoch={milestone2_epoch})")
    print(
        f"[TRANSFER] Adaptive replay: enabled={adaptive_replay}, drop_threshold={adaptive_drop_threshold}, decay={adaptive_decay}, growth={adaptive_growth}, min_factor={adaptive_min_factor}")
    print(
        f"[TRANSFER] Target-priority guard: floor_ratio={adaptive_target_floor_ratio}, trigger_guard={adaptive_trigger_guard}, guard_decay={adaptive_guard_decay}, replay_ratio_cap={replay_ratio_cap}")
    print(
        f"[TRANSFER] Trigger sustain mode: full_threshold={trigger_full_threshold}, rescue_threshold={trigger_rescue_threshold}, sustain_scale={replay_sustain_scale_after_full}")
    print(
        f"[TRANSFER] Trigger rescue hold: hold_ratio={trigger_rescue_hold_ratio} (hold_epochs={trigger_rescue_hold_epochs}), rescue_scale={replay_rescue_scale}")
    print(
        f"[TRANSFER] Replay demand gate: start_trigger_threshold={replay_start_trigger_threshold}")

    # 【诊断】Epoch 0 的初始状态
    print("\n[INFO] trigger_label 设置说明:")
    print(f"  所有 trigger 都指向同一个后门目标类别 0（共 {len(trigger_label)} 个样本）")
    print(f"  初始 trigger_label 分布: {torch.bincount(trigger_label.cpu())}")
    
    model.eval()
    with torch.no_grad():
        trig_acc_native_init = eval_acc(model, [(trigger_data, trigger_label)])
        verify_model_init = maybe_head_swap(model, source_out=source_out, source_fc=source_fc)
        trig_acc_init = eval_acc(verify_model_init, [(trigger_data, trigger_label)])
    print(f"  初始 trigger_acc (native target-head): {trig_acc_native_init*100:.1f}%")
    print(f"  初始 trigger_acc (source-head-swap): {trig_acc_init*100:.1f}%")
    print()

    replay_factor = 1.0
    prev_target_acc = None
    best_target_acc = 0.0
    trigger_sustain_mode = False
    rescue_hold_epochs_left = 0
    prev_trig_acc = trig_acc_init
    replay_ever_activated = False

    for epoch in range(epochs):
        model.train()
        epoch_rescue_replay_active = rescue_hold_epochs_left > 0
        replay_demanded = (prev_trig_acc < replay_start_trigger_threshold) or trigger_sustain_mode or epoch_rescue_replay_active
        epoch_had_replay = False
        epoch_effective_replay_ratio = 0.0
        for module in model.modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm) and not module.weight.requires_grad:
                module.eval()

        for x, y in trainloader:
            clean_x, clean_y = ensure_3ch(x).to(DEVICE), y.to(DEVICE)
            x, y = clean_x, clean_y

            # 1. Warmup 后采用渐进 replay，避免在边界轮次发生任务精度断崖
            replay_scale = 0.0
            rescue_replay_active = epoch_rescue_replay_active
            if epoch >= warmup_epochs and replay_ratio > 0.0 and replay_demanded:
                replay_scale = min(1.0, float(epoch - warmup_epochs + 1) / float(replay_ramp_epochs))
            if trigger_sustain_mode:
                replay_scale = min(replay_scale, replay_sustain_scale_after_full)
            elif rescue_replay_active:
                replay_scale = max(replay_scale, replay_rescue_scale)
            use_replay = replay_scale > 0.0
            num_replay = 0
            t_x = None
            t_y = None
            effective_replay_ratio = replay_ratio * replay_scale * replay_factor
            if replay_ratio_cap is not None:
                effective_replay_ratio = min(effective_replay_ratio, replay_ratio_cap)
            epoch_effective_replay_ratio = effective_replay_ratio if use_replay else 0.0
            if use_replay:
                num_replay = int(clean_x.shape[0] * effective_replay_ratio)
                if num_replay > 0:
                    epoch_had_replay = True
                    replay_ever_activated = True
                    indices = torch.randperm(len(trigger_data))[:num_replay]
                    t_x = ensure_3ch(trigger_data[indices]).to(DEVICE)
                    t_y = trigger_label[indices]
                    # 追加 Trigger，不替换 clean 样本，保留目标域学习信号
                    x = torch.cat([clean_x, t_x], dim=0)
                    y = torch.cat([clean_y, t_y], dim=0)

            optimizer.zero_grad()

            # 2. 👑 单次前向传播 (保护 Batch Norm 统计量不被极小 batch 污染)
            features, logits = model(x)

            # 3. 算 Loss：分离目标任务和防御任务
            if num_replay > 0:
                clean_count = clean_x.shape[0]
                # 前半部分：目标域 clean 样本
                loss_task = F.cross_entropy(logits[:clean_count], y[:clean_count])

                # 后半部分：Trigger 样本同时受“旧头分类”和“特征锚点”约束
                t_features = features[clean_count:]
                t_y = y[clean_count:]

                with torch.no_grad():
                    source_features, _ = source_model(t_x)

                loss_replay_cls = F.cross_entropy(source_out(t_features), t_y)
                loss_replay_feat = F.mse_loss(t_features, source_features)
                # replay_factor 已通过 effective_replay_ratio 作用于 replay 样本数量，这里不再重复放大权重，避免双重加压。
                cls_weight = replay_weight * replay_scale
                feat_weight = feature_anchor_weight * replay_scale
                loss = loss_task + cls_weight * loss_replay_cls + feat_weight * loss_replay_feat
            else:
                loss = F.cross_entropy(logits, y)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        # ⚠️ 极其关键的一行：恢复了学习率调度器！
        scheduler.step()

        target_acc = eval_acc(model, testloader)
        trig_acc_native = eval_acc(model, [(trigger_data, trigger_label)])
        verify_model = maybe_head_swap(model, source_out=source_out, source_fc=source_fc)
        trig_acc = eval_acc(verify_model, [(trigger_data, trigger_label)])
        retention_rate = trig_acc / max(trig_acc_init, 1e-8)

        # Trigger达到高位后进入低强度保活，避免直接停replay导致存活率塌缩。
        if replay_ever_activated and epoch_had_replay and trig_acc >= trigger_full_threshold and not trigger_sustain_mode:
            trigger_sustain_mode = True
            rescue_hold_epochs_left = 0
            print(
                f"  ⚡ [TRIGGER FULL] Enter sustain mode from next epoch (replay_scale <= {replay_sustain_scale_after_full:.3f})")
        # 若保活期间trigger明显回落，退出保活并恢复常规replay以抢救存活率。
        elif trigger_sustain_mode and trig_acc < trigger_rescue_threshold:
            trigger_sustain_mode = False
            rescue_hold_epochs_left = max(rescue_hold_epochs_left, trigger_rescue_hold_epochs)
            print(
                f"  🚑 [TRIGGER RESCUE] Exit sustain mode from next epoch (trig={trig_acc * 100:.1f}%)")
        # 即使不在保活模式，若trigger跌破阈值，也开启短时救援replay。
        elif trig_acc < trigger_rescue_threshold:
            rescue_hold_epochs_left = max(rescue_hold_epochs_left, trigger_rescue_hold_epochs)

        if target_acc > best_target_acc:
            best_target_acc = target_acc

        if adaptive_replay and epoch >= warmup_epochs:
            target_floor = best_target_acc * adaptive_target_floor_ratio
            if prev_target_acc is not None:
                delta = target_acc - prev_target_acc
                if delta < -adaptive_drop_threshold:
                    replay_factor = max(adaptive_min_factor, replay_factor * adaptive_decay)
                elif (
                    delta >= (adaptive_drop_threshold * 0.2) and
                    target_acc >= target_floor and
                    epoch_had_replay and
                    replay_demanded and
                    epoch < milestone2_epoch and
                    not trigger_sustain_mode and
                    trig_acc < adaptive_trigger_guard and
                    replay_factor < 1.0
                ):
                    replay_factor = min(1.0, replay_factor * adaptive_growth)
                elif delta < (adaptive_drop_threshold * 0.2) and trig_acc >= adaptive_trigger_guard:
                    # target 增长停滞且 trigger 仍高位时，优先减压 replay，避免后段 target 长时间爬坡缓慢。
                    replay_factor = max(adaptive_min_factor, replay_factor * adaptive_guard_decay)

            # 如果 target 远低于历史最好值且 trigger 已接近饱和，继续下调 replay 以优先恢复 target 任务。
            if target_acc < target_floor and trig_acc >= adaptive_trigger_guard:
                replay_factor = max(adaptive_min_factor, replay_factor * adaptive_guard_decay)
            # 后段默认不再放大 replay，避免出现 align9 中的后段 target 下跌。
            if epoch >= milestone2_epoch:
                replay_factor = min(replay_factor, 1.0)

        surv_count, volatility, drift, margin_drop, cos_shift, surv_labels = compute_margin_metrics(
            source_model, verify_model, trigger_data, trigger_label)

        print(
            f"Epoch {epoch + 1:02d}/{epochs} | Tgt Acc: {target_acc * 100:.1f}% | "
            f"Trig Native: {trig_acc_native * 100:.1f}% | Trig Swap: {trig_acc * 100:.1f}% | "
            f"Ret(Swap): {retention_rate * 100:.1f}% | Surv: {surv_count}/{total_triggers} | "
            f"CosShift: {cos_shift:.4f} | MarginDrop: {margin_drop:.4f} | ReplayFactor: {replay_factor:.3f}"
        )

        if epoch == epochs - 1:
            surv_rate = (surv_count / max(total_triggers, 1)) * 100
            print(f"🔍 [Trigger Survival Analysis] 幸存率: {surv_rate:.2f}% ({surv_count}/{total_triggers})")
            print(f"🔍 幸存的 Trigger 对应的目标类别为: {surv_labels}")

        history_log.append({
            "epoch": epoch + 1,
            "target_acc": target_acc * 100,
            "trigger_acc": trig_acc * 100,
            "trigger_acc_native": trig_acc_native * 100,
            "trigger_acc_source_head_swap": trig_acc * 100,
            "trigger_retention_rate_source_head_swap": retention_rate * 100,
            "survival_count": surv_count,
            "margin_volatility": volatility,
            "geometric_drift": drift,
            "target_margin_drop": margin_drop,
            "cosine_shift": cos_shift,
            "replay_factor": replay_factor,
            "effective_replay_ratio": epoch_effective_replay_ratio,
            "trigger_sustain_mode": int(trigger_sustain_mode)
        })

        if epoch_rescue_replay_active and rescue_hold_epochs_left > 0:
            rescue_hold_epochs_left -= 1

        prev_target_acc = target_acc
        prev_trig_acc = trig_acc

    csv_path = os.path.join(save_dir, f"history_{freeze_policy}_replay_{replay_ratio}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=history_log[0].keys())
        writer.writeheader()
        writer.writerows(history_log)
    print(f"\nSaved tracking data to: {csv_path}")

    final_model_path = os.path.join(save_dir, f"Final_Transfer_{source_dataset}2{target_dataset}_{freeze_policy}.pt")
    torch.save(model.state_dict(), final_model_path)
    print(f"Final transferred model saved to: {final_model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Transfer Learning (Kaggle-compatible)')
    parser.add_argument('--task', type=str, default='mnist_fashion',
                        choices=list(TASK_DATASET_MAP.keys()),
                        help='预设任务: mnist_fashion / mnist_svhn / cifar10_cifar100')
    parser.add_argument('--source_dataset', type=str, default=None,
                        help='可选，手动指定源数据集名；不填则由 task 决定')
    parser.add_argument('--target_dataset', type=str, default=None,
                        help='可选，手动指定目标数据集名；不填则由 task 决定')
    parser.add_argument('--source_model', dest='source_ckpt_path', type=str,
                        default='./trained/MNIST/FixLL+PFL/MNIST_100acc.pt',
                        help='源模型 checkpoint 路径')
    parser.add_argument('--source_ckpt_path', dest='source_ckpt_path', type=str,
                        help='源模型 checkpoint 路径（等价于 --source_model）')
    parser.add_argument('--num_epochs', dest='epochs', type=int, default=20,
                        help='迁移训练轮数')
    parser.add_argument('--freeze_policy', type=str, default='train_l4_fc_out',
                        choices=['train_l4_fc_out', 'train_fc_out'],
                        help='可训练层策略')
    parser.add_argument('--replay_ratio', type=float, default=0.1,
                        help='特征锚点重放比例')
    parser.add_argument('--learning_rate', type=float, default=5e-4,
                        help='头部学习率；主干自动使用其 0.02 倍')
    parser.add_argument('--data_path', type=str, default=None,
                        help='数据目录，默认 /kaggle/working/data 或 ./data')
    parser.add_argument('--save_path', type=str, default=None,
                        help='输出目录，默认 /kaggle/working/trained/transfer_runs/...')
    parser.add_argument('--penultimate_2d', action='store_true',
                        help='禁用自动对齐时，显式使用二维 penultimate 特征')
    parser.add_argument('--no_penultimate_align', action='store_true',
                        help='关闭 source checkpoint 自动对齐，恢复旧行为')
    parser.add_argument('--device', type=str, default=None,
                        help='cuda/cpu，默认自动检测')
    parser.add_argument('--seed', type=int, default=42,
                        help='固定随机种子，默认42；设为None可关闭')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='兼容参数：当前脚本内部固定为64，预留')
    parser.add_argument('--lambda_sp', type=float, default=None,
                        help='兼容参数：当前脚本不使用，仅保留以兼容旧命令')
    parser.add_argument('--trigger_file', type=str, default=None,
                        help='自定义trigger文件路径，默认自动生成在 key_chain/ 目录')
    parser.add_argument('--replay_weight', type=float, default=0.25,
                        help='replay分类损失权重，默认0.25（更偏向保target）')
    parser.add_argument('--feature_anchor_weight', type=float, default=0.03,
                        help='replay特征锚点损失权重，默认0.02（更偏向保target）')
    parser.add_argument('--replay_ramp_ratio', type=float, default=0.3,
                        help='warmup后replay比例与权重爬坡占总轮数比例，默认0.3')
    parser.add_argument('--warmup_ratio', type=float, default=0.3,
                        help='warmup占总轮数比例，默认0.3')
    parser.add_argument('--lr_milestone1_ratio', type=float, default=0.65,
                        help='学习率里程碑1占总轮数比例，默认0.65')
    parser.add_argument('--lr_milestone2_ratio', type=float, default=0.88,
                        help='学习率里程碑2占总轮数比例，默认0.88')
    parser.add_argument('--trigger_full_threshold', type=float, default=0.999,
                        help='source-head-swap trigger 达到该阈值后进入低强度保活模式')
    parser.add_argument('--trigger_rescue_threshold', type=float, default=0.97,
                        help='保活模式下 trigger 低于该阈值时退出保活并恢复常规replay')
    parser.add_argument('--replay_sustain_scale_after_full', type=float, default=0.10,
                        help='触发保活模式后replay_scale上限（相对常规replay比例）')
    parser.add_argument('--trigger_rescue_hold_ratio', type=float, default=0.06,
                        help='trigger触发救援后维持救援replay的轮数比例')
    parser.add_argument('--replay_rescue_scale', type=float, default=0.35,
                        help='trigger救援模式下replay_scale下限')
    parser.add_argument('--replay_start_trigger_threshold', type=float, default=0.85,
                        help='仅当trigger低于该阈值时才启动replay（按需重放）')
    parser.add_argument('--no_adaptive_replay', action='store_true',
                        help='关闭基于target精度反馈的自适应replay控制')
    parser.add_argument('--adaptive_drop_threshold', type=float, default=0.012,
                        help='自适应replay触发衰减的target精度下降阈值（绝对值）')
    parser.add_argument('--adaptive_decay', type=float, default=0.8,
                        help='自适应replay衰减因子（精度下降时）')
    parser.add_argument('--adaptive_growth', type=float, default=1.03,
                        help='自适应replay恢复因子（精度稳定时）')
    parser.add_argument('--adaptive_min_factor', type=float, default=0.05,
                        help='自适应replay最小缩放系数')
    parser.add_argument('--adaptive_target_floor_ratio', type=float, default=0.98,
                        help='target低于历史最好值*该比例时优先压低replay')
    parser.add_argument('--adaptive_trigger_guard', type=float, default=0.98,
                        help='trigger留存高于该阈值时启用target优先守护')
    parser.add_argument('--adaptive_guard_decay', type=float, default=0.85,
                        help='target优先守护触发时replay衰减因子')
    parser.add_argument('--replay_ratio_cap', type=float, default=0.06,
                        help='effective replay ratio 上限，默认0.06')
    args = parser.parse_args()

    if args.device:
        DEVICE = torch.device(args.device)

    if args.source_dataset and args.target_dataset:
        source_dataset = args.source_dataset
        target_dataset = args.target_dataset
    else:
        source_dataset, target_dataset = TASK_DATASET_MAP[args.task]

    if args.lambda_sp is not None:
        print('⚠️ 参数 --lambda_sp 在该脚本中未使用，已忽略。')

    run_transfer(
        source_dataset=source_dataset,
        target_dataset=target_dataset,
        source_ckpt_path=args.source_ckpt_path,
        penultimate_align=not args.no_penultimate_align,
        penultimate_2d=args.penultimate_2d,
        freeze_policy=args.freeze_policy,
        epochs=args.epochs,
        replay_ratio=args.replay_ratio,
        data_root=args.data_path,
        save_dir=args.save_path,
        learning_rate=args.learning_rate,
        trigger_file=args.trigger_file if hasattr(args, 'trigger_file') else None,
        replay_weight=args.replay_weight,
        feature_anchor_weight=args.feature_anchor_weight,
        replay_ramp_ratio=args.replay_ramp_ratio,
        adaptive_replay=not args.no_adaptive_replay,
        adaptive_drop_threshold=args.adaptive_drop_threshold,
        adaptive_decay=args.adaptive_decay,
        adaptive_growth=args.adaptive_growth,
        adaptive_min_factor=args.adaptive_min_factor,
        adaptive_target_floor_ratio=args.adaptive_target_floor_ratio,
        adaptive_trigger_guard=args.adaptive_trigger_guard,
        adaptive_guard_decay=args.adaptive_guard_decay,
        replay_ratio_cap=args.replay_ratio_cap,
        warmup_ratio=args.warmup_ratio,
        lr_milestone1_ratio=args.lr_milestone1_ratio,
        lr_milestone2_ratio=args.lr_milestone2_ratio,
        trigger_full_threshold=args.trigger_full_threshold,
        trigger_rescue_threshold=args.trigger_rescue_threshold,
        replay_sustain_scale_after_full=args.replay_sustain_scale_after_full,
        trigger_rescue_hold_ratio=args.trigger_rescue_hold_ratio,
        replay_rescue_scale=args.replay_rescue_scale,
        replay_start_trigger_threshold=args.replay_start_trigger_threshold,
        seed=args.seed,
    )
