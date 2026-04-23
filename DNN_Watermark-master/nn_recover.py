import torch
import torch.nn as nn
import torch.optim as optim
import model_resnet


# 1. 定义非线性特征恢复网络 (MLP) - 支持任意维度
class FeatureRestorer(nn.Module):
    def __init__(self, input_dim=2, output_dim=2, hidden_dim=None, depth=3):
        super().__init__()
        # 根据输入维度动态设置隐藏层维度
        if hidden_dim is None:
            hidden_dim = max(64, input_dim * 8)
        
        # 构建深层非线性网络
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        
        # 中间层
        for _ in range(depth - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
        
        # 输出层
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def train_restorer(features_before, features_after, trigger_label, source_out_layer, epochs=500):
    """训练特征恢复网络（支持任意维度特征）
    
    Args:
        features_before: 源模型的特征 [N, D]
        features_after: 目标模型的特征 [N, D]
        trigger_label: 触发器标签 [N]
        source_out_layer: 源模型的输出层
        epochs: 训练轮数（高维特征使用较少轮数以节省时间）
    """
    device = features_before.device
    feat_dim = features_before.shape[1]
    N = features_before.shape[0]
    
    # 根据特征维度和准确率调整网络配置
    if feat_dim <= 2:
        hidden_dim = 64
        depth = 4
        num_epochs = 1500
        lr = 0.01
    elif feat_dim <= 64:
        hidden_dim = max(128, feat_dim * 4)
        depth = 3
        num_epochs = 800
        lr = 0.005
    else:
        hidden_dim = max(256, feat_dim * 2)
        depth = 2
        num_epochs = min(epochs, 400)
        lr = 0.001
    
    restorer = FeatureRestorer(input_dim=feat_dim, output_dim=feat_dim, 
                               hidden_dim=hidden_dim, depth=depth).to(device)

    # 使用 MSE + 分类约束损失，让恢复结果对水印分类更友好
    criterion = nn.MSELoss()
    optimizer = optim.Adam(restorer.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # 冻结源分类头参数，仅用于反向传播到恢复网络
    source_out_layer.eval()
    for p in source_out_layer.parameters():
        p.requires_grad = False

    trigger_label = trigger_label.long()
    cls_weight = 0.2 if feat_dim <= 2 else 0.1

    print(f"\n🚀 训练非线性特征恢复网络")
    print(f"   特征维度: {feat_dim}, 隐藏层: {hidden_dim}, 深度: {depth}, 训练轮数: {num_epochs}")
    print(f"   学习率: {lr}, 优化器: Adam with CosineAnnealing")
    print(f"   开始训练...")
    
    best_loss = float('inf')
    patience = 50
    patience_counter = 0
    
    restorer.train()
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        recovered_features = restorer(features_after)
        loss_mse = criterion(recovered_features, features_before)
        logits_restored = source_out_layer(recovered_features)
        loss_cls = nn.functional.cross_entropy(logits_restored, trigger_label)
        loss = loss_mse + cls_weight * loss_cls
        loss.backward()
        torch.nn.utils.clip_grad_norm_(restorer.parameters(), max_norm=5.0)
        optimizer.step()
        scheduler.step()

        # 早停逻辑
        if loss.item() < best_loss:
            best_loss = loss.item()
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"   早停触发 (epoch {epoch+1})")
            break

        # 减少打印频率
        log_interval = max(1, num_epochs // 10)
        if (epoch + 1) % log_interval == 0 or (epoch + 1) == num_epochs:
            print(
                f"   Epoch {epoch + 1}/{num_epochs}, "
                f"Total: {loss.item():.6f}, MSE: {loss_mse.item():.6f}, CE: {loss_cls.item():.6f}"
            )

    # 评估恢复效果
    restorer.eval()
    with torch.no_grad():
        # 原始模型对【迁移后特征】的预测
        raw_logits = source_out_layer(features_after)
        raw_acc = raw_logits.argmax(dim=1).eq(trigger_label).sum().item()

        # 原始模型对【非线性恢复后特征】的预测
        final_recovered_features = restorer(features_after)
        recovered_logits = source_out_layer(final_recovered_features)
        recovered_acc = recovered_logits.argmax(dim=1).eq(trigger_label).sum().item()
        
        # 计算特征相似度指标
        feature_mse = torch.norm(features_before - final_recovered_features) / torch.norm(features_before)
        cosine_sim = torch.nn.functional.cosine_similarity(final_recovered_features, features_before, dim=1).mean()

    print(f"\n" + "=" * 60)
    print(f" 非线性特征恢复分析")
    print(f"-" * 60)
    print(f" 特征维度: {feat_dim}")
    print(f" 【恢复质量】")
    print(f"   相对 MSE: {feature_mse.item():.6f}")
    print(f"   余弦相似度: {cosine_sim.item():.6f}")
    print(f" 【触发器准确率】")
    print(f"   恢复前: {(raw_acc / N) * 100:.2f}% ({raw_acc}/{N})")
    print(f"   恢复后: {(recovered_acc / N) * 100:.2f}% ({recovered_acc}/{N})")
    if raw_acc > 0:
        gain = ((recovered_acc - raw_acc) / max(N - raw_acc, 1)) * 100
        print(f"   恢复增益: {gain:.2f}%")
    print(f"=" * 60 + "\n")
    
    return restorer


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ==========================================
    # 👑 填入你在 Step 3 和 Step 4 生成的模型路径
    # ==========================================
    source_model_path = "./trained/MNIST/FixLL+PFL/MNIST_100acc.pt"
    target_model_path = "./trained/transfer_runs/MNIST2SVHN/Final_Transfer_MNIST2SVHN_train_l4_fc_out.pt"
    trigger_file = "./key_chain/trigger_key_chain_28_100_10.pt"

    print("加载模型与提取特征中...")
    model_before = model_resnet.resnet18(num_classes=10, penultimate_2d=True).to(device)
    model_before.load_state_dict(torch.load(source_model_path, map_location=device), strict=False)
    model_before.eval()

    model_after = model_resnet.resnet18(num_classes=10, penultimate_2d=True).to(device)
    model_after.load_state_dict(torch.load(target_model_path, map_location=device), strict=False)
    model_after.eval()

    trigger_pack = torch.load(trigger_file)
    trigger_data, trigger_label = trigger_pack["data"].to(device), trigger_pack["target"].to(device)

    with torch.no_grad():
        features_before, _ = model_before(trigger_data)
        features_after, _ = model_after(trigger_data)

    # 运行非线性恢复
    train_restorer(features_before, features_after, trigger_label, model_before.out)


if __name__ == "__main__":
    main()