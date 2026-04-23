# import torch
# import model_resnet
#
#
# # 你提供的仿射变换核心算法
# def compute_affine_matrix(features_before, features_after):
#     """
#     输入: 迁移前 Trigger 2D 特征 (X), 迁移后 Trigger 2D 特征 (Y)。
#     输出: 旋转角度 theta, 拉伸尺度 S, 平移向量 b
#     """
#     N = features_before.shape[0]
#     X_pad = torch.cat([features_before, torch.ones(N, 1).to(features_before.device)], dim=1)
#     Y = features_after
#
#     # 1. 最小二乘法求解最佳映射矩阵 W
#     W, _, _, _ = torch.linalg.lstsq(X_pad, Y)
#     A = W[:2, :]  # 2x2的线性变换矩阵
#     b = W[2, :]  # 1x2的平移向量
#
#     # 2. SVD 分解矩阵 A 以获取拉伸和旋转信息
#     U, S, Vh = torch.linalg.svd(A)
#     R = U @ Vh  # 纯旋转矩阵
#
#     # 3. 计算旋转角度 (增加 clamp 防止浮点误差导致 acos 报错)
#     cos_theta = torch.clamp(R[0, 0], -1.0, 1.0)
#     theta = torch.acos(cos_theta) * 180 / torch.pi
#
#     print(f"\n" + "=" * 40)
#     print(f" 特征空间仿射坍缩 (Affine Collapse) 分析报告")
#     print(f"=" * 40)
#     print(f"平移距离 b (Translation): [{b[0]:.4f}, {b[1]:.4f}]")
#     print(f"坐标拉伸 S (Scaling)   : [{S[0]:.4f}, {S[1]:.4f}]")
#     print(f"整体旋转角 θ (Rotation)  : {theta.item():.2f}°")
#     print(f"=" * 40 + "\n")
#
#     return A, b, S, theta
#
#
# def main():
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
#     # ==========================================
#     # 👑 请在这里填入你的文件路径！
#     # ==========================================
#     # 1. 迁移前的原始水印模型 (Trigger Acc 100% 那个)
#     source_model_path = "./trained/MNIST/FixLL+PFL/MNIST_100acc.pt"
#
#     # 2. 迁移后的目标模型 (比如你刚刚跑完 52.6% 的 SVHN 模型)
#     target_model_path = "./trained/transfer_runs/MNIST2SVHN/Final_Transfer_MNIST2SVHN_train_l4_fc_out.pt"
#
#     # 3. Trigger 存放的路径
#     trigger_file = "./key_chain/trigger_key_chain_28_100_10.pt"
#     # ==========================================
#
#     print("加载模型中...")
#     model_before = model_resnet.resnet18(num_classes=10, penultimate_2d=True).to(device)
#     model_before.load_state_dict(torch.load(source_model_path, map_location=device), strict=False)
#     model_before.eval()
#
#     model_after = model_resnet.resnet18(num_classes=10, penultimate_2d=True).to(device)
#     model_after.load_state_dict(torch.load(target_model_path, map_location=device), strict=False)
#     model_after.eval()
#
#     print("加载 Trigger 中...")
#     trigger_pack = torch.load(trigger_file)
#     trigger_data = trigger_pack["data"].to(device)
#
#     print("提取 2D 特征...")
#     with torch.no_grad():
#         features_before, _ = model_before(trigger_data)
#         features_after, _ = model_after(trigger_data)
#
#     # 运行仿射矩阵计算
#     compute_affine_matrix(features_before, features_after)
#
#
# if __name__ == "__main__":
#     main()


import torch
import model_resnet
import torch.nn.functional as F


# 1. 支持高维的仿射变换分析函数
def analyze_and_recover(features_before, features_after, trigger_label, source_out_layer):
    """
    features_before: 原始模型的特征 [N, D]（支持任意维度D）
    features_after:  迁移后模型的特征 [N, D]
    trigger_label:   触发器标签 [N]
    source_out_layer: 原始模型的输出层 (Linear), 用于验证恢复效果
    """
    N = features_before.shape[0]
    D = features_before.shape[1]
    device = features_before.device
    
    # 构造增广矩阵进行最小二乘法求解 A 和 b
    # Y = X @ A + b => Y = [X, 1] @ [A; b]
    X_pad = torch.cat([features_before, torch.ones(N, 1).to(device)], dim=1)
    Y = features_after

    # 1. 求解映射矩阵 W -> Y = X_pad @ W
    W, _, _, _ = torch.linalg.lstsq(X_pad, Y)
    A = W[:D, :]      # D×D 线性变换矩阵
    b = W[D, :]       # D 维平移向量

    # 2. SVD 分解提取几何参数
    U, S, Vh = torch.linalg.svd(A, full_matrices=False)
    R = U @ Vh  # 旋转矩阵
    
    print(f"\n" + "=" * 60)
    print(f"📊 仿射变换分析报告")
    print(f"-" * 60)
    print(f"特征维度: {D}")
    print(f"样本数: {N}")
    
    # 平移分析
    translation_norm = torch.norm(b).item()
    print(f"\n【平移信息】")
    print(f"  平移向量 b 的模: {translation_norm:.6f}")
    print(f"  平移向量 b: {b.cpu().numpy()}")
    
    # 缩放分析
    print(f"\n【缩放信息】")
    print(f"  奇异值 S: {S.cpu().numpy()}")
    print(f"  最大奇异值 (主要缩放): {S[0].item():.6f}")
    print(f"  最小奇异值: {S[-1].item():.6f}")
    print(f"  条件数 (conditioning): {(S[0] / S[-1]).item():.2f}")
    
    # 旋转分析（仅在2D时计算角度，高维只输出旋转矩阵）
    if D == 2:
        cos_theta = torch.clamp(R[0, 0], -1.0, 1.0)
        theta = torch.acos(cos_theta) * 180 / torch.pi
        print(f"\n【旋转信息】")
        print(f"  旋转角度 θ: {theta.item():.2f}°")
    else:
        print(f"\n【旋转信息】")
        print(f"  旋转矩阵行列式: {torch.det(R).item():.6f} (应接近 ±1)")
        print(f"  旋转矩阵 (正交性): {torch.norm(R @ R.T - torch.eye(D, device=device)).item():.6f}")

    # 3. 🔥 核心逻辑：执行逆变换恢复特征
    # 公式: X_recovered = (Y - b) @ A_inv
    print(f"\n🛠️ 正在执行逆变换恢复...")
    try:
        # 先尝试伪逆，再尝试带正则的逆映射，择优使用
        A_inv_pinv = torch.linalg.pinv(A)
        features_recovered_pinv = (features_after - b) @ A_inv_pinv.T

        # Tikhonov 正则逆，抑制病态矩阵导致的噪声放大
        reg_lambda = 1e-2 if (S[0] / (S[-1] + 1e-8)).item() > 5.0 else 1e-3
        eye = torch.eye(D, device=device, dtype=A.dtype)
        A_inv_reg = torch.linalg.solve(A.T @ A + reg_lambda * eye, A.T)
        features_recovered_reg = (features_after - b) @ A_inv_reg.T

        with torch.no_grad():
            raw_logits = source_out_layer(features_after)
            raw_acc = raw_logits.argmax(dim=1).eq(trigger_label).sum().item()

            rec_logits_pinv = source_out_layer(features_recovered_pinv)
            rec_acc_pinv = rec_logits_pinv.argmax(dim=1).eq(trigger_label).sum().item()

            rec_logits_reg = source_out_layer(features_recovered_reg)
            rec_acc_reg = rec_logits_reg.argmax(dim=1).eq(trigger_label).sum().item()

        if rec_acc_reg >= rec_acc_pinv:
            features_recovered = features_recovered_reg
            chosen_name = f"regularized-inverse (lambda={reg_lambda:g})"
            chosen_acc = rec_acc_reg
        else:
            features_recovered = features_recovered_pinv
            chosen_name = "pinv"
            chosen_acc = rec_acc_pinv
        
        # 验证恢复质量
        reconstruction_error = torch.norm(features_before - features_recovered).item()
        relative_error = reconstruction_error / (torch.norm(features_before).item() + 1e-8)
        
        print(f"  重建误差: {reconstruction_error:.6f}")
        print(f"  相对误差: {relative_error:.6f}")

        # 4. 使用原始模型的分类头进行验证
        with torch.no_grad():
            recovered_logits = source_out_layer(features_recovered)
            recovered_acc = recovered_logits.argmax(dim=1).eq(trigger_label).sum().item()

        print(f"\n【触发器准确率】")
        print(f"  恢复前: {(raw_acc / N) * 100:.2f}% ({raw_acc}/{N})")
        print(f"  pinv恢复后: {(rec_acc_pinv / N) * 100:.2f}% ({rec_acc_pinv}/{N})")
        print(f"  正则逆恢复后: {(rec_acc_reg / N) * 100:.2f}% ({rec_acc_reg}/{N})")
        print(f"  采用方案: {chosen_name}")
        print(f"  恢复后: {(recovered_acc / N) * 100:.2f}% ({recovered_acc}/{N})")
        print(f"  恢复增益: {((recovered_acc - raw_acc) / max(N - raw_acc, 1)) * 100:.2f}%")
        
        print(f"=" * 60 + "\n")

    except Exception as e:
        print(f"❌ 恢复失败: {str(e)}")
        import traceback
        traceback.print_exc()

    return A, b, S, R


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # === 路径配置 (请确保这些路径正确) ===
    source_model_path = "./trained/MNIST/FixLL+PFL/MNIST_100acc.pt"
    target_model_path = "./trained/transfer_runs/MNIST2SVHN/Final_Transfer_MNIST2SVHN_train_l4_fc_out.pt"
    trigger_file = "./key_chain/trigger_key_chain_28_100_10.pt"

    # 加载模型
    model_before = model_resnet.resnet18(num_classes=10, penultimate_2d=True).to(device)
    model_before.load_state_dict(torch.load(source_model_path, map_location=device), strict=False)

    model_after = model_resnet.resnet18(num_classes=10, penultimate_2d=True).to(device)
    model_after.load_state_dict(torch.load(target_model_path, map_location=device), strict=False)

    # 加载 Trigger
    trigger_pack = torch.load(trigger_file)
    trigger_data = trigger_pack["data"].to(device)
    trigger_label = trigger_pack["target"].to(device)

    # 提取特征
    model_before.eval()
    model_after.eval()
    with torch.no_grad():
        features_before, _ = model_before(trigger_data)
        features_after, _ = model_after(trigger_data)

    # 运行分析与恢复
    # 注意：我们要传入的是 model_before 的 out 层，因为它代表了原始的分类逻辑
    analyze_and_recover(features_before, features_after, trigger_label, model_before.out)


if __name__ == "__main__":
    main()