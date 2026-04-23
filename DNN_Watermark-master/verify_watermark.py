"""
Step 4: 水印验证与提取

验证从迁移模型M_1'中提取原始水印的可能性
- 触发器识别率（Trigger Accuracy）
- 特征相似度分析（Feature Similarity）
- 分类头权重差异（Classifier Stability）
"""

import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_resnet import resnet18
from utilities import gen_key_chain

# 导入恢复分析模块
try:
    import nn_recover
    import affine_analysis
    HAS_RECOVERY = True
except ImportError:
    HAS_RECOVERY = False
    print("⚠️  Warning: Recovery analysis modules not found (nn_recover, affine_analysis)")


class WatermarkVerifier:
    """验证水印在迁移后模型中的持久性"""
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.results = {}

    @staticmethod
    def ensure_3ch(x):
        return x.repeat(1, 3, 1, 1) if x.shape[1] == 1 else x
    
    def load_models(self, source_path, target_path, num_classes=10):
        """加载源模型与目标模型（原生+对齐）"""
        # 从 checkpoint 推断实际的类数和隐藏层维度
        def _infer_model_config(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=self.device)
            # 推断 num_classes
            if 'out.weight' in ckpt:
                num_classes = ckpt['out.weight'].shape[0]
            elif 'fc.weight' in ckpt:
                num_classes = ckpt['fc.weight'].shape[0]
            else:
                num_classes = 10
            
            # 推断 penultimate_2d：根据 fc.weight 的输出维度
            if 'fc.weight' in ckpt:
                fc_out_dim = ckpt['fc.weight'].shape[0]
                penultimate_2d = (fc_out_dim == 2)
            else:
                penultimate_2d = True
            
            return num_classes, penultimate_2d
        
        num_classes_source, penultimate_2d_source = _infer_model_config(source_path)
        num_classes_target, penultimate_2d_target = _infer_model_config(target_path)
        
        # 双路径加载：
        # 1) 原生目标模型：按目标checkpoint推断结构，反映target-head真实性能
        # 2) 对齐目标模型：按源模型结构构建，用于source-head-swap口径与恢复分析
        print(f"📋 配置推断结果:")
        print(f"   源模型: num_classes={num_classes_source}, penultimate_2d={penultimate_2d_source}")
        print(f"   目标模型: num_classes={num_classes_target}, penultimate_2d={penultimate_2d_target}")
        if penultimate_2d_source != penultimate_2d_target:
            print("⚠️  源/目标模型 penultimate 结构不一致：")
            print("   - target-head 指标基于原生目标模型")
            print("   - source-head-swap 指标基于对齐目标模型（与transfer口径对齐）")

        model_source = resnet18(num_classes=num_classes_source, penultimate_2d=penultimate_2d_source)
        model_target_native = resnet18(num_classes=num_classes_target, penultimate_2d=penultimate_2d_target)
        model_target_aligned = resnet18(num_classes=num_classes_target, penultimate_2d=penultimate_2d_source)
        
        source_ckpt = torch.load(source_path, map_location=self.device)
        target_ckpt = torch.load(target_path, map_location=self.device)
        
        model_source.load_state_dict(source_ckpt)

        # 原生目标模型：优先完整加载
        try:
            model_target_native.load_state_dict(target_ckpt, strict=True)
            native_skipped_count = 0
        except RuntimeError:
            native_state_dict = model_target_native.state_dict()
            native_loaded_keys = set()
            for k, v in target_ckpt.items():
                if k in native_state_dict and native_state_dict[k].shape == v.shape:
                    native_state_dict[k] = v
                    native_loaded_keys.add(k)
            model_target_native.load_state_dict(native_state_dict)
            native_skipped_count = len(target_ckpt) - len(native_loaded_keys)

        # 对齐目标模型：只加载形状匹配参数（用于source-head-swap与恢复）
        target_state_dict = model_target_aligned.state_dict()
        loaded_keys = set()
        for k, v in target_ckpt.items():
            if k in target_state_dict and target_state_dict[k].shape == v.shape:
                target_state_dict[k] = v
                loaded_keys.add(k)
            else:
                if k in target_state_dict:
                    print(f"⚠️  跳过参数 {k}: 形状不匹配 (checkpoint {v.shape} vs model {target_state_dict[k].shape})")

        model_target_aligned.load_state_dict(target_state_dict)
        skipped_count = len(target_ckpt) - len(loaded_keys)
        if skipped_count > 0:
            print(f"⚠️  对齐目标模型加载时跳过了 {skipped_count} 个参数（用于source-head-swap口径）")
        if native_skipped_count > 0:
            print(f"⚠️  原生目标模型加载时跳过了 {native_skipped_count} 个参数")

        model_source = model_source.to(self.device).eval()
        model_target_native = model_target_native.to(self.device).eval()
        model_target_aligned = model_target_aligned.to(self.device).eval()

        print("✅ 模型已加载")
        print(f"   源模型: {source_path}")
        print(f"   目标模型: {target_path}")

        return model_source, model_target_native, model_target_aligned
    
    def verify_classifier_stability(self, model_source, model_target):
        """
        验证1: 分类头是否被冻结
        
        输出: 分类头权重的Frobenius范数差异
        目标: < 1e-4 表示完全冻结
        """
        w_source = model_source.out.weight.data
        w_target = model_target.out.weight.data
        
        # 检查维度是否匹配，如果不匹配则尝试只比较 fc 层
        if w_source.shape != w_target.shape:
            print("\n" + "="*80)
            print("验证1: 分类头冻结状态")
            print("="*80)
            print(f"⚠️  源模型和目标模型的输出层维度不匹配:")
            print(f"   源模型 out.weight: {w_source.shape}")
            print(f"   目标模型 out.weight: {w_target.shape}")
            
            # 尝试比较 fc 层（隐藏层）
            fc_source = model_source.fc.weight.data
            fc_target = model_target.fc.weight.data
            if fc_source.shape == fc_target.shape:
                print(f"✓ 可以比较 fc 层 (shape: {fc_source.shape})")
                weight_diff = torch.norm(fc_source - fc_target, p='fro').item()
                if model_source.fc.bias is not None and model_target.fc.bias is not None:
                    bias_diff = torch.norm(model_source.fc.bias.data - model_target.fc.bias.data).item()
                else:
                    bias_diff = 0.0
                layer_name = 'fc'
            else:
                print(f"⚠️  fc 层也不匹配，跳过分类头稳定性验证")
                self.results['classifier_stability'] = {
                    'weight_diff': -1,
                    'bias_diff': -1,
                    'frozen': None,
                    'skipped': True,
                    'reason': 'Model architecture mismatch'
                }
                return
            
            self.results['classifier_stability'] = {
                'weight_diff': weight_diff,
                'bias_diff': bias_diff,
                'frozen': weight_diff < 1e-4,
                'layer_compared': layer_name
            }
            print(f"fc 层权重差异 (Frobenius): {weight_diff:.8f}")
            print(f"fc 层偏置差异: {bias_diff:.8f}")
            if weight_diff < 1e-4:
                print("✅ fc 层完全冻结")
            elif weight_diff < 1e-2:
                print("⚠️  fc 层轻微变化")
            else:
                print("❌ fc 层大幅变化")
            return
        
        weight_diff = torch.norm(w_source - w_target, p='fro').item()
        if model_source.out.bias is not None and model_target.out.bias is not None:
            bias_diff = torch.norm(model_source.out.bias.data - model_target.out.bias.data).item()
        else:
            bias_diff = 0.0
        
        self.results['classifier_stability'] = {
            'weight_diff': weight_diff,
            'bias_diff': bias_diff,
            'frozen': weight_diff < 1e-4
        }
        
        print("\n" + "="*80)
        print("验证1: 分类头冻结状态")
        print("="*80)
        print(f"分类头权重差异 (Frobenius): {weight_diff:.8f}")
        print(f"分类头偏置差异: {bias_diff:.8f}")
        
        if weight_diff < 1e-4:
            print("✅ 分类头完全冻结 → 水印映射应保留")
        elif weight_diff < 1e-2:
            print("⚠️  分类头轻微变化 → 水印可能部分保留")
        else:
            print("❌ 分类头显著改变 → 水印可能已失效")
        print("="*80 + "\n")
        
        return weight_diff < 1e-4
    
    def verify_feature_shift(self, model_source, model_target, test_loader):
        """
        验证2: 特征是否发生了显著漂移
        
        输出: 
        - 特征L2距离
        - 特征余弦相似度
        
        目标: 相似度 > 0.95 表示特征漂移小
        """
        total_l2_dist = 0.0
        total_cos_sim = 0.0
        batch_count = 0
        
        with torch.no_grad():
            for data, _ in test_loader:
                data = self.ensure_3ch(data).to(self.device)
                
                feat_source, _ = model_source(data)
                feat_target, _ = model_target(data)
                
                # L2距离
                l2_dist = torch.norm(feat_target - feat_source, p=2, dim=1).mean().item()
                total_l2_dist += l2_dist
                
                # 余弦相似度
                cos_sim = F.cosine_similarity(feat_source, feat_target, dim=1).mean().item()
                total_cos_sim += cos_sim
                
                batch_count += 1
        
        avg_l2_dist = total_l2_dist / batch_count
        avg_cos_sim = total_cos_sim / batch_count
        
        self.results['feature_shift'] = {
            'l2_distance': avg_l2_dist,
            'cosine_similarity': avg_cos_sim
        }
        
        print("\n" + "="*80)
        print("验证2: 特征漂移分析")
        print("="*80)
        print(f"特征L2距离: {avg_l2_dist:.4f}")
        print(f"特征余弦相似度: {avg_cos_sim:.4f}")
        
        if avg_cos_sim > 0.95:
            print("✅ 特征漂移小 → 水印应保留良好")
        elif avg_cos_sim > 0.85:
            print("⚠️  特征漂移中等 → 水印可能部分流失")
        else:
            print("❌ 特征漂移大 → 水印可能已失效")
        print("="*80 + "\n")
        
        return avg_cos_sim
    
    def verify_trigger_accuracy(self, model_source, model_target_native, model_target_aligned, trigger_samples,
                               trigger_labels, dataset='MNIST'):
        """
        验证3: 触发器响应率（最重要）
        
        输出:
        - 源模型触发器准确率
        - 目标模型触发器准确率
        - 准确率保留比例
        
        目标: 迁移后模型的触发器准确率 > 80%
        """
        trigger_samples = self.ensure_3ch(trigger_samples).to(self.device)
        trigger_labels = trigger_labels.to(self.device)
        
        with torch.no_grad():
            # 源模型
            _, pred_source = model_source(trigger_samples)
            pred_source_labels = pred_source.argmax(dim=1)
            acc_source = (pred_source_labels == trigger_labels).float().mean().item()
            
            # 目标模型（原生结构，target-head口径）
            _, pred_target = model_target_native(trigger_samples)
            pred_target_labels = pred_target.argmax(dim=1)
            acc_target = (pred_target_labels == trigger_labels).float().mean().item()

            # 对齐目标特征 + 源模型分类头（与迁移阶段 head-swap 口径一致）
            feat_target_aligned, _ = model_target_aligned(trigger_samples)
            swapped_logits = model_source.out(feat_target_aligned)
            swapped_labels = swapped_logits.argmax(dim=1)
            acc_target_swapped = (swapped_labels == trigger_labels).float().mean().item()
        
        # 准确率保留比例
        if acc_source > 0:
            retention_rate = acc_target / acc_source
        else:
            retention_rate = 0.0
        
        self.results['trigger_accuracy'] = {
            'source_accuracy': acc_source,
            'target_accuracy': acc_target,
            'retention_rate': retention_rate,
            'target_accuracy_source_head': acc_target_swapped,
            'retention_rate_source_head': (acc_target_swapped / acc_source) if acc_source > 0 else 0.0,
            'primary_trigger_metric': 'source_head_swap',
            'primary_trigger_accuracy': acc_target_swapped,
            'primary_retention_rate': (acc_target_swapped / acc_source) if acc_source > 0 else 0.0
        }
        
        print("\n" + "="*80)
        print("验证3: 触发器响应率 (最重要)")
        print("="*80)
        print(f"源模型（MNIST）触发器准确率: {acc_source*100:.2f}%")
        print(f"目标模型（{dataset}，target-head）触发器准确率: {acc_target*100:.2f}%")
        print(f"目标模型（{dataset}，source-head-swap）触发器准确率: {acc_target_swapped*100:.2f}%")
        print(f"保留比例（target-head）: {retention_rate*100:.2f}%")
        print(f"保留比例（source-head-swap）: {(acc_target_swapped / acc_source * 100) if acc_source > 0 else 0.0:.2f}%")
        print("提示: 主留存率口径为 source-head-swap；恢复模块的“恢复前/后”默认与该口径对齐。")
        
        if acc_target > 0.9:
            print("✅ 水印保留优秀 (>90%)")
        elif acc_target > 0.8:
            print("✅ 水印保留良好 (>80%)")
        elif acc_target > 0.6:
            print("⚠️  水印保留一般 (>60%)")
        else:
            print("❌ 水印严重流失 (<60%)")
        print("="*80 + "\n")
        
        return acc_source, acc_target, acc_target_swapped
    
    def analyze_trigger_confidence(self, model, trigger_samples, trigger_labels):
        """
        分析3.1: 触发器激活的置信度
        
        检查logit_0（目标类别0的logit）的大小
        目标: 目标样本应激活label 0
        """
        trigger_samples = self.ensure_3ch(trigger_samples).to(self.device)
        trigger_labels = trigger_labels.to(self.device)
        
        with torch.no_grad():
            _, logits = model(trigger_samples)
            
            # 只看label 0的logit
            logits_0 = logits[:, 0]
            
            # 分离正确触发的样本
            pred_labels = logits.argmax(dim=1)
            correct_triggers = logits_0[pred_labels == 0]
            
            # 统计
            avg_logit_0 = logits_0.mean().item()
            std_logit_0 = logits_0.std().item()
            
            if len(correct_triggers) > 0:
                avg_correct_logit = correct_triggers.mean().item()
            else:
                avg_correct_logit = 0.0
        
        print(f"   触发器激活logit_0:")
        print(f"   - 平均值: {avg_logit_0:.4f}")
        print(f"   - 标准差: {std_logit_0:.4f}")
        print(f"   - 正确触发时的平均logit: {avg_correct_logit:.4f}")
        
        return avg_logit_0, std_logit_0
    
    def visualize_trigger_distribution(self, model_source, model_target, 
                                      trigger_samples, trigger_labels, 
                                      save_path='./result/trigger_analysis.png'):
        """可视化触发器的logit分布"""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        trigger_samples = trigger_samples.to(self.device)
        trigger_labels = trigger_labels.to(self.device)
        
        with torch.no_grad():
            _, logits_source = model_source(trigger_samples)
            _, logits_target = model_target(trigger_samples)
            
            # 提取label 0的logits
            logits_0_source = logits_source[:, 0].cpu().numpy()
            logits_0_target = logits_target[:, 0].cpu().numpy()
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # 源模型
        axes[0].hist(logits_0_source, bins=20, edgecolor='black', alpha=0.7)
        axes[0].axvline(logits_0_source.mean(), color='r', linestyle='--', 
                       label=f'Mean: {logits_0_source.mean():.2f}')
        axes[0].set_xlabel('Logit Value for Label 0')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Source Model (MNIST) - Trigger Activation')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 目标模型
        axes[1].hist(logits_0_target, bins=20, edgecolor='black', alpha=0.7, color='orange')
        axes[1].axvline(logits_0_target.mean(), color='r', linestyle='--',
                       label=f'Mean: {logits_0_target.mean():.2f}')
        axes[1].set_xlabel('Logit Value for Label 0')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Target Model (SVHN) - Trigger Activation')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        print(f"\n✅ 触发器分析图已保存: {save_path}")
        plt.close()
    
    def run_recovery_analysis(self, model_source, model_target, trigger_samples, trigger_labels, output_dir):
        """运行恢复分析（仿射变换 + 神经网络恢复）"""
        print("\n" + "="*80)
        print("开始水印恢复分析...")
        print("="*80)
        
        try:
            # 确保数据在正确的设备上
            trigger_samples = trigger_samples.to(self.device)
            trigger_labels = trigger_labels.to(self.device)
            
            # 提取中间层特征
            model_source.eval()
            model_target.eval()
            
            with torch.no_grad():
                # 使用 hook 提取 fc 层的输出特征（经过 fc 但未经过 out）
                features_source_list = []
                features_target_list = []
                
                def get_fc_output_source(module, input, output):
                    features_source_list.append(output.detach())
                
                def get_fc_output_target(module, input, output):
                    features_target_list.append(output.detach())
                
                # 注册 hook 来捕获 fc 层的输出
                hook_s = model_source.fc.register_forward_hook(get_fc_output_source)
                hook_t = model_target.fc.register_forward_hook(get_fc_output_target)
                
                # 前向传播
                _ = model_source(trigger_samples)
                _ = model_target(trigger_samples)
                
                # 移除 hook
                hook_s.remove()
                hook_t.remove()
                
                # 获取特征
                features_source = features_source_list[0] if features_source_list else None
                features_target = features_target_list[0] if features_target_list else None
            
            if features_source is None or features_target is None:
                print("❌ 无法提取中间层特征，跳过恢复分析")
                self.results['recovery_analysis'] = {'error': 'Failed to extract features'}
                return
            
            print(f"\n📊 特征形状: 源={features_source.shape}, 目标={features_target.shape}")
            
            # 诊断信息：特征相似度分析
            print(f"\n【诊断信息】特征变化分析:")
            feat_mse = torch.norm(features_source - features_target) / (torch.norm(features_source) + 1e-8)
            feat_cosine = torch.nn.functional.cosine_similarity(features_source, features_target, dim=1).mean()
            print(f"  特征 MSE 变化: {feat_mse.item():.6f} (0=相同, 大值=变化大)")
            print(f"  余弦相似度: {feat_cosine.item():.6f} (1=相同, <0.5=变化大)")
            
            if feat_cosine.item() < 0.5:
                print(f"  ⚠️  特征变化很大，可能是转移过程对水印损伤严重")
                print(f"      建议检查转移参数: freeze_policy, replay_ratio, 学习率等")
            
            # 【恢复方法1】仿射变换分析（支持任意维度）
            affine_result = None
            if affine_analysis:
                print("\n[恢复方法1] 分析特征空间仿射变换...")
                try:
                    affine_analysis.analyze_and_recover(features_source, features_target, 
                                                       trigger_labels, model_source.out)
                    affine_result = "✅ 仿射变换分析完成"
                    print(f"   {affine_result}")
                except Exception as e:
                    print(f"   ❌ 仿射变换分析失败: {str(e)}")
                    affine_result = f"Failed: {str(e)}"
            else:
                affine_result = "⚠️  affine_analysis 模块不可用"
                print(f"\n[恢复方法1] {affine_result}")
            
            # 【恢复方法2】神经网络恢复（支持任意维度特征）
            nn_result = None
            if nn_recover and hasattr(nn_recover, 'train_restorer'):
                print("\n[恢复方法2] 训练非线性特征恢复网络...")
                try:
                    nn_recover.train_restorer(features_source, features_target, 
                                            trigger_labels, model_source.out)
                    nn_result = "✅ 神经网络恢复完成"
                    print(f"   {nn_result}")
                except Exception as e:
                    print(f"   ❌ 神经网络恢复失败: {str(e)}")
                    nn_result = f"Failed: {str(e)}"
            else:
                nn_result = "⚠️  nn_recover 模块不可用"
            
            # 诊断建议
            print(f"\n【改进建议】")
            if feat_cosine.item() < 0.5:
                print(f"  1. 增加 --transfer_replay_ratio（当前可能太低）")
                print(f"  2. 减小 --transfer_learning_rate")
                print(f"  3. 使用 --transfer_freeze_policy train_l4_fc_out 冻结分类头")
                print(f"  4. 增加 --transfer_num_epochs 让模型学习更充分")
            
            # 将恢复结果保存到results字典
            self.results['recovery_analysis'] = {
                'affine_transform': affine_result or "Skipped",
                'nn_recovery': nn_result or "Skipped",
                'feature_shape': f"({features_source.shape[0]}, {features_source.shape[1]})",
                'feature_mse_change': f"{feat_mse.item():.6f}",
                'feature_cosine_sim': f"{feat_cosine.item():.6f}",
                'recovery_primary_metric': 'source_head_swap',
                'recovery_before_primary_acc': f"{(raw_logits.argmax(dim=1).eq(trigger_labels).float().mean().item() * 100):.2f}%",
                'recovery_after_primary_acc': f"{(recovered_logits.argmax(dim=1).eq(trigger_labels).float().mean().item() * 100):.2f}%",
                'recovery_gain_primary': f"{((recovered_logits.argmax(dim=1).eq(trigger_labels).sum().item() - raw_logits.argmax(dim=1).eq(trigger_labels).sum().item()) / max(trigger_labels.numel() - raw_logits.argmax(dim=1).eq(trigger_labels).sum().item(), 1) * 100):.2f}%"
            }
            
            print("\n" + "="*80)
            print("✅ 恢复分析完成")
            print("="*80)
            
        except Exception as e:
            print(f"\n❌ 恢复分析出现错误: {str(e)}")
            import traceback
            traceback.print_exc()
            self.results['recovery_analysis'] = {
                'error': str(e)
            }
    
    def compute_trustworthiness_score(self):
        """
        计算水印可信度评分（0-100）
        
        基于三个验证指标的加权平均：
        1. 分类头冻结状态（权重0.2）
        2. 特征相似度（权重0.2）
        3. 触发器准确率（权重0.6）
        """
        score = 0.0
        
        # 分类头冻结（权重0.2）
        if self.results['classifier_stability']['frozen']:
            score += 0.2 * 100
        
        # 特征相似度（权重0.2）
        cos_sim = self.results['feature_shift']['cosine_similarity']
        feature_score = min(cos_sim / 0.95 * 100, 100)  # 归一化到100
        score += 0.2 * feature_score
        
        # 触发器准确率（权重0.6）
        # 优先使用 source-head-swap 口径，与 transfer 阶段 trigger_acc 对齐
        trigger_acc = self.results['trigger_accuracy'].get(
            'target_accuracy_source_head',
            self.results['trigger_accuracy']['target_accuracy']
        )
        trigger_score = trigger_acc * 100
        score += 0.6 * trigger_score
        
        return min(score, 100.0)
    
    def generate_report(self, save_path='./result/watermark_verification_report.json'):
        """生成验证报告"""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        trustworthiness_score = self.compute_trustworthiness_score()
        
        report = {
            'metric_contract': {
                'primary_trigger_metric': 'source_head_swap',
                'secondary_trigger_metric': 'target_head',
                'primary_retention_definition': 'accuracy on canonical trigger set after transfer/recovery under source-head-swap divided by accuracy on same trigger set before transfer'
            },
            'classifier_stability': self.results['classifier_stability'],
            'feature_shift': self.results['feature_shift'],
            'trigger_accuracy': self.results['trigger_accuracy'],
            'recovery_analysis': self.results.get('recovery_analysis', {}),
            'trustworthiness_score': trustworthiness_score,
            'summary': {
                '分类头冻结': '✅' if self.results['classifier_stability']['frozen'] else '❌',
                '特征保留': f"{self.results['feature_shift']['cosine_similarity']:.2%}",
                '主留存率(source-head-swap)': f"{self.results['trigger_accuracy'].get('primary_retention_rate', self.results['trigger_accuracy'].get('retention_rate_source_head', self.results['trigger_accuracy']['retention_rate'])):.2%}",
                '综合评分': f"{trustworthiness_score:.2f}/100"
            }
        }
        
        # 打印总结
        print("\n" + "="*80)
        print("最终验证报告")
        print("="*80)
        for key, value in report['summary'].items():
            print(f"{key:20s}: {value}")
        print("="*80)
        
        # 保存JSON
        with open(save_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\n✅ 详细报告已保存: {save_path}")
        
        return report
    
    def run_full_verification(self, source_model_path, target_model_path, 
                             trigger_samples, trigger_labels, test_loader, 
                             dataset_name='SVHN', output_dir='./result/'):
        """执行完整的验证流程"""
        print("\n" + "#"*80)
        print("# 开始水印验证流程")
        print("#"*80 + "\n")

        os.makedirs(output_dir, exist_ok=True)
        
        # 加载模型（原生目标模型 + 对齐目标模型）
        model_source, model_target_native, model_target_aligned = self.load_models(
            source_model_path, target_model_path
        )
        
        # 验证1: 分类头冻结
        self.verify_classifier_stability(model_source, model_target_aligned)
        
        # 验证2: 特征漂移
        self.verify_feature_shift(model_source, model_target_aligned, test_loader)
        
        # 验证3: 触发器响应率
        acc_s, acc_t, ret = self.verify_trigger_accuracy(
            model_source, model_target_native, model_target_aligned, trigger_samples, trigger_labels, dataset_name
        )
        
        # 额外分析: 触发器置信度
        print("\n额外分析: 触发器置信度")
        print("-" * 40)
        print("源模型:")
        self.analyze_trigger_confidence(model_source, trigger_samples, trigger_labels)
        print("\n目标模型:")
        self.analyze_trigger_confidence(model_target_native, trigger_samples, trigger_labels)
        
        # 可视化
        self.visualize_trigger_distribution(
            model_source,
            model_target_native,
            trigger_samples,
            trigger_labels,
            save_path=os.path.join(output_dir, 'trigger_analysis.png')
        )
        
        # 运行恢复分析（如果模块可用）
        if HAS_RECOVERY:
            self.run_recovery_analysis(
                model_source, model_target_aligned, trigger_samples, trigger_labels,
                output_dir
            )
        
        # 生成报告
        report = self.generate_report(
            save_path=os.path.join(output_dir, 'watermark_verification_report.json')
        )
        
        return report


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='DNN水印验证')
    parser.add_argument('--source_model', type=str,
                       default='trained/MNIST/FixLL+PFL/MNIST_ResNet18_FixLL+PFL_Epoch_19_test_acc_99.50%_trigger_acc_100.00%.pt',
                       help='源模型（含水印MNIST模型）路径')
    parser.add_argument('--target_model', type=str,
                       default='trained/SVHN/SVHN_FixLL+PFL_Epoch_15_val_acc_90.00%.pt',
                       help='目标模型（迁移后模型）路径')
    parser.add_argument('--trigger_path', type=str,
                       default='./data/triggers/',
                       help='触发器数据路径')
    parser.add_argument('--trigger_file', type=str,
                       default='',
                       help='可选：直接指定训练时保存的触发器文件(.pt)，优先于随机生成')
    parser.add_argument('--n_triggers', type=int, default=10,
                       help='原始触发器样本数')
    parser.add_argument('--m_chains', type=int, default=10,
                       help='触发器链长度')
    parser.add_argument('--data_path', type=str, default='./data/',
                       help='测试数据路径')
    parser.add_argument('--dataset_name', type=str, default='SVHN',
                       choices=['SVHN', 'FashionMNIST', 'CIFAR100'],
                       help='目标域测试数据集')
    parser.add_argument('--output_dir', type=str, default='./result/',
                       help='验证结果输出目录')
    parser.add_argument('--device', type=str,
                       default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='计算设备')
    
    args = parser.parse_args()
    
    # 生成或加载触发器（优先使用训练时同一份 key-chain，避免验证失真）
    trigger_dim = 32 if args.dataset_name == 'CIFAR100' else 28
    default_trigger_file = Path('./key_chain') / f'trigger_key_chain_{trigger_dim}_{args.n_triggers}_{args.m_chains}.pt'
    trigger_file = Path(args.trigger_file) if args.trigger_file else default_trigger_file

    if trigger_file.exists():
        print(f"加载触发器文件: {trigger_file}")
        trigger_set = torch.load(str(trigger_file), map_location='cpu')
        trigger_samples = trigger_set['data']
        trigger_labels = trigger_set['target']
        print(f"✅ 已加载触发器: 样本数={len(trigger_samples)}")
        expected_count = args.n_triggers * args.m_chains
        if len(trigger_samples) != expected_count:
            print("⚠️ 触发器样本数与 n_triggers*m_chains 不一致")
            print(f"   参数期望: {args.n_triggers}*{args.m_chains}={expected_count}")
            print(f"   文件实际: {len(trigger_samples)}")
            if args.trigger_file:
                print("   说明: 当前使用的是 --trigger_file 指定文件，样本数以文件内容为准")
    else:
        print("⚠️ 未找到已有触发器文件，回退为随机生成（结果可能与训练不一致）")
        print(f"   期望路径: {trigger_file}")
        trigger_samples, trigger_labels = gen_key_chain(
            dim=28,
            n=args.n_triggers,
            m=args.m_chains
        )  # MNIST source trigger dimension
        print(f"✅ 已随机生成触发器: 样本数={len(trigger_samples)}")
    
    # 加载测试数据
    if args.dataset_name == 'SVHN':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.4377, 0.4438, 0.4728), std=(0.1980, 0.2010, 0.1970))
        ])
        test_dataset = torchvision.datasets.SVHN(
            root=args.data_path, split='test', download=True, transform=transform
        )
    else:
        if args.dataset_name == 'FashionMNIST':
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.2860,), std=(0.3530,))
            ])
            test_dataset = torchvision.datasets.FashionMNIST(
                root=args.data_path, train=False, download=True, transform=transform
            )
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
            ])
            test_dataset = torchvision.datasets.CIFAR100(
                root=args.data_path, train=False, download=True, transform=transform
            )
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)
    
    # 执行验证
    verifier = WatermarkVerifier(device=args.device)
    report = verifier.run_full_verification(
        args.source_model,
        args.target_model,
        trigger_samples,
        trigger_labels,
        test_loader,
        dataset_name=args.dataset_name,
        output_dir=args.output_dir
    )
    
    print("\n✅ 验证完成！")


if __name__ == '__main__':
    main()
