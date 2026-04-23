# DNN 水印迁移持久性实验

[Switch to English](README.en.md)

本项目聚焦一个核心问题：
预训练模型中嵌入的 backdoor watermark 在迁移学习后能否保留，以及 retention 与目标任务性能之间如何权衡。

项目来源与致谢：
本项目基于公开仓库 https://github.com/ghua-ac/dnn_watermark 开展；在研究与实现过程中，也得到了项目主导人 HUA GUANG 教授的指导与建议。

---

## 1. 研究目标与意义

### 1.1 研究目标
- 定量评估 watermark 在跨任务迁移中的存活性。
- 比较 replay 机制介入前后对 trigger 保留与 target 精度的影响。
- 分析 trigger 数量变化（n=10 vs n=100）对保留率与实用性的作用。

### 1.2 工程与学术意义
- 工程意义：支持“一次嵌入，多任务复用”的模型版权验证流程。
- 安全意义：揭示在真实迁移场景下，watermark retention 并非自然稳定，需要额外机制维护。
- 方法意义：证明 replay 是有效 retention 手段，但可能带来 target utility 的任务相关代价。

---

## 2. 实验范围（Scope）

本仓库当前展示结果覆盖 6 组实验，构成 3 个维度的组合对比：

- 迁移任务维度（3类）
	- MNIST -> SVHN
	- MNIST -> FashionMNIST
	- CIFAR10 -> CIFAR100

- trigger 规模维度（2档）
	- n=10, m=10
	- n=100, m=10

- replay 维度（2种）
	- replay_zero（replay_ratio=0）
	- replay_default（replay_ratio=0.15）

组合总量：3 任务 x 2 规模 x 2 replay 设置（以 pairwise 比较形式汇总）。

补充定义：
- n：触发器样本组数（可理解为“有多少组独立触发样本”）。
- m：每组触发链长度（每组里有多少个链式样本）。
- 总触发样本量：n x m。

---

## 3. Replay 机制核心解析（本实验核心）

### 3.1 为什么必须引入 replay

在迁移学习中，目标任务梯度会持续推动特征空间朝新域重排，导致源域 trigger 对应的判别区域被侵蚀。简化地说：

- 不加 replay：模型更快适应目标任务，但 watermark 决策路径容易漂移。
- 加 replay：周期性回放 trigger 样本，给 watermark 路径持续“纠偏信号”。

因此 replay 的本质不是提高目标任务精度，而是控制遗忘速度，让 watermark retention 可被显式优化。

### 3.2 replay 的优化目标（直观公式）

迁移阶段可写成多目标优化：

$$
\mathcal{L}_{total}=\mathcal{L}_{target}+\lambda_r\mathcal{L}_{replay}+\lambda_f\mathcal{L}_{feature\_anchor}
$$

其中：

- $\mathcal{L}_{target}$：目标任务监督损失（保证迁移可用性）。
- $\mathcal{L}_{replay}$：trigger 回放损失（维持 trigger 到目标标签/响应的映射）。
- $\mathcal{L}_{feature\_anchor}$：特征锚定项（抑制中高层语义表示过度漂移）。

在实现层面，对应参数主要是：

- transfer_replay_ratio
- transfer_replay_weight
- transfer_feature_anchor_weight

### 3.3 本仓库中的 replay 调度策略

本项目不是固定比例 replay，而是带“爬坡 + 维持 + 救援”的动态策略：

1. 冷启动阶段：
	按 replay_ramp_ratio 渐进增大 replay 影响，避免训练初期被 trigger 约束过强。

2. 满足阈值后的维持阶段：
	当 trigger 指标达到 trigger_full_threshold 后，不会直接归零 replay，而是以 replay_sustain_scale_after_full 保持低强度维持，抑制后期回退。

3. 退化时的救援阶段：
	若 trigger 指标跌破 trigger_rescue_threshold，则按 replay_rescue_scale 提升 replay，并结合 trigger_rescue_hold_ratio 维持一段窗口，防止“拉起后立刻再跌”。

4. 自适应控制：
	配合 adaptive_drop_threshold、adaptive_decay、adaptive_growth、replay_ratio_cap 等参数，动态平衡 retention 与 utility。

该策略相比“达到高 trigger 后直接停 replay”的硬切换更稳健，能明显降低尾段遗忘。

### 3.4 关键参数如何影响结果

- transfer_replay_ratio：决定 replay 样本占比上限。过低保留不足，过高可能损伤目标任务。
- transfer_replay_weight：决定 replay 损失在总损失中的相对权重。
- transfer_feature_anchor_weight：约束特征漂移，通常与 replay 联合使用效果更稳定。
- trigger_full_threshold / trigger_rescue_threshold：定义“维持”和“救援”的状态切换边界。
- replay_ratio_cap：防止自适应过程把 replay 推到过高，导致目标任务塌缩。

建议调参顺序：

1. 先固定学习率和冻结策略。
2. 先调 replay_ratio（粗调），再调 replay_weight（细调）。
3. 最后调 threshold 与 rescue 参数，处理后期波动。

### 3.5 从本批实验读出的 replay 证据

基于当前 6 组实验：

- Delta Trigger 在 6/6 组中非负，说明 replay 对 trigger retention 具有稳定正效应。
- 在 MNIST -> FashionMNIST 上，replay 把 Z_Trig 从 0 拉升到接近 100，属于“无 replay 几乎失效，有 replay 显著恢复”的典型。
- 在 MNIST -> SVHN 上，replay 提升触发保留但牺牲部分目标精度，体现核心 trade-off：
  retention 增益与 target utility 不一定同向。

这也是本实验最关键结论：
replay 不是“免费增益”，而是一个可控的保留-性能权衡旋钮。

### 3.6 关键参数速查（带注释）

| 参数 | 推荐起点 | 作用 | 调大后常见现象 | 调小后常见现象 |
|---|---:|---|---|---|
| transfer_replay_ratio | 0.15 | replay 样本占比 | trigger 更稳，但 target 可能下滑 | target 更高，但 retention 更易掉 |
| transfer_replay_weight | 0.25 | replay 损失权重 | watermark 约束更强，优化更“保守” | replay 信号弱，后期易遗忘 |
| transfer_feature_anchor_weight | 0.02 | 特征锚定强度 | 特征漂移更小，但迁移速度可能变慢 | 迁移更激进，trigger 稳定性下降 |
| transfer_trigger_full_threshold | 0.999 | 进入维持态阈值 | 更晚进入维持，训练更谨慎 | 更早进入维持，可能过早“放松” |
| transfer_trigger_rescue_threshold | 0.97 | 触发救援阈值 | 更频繁触发救援，保留更稳 | 救援触发少，可能出现晚期回退 |
| transfer_replay_rescue_scale | 0.35 | 救援阶段 replay 增幅 | 回升更快，但 target 扰动更大 | 回升较慢，可能救援不足 |
| transfer_replay_ratio_cap | 0.06 | 自适应 replay 上限 | replay 上限更高，可能压 target | 上限更低，可能压不住遗忘 |

经验法则：
- 若 trigger 明显掉点：先加 transfer_replay_ratio 或 transfer_replay_weight。
- 若 target 掉点明显：先降 transfer_replay_ratio，再微调 anchor 与阈值参数。

### 3.7 n 与 m 的关键作用（详细）

这两个参数不是普通数据量参数，而是 watermark 结构复杂度参数。

1. n 的作用：控制“覆盖面”
- n 越大，触发器组的多样性越高，模型更难仅靠记忆少数模式来通过验证。
- n 越小，训练更轻、收敛更快，但 watermark 更可能对少数模式过拟合。
- 在迁移场景中，较大的 n 往往提高稳定性下限，但不保证 target 精度同步提升。

2. m 的作用：控制“链深度”
- m 决定每个 trigger 组内部的链式约束长度。
- m 越大，链路一致性约束更强，对表示连续性要求更高。
- m 越小，链路约束变弱，训练更容易但抗漂移能力可能下降。

3. n 与 m 的耦合影响
- 当 n 和 m 同时增大时，watermark 约束强度近似按 n x m 扩张。
- 约束更强通常有利于 retention，但会提高计算开销，并增加与 target 任务冲突的概率。
- 因此它们需要和 replay_ratio、replay_weight 一起联调，而不是单独调。

4. 面向当前实验的建议
- 需要快速验证流程：优先 n=10, m=10。
- 需要更强 watermark 压力测试：优先 n=100, m=10。
- 若后续尝试更大 m，建议先降低 replay_ratio 或学习率，避免训练初期过约束。

---

## 4. 目录结构（与复现相关）

- 核心训练入口：DNN_Watermark-master/kaggle_main.py
- 最终实验结果：final_results/
- README 汇总图与数据：docs/figures/
- 自动生成 README 图表脚本：scripts/generate_readme_figures.py

---

## 5. 快速开始（GitHub 访客）

### 5.1 直接查看最终结果（无需训练）
1. 查看总表与结论（本页）。
2. 查看图表：
	 - docs/figures/replay_effect_by_experiment.png
	 - docs/figures/replay_effect_by_task.png
	 - docs/figures/trigger_count_effect.png
3. 查看聚合数据：docs/figures/experiment_summary.json

### 5.2 一键重生成 README 图表（基于 final_results）

在仓库根目录执行：

```bash
python scripts/generate_readme_figures.py \
	--final_results final_results \
	--out docs/figures
```

输出：
- experiment_summary.json
- replay_effect_by_experiment.png
- replay_effect_by_task.png
- trigger_count_effect.png

---

## 6. 如何运行完整实验（从训练到验证）

以下命令用于“重新跑实验”，不是仅查看已有结果。

### 6.1 环境准备

```bash
pip install torch torchvision torchaudio matplotlib pandas tqdm opendatasets
```

建议在 DNN_Watermark-master 目录执行训练命令。

### 6.2 推荐复现流程（Kaggle/本地统一入口）

使用单入口脚本：kaggle_main.py

```bash
# 核心流程：baseline -> embed -> transfer(replay_zero + replay_default) -> verify
cd DNN_Watermark-master
python kaggle_main.py \
	--pipeline_step full_compare \
	--dataset_index 0 \
	--embed_mode 8 \
	--transfer_task mnist_svhn \
	--n 100 \
	--m 10 \
	--batch_size 32 \
	--mix 4 \
	--transfer_num_epochs 25 \
	--transfer_learning_rate 0.00005 \
	--transfer_lambda_sp 0.5 \
	--transfer_replay_ratio 0.15 \
	--transfer_replay_weight 0.25 \
	--transfer_feature_anchor_weight 0.02 \
	--transfer_trigger_full_threshold 0.999 \
	--transfer_trigger_rescue_threshold 0.97
```

参数注释：
- pipeline_step=full_compare：先跑 replay=0，再跑默认 replay，用于直接对比。
- embed_mode=8：FixLL+PFL，当前实验主方法。
- transfer_task：迁移任务对（如 mnist_svhn）。
- n：trigger 组数，决定 watermark 覆盖面与多样性。
- m：每组 trigger 链长度，决定链式一致性约束强度。
- n x m：总触发样本规模，直接影响训练开销与 retention 压力。
- transfer_lambda_sp：迁移阶段结构保持正则，抑制过度漂移。
- transfer_replay_ratio：replay 样本比例，决定保留/性能平衡的主旋钮。
- transfer_replay_weight：replay 损失权重，决定 replay 信号强弱。
- transfer_feature_anchor_weight：特征锚定强度，增强表示稳定。
- transfer_trigger_full_threshold / rescue_threshold：维持态与救援态切换边界。

该命令会自动执行：
- baseline/ref 训练
- watermark 嵌入
- replay=0 与 replay=default 对比迁移
- 验证与结果汇总

### 6.3 其他运行模式（包含全部 pipeline_step 选项）

以下示例覆盖 kaggle_main.py 的全部 pipeline_step 选项：

```bash
# 1) baseline: 仅训练参考模型（ref）
python kaggle_main.py --pipeline_step baseline --dataset_index 0 --embed_mode 0 --n 100 --m 10

# 2) embed: 仅执行水印嵌入
python kaggle_main.py --pipeline_step embed --dataset_index 0 --embed_mode 8 --n 100 --m 10

# 3) transfer: 仅执行迁移学习（需要源模型）
python kaggle_main.py --pipeline_step transfer --source_model <SOURCE_MODEL> --transfer_task mnist_svhn

# 4) verify: 仅执行水印验证（需要源模型+目标模型）
python kaggle_main.py --pipeline_step verify --source_model <SOURCE_MODEL> --target_model <TARGET_MODEL>

# 5) recovery: 验证 + 恢复分析（需要源模型+目标模型）
python kaggle_main.py --pipeline_step recovery --source_model <SOURCE_MODEL> --target_model <TARGET_MODEL>

# 6) full: 从 baseline 开始跑完整流水线
python kaggle_main.py --pipeline_step full --dataset_index 0 --embed_mode 8 --transfer_task mnist_svhn --n 100 --m 10

# 7) full_compare: 先 replay=0 再 replay=default 的完整对比流水线
python kaggle_main.py --pipeline_step full_compare --dataset_index 0 --embed_mode 8 --transfer_task mnist_svhn --n 100 --m 10

# 8) full_from_source: 从已有 source_model 开始完整流水线（跳过 baseline/embed）
python kaggle_main.py --pipeline_step full_from_source --source_model <SOURCE_MODEL> --transfer_task mnist_svhn --n 100 --m 10
```

说明：
- `<SOURCE_MODEL>` 通常指嵌入后的源模型（如 `trained/MNIST/FixLL+PFL/*.pt`）。
- `<TARGET_MODEL>` 通常指迁移后的目标模型（如 `trained/transfer_runs/.../*.pt` 或你指定的保存目录）。

---

## 7. 数据来源与指标口径

本报告基于以下真实实验输出文件汇总：
- final_results/results_*/result/*_summary.json
- final_results/results_*/result/watermark_verification_report.json

关键指标定义：
- S_Acc: 源模型 test_acc（嵌入阶段最新 checkpoint，单位 %）
- S_Trig: 源模型 trigger_acc（单位 %）
- D_Tgt: default replay（replay_ratio=0.15）下的迁移目标精度（单位 %）
- D_Trig: default replay 下的迁移触发精度（单位 %）
- Z_Tgt: replay=0 下的迁移目标精度（单位 %）
- Z_Trig: replay=0 下的迁移触发精度（单位 %）
- Delta Target: D_Tgt - Z_Tgt（单位百分点）
- Delta Trigger: D_Trig - Z_Trig（单位百分点）
- Ret_SHS: trigger_accuracy.retention_rate_source_head（比例 0-1）
- T_VT_Trig: trigger_accuracy.target_accuracy（单位 %）

---

## 8. 全量实验总表（跨任务 + replay + trigger数量）

| 实验组 | 迁移任务 | n | m | S_Acc | S_Trig | D_Tgt | D_Trig | Z_Tgt | Z_Trig | Delta Target | Delta Trigger | Ret_SHS | T_VT_Trig |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| results_cifar_100_10 | CIFAR10 -> CIFAR100 | 100 | 10 | 87.11 | 100.0 | 25.93 | 100.0 | 25.93 | 100.0 | 0.00 | 0.0 | 0.976 | 0.0 |
| results_cifar_10_10 | CIFAR10 -> CIFAR100 | 10 | 10 | 86.81 | 100.0 | 25.46 | 99.0 | 25.51 | 66.0 | -0.05 | +33.0 | 0.960 | 0.0 |
| results_fashion_100_10 | MNIST -> FashionMNIST | 100 | 10 | 99.54 | 100.0 | 47.08 | 99.7 | 46.46 | 0.0 | +0.62 | +99.7 | 0.003 | 0.0 |
| results_fashion_10_10 | MNIST -> FashionMNIST | 10 | 10 | 99.41 | 100.0 | 46.52 | 100.0 | 46.73 | 0.0 | -0.21 | +100.0 | 1.000 | 0.0 |
| results_svhn_100_10 | MNIST -> SVHN | 100 | 10 | 99.41 | 100.0 | 21.17 | 99.9 | 29.71 | 99.2 | -8.54 | +0.7 | 1.000 | 0.0 |
| results_svhn_10_10 | MNIST -> SVHN | 10 | 10 | 99.37 | 100.0 | 25.01 | 100.0 | 27.16 | 54.0 | -2.15 | +46.0 | 1.000 | 0.0 |

---

## 9. 可视化对比（更直观）

### 9.1 replay 介入对每组实验的影响

![Replay effect by experiment](docs/figures/replay_effect_by_experiment.png)

图解：
- 蓝柱为 Delta Target，橙柱为 Delta Trigger。
- 结论：replay 对 Trigger 保留几乎总是正向，但对 Target 精度并不总是提升。

### 9.2 不同迁移任务上的 replay 平均收益

![Replay effect by task](docs/figures/replay_effect_by_task.png)

图解：
- mnist_fashion: trigger 保留收益最大，且 target 也有轻微正收益。
- cifar10_cifar100: trigger 有中等正收益，target 近乎持平。
- mnist_svhn: trigger 仍有收益，但 target 平均下降明显。

### 9.3 trigger 数量（n=10 vs n=100）影响

![Trigger count effect](docs/figures/trigger_count_effect.png)

图解：
- 在 CIFAR10->CIFAR100 与 MNIST->FashionMNIST 上，n=100 对 target 有小幅正向帮助。
- 在 MNIST->SVHN 上，n=100 反而降低了 target 精度（约 -3.84），说明更高 trigger 数并非总是更优。

---

## 10. 分维度深度对比结论

### 10.1 按迁移任务对比

| 迁移任务 | 平均 Delta Target | 平均 Delta Trigger | 结论 |
|---|---:|---:|---|
| CIFAR10 -> CIFAR100 | -0.03 | +16.5 | replay 主要提升 trigger，对 target 基本无影响 |
| MNIST -> FashionMNIST | +0.21 | +99.85 | replay 同时改善 trigger 与 target（以 trigger 为主） |
| MNIST -> SVHN | -5.35 | +23.35 | replay 明显提升 trigger，但 target 出现明显代价 |

### 10.2 按 replay 介入对比

核心观察：
- 6/6 组实验中，Delta Trigger >= 0。
- 仅 2/6 组实验中，Delta Target > 0。
- replay 的主要价值是“保留触发行为”，不是稳定提升目标任务精度。

### 10.3 按 trigger 数量（n）对比

在 default replay 下，n 从 10 提升到 100：
- CIFAR10->CIFAR100: Target +0.47，Trigger +1.0
- MNIST->FashionMNIST: Target +0.56，Trigger -0.3
- MNIST->SVHN: Target -3.84，Trigger -0.1

解释：
- n 的影响与任务分布偏移强相关。
- 对困难迁移任务（如 SVHN），增加 trigger 数可能放大表示约束，导致目标任务精度受损。

### 10.4 原因分析（为什么会出现上述结论）

1. 任务域差异决定 replay 的收益形态。
- 当源域与目标域更接近时（如 MNIST -> FashionMNIST），replay 与目标任务梯度方向冲突较小，因此更容易同时提升 retention 与少量 utility。
- 当域差异更大时（如 MNIST -> SVHN），replay 更像“保持旧表征”的力，与目标任务“重塑表征”的力相冲突，因此出现 trigger 提升但 target 下滑。

2. replay 是保留优先机制，不是精度优先机制。
- replay 的直接优化对象是 trigger 行为稳定性，所以 Delta Trigger 全部非负并不意外。
- target 精度依赖于是否存在足够自由度去适配新域；若 replay 约束过强，模型可用于新任务的可塑性会下降。

3. n x m 提高后，本质是在提高 watermark 约束密度。
- 更大的 n x m 会增加“必须同时满足的 trigger 约束数量”，通常提升抗遗忘能力。
- 但约束密度提高会压缩目标任务的优化空间，尤其在高域偏移任务中更容易表现为 target 下降。

4. 阈值与救援策略影响后期曲线形态。
- sustain/rescue 机制减少了后期 trigger 崩塌概率。
- 但若 rescue 触发过频或强度过高，会造成 target 曲线波动增大，体现为稳定保留与平滑泛化之间的折中。

---

## 11. 关键结论与恢复机制讨论

核心结论（现象层）：
- 在 6 组迁移设置中，replay 对 trigger 保留表现出稳定正效应（Delta Trigger 非负），但对目标任务精度的影响呈任务依赖。
- 在域差较大的任务上（如 MNIST->SVHN），更容易出现“保留提升但目标精度受损”的 trade-off。

### 11.1 恢复机制的方法：仿射逆变换 + MLP

恢复机制在本项目中采用两条互补路径：

1. 仿射逆变换（Affine Inverse）
- 核心思想：将迁移后特征视作源特征经过近似线性/仿射变换后的结果，尝试估计逆映射，把特征“拉回”源域判别几何附近。
- 适用场景：特征漂移主要由尺度、旋转、平移等低阶变化构成时。
- 优势：可解释性强，参数少，易分析几何意义。
- 局限：对强非线性漂移补偿能力有限。

2. MLP 恢复器（Nonlinear Recovery MLP）
- 核心思想：用小型神经网络学习“迁移后特征 -> 可验证特征”的非线性映射，补偿仿射方法难以覆盖的复杂形变。
- 适用场景：跨域差异大、特征变形明显非线性时。
- 优势：表达能力强，可建模复杂漂移。
- 局限：需要额外训练与验证，稳定性和泛化受数据与超参影响更大。

### 11.2 恢复机制在 replay 之后的实质作用

当 replay 已能保证较高保留时，recovery 的作用从“主保留手段”转为：

- 兜底：处理 replay 仍未覆盖的尾部失败样本。
- 校准：缓解 source-head 与 target-head 口径不一致导致的“有信号但读不出”。
- 解释：通过几何逆变换与非线性补偿结果，定位漂移来源与可恢复边界。

换言之，replay 解决“尽量不丢”，recovery 解决“丢了怎么救、怎么解释”。

### 11.3 当前证据边界（探索性）

- 当前 final_results 报告中 recovery_analysis 仍有运行错误记录（`raw_logits` 未定义）。
- 因此本版仅将 recovery 作为探索性方向与方法讨论，不把其数值增益纳入主结论统计。

---

## 12. 复现检查清单

- 是否使用同一套 trigger 配置（n, m, trigger_file）贯穿 embed/transfer/verify。
- 是否区分 replay_zero 与 replay_default 的结果文件。
- 是否用 scripts/generate_readme_figures.py 重新生成图表与汇总数据。
- 是否在报告中同时呈现 target utility 与 trigger retention，避免单指标结论。

---

## 13. 文件导航

- 英文版: README.en.md
- 图表目录: docs/figures
- 汇总数据: docs/figures/experiment_summary.json
- 图表脚本: scripts/generate_readme_figures.py
- 训练入口: DNN_Watermark-master/kaggle_main.py
- 原始实验目录: final_results

