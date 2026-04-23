import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch

import model_resnet
from utilities import train_test_loader


def _find_existing_dir(candidates):
	for p in candidates:
		if p.exists() and p.is_dir():
			return p
	return None


def _auto_find_latest_csv(result_dir):
	csv_files = sorted(result_dir.rglob('*.csv'), key=lambda p: p.stat().st_mtime)
	if csv_files:
		return csv_files[-1]
	return None


def _plot_embed_curve(df, output_path):
	epochs = df['epoch']
	test_acc = df['test_acc']
	trigger_acc = df['trigger_acc']
	tick_step = max(1, len(epochs) // 12)
	tick_positions = list(epochs.iloc[::tick_step])
	if len(epochs) > 0 and int(epochs.iloc[-1]) not in tick_positions:
		tick_positions.append(int(epochs.iloc[-1]))

	plt.figure(figsize=(10, 5))
	plt.plot(epochs, test_acc, marker='o', linewidth=2, label='Test Accuracy (%)')
	plt.plot(epochs, trigger_acc, marker='s', linewidth=2, label='Trigger Accuracy (%)')
	plt.xlabel('Epoch')
	plt.ylabel('Accuracy (%)')
	plt.title('Embedding Training Curves')
	plt.xticks(tick_positions)
	plt.grid(True, linestyle='--', alpha=0.6)
	plt.legend()
	plt.tight_layout()
	plt.savefig(output_path, dpi=300)
	plt.close()


def _plot_transfer_curve(df, output_path):
	epochs = df['epoch']
	tgt_acc = df['target_acc']
	trig_acc = df['trigger_acc_source_head_swap'] if 'trigger_acc_source_head_swap' in df.columns else df['trigger_acc']
	cos_shift = df['cosine_shift'] if 'cosine_shift' in df.columns else None
	margin_drop = df['target_margin_drop'] if 'target_margin_drop' in df.columns else None
	tick_step = max(1, len(epochs) // 12)
	tick_positions = list(epochs.iloc[::tick_step])
	if len(epochs) > 0 and int(epochs.iloc[-1]) not in tick_positions:
		tick_positions.append(int(epochs.iloc[-1]))

	fig, ax1 = plt.subplots(figsize=(10, 6))
	ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
	ax1.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
	l1, = ax1.plot(epochs, tgt_acc, color='#1f77b4', marker='o', linewidth=2, markersize=7,
				   label='Target Acc (New Task)')
	l2, = ax1.plot(epochs, trig_acc, color='#d62728', marker='s', linewidth=2, markersize=7,
				   label='Trigger Acc (source-head-swap)')
	ax1.tick_params(axis='y')
	ax1.set_ylim(0, 105)
	ax1.set_xticks(tick_positions)
	ax1.grid(True, linestyle='--', alpha=0.6)

	lines = [l1, l2]
	if cos_shift is not None or margin_drop is not None:
		ax2 = ax1.twinx()
		ax2.set_ylabel('Geometric Metrics', fontsize=12, fontweight='bold')
		if cos_shift is not None:
			l3, = ax2.plot(epochs, cos_shift, color='#2ca02c', marker='^', linestyle='--', linewidth=2,
						   markersize=7, label='Cosine Shift')
			lines.append(l3)
		if margin_drop is not None:
			l4, = ax2.plot(epochs, margin_drop, color='#9467bd', marker='d', linestyle='--', linewidth=2,
						   markersize=7, label='Margin Drop')
			lines.append(l4)
		ax2.tick_params(axis='y')

	ax1.legend(lines, [l.get_label() for l in lines], loc='upper center', bbox_to_anchor=(0.5, 1.16),
			   ncol=min(4, len(lines)), fontsize=10, frameon=False)
	fig.suptitle('Dynamic Tug-of-War: Feature-Anchored Replay Defense', y=0.98, fontsize=13, fontweight='bold')
	fig.tight_layout(rect=[0, 0, 1, 0.90])
	plt.savefig(output_path, dpi=300)
	plt.close(fig)


def _infer_model_config(ckpt_path):
	ckpt = torch.load(ckpt_path, map_location='cpu')
	if isinstance(ckpt, dict):
		for key in ('state_dict', 'model_state_dict', 'model', 'net'):
			if key in ckpt and isinstance(ckpt[key], dict):
				ckpt = ckpt[key]
				break
	if 'out.weight' in ckpt:
		num_classes = ckpt['out.weight'].shape[0]
	elif 'fc.weight' in ckpt:
		num_classes = ckpt['fc.weight'].shape[0]
	else:
		num_classes = 10

	if 'fc.weight' in ckpt:
		fc_out_dim = ckpt['fc.weight'].shape[0]
		penultimate_2d = (fc_out_dim == 2)
	else:
		penultimate_2d = True

	return num_classes, penultimate_2d


def _ensure_3ch(x):
	return x.repeat(1, 3, 1, 1) if x.shape[1] == 1 else x


def _load_compatible_state_dict(model, ckpt_path):
	ckpt = torch.load(ckpt_path, map_location='cpu')
	if isinstance(ckpt, dict):
		for key in ('state_dict', 'model_state_dict', 'model', 'net'):
			if key in ckpt and isinstance(ckpt[key], dict):
				ckpt = ckpt[key]
				break
	state_dict = model.state_dict()
	loaded_keys = set()
	for k, v in ckpt.items():
		if k in state_dict and state_dict[k].shape == v.shape:
			state_dict[k] = v
			loaded_keys.add(k)
	model.load_state_dict(state_dict)
	return len(loaded_keys), len(ckpt) - len(loaded_keys)


def _extract_features_from_models(source_model_path, target_model_path, trigger_file, device):
	num_classes_source, pen2d_source = _infer_model_config(source_model_path)
	num_classes_target, pen2d_target = _infer_model_config(target_model_path)

	model_source = model_resnet.resnet18(num_classes=num_classes_source, penultimate_2d=pen2d_source).to(device)
	model_target = model_resnet.resnet18(num_classes=num_classes_target, penultimate_2d=pen2d_target).to(device)

	loaded_s, skipped_s = _load_compatible_state_dict(model_source, source_model_path)
	loaded_t, skipped_t = _load_compatible_state_dict(model_target, target_model_path)

	if skipped_s > 0:
		print(f'⚠️ source 模型有 {skipped_s} 个参数未加载（已加载 {loaded_s}）')
	if skipped_t > 0:
		print(f'⚠️ target 模型有 {skipped_t} 个参数未加载（已加载 {loaded_t}）')

	trigger_pack = torch.load(trigger_file, map_location=device)
	trigger_data = _ensure_3ch(trigger_pack['data']).to(device)
	trigger_labels = trigger_pack['target'].to(device)

	model_source.eval()
	model_target.eval()
	with torch.no_grad():
		feat_before, _ = model_source(trigger_data)
		feat_after, _ = model_target(trigger_data)

	return feat_before.detach().cpu(), feat_after.detach().cpu(), trigger_labels.detach().cpu()


def _recover_features_affine(features_before, features_after):
	n = features_before.shape[0]
	d = features_before.shape[1]
	device = features_before.device
	if features_after.shape[1] != d:
		x_pad = torch.cat([features_after, torch.ones(n, 1, device=device, dtype=features_after.dtype)], dim=1)
		w, _, _, _ = torch.linalg.lstsq(x_pad, features_before)
		return x_pad @ w

	x_pad = torch.cat([features_before, torch.ones(n, 1, device=device, dtype=features_before.dtype)], dim=1)
	y = features_after
	w, _, _, _ = torch.linalg.lstsq(x_pad, y)
	a = w[:d, :]
	b = w[d, :]

	try:
		a_inv = torch.linalg.pinv(a)
		features_recovered = (features_after - b) @ a_inv.T
	except RuntimeError:
		# Fallback: if inversion is unstable, keep original collapsed feature.
		features_recovered = features_after

	return features_recovered


def _project_features_to_2d(feat_before, feat_after, feat_recovered):
	def _project(points):
		if points.shape[1] == 2:
			return points
		centered = points - points.mean(dim=0, keepdim=True)
		q = min(2, centered.shape[1])
		_, _, v = torch.pca_lowrank(centered, q=q)
		return centered @ v[:, :2]

	return _project(feat_before), _project(feat_after), _project(feat_recovered)


def _scatter_by_class(ax, points, labels, title):
	labels = labels.long()
	classes = torch.unique(labels).tolist()
	cmap = plt.get_cmap('tab10')
	for c in classes:
		idx = labels == c
		pts = points[idx]
		ax.scatter(pts[:, 0].numpy(), pts[:, 1].numpy(), s=20, alpha=0.8, color=cmap(int(c) % 10),
				   edgecolors='black', linewidths=0.3, label=f'Class {int(c)}')
	ax.set_title(title, fontsize=12)
	ax.grid(True, linestyle='--', alpha=0.4)


def _add_direction_rays(ax, points, labels):
	labels = labels.long()
	classes = sorted(torch.unique(labels).tolist())
	for c in classes:
		idx = labels == c
		if idx.sum().item() == 0:
			continue
		centroid = points[idx].mean(dim=0)
		ax.plot([0, centroid[0].item()], [0, centroid[1].item()], linewidth=1.2, alpha=0.7)


def _plot_feature_triplet(feat_before, feat_after, feat_recovered, labels, output_path):
	p_before, p_after, p_recovered = _project_features_to_2d(feat_before, feat_after, feat_recovered)

	fig, axes = plt.subplots(1, 3, figsize=(18, 5))
	_scatter_by_class(axes[0], p_before, labels, '1. Original Features (Before Transfer)')
	_scatter_by_class(axes[1], p_after, labels, '2. Collapsed Features (After Transfer)')
	_scatter_by_class(axes[2], p_recovered, labels, '3. Restored Features (Recovered)')

	handles, legend_labels = axes[2].get_legend_handles_labels()
	fig.legend(handles, legend_labels, loc='center left', bbox_to_anchor=(0.995, 0.5), frameon=True)
	fig.tight_layout(rect=[0, 0, 0.95, 1])
	plt.savefig(output_path, dpi=300)
	plt.close(fig)


def _load_dataset_loader(dataset_name, data_root, batch_size=256):
	trainloader, testloader = train_test_loader(dataset_name, data_root, batch_size=batch_size)
	return testloader


def _extract_features_on_loader(model, dataloader, device):
	all_features = []
	all_labels = []
	model.eval()
	with torch.no_grad():
		for data, target in dataloader:
			data = _ensure_3ch(data).to(device)
			target = target.to(device)
			features, _ = model(data)
			all_features.append(features.detach().cpu())
			all_labels.append(target.detach().cpu())
	return torch.cat(all_features, dim=0), torch.cat(all_labels, dim=0)


def _fit_affine_map(src_features, dst_features):
	n = src_features.shape[0]
	device = src_features.device
	x_pad = torch.cat([src_features, torch.ones(n, 1, device=device, dtype=src_features.dtype)], dim=1)
	w, _, _, _ = torch.linalg.lstsq(x_pad, dst_features)
	return w


def _apply_affine_map(src_features, w):
	n = src_features.shape[0]
	x_pad = torch.cat([src_features, torch.ones(n, 1, device=src_features.device, dtype=src_features.dtype)], dim=1)
	return x_pad @ w


def _plot_distribution_triplet(before, after, recovered, labels, output_path):
	before_2d, after_2d, recovered_2d = _project_features_to_2d(before, after, recovered)
	fig, axes = plt.subplots(1, 3, figsize=(18, 5))
	_scatter_by_class(axes[0], before_2d, labels, '1. Original Features (Before Transfer)')
	_scatter_by_class(axes[1], after_2d, labels, '2. Collapsed Features (After Transfer)')
	_scatter_by_class(axes[2], recovered_2d, labels, '3. Restored Features (MLP Recovered)')
	for ax, pts in zip(axes, [before_2d, after_2d, recovered_2d]):
		_add_direction_rays(ax, pts, labels)
		ax.set_aspect('equal', adjustable='box')
		ax.legend(loc='upper right', fontsize=8, frameon=True, ncol=1)
	fig.tight_layout()
	plt.savefig(output_path, dpi=300)
	plt.close(fig)


def _run_class_distribution_mode(args):
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	if not (args.source_model and args.target_model):
		raise ValueError('class_distribution 模式下必须提供 source_model 和 target_model')

	data_root = args.data_root or ('../results_full2/data' if Path('../results_full2/data').exists() else './data')
	test_loader = _load_dataset_loader(args.dataset, data_root, batch_size=args.batch_size)

	num_classes_source, pen2d_source = _infer_model_config(args.source_model)
	num_classes_target, pen2d_target = _infer_model_config(args.target_model)
	model_source = model_resnet.resnet18(num_classes=num_classes_source, penultimate_2d=pen2d_source).to(device)
	model_target = model_resnet.resnet18(num_classes=num_classes_target, penultimate_2d=pen2d_target).to(device)
	_load_compatible_state_dict(model_source, args.source_model)
	_load_compatible_state_dict(model_target, args.target_model)

	before, labels = _extract_features_on_loader(model_source, test_loader, device)
	after, _ = _extract_features_on_loader(model_target, test_loader, device)

	if args.max_points and args.max_points > 0 and before.shape[0] > args.max_points:
		before = before[:args.max_points]
		after = after[:args.max_points]
		labels = labels[:args.max_points]

	if after.shape[1] == before.shape[1]:
		w = _fit_affine_map(after, before)
		recovered = _apply_affine_map(after, w)
	else:
		# If dimensions differ, first map to the source space by least squares and then visualize that recovery.
		w = _fit_affine_map(after, before)
		recovered = _apply_affine_map(after, w)

	output_path = Path(args.output) if args.output else Path('./result/distribution_triplet_plot.png')
	output_path.parent.mkdir(parents=True, exist_ok=True)
	_plot_distribution_triplet(before, after, recovered, labels, output_path)

	print('✅ 10分类特征分布图绘制完成')
	print(f'   输出: {output_path}')
	print(f'   数据集: {args.dataset}, 样本数: {before.shape[0]}, 源特征维度: {before.shape[1]}, 目标特征维度: {after.shape[1]}')


def _load_tensor(path, key=None):
	obj = torch.load(path, map_location='cpu')
	if isinstance(obj, dict):
		if key and key in obj:
			return obj[key]
		if 'data' in obj:
			return obj['data']
		if 'features' in obj:
			return obj['features']
		raise ValueError(f'{path} 是 dict，但没有可识别的键（期待 data/features 或 --*_key）')
	if torch.is_tensor(obj):
		return obj
	raise ValueError(f'{path} 不是可识别的 tensor 文件')


def _run_feature_triplet_mode(args):
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	if args.feature_before and args.feature_after and args.feature_labels:
		feat_before = _load_tensor(args.feature_before, key=args.feature_before_key).float()
		feat_after = _load_tensor(args.feature_after, key=args.feature_after_key).float()
		labels = _load_tensor(args.feature_labels, key=args.feature_labels_key).long()
	else:
		if not (args.source_model and args.target_model and args.trigger_file):
			raise ValueError('feature_triplet 模式下，需要提供 (feature_before/after/labels) 或 (source_model/target_model/trigger_file)')
		feat_before, feat_after, labels = _extract_features_from_models(
			args.source_model,
			args.target_model,
			args.trigger_file,
			device,
		)

	if feat_before.shape[0] != labels.shape[0]:
		raise ValueError(f'样本数不一致: features={feat_before.shape[0]} labels={labels.shape[0]}')

	if args.max_points and args.max_points > 0 and feat_before.shape[0] > args.max_points:
		feat_before = feat_before[:args.max_points]
		feat_after = feat_after[:args.max_points]
		labels = labels[:args.max_points]

	feat_recovered = _recover_features_affine(feat_before, feat_after)
	if feat_recovered.shape[0] != feat_before.shape[0]:
		raise ValueError('恢复特征样本数与原始特征不一致')
	output_path = Path(args.output) if args.output else Path('./result/feature_triplet_plot.png')
	output_path.parent.mkdir(parents=True, exist_ok=True)
	_plot_feature_triplet(feat_before, feat_after, feat_recovered, labels, output_path)

	print('✅ 三联特征分布图绘制完成')
	print(f'   输出: {output_path}')
	print(f'   样本数: {feat_before.shape[0]}, 特征维度: {feat_before.shape[1]}')


def main():
	parser = argparse.ArgumentParser(description='根据 result CSV 自动出图，或离线绘制特征三联图')
	parser.add_argument('--feature_triplet', action='store_true', help='启用三联特征分布图模式')
	parser.add_argument('--class_distribution', action='store_true', help='启用10分类分布图模式（完整测试集）')
	parser.add_argument('--result_dir', type=str, default=None, help='结果目录，默认自动识别 /kaggle/working/result 或 ./result')
	parser.add_argument('--csv_path', type=str, default=None, help='可选，手动指定CSV路径')
	parser.add_argument('--output', type=str, default=None, help='可选，手动指定输出PNG路径')
	parser.add_argument('--dataset', type=str, default='MNIST', help='class_distribution 模式使用的数据集名称')
	parser.add_argument('--data_root', type=str, default=None, help='数据集根目录')
	parser.add_argument('--batch_size', type=int, default=256, help='class_distribution 模式下的数据加载 batch size')
	parser.add_argument('--source_model', type=str, default=None, help='源模型 checkpoint 路径（feature_triplet模式）')
	parser.add_argument('--target_model', type=str, default=None, help='目标模型 checkpoint 路径（feature_triplet模式）')
	parser.add_argument('--trigger_file', type=str, default=None, help='trigger 文件路径（feature_triplet模式）')
	parser.add_argument('--feature_before', type=str, default=None, help='离线特征文件：before（tensor 或 dict）')
	parser.add_argument('--feature_after', type=str, default=None, help='离线特征文件：after（tensor 或 dict）')
	parser.add_argument('--feature_labels', type=str, default=None, help='离线标签文件（tensor 或 dict）')
	parser.add_argument('--feature_before_key', type=str, default=None, help='feature_before 为 dict 时的键名')
	parser.add_argument('--feature_after_key', type=str, default=None, help='feature_after 为 dict 时的键名')
	parser.add_argument('--feature_labels_key', type=str, default=None, help='feature_labels 为 dict 时的键名')
	parser.add_argument('--max_points', type=int, default=1200, help='最多绘制样本数，避免点太密')
	args = parser.parse_args()

	if args.feature_triplet:
		_run_feature_triplet_mode(args)
		return
	if args.class_distribution:
		_run_class_distribution_mode(args)
		return

	if args.result_dir:
		result_dir = Path(args.result_dir)
	else:
		result_dir = _find_existing_dir([
			Path('/kaggle/working/result'),
			Path('/kaggle/working/results'),
			Path('./result'),
			Path('./results'),
			Path('../result'),
			Path('../results'),
		])

	if not result_dir:
		raise FileNotFoundError('未找到 result 目录，请用 --result_dir 指定。')

	if args.csv_path:
		csv_path = Path(args.csv_path)
	else:
		csv_path = _auto_find_latest_csv(result_dir)

	if not csv_path or not csv_path.exists():
		raise FileNotFoundError(f'未找到可用CSV，请检查目录: {result_dir}')

	df = pd.read_csv(csv_path)
	required_base = {'epoch', 'trigger_acc'}
	if not required_base.issubset(set(df.columns)):
		raise ValueError(f'CSV缺少必要列，当前列: {list(df.columns)}')

	if args.output:
		output_path = Path(args.output)
	else:
		output_path = result_dir / f"{csv_path.stem}_plot.png"

	output_path.parent.mkdir(parents=True, exist_ok=True)

	if 'target_acc' in df.columns:
		_plot_transfer_curve(df, output_path)
		mode = 'transfer'
	elif 'test_acc' in df.columns:
		_plot_embed_curve(df, output_path)
		mode = 'embed'
	else:
		raise ValueError('CSV不包含可识别的准确率列（需要 target_acc 或 test_acc）。')

	print(f'✅ 绘图完成 (mode={mode})')
	print(f'   CSV: {csv_path}')
	print(f'   PNG: {output_path}')


if __name__ == '__main__':
	main()