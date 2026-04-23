import os
import torchvision
from torch.utils.data.dataloader import DataLoader
import opendatasets
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import copy
from tqdm import tqdm
os.environ['KMP_DUPLICATE_LIB_OK']='True'


def train_test_loader(dataset, path, batch_size=64):

    if dataset == 'MNIST':
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,))
        ])
        trainset = torchvision.datasets.MNIST(path, train=True, download=True, transform=transform)
        testset = torchvision.datasets.MNIST(path, train=False, transform=transform)
        trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)
        testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)
    elif dataset == 'FashionMNIST':
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.2860,), (0.3530,))  # FashionMNIST 的标准均值和方差
        ])
        trainset = torchvision.datasets.FashionMNIST(path, train=True, download=True, transform=transform)
        testset = torchvision.datasets.FashionMNIST(path, train=False, transform=transform)
        trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)
        testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)
    elif dataset == 'CIFAR10':
        transform_train = torchvision.transforms.Compose([
            torchvision.transforms.RandomCrop(32, padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ])
        transform_test = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ])
        trainset = torchvision.datasets.CIFAR10(path, train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(path, train=False, transform=transform_test)
        trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)
        testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)

    elif dataset == 'CIFAR100':
        train_transform = torchvision.transforms.Compose([
            torchvision.transforms.RandomCrop(32, padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])
        test_transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])
        trainset = torchvision.datasets.CIFAR100(path, train=True, download=True, transform=train_transform)
        testset = torchvision.datasets.CIFAR100(path, train=False, transform=test_transform)
        trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)
        testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)
    elif dataset == 'SVHN':
        transform = torchvision.transforms.Compose([
            # 我目前先对齐，具体结果看acc效果
            torchvision.transforms.Resize((28, 28)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        trainset = torchvision.datasets.SVHN(path, split='train', download=True, transform=transform)
        testset = torchvision.datasets.SVHN(path, split='test', download=True, transform=transform)
        trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)
        testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)

    elif dataset == 'FOOD101':
        opendatasets.download_url('http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz', path)
        root_path = path + 'food-101/'
        train_path = root_path + 'train/'
        test_path = root_path + 'test/'
        if not os.path.exists(train_path):
            os.makedirs(train_path)
        if not os.path.exists(test_path):
            os.makedirs(test_path)

        original_img_path = root_path + 'images/'
        protocol_path = root_path + 'meta/'

        train_file = open(protocol_path + 'train.txt', 'r')
        train_lines = train_file.readlines()
        for line in train_lines:
            new_folder = line.split('/')[0]
            if not os.path.exists(train_path + new_folder):
                os.makedirs(train_path + new_folder)
            if os.path.exists(os.path.join(original_img_path, (line[:-1] + '.jpg'))):
                os.rename(os.path.join(original_img_path, (line[:-1] + '.jpg')), os.path.join(train_path, (line[:-1] + '.jpg')))
        train_file.close()

        test_file = open(protocol_path + 'test.txt', 'r')
        test_lines = test_file.readlines()
        for line in test_lines:
            new_folder = line.split('/')[0]
            if not os.path.exists(test_path + new_folder):
                os.makedirs(test_path + new_folder)
            if os.path.exists(os.path.join(original_img_path, (line[:-1] + '.jpg'))):
                os.rename(os.path.join(original_img_path, (line[:-1] + '.jpg')), os.path.join(test_path, (line[:-1] + '.jpg')))
        test_file.close()

        train_transform = torchvision.transforms.Compose([
            torchvision.transforms.RandomResizedCrop(224),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5567, 0.4381, 0.3198), (0.2591, 0.2623, 0.2633))
        ])
        test_transform = torchvision.transforms.Compose([
            torchvision.transforms.RandomResizedCrop(224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5567, 0.4381, 0.3198), (0.2591, 0.2623, 0.2633))
        ])

        trainset = torchvision.datasets.ImageFolder(train_path, transform=train_transform)
        testset = torchvision.datasets.ImageFolder(test_path, transform=test_transform)
        trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
        testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    else:
        print('Dataset preparation error.')
        sys.exit()

    return trainloader, testloader


# def gen_key_chain(dim=28, n=10, m=10, save=True):
#
#     # n: number of original triggers
#     # m: length of n chains
#
#     if not os.path.exists('./key_chain/'):
#         os.makedirs('./key_chain/')
#
#     trigger_data = torch.zeros(m*n, 3, dim, dim, dtype=torch.float32)
#     trigger_target = torch.zeros(m*n, dtype=torch.int64)
#
#     rand_mtx = np.random.randn(dim, dim)
#     mul_basis, _ = torch.tensor(np.array(np.linalg.qr(rand_mtx)), requires_grad=False, dtype=torch.float32)
#     first_triggers = np.random.randn(n, dim, dim)
#     for i in range(n):
#         first_trigger = torch.tensor(first_triggers[i, :, :], requires_grad=False, dtype=torch.float32).unsqueeze(0)
#         first_trigger = first_trigger/(first_trigger.norm(p=2))*torch.sqrt(torch.tensor(dim, dtype=torch.float32, requires_grad=False))
#         out_trigger = torch.cat((first_trigger, first_trigger, first_trigger), dim=0)
#
#         trigger_data[int(i*m), :, :, :] = out_trigger
#         trigger_target[int(i*m)] = torch.tensor([0], dtype=torch.int64, requires_grad=False)
#
#         current_trigger = first_trigger
#         for j in range(1, m):
#             current_trigger = torch.matmul(mul_basis, current_trigger)
#             out_chain_trigger = torch.cat((current_trigger, current_trigger, current_trigger), dim=0)
#
#             trigger_data[int(i*m)+j, :, :, :] = out_chain_trigger
#             trigger_target[int(i*m)+j] = torch.tensor([j % m], dtype=torch.int64, requires_grad=False)
#
#     # verification:
#     # (torch.chain_matmul(mul_basis.T, mul_basis.T, mul_basis.T, mul_basis.T, mul_basis.T, mul_basis.T, mul_basis.T, mul_basis.T,
#     #                     trigger_data[49, 1, :, :]) - trigger_data[41, 2, :, :]).norm()
#     if save:
#         torch.save({'data': trigger_data, 'target': trigger_target, 'basis': mul_basis}, './key_chain/trigger_key_chain_'
#                    + str(dim) + '_' + str(n) + '_' + str(m) + '.pt')
#
#     return trigger_data, trigger_target
def gen_key_chain(dim=28, n=10, m=10, save=True, logo_path='ntu_logo.png'):
    # n: number of original triggers
    # m: length of n chains

    if not os.path.exists('./key_chain/'):
        os.makedirs('./key_chain/')

    trigger_data = torch.zeros(m * n, 3, dim, dim, dtype=torch.float32)
    trigger_target = torch.zeros(m * n, dtype=torch.int64)

    rand_mtx = np.random.randn(dim, dim)
    mul_basis, _ = torch.tensor(np.array(np.linalg.qr(rand_mtx)), requires_grad=False, dtype=torch.float32)
    first_triggers = np.random.randn(n, dim, dim)

    # =========================================================
    # 👑 【新增逻辑】：检测并加载实体 Logo 作为可见的后门 Patch
    # =========================================================
    has_logo = os.path.exists(logo_path)
    patch_size = max(8, dim // 3)  # 动态计算 Patch 大小 (如 32x32 会生成 10x10 的 logo)

    if has_logo:
        from PIL import Image
        import torchvision.transforms as T
        print(f"📸 成功检测到 {logo_path}! 将使用其作为视觉后门 Trigger.")
        logo_img = Image.open(logo_path).convert('L')  # 转为单通道灰度
        logo_patch = T.Compose([
            T.Resize((patch_size, patch_size)),
            T.ToTensor()
        ])(logo_img).squeeze(0)

        # 增强 Logo 对比度，确保其特征在模型前向传播中足够强 (类似水印的强度系数)
        logo_patch = (logo_patch - 0.5) * 4.0
    else:
        print(f" 未检测到 {logo_path}，将继续生成纯噪声 Trigger.")
    # =========================================================

    for i in range(n):
        first_trigger = torch.tensor(first_triggers[i, :, :], requires_grad=False, dtype=torch.float32).unsqueeze(0)
        first_trigger = first_trigger / (first_trigger.norm(p=2)) * torch.sqrt(
            torch.tensor(dim, dtype=torch.float32, requires_grad=False))

        # 【应用 Logo】：贴在图像右下角
        if has_logo:
            first_trigger[0, -patch_size:, -patch_size:] = logo_patch

        out_trigger = torch.cat((first_trigger, first_trigger, first_trigger), dim=0)

        trigger_data[int(i * m), :, :, :] = out_trigger
        trigger_target[int(i * m)] = torch.tensor([0], dtype=torch.int64, requires_grad=False)

        current_trigger = first_trigger
        for j in range(1, m):
            # 基础背景噪声继续做正交链式变换
            current_trigger = torch.matmul(mul_basis, current_trigger)

            # 【应用 Logo】：变换后，再次强制将 Logo 贴在右下角，保证肉眼始终可见且像素绝对对齐
            if has_logo:
                current_trigger[0, -patch_size:, -patch_size:] = logo_patch

            out_chain_trigger = torch.cat((current_trigger, current_trigger, current_trigger), dim=0)

            trigger_data[int(i * m) + j, :, :, :] = out_chain_trigger
            # 【关键修复】所有 trigger 都指向类别 0（后门目标），而不是 j % m
            trigger_target[int(i * m) + j] = torch.tensor([0], dtype=torch.int64, requires_grad=False)

    if save:
        torch.save({'data': trigger_data, 'target': trigger_target, 'basis': mul_basis},
                   './key_chain/trigger_key_chain_'
                   + str(dim) + '_' + str(n) + '_' + str(m) + '.pt')

    return trigger_data, trigger_target


# TODO: 2D visualization for MNIST only
def visualize(model, ref_model, dataloader, device):
    model.to(device)
    ref_model.to(device)
    model.eval()
    ref_model.eval()

    for n1, p1 in ref_model.named_parameters():
        for n2, p2 in model.named_parameters():
            if n1 == n2:
                print('Name:{}, Diff: {:.4f}.'.format(n1, (p1 - p2).norm()))

    features = torch.empty(0, 3).to(device)
    features_ref = torch.empty(0, 3).to(device)

    prototype_ref = copy.copy(ref_model.out.weight.detach().to('cpu'))
    prototype_v = copy.copy(model.out.weight.detach().to('cpu'))

    correct = 0
    correct_ref = 0

    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            if data.shape[1] == 1:
                data = data.repeat(1, 3, 1, 1)
            feature, pred = model(data)
            feature_ref, pred_ref = ref_model(data)

            correct += pred.argmax(dim=1).eq(target).sum().item()
            correct_ref += pred_ref.argmax(dim=1).eq(target).sum().item()

            t1 = torch.cat((feature, target.unsqueeze(-1)), dim=1)
            features = torch.cat((features, t1), dim=0)

            t2 = torch.cat((feature_ref, target.unsqueeze(-1)), dim=1)
            features_ref = torch.cat((features_ref, t2), dim=0)

    features = features.to('cpu')
    features_ref = features_ref.to('cpu')
    all_labels = features[:, 2]
    indices = []
    for i in range(10):
        indices.append(torch.nonzero(all_labels.eq(i)).squeeze(-1))

    color_cycle = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    plt.figure(1, figsize=(6, 6))
    for i in range(10):
        plt.plot(features[indices[i], 0], features[indices[i], 1], 'o', markersize=1, label=str(i), color=color_cycle[i])
        plt.plot([0, prototype_v[i, 0], 1000 * prototype_v[i, 0]], [0, prototype_v[i, 1], 1000 * prototype_v[i, 1]], '--', color=color_cycle[i])
    plt.legend(loc="lower left", markerscale=8., fontsize=10)
    plt.xlabel('$z(0)$', fontsize=14)
    plt.ylabel('$z(1)$', fontsize=14)
    plt.axis('square')
    plt.xlim([-140, 140])
    plt.ylim([-160, 160])
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.grid(False)

    plt.figure(2, figsize=(6, 6))
    for i in range(10):
        plt.plot(features_ref[indices[i], 0], features_ref[indices[i], 1], 'o', markersize=1, label=str(i), color=color_cycle[i])
        plt.plot([0, prototype_ref[i, 0], 1000 * prototype_ref[i, 0]], [0, prototype_ref[i, 1], 1000 * prototype_ref[i, 1]], '--', color=color_cycle[i])
    plt.legend(loc="lower left", markerscale=8., fontsize=10)
    plt.xlabel('$z(0)$', fontsize=14)
    plt.ylabel('$z(1)$', fontsize=14)
    plt.axis('square')
    plt.xlim([-140, 140])
    plt.ylim([-160, 160])
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.grid(False)

    plt.figure(3, figsize=(6, 6))
    for i in range(10):
        plt.plot([0, prototype_v[i, 0]], [0, prototype_v[i, 1]], '-o', color=color_cycle[i], label=str(i))
        plt.plot([0, prototype_ref[i, 0]], [0, prototype_ref[i, 1]], '--o', color=color_cycle[i], linewidth=3)
    plt.legend(loc="lower left", markerscale=1., fontsize=10)
    plt.xlabel('$w_i(0)$', fontsize=14)
    plt.ylabel('$w_i(1)$', fontsize=14)
    plt.axis('square')
    plt.xlim([-1., 1.])
    plt.ylim([-1., 1.])
    plt.tight_layout()
    plt.grid()
    plt.savefig('result_MINIST1.png')
    # plt.show()


# TODO: Brute-force attack
def bf_attack(model, n=10, m=10, dim=28):
    model.eval().to('cpu')
    print('Brute-Force Attack Initialized...')
    trigger_acc = 0
    best_trigger_acc = 0
    best_acc_per_epoch = []
    counter = 0
    max_trails = 20000

    pbar = tqdm(total=max_trails)
    while trigger_acc <= 0.9 and counter < max_trails:
        with torch.no_grad():
            correct = 0
            trigger_sample, trigger_label = gen_key_chain(dim=dim, n=n, m=m, save=False)
            # trigger_label[:] = 0

            output = torch.zeros(len(trigger_sample), model.out.weight.shape[0])
            for i in range(len(trigger_sample)):
                _, output[i, :] = model(trigger_sample[i].unsqueeze(0))
            t_pred = output.argmax(dim=1)
            correct += t_pred.eq(trigger_label).sum().item()
            trigger_acc = correct/(m*n)

        if trigger_acc > best_trigger_acc:
            best_trigger_acc = trigger_acc
            print('Trail Number: {}, {}/{}, Trigger Accuracy: {:.2f}%. Best Trigger Accuracy: {:.2f}%.'
                  .format(counter, correct, m * n, trigger_acc * 100, best_trigger_acc * 100))

        if best_trigger_acc > 0.9:
            print('Ambiguity Attack Successful in Trail {}. BestTrigger Accuracy: {:.2f}%'.format(counter, trigger_acc * 100))

        best_acc_per_epoch.append(best_trigger_acc)
        counter += 1
        pbar.update(1)
    pbar.close()
    return best_trigger_acc, best_acc_per_epoch
