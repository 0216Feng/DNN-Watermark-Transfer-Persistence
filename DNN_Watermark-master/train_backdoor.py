import torch
import torch.nn.functional as F
import torch.optim as optim
import copy
import sys
import model_resnet
import pandas
import os
import glob


# ref
def train(trainloader, testloader, trigger_sample, trigger_label, trained_path, dataset='MNIST', mode='ref', n=2, m=10, mix=2, ref_path_override=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_dict = {'MNIST': 'ResNet18', 'CIFAR10': 'ResNet18', 'CIFAR100': 'ResNet50', 'FOOD101': 'EfficientNet-B0', 'SVHN': 'ResNet18'}

    def resolve_checkpoint_path(candidates, dataset_name, branch_name='default'):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        probed = []

        for candidate in candidates:
            base_candidates = [candidate] if os.path.isabs(candidate) else [candidate, os.path.join(script_dir, candidate)]
            for path_candidate in base_candidates:
                if any(ch in path_candidate for ch in ['*', '?', '[']):
                    probed.extend(sorted(glob.glob(path_candidate)))
                else:
                    probed.append(path_candidate)

        unique_paths = []
        seen = set()
        for p in probed:
            norm_p = os.path.normpath(p)
            if norm_p in seen:
                continue
            seen.add(norm_p)
            unique_paths.append(norm_p)

        for p in unique_paths:
            if os.path.exists(p):
                print('[{}][{}] Using reference checkpoint: {}'.format(dataset_name, branch_name, p))
                return p

        searched = '\n  - ' + '\n  - '.join(unique_paths) if unique_paths else ' (none)'
        raise FileNotFoundError(
            'No reference checkpoint found for dataset={} (branch={}). Searched:{}\n'
            'Please upload or copy a valid checkpoint into ref_models/ or trained/<dataset>/ref/.'.format(
                dataset_name, branch_name, searched
            )
        )

    def maybe_save_best_checkpoint(model_obj, candidate_path, candidate_score, best_state, tag):
        if (best_state['score'] is None) or (candidate_score > best_state['score']):
            # 仅保留最优 checkpoint，避免大量文件占用
            if best_state['path'] and os.path.exists(best_state['path']):
                os.remove(best_state['path'])
            torch.save(model_obj.state_dict(), candidate_path)
            best_state['score'] = candidate_score
            best_state['path'] = candidate_path
            print('[{}] Best checkpoint updated: {}'.format(tag, candidate_path))
            return True
        return False

    # ref models used in [1]
    if dataset == 'MNIST':
        model = model_resnet.resnet18(num_classes=10, penultimate_2d=True)
        ref_model = model_resnet.resnet18(num_classes=10, penultimate_2d=True)
        ref_path_candidates = [
            './ref_models/deep_fidelity/MNIST_ref.pt',
            './ref_models/MNIST_ResNet18_ref_*.pt',
            './ref_models/deep_fidelity/MNIST_ResNet18_Host_Model_*.pt'
        ]

    elif dataset == 'CIFAR10':
        model = model_resnet.resnet18(num_classes=10, penultimate_2d=False)
        ref_model = model_resnet.resnet18(num_classes=10, penultimate_2d=False)
        ref_path_candidates = [
            './trained/CIFAR10/ref/cifar10_best.pt',
            './ref_models/CIFAR10_ResNet18_ref_*.pt',
            './ref_models/deep_fidelity/CIFAR10_ResNet18_Host_Model_*.pt'
        ]

    elif dataset == 'CIFAR100':
        # from homura.vision.models.cifar_resnet import wrn28_10  # manually change last layer name to 'out', disable bias, adjust output
        # model = wrn28_10(num_classes=100)
        # ref_model = wrn28_10(num_classes=100)
        # ref_path = 'ref_models/CIFAR100_wrn28_10_ref_Epoch_85_test_acc_75.40%_trigger_acc_0.00%.pt'
        import torchvision.models as models
        import torch.nn as nn
        # 调用标准的 ResNet50 (特征提取层输出维度为 2048)
        model = models.resnet50(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, 100)
        model.out = model.fc  # 绑定 out 属性以兼容你现有的前向传播代码

        ref_model = models.resnet50(pretrained=False)
        ref_model.fc = nn.Linear(ref_model.fc.in_features, 100)
        ref_model.out = ref_model.fc
        ref_path_candidates = [
            './ref_models/CIFAR100_ResNet50_ref.pt',
            './ref_models/deep_fidelity/CIFAR100_wrn28_10_ref_*.pt'
        ]

    elif dataset == 'FOOD101':
        from efficientnet_pytorch import EfficientNet  # manually change last layer name to 'out', disable bias, adjust output
        model = EfficientNet.from_name('efficientnet-b0', num_classes=101)
        ref_model = EfficientNet.from_name('efficientnet-b0', num_classes=101)
        ref_path_candidates = [
            os.path.join('ref_models', 'FOOD101_EfficientNet-B0_ref_Epoch_92_test_acc_74.25%_trigger_acc_0.00%.pt'),
            './ref_models/FOOD101_EfficientNet-B0_ref_*.pt'
        ]
    elif dataset == 'SVHN':
        model = model_resnet.resnet18(num_classes=10, penultimate_2d=True)
        ref_model = model_resnet.resnet18(num_classes=10, penultimate_2d=True)
        # SVHN 作为目标数据集，优先使用 MNIST reference
        ref_path_candidates = [
            './ref_models/deep_fidelity/MNIST_ref.pt',
            './ref_models/MNIST_ResNet18_ref_*.pt',
            './ref_models/deep_fidelity/MNIST_ResNet18_Host_Model_*.pt'
        ]
    else:
        print('Dataset preparation error.')
        sys.exit()

    ref_path = None
    # ref 模式本身就是从头训练 host/reference，不应强制依赖外部 checkpoint。
    if mode != 'ref':
        if ref_path_override:
            ref_path = resolve_checkpoint_path([ref_path_override], dataset, 'override')
        else:
            ref_path = resolve_checkpoint_path(ref_path_candidates, dataset, 'base')

    num_params = sum(i.numel() for i in model.parameters() if i.requires_grad)
    print('Model number of parameters: {:.0f}'.format(num_params))

    # TODO: Train ref (host) model
    if mode == 'ref':
        print('Dataset: {}, model: {}, mode: {}, training started using {}...'.format(dataset, model_dict[dataset], mode, str(device)))
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        best_ckpt_state = {'score': None, 'path': None}
        history_rows = []
        sub_path = os.path.join(trained_path, dataset, mode)
        if not os.path.exists(sub_path):
            os.makedirs(sub_path)
        if dataset == 'MNIST':
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10], gamma=0.1)
            num_epochs = 20
        elif dataset == 'CIFAR10':
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[25], gamma=0.1)
            num_epochs = 50
        elif dataset == 'CIFAR100':
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50], gamma=0.1)
            num_epochs = 100
        elif dataset == 'SVHN':
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50], gamma=0.1)
            num_epochs = 20
        else:  # FOOD101
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50], gamma=0.1)
            num_epochs = 100

        for epoch in range(num_epochs):
            model.to(device)
            model.train()
            for batch_idx, (data, target) in enumerate(trainloader):
                data, target = data.to(device), target.to(device)
                if data.shape[1] == 1:
                    data = data.repeat(1, 3, 1, 1)
                optimizer.zero_grad()
                _, output = model(data)
                loss = F.cross_entropy(output, target)
                loss.backward()
                optimizer.step()
            scheduler.step()

            model.eval()
            correct = 0
            with torch.no_grad():
                for data, target in testloader:
                    data, target = data.to(device), target.to(device)
                    if data.shape[1] == 1:
                        data = data.repeat(1, 3, 1, 1)
                    _, output = model(data)
                    pred = output.argmax(dim=1)
                    correct += pred.eq(target.view_as(pred)).sum().item()
                acc = correct / len(testloader.dataset)

                t_correct = 0
                t_output = torch.zeros(len(trigger_sample), model.out.weight.shape[0], device=device)
                for i in range(len(trigger_sample)):
                    _, t_output[i, :] = model(trigger_sample[i].unsqueeze(0).to(device))
                t_pred = t_output.argmax(dim=1)
                # pytorch库中的函数来
                t_correct += t_pred.eq(trigger_label.to(device)).sum().item()

                model_str = sub_path + '/' + dataset + '_' + model_dict[dataset] + '_' + mode + '_Epoch_{:.0f}_test_acc_{:.2f}%_trigger_acc_{:.2f}%.pt'\
                    .format(epoch, acc * 100, t_correct / (m * n) * 100)
                current_score = (acc * 100, t_correct / (m * n) * 100)
                maybe_save_best_checkpoint(model, model_str, current_score, best_ckpt_state, '{}-{}'.format(dataset, mode))

                history_rows.append({
                    'epoch': int(epoch),
                    'test_acc': float(acc * 100),
                    'trigger_acc': float(t_correct / (m * n) * 100),
                    'checkpoint': best_ckpt_state['path'] if best_ckpt_state['path'] else ''
                })

                print('{}, Mode: {}, Epoch: {:.0f}, testing acc: {:.2f}%, trigger Acc: {:.0f}/{:.0f}.'.format(dataset, mode, epoch, acc * 100, t_correct, m * n))
            torch.cuda.empty_cache()

        if history_rows:
            pandas.DataFrame(history_rows).to_csv(os.path.join(sub_path, 'epoch_history.csv'), index=False)

    # TODO: Backdoor embed from scratch
    elif mode == 'scratch':
        print('Dataset: {}, model: {}, mode: {}, training started using {}...'.format(dataset, model_dict[dataset], mode, str(device)))
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        best_ckpt_state = {'score': None, 'path': None}
        
        # load ref model for comparison only, since it is embed from scratch
        check_point = torch.load(ref_path)
        ref_model.load_state_dict(check_point, strict=False)
        
        if dataset == 'MNIST':
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10], gamma=0.1)
            num_epochs = 20
        elif dataset == 'CIFAR10':
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[25], gamma=0.1)
            num_epochs = 50
        elif dataset == 'CIFAR100':
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50], gamma=0.1)
            num_epochs = 100
        else:
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50], gamma=0.1)
            num_epochs = 100

        for epoch in range(num_epochs):
            model.to(device)
            model.train()
            for batch_idx, (data, target) in enumerate(trainloader):
                data, target = data.to(device), target.to(device)
                if data.shape[1] == 1:
                    data = data.repeat(1, 3, 1, 1)

                trigger_index = torch.randperm(m * n)
                t_data = trigger_sample[trigger_index[0:mix], :, :, :].to(device)
                t_target = trigger_label[trigger_index[0:mix]].to(device)
                data = torch.cat((data, t_data), dim=0)
                target = torch.cat((target, t_target), dim=0)

                optimizer.zero_grad()
                _, output = model(data)
                loss = F.cross_entropy(output, target)
                loss.backward()
                optimizer.step()
            scheduler.step()

            model.eval()
            correct = 0
            with torch.no_grad():
                for data, target in testloader:
                    data, target = data.to(device), target.to(device)
                    if data.shape[1] == 1:
                        data = data.repeat(1, 3, 1, 1)
                    _, output = model(data)
                    pred = output.argmax(dim=1)
                    correct += pred.eq(target.view_as(pred)).sum().item()
                acc = correct / len(testloader.dataset)

                t_correct = 0
                t_output = torch.zeros(len(trigger_sample), model.out.weight.shape[0], device=device)
                for i in range(len(trigger_sample)):
                    _, t_output[i, :] = model(trigger_sample[i].unsqueeze(0).to(device))
                t_pred = t_output.argmax(dim=1)
                t_correct += t_pred.eq(trigger_label.to(device)).sum().item()

                prototype_ref = copy.copy(ref_model.out.weight.detach())
                prototype = copy.copy(model.to('cpu').out.weight.detach())
                lp = (prototype_ref - prototype).norm()

                sub_path = os.path.join(trained_path, dataset, mode)
                if not os.path.exists(sub_path):
                    os.makedirs(sub_path)

                model_str = sub_path + '/' + dataset + '_' + model_dict[dataset] + '_' + mode + \
                    '_Epoch_{:.0f}_test_acc_{:.2f}%_trigger_acc_{:.2f}%_n_{:.0f}_m_{:.0f}_mix_({:.0f}){:.0f}_LP_{:.4f}.pt' \
                    .format(epoch, acc * 100, t_correct / (m * n) * 100, n, m, trainloader.batch_size, mix, lp)
                current_score = (t_correct / (m * n) * 100, acc * 100)
                maybe_save_best_checkpoint(model, model_str, current_score, best_ckpt_state, '{}-{}'.format(dataset, mode))

                print('{}, Mode: {}, Epoch: {:.0f}, testing acc: {:.2f}%, trigger Acc: {:.0f}/{:.0f}, LP: {:.4f}, mix: ({:.0f}){:.0f}.'
                      .format(dataset, mode, epoch, acc * 100, t_correct, m * n, lp, trainloader.batch_size, mix))
            torch.cuda.empty_cache()

    # TODO: FTAL and FTLL
    elif (mode == 'FTAL') or (mode == 'FTLL'):
        print('Dataset: {}, model: {}, mode: {}, training started using {}...'.format(dataset, model_dict[dataset], mode, str(device)))
        optimizer = optim.Adam(model.parameters(), lr=0.0001)  # Fine-tuning based, no scheduler
        best_ckpt_state = {'score': None, 'path': None}

        check_point = torch.load(ref_path)
        model.load_state_dict(check_point, strict=False)
        ref_model.load_state_dict(check_point, strict=False)

        if dataset == 'MNIST':
            num_epochs = 20
        elif dataset == 'CIFAR10':
            num_epochs = 50
        elif dataset == 'CIFAR100':
            num_epochs = 50
        else:
            num_epochs = 50

        for epoch in range(num_epochs):
            model = model.to(device)
            model.eval()
            model = freeze_bn(model)

            if mode == 'FTLL':
                # Freeze all except last layer
                model.eval()
                for param in model.parameters():
                    param.requires_grad = False
                model.out.weight.requires_grad = True

            for batch_idx, (data, target) in enumerate(trainloader):
                data, target = data.to(device), target.to(device)
                if data.shape[1] == 1:
                    data = data.repeat(1, 3, 1, 1)

                trigger_index = torch.randperm(m * n)
                t_data = trigger_sample[trigger_index[0:mix], :, :, :].to(device)
                t_target = trigger_label[trigger_index[0:mix]].to(device)
                data = torch.cat((data, t_data), dim=0)
                target = torch.cat((target, t_target), dim=0)

                optimizer.zero_grad()
                _, output = model(data)
                loss = F.cross_entropy(output, target)
                loss.backward()
                optimizer.step()

            model.eval()
            correct = 0
            with torch.no_grad():
                for data, target in testloader:
                    data, target = data.to(device), target.to(device)
                    if data.shape[1] == 1:
                        data = data.repeat(1, 3, 1, 1)
                    _, output = model(data)
                    pred = output.argmax(dim=1)
                    correct += pred.eq(target.view_as(pred)).sum().item()
                acc = correct / len(testloader.dataset)

                t_correct = 0
                t_output = torch.zeros(len(trigger_sample), model.out.weight.shape[0], device=device)
                for i in range(len(trigger_sample)):
                    _, t_output[i, :] = model(trigger_sample[i].unsqueeze(0).to(device))
                t_pred = t_output.argmax(dim=1)
                t_correct += t_pred.eq(trigger_label.to(device)).sum().item()

                prototype_ref = copy.copy(ref_model.out.weight.detach())
                prototype = copy.copy(model.to('cpu').out.weight.detach())
                lp = (prototype_ref - prototype).norm()

                sub_path = os.path.join(trained_path, dataset, mode)
                if not os.path.exists(sub_path):
                    os.makedirs(sub_path)

                model_str = sub_path + '/' + dataset + '_' + model_dict[dataset] + '_' + mode + \
                    '_Epoch_{:.0f}_test_acc_{:.2f}%_trigger_acc_{:.2f}%_n_{:.0f}_m_{:.0f}_mix_({:.0f}){:.0f}_LP_{:.4f}.pt' \
                    .format(epoch, acc * 100, t_correct / (m * n) * 100, n, m, trainloader.batch_size, mix, lp)
                current_score = (t_correct / (m * n) * 100, acc * 100)
                maybe_save_best_checkpoint(model, model_str, current_score, best_ckpt_state, '{}-{}'.format(dataset, mode))

                print('{}, Mode: {}, Epoch: {:.0f}, testing acc: {:.2f}%, trigger Acc: {:.0f}/{:.0f}, LP: {:.4f}, mix: ({:.0f}){:.0f}.'
                      .format(dataset, mode, epoch, acc * 100, t_correct, m * n, lp, trainloader.batch_size, mix))
            torch.cuda.empty_cache()

    # TODO: FTAL with PGL
    elif mode == 'FTAL+PGR':
        print('Dataset: {}, model: {}, mode: {}, training started using {}...'.format(dataset, model_dict[dataset], mode, str(device)))
        optimizer = optim.Adam(model.parameters(), lr=0.0001)  # Fine-tuning based, no scheduler
        best_ckpt_state = {'score': None, 'path': None}
        
        check_point = torch.load(ref_path)
        model.load_state_dict(check_point, strict=False)
        ref_model.load_state_dict(check_point, strict=False)
        ref_model.eval()

        const_prototype = copy.deepcopy(model.out.weight.detach()).to(device)

        if dataset == 'MNIST':
            num_epochs = 20
        elif dataset == 'CIFAR10':
            num_epochs = 50
        elif dataset == 'CIFAR100':
            num_epochs = 50
        else:
            num_epochs = 50

        for epoch in range(num_epochs):
            model = model.to(device)
            model.eval()
            model = freeze_bn(model)

            for batch_idx, (data, target) in enumerate(trainloader):
                data, target = data.to(device), target.to(device)
                if data.shape[1] == 1:
                    data = data.repeat(1, 3, 1, 1)

                trigger_index = torch.randperm(m * n)
                t_data = trigger_sample[trigger_index[0:mix], :, :, :].to(device)
                t_target = trigger_label[trigger_index[0:mix]].to(device)
                data = torch.cat((data, t_data), dim=0)
                target = torch.cat((target, t_target), dim=0)

                optimizer.zero_grad()
                _, output = model(data)

                guided_loss = (model.out.weight - const_prototype).norm(p=2, dim=1).sum()
                alpha = 0.1
                loss = F.cross_entropy(output, target) + alpha * guided_loss
                loss.backward()
                optimizer.step()

            model.eval()
            correct = 0
            with torch.no_grad():
                for data, target in testloader:
                    data, target = data.to(device), target.to(device)
                    if data.shape[1] == 1:
                        data = data.repeat(1, 3, 1, 1)
                    _, output = model(data)
                    pred = output.argmax(dim=1)
                    correct += pred.eq(target.view_as(pred)).sum().item()
                acc = correct / len(testloader.dataset)

                t_correct = 0
                t_output = torch.zeros(len(trigger_sample), model.out.weight.shape[0], device=device)
                for i in range(len(trigger_sample)):
                    _, t_output[i, :] = model(trigger_sample[i].unsqueeze(0).to(device))
                t_pred = t_output.argmax(dim=1)
                t_correct += t_pred.eq(trigger_label.to(device)).sum().item()

                prototype_ref = copy.copy(ref_model.out.weight.detach())
                prototype = copy.copy(model.to('cpu').out.weight.detach())
                lp = (prototype_ref - prototype).norm()

                sub_path = os.path.join(trained_path, dataset, mode)
                if not os.path.exists(sub_path):
                    os.makedirs(sub_path)

                model_str = sub_path + '/' + dataset + '_' + model_dict[dataset] + '_' + mode + \
                    '_Epoch_{:.0f}_test_acc_{:.2f}%_trigger_acc_{:.2f}%_n_{:.0f}_m_{:.0f}_mix_({:.0f}){:.0f}_LP_{:.4f}.pt' \
                    .format(epoch, acc * 100, t_correct / (m * n) * 100, n, m, trainloader.batch_size, mix, lp)
                current_score = (t_correct / (m * n) * 100, acc * 100)
                maybe_save_best_checkpoint(model, model_str, current_score, best_ckpt_state, '{}-{}'.format(dataset, mode))

                print('{}, Mode: {}, Epoch: {:.0f}, testing acc: {:.2f}%, trigger Acc: {:.0f}/{:.0f}, LP: {:.4f}, mix: ({:.0f}){:.0f}.'
                      .format(dataset, mode, epoch, acc * 100, t_correct, m * n, lp, trainloader.batch_size, mix))
            torch.cuda.empty_cache()

    # TODO: Deep fidelity implementations [2]
    else:
        print('Dataset: {}, model: {}, mode: {}, training started using {}...'.format(dataset, model_dict[dataset], mode, str(device)))
        optimizer = optim.Adam(model.parameters(), lr=0.0001)
        best_ckpt_state = {'score': None, 'path': None}
        history_rows = []
        num_epochs = 50
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(num_epochs / 2)], gamma=0.1)

        # Use deep-fidelity references when available; otherwise fallback to base references.
        if dataset == 'MNIST':
            deep_ref_candidates = [
                './ref_models/deep_fidelity/MNIST_ref.pt',
                './ref_models/deep_fidelity/MNIST_ResNet18_Host_Model_*.pt',
                './ref_models/MNIST_ResNet18_ref_*.pt'
            ]
        elif dataset == 'CIFAR10':
            deep_ref_candidates = [
                './ref_models/deep_fidelity/CIFAR10_ResNet18_Host_Model_*.pt',
                './trained/CIFAR10/ref/cifar10_best.pt',
                './ref_models/CIFAR10_ResNet18_ref_*.pt'
            ]
        else:
            deep_ref_candidates = [
                './ref_models/deep_fidelity/CIFAR100_wrn28_10_ref_*.pt',
                './ref_models/CIFAR100_ResNet50_ref.pt'
            ]

        if ref_path_override:
            deep_ref_path = resolve_checkpoint_path([ref_path_override], dataset, 'deep_fidelity_override')
        else:
            deep_ref_path = resolve_checkpoint_path(deep_ref_candidates, dataset, 'deep_fidelity')

        check_point = torch.load(deep_ref_path)
        model.load_state_dict(check_point, strict=True)
        ref_model.load_state_dict(check_point, strict=True)
        ref_model.eval()

        validation_acc = torch.zeros(num_epochs, )
        trigger_acc = torch.zeros(num_epochs, )

        weight_loss = torch.zeros(num_epochs, )
        feature_loss = torch.zeros(num_epochs,)
        kld_loss = torch.zeros(num_epochs,)
        prototype_loss = torch.zeros(num_epochs,)

        # 早停条件：trigger达到100%且target(test) acc连续4轮不提升
        early_stop_patience = 4
        best_target_acc = -1.0
        no_improve_epochs = 0
        actual_epochs = num_epochs

        for epoch in range(num_epochs):
            model.eval()
            model.to(device)
            ref_model.to(device)
            model = freeze_bn(model) 
            # 将模型的最后一层锁死
            if (mode == 'FixLL') or (mode == 'FixLL+TWL') or (mode == 'FixLL+PFL') or (mode == 'FixLL+SPL'):
                model.out.weight.requires_grad = False # FixLL

            for batch_idx, (data, target) in enumerate(trainloader):
                data, target = data.to(device), target.to(device)
                if data.shape[1] == 1:
                    data = data.repeat(1, 3, 1, 1)
                trigger_index = torch.randperm(m * n)
                t_data = trigger_sample[trigger_index[0:mix], :, :, :].to(device)
                t_target = trigger_label[trigger_index[0:mix]].to(device)

                optimizer.zero_grad()

                if (mode == 'FTAL+TWL') or (mode == 'FixLL+TWL'):
                    data = torch.cat((data, t_data), dim=0)
                    target = torch.cat((target, t_target), dim=0)
                    _, output = model(data)
                    twl = torch.tensor(0., requires_grad=True).to(device)
                    for param1, param2 in zip(model.parameters(), ref_model.parameters()):
                        twl += ((param1 - param2).norm(p=2)) ** 2
                    loss = F.cross_entropy(output, target) + 0.01 * twl  # 0.01

                elif mode == 'FixLL+PFL':
                    feature, _ = model(data)
                    host_feature, _ = ref_model(data)
                    _, t_output = model(t_data)
                    l_trigger = F.cross_entropy(t_output, t_target)
                    l_feature = F.mse_loss(feature, host_feature, reduction='sum') / trainloader.batch_size
                    loss = l_trigger + 0.01 * l_feature  # paper setting 0.01

                elif mode == 'FixLL+SPL':
                    _, output = model(data)
                    _, host_output = ref_model(data)
                    _, t_output = model(t_data)
                    l_trigger = F.cross_entropy(t_output, t_target)
                    l_kl = F.kl_div(F.log_softmax(output, dim=1), F.softmax(host_output, dim=1), reduction='batchmean')
                    loss = l_trigger + 1000 * l_kl  # paper setting 1000

                else:
                    data = torch.cat((data, t_data), dim=0)
                    target = torch.cat((target, t_target), dim=0)
                    _, output = model(data)
                    loss = F.cross_entropy(output, target)

                loss.backward()
                optimizer.step()
            scheduler.step()

            model.eval()
            correct = 0
            with torch.no_grad():
                for data, target in testloader:
                    data, target = data.to(device), target.to(device)
                    if data.shape[1] == 1:
                        data = data.repeat(1, 3, 1, 1)
                    _, output = model(data)
                    pred = output.argmax(dim=1)
                    correct += pred.eq(target.view_as(pred)).sum().item()
                acc = correct / len(testloader.dataset)

                t_correct = 0
                t_output = torch.zeros(len(trigger_sample), model.out.weight.shape[0], device=device)
                for i in range(len(trigger_sample)):
                    _, t_output[i, :] = model(trigger_sample[i].unsqueeze(0).to(device))
                t_pred = t_output.argmax(dim=1)
                t_correct += t_pred.eq(trigger_label.to(device)).sum().item()

                prototype_ref = copy.copy(ref_model.out.weight.detach())
                prototype = copy.copy(model.out.weight.detach())
                lp = (prototype_ref - prototype).norm(p=2) ** 2  # in deep fidelity we set this quantity as squared

                sub_path = os.path.join(trained_path, dataset, mode)
                if not os.path.exists(sub_path):
                    os.makedirs(sub_path)

                model_str = sub_path + '/' + dataset + '_' + model_dict[dataset] + '_' + mode + \
                    '_Epoch_{:.0f}_test_acc_{:.2f}%_trigger_acc_{:.2f}%_n_{:.0f}_m_{:.0f}_mix_({:.0f}){:.0f}_LP_{:.4f}.pt' \
                    .format(epoch, acc * 100, t_correct / (m * n) * 100, n, m, trainloader.batch_size, mix, lp)
                current_score = (t_correct / (m * n) * 100, acc * 100)
                maybe_save_best_checkpoint(model, model_str, current_score, best_ckpt_state, '{}-{}'.format(dataset, mode))

                history_rows.append({
                    'epoch': int(epoch),
                    'test_acc': float(acc * 100),
                    'trigger_acc': float(t_correct / (m * n) * 100),
                    'checkpoint': best_ckpt_state['path'] if best_ckpt_state['path'] else ''
                })

                print('{}, Mode: {}, Epoch: {:.0f}, testing acc: {:.2f}%, trigger Acc: {:.0f}/{:.0f}, LP: {:.4f}, mix: ({:.0f}){:.0f}.'
                      .format(dataset, mode, epoch, acc * 100, t_correct, m * n, lp, trainloader.batch_size, mix))

                fea_loss = 0.0
                kl_loss = 0.0
                for data, target in trainloader:
                    if data.shape[1] == 1:
                        data = data.repeat(1, 3, 1, 1)
                    data, target = data.to(device), target.to(device)
                    fea, pre = model(data)
                    fea_ref, pre_ref = ref_model(data)
                    fea_loss += F.mse_loss(fea, fea_ref, reduction='sum')
                    kl_loss += F.kl_div(F.log_softmax(pre, dim=1), F.softmax(pre_ref, dim=1), reduction='sum')

                twl_new = 0.0
                for param1, param2 in zip(model.to(device).parameters(), ref_model.parameters()):
                    twl_new += ((param1 - param2).norm(p=2)) ** 2

                validation_acc[epoch] = acc * 100
                trigger_acc[epoch] = t_correct / (m*n) * 100

                weight_loss[epoch] = twl_new
                feature_loss[epoch] = fea_loss / len(trainloader.dataset)
                kld_loss[epoch] = kl_loss / len(trainloader.dataset)
                prototype_loss[epoch] = lp

                log_str = dataset + '_' + model_dict[dataset] + \
                    '_mix_({:.0f}){:.0f}_Epoch_{:.0f}_test_acc_{:.2f}%_trigger_acc_{:.2f}%_TWL_{:.4f}_PFL_{:.4f}_SPL_{:.4f}_LP_{:.4f}' \
                    .format(trainloader.batch_size, mix, epoch, acc * 100, t_correct / (m*n) * 100, twl_new,
                            fea_loss / len(trainloader.dataset), kl_loss / len(trainloader.dataset), lp)
                df = pandas.DataFrame([log_str])
                df.to_csv(sub_path + '/embed_log.txt', sep=' ', mode='a', header=False, index=False)
                print(log_str)

                # 双条件早停：必须先达到100% trigger，再看target acc是否连续4轮无提升
                current_target_acc = acc * 100
                if current_target_acc > best_target_acc:
                    best_target_acc = current_target_acc
                    no_improve_epochs = 0
                else:
                    no_improve_epochs += 1

                trigger_is_full = (t_correct == (m * n))
                if trigger_is_full and no_improve_epochs >= early_stop_patience:
                    actual_epochs = epoch + 1
                    print(
                        'Early stopping triggered at epoch {:.0f}: trigger reached 100% and '
                        'target acc has not improved for {:.0f} consecutive epochs.'.format(
                            epoch, early_stop_patience
                        )
                    )
                    break

            torch.cuda.empty_cache()

        if history_rows:
            pandas.DataFrame(history_rows).to_csv(os.path.join(sub_path, 'epoch_history.csv'), index=False)

        performance_vs_epoch_data = {
            'number_of_epochs': actual_epochs,
            'validation_accuracy': validation_acc,
            'trigger_accuracy': trigger_acc,
            'total_weight_loss': weight_loss,
            'penultimate_feature_loss': feature_loss,
            'softmax_probability_loss': kld_loss,
            'prototype_loss': lp
        }
        performance_str = sub_path + '/' + dataset + '_' + model_dict[dataset] + '_' + mode + '.pt'
        torch.save(performance_vs_epoch_data, performance_str)


def freeze_bn(model):
    for p in model.modules():
        if isinstance(p, torch.nn.BatchNorm2d):
            p.eval()   # freeze running mean and var
            p.weight.requires_grad = False
            p.bias.requires_grad = False
    return model