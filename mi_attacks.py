import torch
import torch.nn as nn
from torchvision import datasets, transforms

import numpy as np
import pickle
import os

import warnings
warnings.filterwarnings("ignore")

# dependent package from https://github.com/yigitcankaya/augmentation_mia.

def split_indices(indices, first_split_size):
    first_split_indices = np.random.choice(indices, size=first_split_size, replace=False)
    second_split_indices = np.array([x for x in indices if x not in first_split_indices])
    
    return first_split_indices, second_split_indices

def apply_avg_and_best_attacks(train_losses, test_losses, idx):
    train_in_atk_train_idx, train_in_atk_test_idx, test_in_atk_train_idx, test_in_atk_test_idx = idx

    avg_loss_train = np.mean(train_losses[train_in_atk_train_idx])
    avg_train_memberships = yeom_mi_attack(train_losses[train_in_atk_test_idx], avg_loss_train)
    avg_test_memberships = yeom_mi_attack(test_losses[test_in_atk_test_idx], avg_loss_train)
    avg_yeom_mi_advantage = mi_success(avg_train_memberships, avg_test_memberships, print_details=False)

    avg_results = (avg_loss_train, avg_train_memberships, avg_test_memberships, avg_yeom_mi_advantage)

    best_threshold = yeom_w_get_best_threshold(train_losses[train_in_atk_train_idx], test_losses[test_in_atk_train_idx])
    best_train_memberships = yeom_mi_attack(train_losses[train_in_atk_test_idx], best_threshold)
    best_test_memberships = yeom_mi_attack(test_losses[test_in_atk_test_idx], best_threshold)
    best_yeom_mi_advantage = mi_success(best_train_memberships, best_test_memberships, print_details=False)

    best_results = (best_threshold, best_train_memberships, best_test_memberships, best_yeom_mi_advantage)

    return avg_results, best_results


def take_subset_from_datasets(datasets, seed, n_attacker_train, n_attacker_test, batch_size=1000, device='cpu'):

    np.random.seed(seed)
    train_indices = np.random.choice(len(datasets[0].data), size=n_attacker_train + n_attacker_test, replace=False)
    test_indices = np.random.choice(len(datasets[1].data), size=n_attacker_train + n_attacker_test, replace=False)

    train_in_atk_test_idx, train_in_atk_train_idx = split_indices(train_indices, n_attacker_test)
    test_in_atk_test_idx, test_in_atk_train_idx = split_indices(test_indices, n_attacker_test)

    train_data = datasets[0].data[np.concatenate((train_in_atk_train_idx, train_in_atk_test_idx))].cpu().detach().numpy()
    train_labels = datasets[0].labels[np.concatenate((train_in_atk_train_idx, train_in_atk_test_idx))].cpu().detach().numpy()

    test_data = datasets[1].data[np.concatenate((test_in_atk_train_idx, test_in_atk_test_idx))].cpu().detach().numpy()
    test_labels = datasets[1].labels[np.concatenate((test_in_atk_train_idx, test_in_atk_test_idx))].cpu().detach().numpy()

    train_ds = ManualData(train_data, train_labels)
    train_ds.train = False

    test_ds = ManualData(test_data, test_labels)
    test_ds.train = False

    train_loader = get_loader(train_ds, shuffle=False, batch_size=batch_size, device=device)
    test_loader = get_loader(test_ds, shuffle=False, batch_size=batch_size, device=device)

    train_in_atk_train_idx, train_in_atk_test_idx = np.arange(len(train_in_atk_train_idx)), np.arange(len(train_in_atk_train_idx), len(train_data))
    test_in_atk_train_idx, test_in_atk_test_idx = np.arange(len(test_in_atk_train_idx)), np.arange(len(test_in_atk_train_idx), len(test_data))

    idx = (train_in_atk_train_idx, train_in_atk_test_idx, test_in_atk_train_idx, test_in_atk_test_idx)
    
    return (train_loader, test_loader), idx


def apply_mi_attack(model, loaders, idx, save_path, n_attacker_train=100, seed=0, device='cpu'):

    results = {}
    results_path = os.path.join(save_path, f'mi_results_ntrain_{n_attacker_train}_randseed_{seed}.pickle')
    
    train_top1, train_top5 = test_clf(model, loaders[0], device)
    test_top1, test_top5 = test_clf(model, loaders[1], device)
    
    train_losses = get_clf_losses(model, loaders[0], device=device)
    test_losses = get_clf_losses(model, loaders[1], device=device)

    # apply vanilla yeom attacks
    avg_results, best_results = apply_avg_and_best_attacks(train_losses, test_losses, idx)
    avg_loss_train, avg_train_memberships, avg_test_memberships, avg_yeom_adv = avg_results
    best_threshold, best_train_memberships, best_test_memberships, best_yeom_adv = best_results

    results['train_top1'], results['train_top5'], results['test_top1'], results['test_top5'] = train_top1, train_top5, test_top1, test_top5
    results['avg_yeom_adv'], results['best_yeom_adv'], results['avg_threshold'], results['best_threshold'] = avg_yeom_adv, best_yeom_adv, avg_loss_train, best_threshold
    results['avg_train_memberships'], results['avg_test_memberships'] = avg_train_memberships, avg_test_memberships
    results['best_train_memberships'], results['best_test_memberships'] = best_train_memberships, best_test_memberships
    results['std_train_losses'], results['std_test_losses'] = train_losses, test_losses
    results['attack_idx'] = idx

    with open(results_path, 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)


    print('Train Top1: {0:.3f}%, Train Top5: {1:.3f}%, Test Top1: {2:.3f}%, Test Top5: {3:.3f}%'.format(results['train_top1'], results['train_top5'], results['test_top1'], results['test_top5']))
    print('Avg Yeom MI Advantage: {0:.2f}'.format(results['avg_yeom_adv']))
    print('Best Yeom MI Advantage: {0:.2f}'.format(results['best_yeom_adv']))

    return results


def mi_success(train_memberships, test_memberships, print_details=True):
    tp = np.sum(train_memberships)
    fp = np.sum(test_memberships)
    fn = len(train_memberships) - tp
    tn = len(test_memberships) - fp

    # yeom's membership inference advantage
    acc = 100*(tp + tn) / (tp + fp + tn + fn)
    advantage = 2*(acc - 50)

    if print_details:
        precision = 100*(tp/(tp+fp)) if (tp+fp) > 0 else 0
        recall = 100*(tp/(tp+fn)) if (tp+fn) > 0 else 0

        print('Adversary Advantage: {0:.3f}%, Accuracy: {1:.3f}%, Precision : {2:.3f}%, Recall: {3:.3f}%'.format(advantage,  acc, precision, recall))
        print('In training: {}/{}, In testing: {}/{}'.format(tp, len(train_memberships), tn, len(test_memberships)))

    return advantage

# YEOM et all's membership inference attack using pred loss
def yeom_mi_attack(losses, avg_loss):
    memberships = (losses < avg_loss).astype(int)
    return memberships


def yeom_w_get_best_threshold(train_losses, test_losses):    
    advantages = []

    mean_loss = np.mean(train_losses)
    std_dev = np.std(train_losses)

    coeffs = np.linspace(-5,5,num=1001, endpoint=True)

    for coeff in coeffs:
        cur_threshold = mean_loss + std_dev*coeff
        cur_yeom_mi_advantage = mi_success(yeom_mi_attack(train_losses, cur_threshold),  yeom_mi_attack(test_losses, cur_threshold), print_details=False)
        advantages.append(cur_yeom_mi_advantage)

    best_threshold = mean_loss + std_dev*coeffs[np.argmax(advantages)]

    return best_threshold

def get_clf_losses(clf, loader, device='cpu'):

    clf_loss_func = nn.NLLLoss(reduction='none')

    losses = np.zeros(loader_inst_counter(loader))
    cur_idx = 0

    clf.eval()
    with torch.no_grad():
        for batch in loader:
            b_x = batch[0].to(device, dtype=torch.float)
            b_y = batch[1].to(device, dtype=torch.long)

            output = clf(b_x)
            losses[cur_idx:cur_idx+len(b_x)] = clf_loss_func(output, b_y).flatten().cpu().detach().numpy()
            cur_idx += len(b_x)

    
    return losses

def test_clf(clf, loader, device='cpu'):
    clf.eval()
   
    top1 = AverageMeter()
    top5 = AverageMeter()

    with torch.no_grad():
        for x, y in loader:
            b_x = x.to(device, dtype=torch.float)
            b_y = y.to(device, dtype=torch.long)

            clf_output = clf(b_x)
            
            accs = accuracy(clf_output, b_y, topk=(1, 5))
            top5.update(accs[1], b_x.size(0))
            top1.update(accs[0], b_x.size(0))

    top1_acc = top1.avg
    top5_acc = top5.avg

    return top1_acc, top5_acc

class ManualData(torch.utils.data.Dataset):
    def __init__(self, data, labels, device='cpu'):
        self.data = torch.from_numpy(data).to(device, dtype=torch.float)
        self.device = device
        self.labels = torch.from_numpy(labels).to(device, dtype=torch.long)
        self.train = True

        self.transforms = None
        self.gaussian_std = None


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]

        if self.train:
            if self.transforms is not None:
                data = self.transforms(data)
            
            if self.gaussian_std is not None:
                data = torch.clamp(data + torch.randn(data.size(), device=self.device) * self.gaussian_std, min=0, max=1)

        return (data, self.labels[idx])

def get_loader(dataset, shuffle=True, batch_size=128, device='cpu'):

    if device == 'cpu':
        num_workers = 4
    else:
        num_workers = 0


    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        
    return loader

def get_cifar100_datasets(root, device='cpu'):
    t = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.CIFAR100(root, train=True, download=True, transform=t)
    test_dataset = datasets.CIFAR100(root, train=False, download=True, transform=t)
        
    train_data, test_data = (train_dataset.data/ 255) , (test_dataset.data / 255)
    train_data, test_data = train_data.transpose((0, 3, 1, 2)), test_data.transpose((0,3,1,2))  
    train_data, test_data = (train_data-0.5)/0.5, (test_data-0.5)/0.5
    train_labels, test_labels = np.array(train_dataset.targets), np.array(test_dataset.targets)
    
    train_dataset = ManualData(train_data, train_labels, device)
    test_dataset = ManualData(test_data, test_labels, device)
    
    return train_dataset, test_dataset

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].flatten().float().sum(0, keepdim=True)
            acc = float(correct_k.mul_(100.0 / batch_size))
            res.append(acc)
        return res

def loader_inst_counter(loader):
    num_instances = 0
    for batch in loader:
        num_instances += len(batch[1])
    return num_instances

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n  
        self.count += n
        self.avg = self.sum / self.count