'''
The original source code can be found in
https://github.com/HobbitLong/SupContrast/blob/master/main_supcon.py
'''

from __future__ import print_function

import os
import copy
import sys
import argparse
import shutil
import time
import math
import random
import numpy as np

# import tensorboard_logger as tb_logger
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torchvision import transforms, datasets
from torch.utils.data import Subset, Dataset

from datasets import TinyImagenet
from util import TwoCropTransform, AverageMeter
from util import adjust_learning_rate, warmup_learning_rate
from util import set_optimizer, save_model, load_model
from networks.resnet_big import SupConResNet
from losses_negative_only import SupConLoss
# from gpytorch.kernels import ScaleKernel, RBFKernel, LinearKernel

try:
    import apex
    from apex import amp, optimizers
except ImportError:
    pass


def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--target_task', type=int, default=0)

    parser.add_argument('--resume_target_task', type=int, default=None)

    parser.add_argument('--end_task', type=int, default=None)

    parser.add_argument('--replay_policy', type=str, choices=['random'], default='random')

    parser.add_argument('--weight', type=str, choices=['alignment', 'distillation', 'distribution'], default='')

    parser.add_argument('--alpha', type=float, default=1.0)

    parser.add_argument('--beta', type=float, default=1.0)

    parser.add_argument('--gamma', type=float, default=1.0)

    parser.add_argument('--mem_size', type=int, default=200)

    parser.add_argument('--cls_per_task', type=int, default=2)

    parser.add_argument('--distill_power', type=float, default=1.0)

    parser.add_argument('--current_temp', type=float, default=0.2,
                        help='temperature for loss function')

    parser.add_argument('--past_temp', type=float, default=0.01,
                        help='temperature for loss function')

    parser.add_argument('--print_freq', type=int, default=10,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=1000,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='number of training epochs')
    parser.add_argument('--start_epoch', type=int, default=None)

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.05,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='700,800,900',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')

    # model dataset
    parser.add_argument('--model', type=str, default='resnet18')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100', 'tiny-imagenet', 'path'], help='dataset')
    parser.add_argument('--mean', type=str, help='mean of dataset in path in form of str tuple')
    parser.add_argument('--std', type=str, help='std of dataset in path in form of str tuple')
    parser.add_argument('--data_folder', type=str, default='/datasets/work/d61-eif/source/', help='path to custom dataset')
    parser.add_argument('--size', type=int, default=32, help='parameter for RandomResizedCrop')


    # temperature
    parser.add_argument('--temp', type=float, default=0.07,
                        help='temperature for loss function')

    # other setting
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--syncBN', action='store_true',
                        help='using synchronized batch normalization')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')
    parser.add_argument('--trial', type=str, default='0',
                        help='id for recording multiple runs')

    parser.add_argument('--dist_weight', type=float, default=0.9,
                        help='learning rate')

    opt = parser.parse_args()

    opt.save_freq = opt.epochs // 2


    if opt.dataset == 'cifar10':
        opt.n_cls = 10
        opt.cls_per_task = 2
        opt.size = 32
    elif opt.dataset == 'cifar100':
        opt.n_cls = 100
        opt.cls_per_task = 10
        opt.size = 32
    elif opt.dataset == 'tiny-imagenet':
        opt.n_cls = 200
        opt.cls_per_task = 20
        opt.size = 64
    else:
        pass


    # check if dataset is path that passed required arguments
    if opt.dataset == 'path':
        assert opt.data_folder is not None \
            and opt.mean is not None \
            and opt.std is not None

    # set the path according to the environment
    if opt.data_folder is None:
        opt.data_folder = '~/data/'
    opt.model_path = './save_{}_{}_rbf/{}_models'.format(opt.replay_policy, opt.mem_size, opt.dataset)
    opt.tb_path = './save_{}_{}_rbf/{}_tensorboard'.format(opt.replay_policy, opt.mem_size, opt.dataset)
    opt.log_path = './save_{}_{}_rbf/logs'.format(opt.replay_policy, opt.mem_size, opt.dataset)

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    # opt.model_name = '{}_{}_{}_lr_{}_decay_{}_bsz_{}_temp_{}_trial_{}_{}_{}_{}_{}_{}'.\
    #     format(opt.dataset, opt.size, opt.model, opt.learning_rate,
    #            opt.weight_decay, opt.batch_size, opt.temp,
    #            opt.trial,
    #            opt.start_epoch if opt.start_epoch is not None else opt.epochs, opt.epochs,
    #            opt.current_temp,
    #            opt.past_temp,
    #            opt.distill_power
    #            )
    
    opt.model_name = '{}_{}_{}_{}_bsz_{}_temp_{}'.format(opt.dataset, opt.size, opt.model, opt.learning_rate, opt.batch_size, opt.temp)

    if opt.cosine:
        opt.model_name = '{}_cosine'.format(opt.model_name)

    # warm-up for large-batch training,
    if opt.batch_size > 256:
        opt.warm = True
    if opt.warm:
        opt.model_name = '{}_warm'.format(opt.model_name)
        opt.warmup_from = 0.01
        opt.warm_epochs = 10
        if opt.cosine:
            eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
            opt.warmup_to = eta_min + (opt.learning_rate - eta_min) * (
                    1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)) / 2
        else:
            opt.warmup_to = opt.learning_rate

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    opt.log_folder = os.path.join(opt.log_path, opt.model_name)
    if not os.path.isdir(opt.log_folder):
        os.makedirs(opt.log_folder)

    return opt


def set_replay_samples(opt, model, prev_indices=None):

    is_training = model.training
    model.eval()

    class IdxDataset(Dataset):
        def __init__(self, dataset, indices):
            self.dataset = dataset
            self.indices = indices
        def __len__(self):
            return len(self.dataset)
        def __getitem__(self, idx):
            return self.indices[idx], self.dataset[idx]

    # construct data loader
    val_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    if opt.dataset == 'cifar10':
        subset_indices = []
        val_dataset = datasets.CIFAR10(root=opt.data_folder,
                                         transform=val_transform,
                                         download=True)
        val_targets = np.array(val_dataset.targets)
    elif opt.dataset == 'cifar100':
        subset_indices = []
        val_dataset = datasets.CIFAR100(root=opt.data_folder,
                                         transform=val_transform,
                                         download=True)
        val_targets = np.array(val_dataset.targets)
    elif opt.dataset == 'tiny-imagenet':
        subset_indices = []
        val_dataset = TinyImagenet(root=opt.data_folder,
                                    transform=val_transform,
                                    download=True)
        val_targets = val_dataset.targets

    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))

    if prev_indices is None:
        prev_indices = []
        observed_classes = list(range(0, opt.target_task*opt.cls_per_task))
    else:
        shrink_size = ((opt.target_task - 1) * opt.mem_size / opt.target_task)
        if len(prev_indices) > 0:
            unique_cls = np.unique(val_targets[prev_indices])
            _prev_indices = prev_indices
            prev_indices = []

            for c in unique_cls:
                mask = val_targets[_prev_indices] == c
                size_for_c = shrink_size / len(unique_cls)
                p = size_for_c - (shrink_size // len(unique_cls))
                if random.random() < p:
                    size_for_c = math.ceil(size_for_c)
                else:
                    size_for_c = math.floor(size_for_c)

                prev_indices += torch.tensor(_prev_indices)[mask][torch.randperm(mask.sum())[:size_for_c]].tolist()

            print(np.unique(val_targets[prev_indices], return_counts=True))
        observed_classes = list(range(max(opt.target_task-1, 0)*opt.cls_per_task, (opt.target_task)*opt.cls_per_task))

    if len(observed_classes) == 0:
        return prev_indices

    observed_indices = []
    for tc in observed_classes:
        observed_indices += np.where(val_targets == tc)[0].tolist()


    val_observed_targets = val_targets[observed_indices]
    val_unique_cls = np.unique(val_observed_targets)


    selected_observed_indices = []
    for c_idx, c in enumerate(val_unique_cls):
        size_for_c_float = ((opt.mem_size - len(prev_indices) - len(selected_observed_indices)) / (len(val_unique_cls) - c_idx))
        p = size_for_c_float -  ((opt.mem_size - len(prev_indices) - len(selected_observed_indices)) // (len(val_unique_cls) - c_idx))
        if random.random() < p:
            size_for_c = math.ceil(size_for_c_float)
        else:
            size_for_c = math.floor(size_for_c_float)
        mask = val_targets[observed_indices] == c
        selected_observed_indices += torch.tensor(observed_indices)[mask][torch.randperm(mask.sum())[:size_for_c]].tolist()
    print(np.unique(val_targets[selected_observed_indices], return_counts=True))


    model.is_training = is_training

    return prev_indices + selected_observed_indices


def set_loader(opt, replay_indices):
    # construct data loader
    if opt.dataset == 'cifar10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    elif opt.dataset == 'cifar100':
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
    elif opt.dataset == 'tiny-imagenet':
        mean = (0.4802, 0.4480, 0.3975)
        std = (0.2770, 0.2691, 0.2821)
    elif opt.dataset == 'path':
        mean = eval(opt.mean)
        std = eval(opt.mean)
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))


    normalize = transforms.Normalize(mean=mean, std=std)
    train_transform = transforms.Compose([
        transforms.Resize(size=(opt.size, opt.size)),
        transforms.RandomResizedCrop(size=opt.size, scale=(0.1 if opt.dataset=='tiny-imagenet' else 0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=opt.size//20*2+1, sigma=(0.1, 2.0))], p=0.5 if opt.size>32 else 0.0),
        transforms.ToTensor(),
        normalize,
    ])


    target_classes = list(range(opt.target_task*opt.cls_per_task, (opt.target_task+1)*opt.cls_per_task))
    print(target_classes)

    if opt.dataset == 'cifar10':
        subset_indices = []
        _train_dataset = datasets.CIFAR10(root=opt.data_folder,
                                         transform=TwoCropTransform(train_transform),
                                         download=True)
        for tc in target_classes:
            target_class_indices = np.where(np.array(_train_dataset.targets) == tc)[0]
            subset_indices += np.where(np.array(_train_dataset.targets) == tc)[0].tolist()

        subset_indices += replay_indices

        train_dataset =  Subset(_train_dataset, subset_indices)
        print('Dataset size: {}'.format(len(subset_indices)))
        uk, uc = np.unique(np.array(_train_dataset.targets)[subset_indices], return_counts=True)
        print(uc[np.argsort(uk)])

    elif opt.dataset == 'cifar100':
        subset_indices = []
        _train_dataset = datasets.CIFAR100(root=opt.data_folder,
                                         transform=TwoCropTransform(train_transform),
                                         download=True)
        for tc in target_classes:
            target_class_indices = np.where(np.array(_train_dataset.targets) == tc)[0]
            subset_indices += np.where(np.array(_train_dataset.targets) == tc)[0].tolist()

        subset_indices += replay_indices

        train_dataset =  Subset(_train_dataset, subset_indices)
        print('Dataset size: {}'.format(len(subset_indices)))
        uk, uc = np.unique(np.array(_train_dataset.targets)[subset_indices], return_counts=True)
        print(uc[np.argsort(uk)])

    elif opt.dataset == 'tiny-imagenet':
        subset_indices = []
        _train_dataset = TinyImagenet(root=opt.data_folder,
                                          transform=TwoCropTransform(train_transform),
                                          download=True)
        for tc in target_classes:
            target_class_indices = np.where(_train_dataset.targets == tc)[0]
            subset_indices += np.where(_train_dataset.targets == tc)[0].tolist()

        subset_indices += replay_indices

        train_dataset =  Subset(_train_dataset, subset_indices)
        print('Dataset size: {}'.format(len(subset_indices)))
        uk, uc = np.unique(np.array(_train_dataset.targets)[subset_indices], return_counts=True)
        print(uc[np.argsort(uk)])

    elif opt.dataset == 'path':
        train_dataset = datasets.ImageFolder(root=opt.data_folder,
                                            transform=TwoCropTransform(train_transform))
    else:
        raise ValueError(opt.dataset)

    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size, shuffle=(train_sampler is None),
        num_workers=opt.num_workers, pin_memory=True, sampler=train_sampler)

    return train_loader, subset_indices



def set_model(opt):
    model = SupConResNet(name=opt.model)
    criterion = SupConLoss(temperature=opt.temp)

    # enable synchronized Batch Normalization
    if opt.syncBN:
        model = apex.parallel.convert_syncbn_model(model)

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model.encoder = torch.nn.DataParallel(model.encoder)
        model = model.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

    return model, criterion


def compute_reconstructed_feature(feature1_, feature2_, labels1=None, labels2=None, dim=20):
        # assert feature1_.shape == feature2_.shape
        batch_size = feature1_.shape[0]
        feature1_ = feature1_ + torch.zeros_like(feature1_).normal_(0, 0.01)
        feature2_ = feature2_ + torch.zeros_like(feature2_).normal_(0, 0.01)

        # print(feature1_.shape, feature2_.shape)
        if labels2 is not None:
            labels_unique = torch.unique(labels2)
            label_ind1 = [[i for i, value in enumerate(labels1) if value == x] for x in labels_unique]
            label_ind2 = [[i for i, value in enumerate(labels2) if value == x] for x in labels_unique]
 
            hyperplane_x = []
            hyperplane_y = []
            dist = []

            for label_ind in range(len(label_ind2)):
                # print(len(labels), label)
                feature1 = feature1_[label_ind1[label_ind]]
                feature2 = feature2_[label_ind2[label_ind]]
                # print("Original input shape:{}".format(feature1.shape))
		
                samples1 = torch.transpose(feature1, 0, 1)
                samples2 = torch.transpose(feature2, 0, 1)
                # print("SVD input shape:{}".format(samples1.shape))
		
                u_x, s_x, v_x = torch.linalg.svd(samples1, full_matrices=False)
                u_y, s_y, v_y = torch.linalg.svd(samples2, full_matrices=False)
                # print("Original subspaces shape:{}, selected:{}, final subspaces shape:{}".format(u_x.shape, dim, u_x[:, 0:dim].shape))
		
                dist_per_class = torch.frobenius_norm(torch.mm(torch.mm(u_x[:, 0:dim], torch.transpose(u_x[:, 0:dim], 0, 1)), samples1) - torch.mm(torch.mm(u_y[:, 0:dim], torch.transpose(u_y[:, 0:dim], 0, 1)), samples1)) ** 2
                ## dist_per_class = (1/feature1.shape[0])*(torch.frobenius_norm(torch.mm(torch.transpose(u_x[:, 0:dim], 0, 1), u_y[:, 0:dim])) ** 2)
                dist.append(dist_per_class)

                # u_x, s_x, v_x = torch.linalg.svd(feature1, full_matrices=False)
                # u_y, s_y, v_y = torch.linalg.svd(feature2, full_matrices=False)

                # hyperplane_x.append(u_x[:,0:dim])
                # hyperplane_y.append(u_y[:,0:dim])
                # print(torch.mm(u_x, u_x.T))
                # dist.append(torch.frobenius_norm(torch.mm(u_x[:, 0:dim], torch.transpose(u_y[:, 0:dim], 0, 1))))
                
            return torch.tensor(dist).mean()

            # combined_hyperplane_x = torch.stack(hyperplane_x, dim=0)
            # combined_hyperplane_y = torch.stack(hyperplane_y, dim=0)
            # print(combined_hyperplane_x.shape, combined_hyperplane_y.shape)
            # dist = torch.frobenius_norm(torch.mm(torch.transpose(combined_hyperplane_x, 0, 1), combined_hyperplane_y))
            # return dist

        feature1 = torch.transpose(feature1_, 0, 1)
        feature2 = torch.transpose(feature2_, 0, 1)
        u_x, s_x, v_x = torch.linalg.svd(feature1, full_matrices=False)
        u_y, s_y, v_y = torch.linalg.svd(feature2, full_matrices=False)
        # print(u_x.size(), u_y.size())
        return torch.frobenius_norm(torch.mm(torch.mm(u_x[:, 0:dim], torch.transpose(u_x[:, 0:dim], 0, 1)), feature1) - torch.mm(torch.mm(u_y[:, 0:dim], torch.transpose(u_y[:, 0:dim], 0, 1)), feature1)) ** 2
        #### return (1/batch_size)*(torch.frobenius_norm(torch.mm(torch.mm(u_x[:, 0:dim], torch.transpose(u_x[:, 0:dim], 0, 1)), feature1) - torch.mm(torch.mm(u_y[:, 0:dim], torch.transpose(u_y[:, 0:dim], 0, 1)), feature1)) ** 2)

        # print(torch.mm(u_x, u_x.T))
        # u_x, s_x, v_x = torch.linalg.svd(feature1_, full_matrices=False)
        # u_y, s_y, v_y = torch.linalg.svd(feature2_, full_matrices=False)
        # return torch.frobenius_norm(torch.mm(u_x[:, 0:dim], torch.transpose(u_y[:, 0:dim], 0, 1)))


def compute_subspace_projection(feature, labels=None, dim=100):
        batch_size = feature.shape[0]
        # print(feature.shape)
        dim = feature.shape[1]//4
        feature = feature + torch.zeros_like(feature).normal_(0, 0.01)
        u = {}
        projected_feature = torch.zeros(dim, batch_size)
        
        if labels is not None:
            labels_unique = torch.unique(labels)
            label_inds = [[i for i, value in enumerate(labels) if value == x] for x in labels_unique]

            for label_ind in range(len(labels_unique)):
                ind = label_inds[label_ind]
                samples = feature[ind]
		
                samples = torch.transpose(samples, 0, 1)
		
                u_x, s_x, v_x = torch.linalg.svd(samples, full_matrices=False)
                projected_feature[ind] = torch.mm(torch.transpose(u_x[:, 0:dim], 0, 1), samples)
                
            return torch.transpose(projected_feature, 0, 1)
            
        samples = torch.transpose(feature, 0, 1)
        u_x, s_x, v_x = torch.linalg.svd(samples, full_matrices=False)
        projected_feature = torch.mm(torch.transpose(u_x[:, 0:dim], 0, 1), samples)
        return torch.transpose(projected_feature, 0, 1)


def update_old_model(net, old_net, m_=.001):
    net.eval()
    for param_q, param_k in zip(net.parameters(), old_net.parameters()):
        param_k.data = param_k.data * (1. - m_) + param_q.data * m_
    net.train()


# Features Sparsification loss defined in SDR: https://arxiv.org/abs/2103.06342
class FeaturesSparsificationLoss(nn.Module):
    def __init__(self, lfs_normalization='softmax', lfs_shrinkingfn='squared', lfs_loss_fn_touse='entropy', mask=False, reduction='mean'):
        super().__init__()
        self.mask = mask
        self.lfs_normalization = lfs_normalization
        self.lfs_shrinkingfn = lfs_shrinkingfn
        self.lfs_loss_fn_touse = lfs_loss_fn_touse
        self.eps = 1e-15
        self.reduction = reduction

    def forward(self, features, labels, val=False):
        outputs = torch.tensor(0.)

        if not val:
            
            if self.lfs_normalization == 'L1':
                features_norm = F.normalize(features, p=1, dim=1)
            elif self.lfs_normalization == 'L2':
                features_norm = F.normalize(features, p=2, dim=1)
            elif self.lfs_normalization == 'max_foreachfeature':
                features_norm = features / (torch.max(features, dim=1, keepdim=True).values + self.eps)
            elif self.lfs_normalization == 'max_maskedforclass':
                labels = labels.unsqueeze(dim=1)
                labels_down = labels  # (F.interpolate(input=labels.double(), size=(features.shape[2], features.shape[3]), mode='nearest')).long()
                features_norm = torch.zeros_like(features)
                classes = torch.unique(labels_down)
                if classes[-1] == 0:
                    classes = classes[:-1]
                for cl in classes:
                    cl_mask = labels_down == cl
                    features_norm += (features / (torch.max(features[cl_mask.expand(-1, features.shape[1], -1, -1)]) + self.eps)) * cl_mask.float()
            elif self.lfs_normalization == 'max_overall':
                features_norm = features / (torch.max(features) + self.eps)
            elif self.lfs_normalization == 'softmax':
                features_norm = torch.softmax(features, dim=1)

            if features_norm.sum() > 0:
                if self.lfs_shrinkingfn == 'squared':
                    shrinked_value = torch.sum(features_norm**2, dim=1, keepdim=True)
                if self.lfs_shrinkingfn == 'power3':
                    shrinked_value = torch.sum(features_norm ** 3, dim=1, keepdim=True)
                elif self.lfs_shrinkingfn == 'exponential':
                    shrinked_value = torch.sum(torch.exp(features_norm), dim=1, keepdim=True)

                summed_value = torch.sum(features_norm, dim=1, keepdim=True)

                if self.lfs_loss_fn_touse == 'ratio':
                    outputs = shrinked_value / (summed_value + self.eps)
                elif self.lfs_loss_fn_touse == 'lasso':  # NB: works at features space directly
                    outputs = torch.norm(features, 1) / 2  # simple L1 (Lasso) regularization
                elif self.lfs_loss_fn_touse == 'max_minus_ratio':
                    # TODO: other loss functions to be considered
                    # outputs = summed_value - shrinked_value / summed_value
                    pass
                elif self.lfs_loss_fn_touse == 'entropy':  # NB: works only with probabilities (i.e. with L1 or softmax as normalization)
                    outputs = torch.sum(- features_norm * torch.log(features_norm + 1e-10), dim=1)

        if self.reduction == 'mean':
            return outputs.mean()
        elif self.reduction == 'sum':
            return outputs.sum()


def train(train_loader, model, model2, criterion, sparsity_criterion, optimizer, epoch, opt):


    """one epoch training"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    distill = AverageMeter()

    end = time.time()
    for idx, (images, labels) in enumerate(train_loader):
        data_time.update(time.time() - end)

        images = torch.cat([images[0], images[1]], dim=0)
        if torch.cuda.is_available():
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]

        with torch.no_grad():
            prev_task_mask = labels < opt.target_task * opt.cls_per_task
            prev_task_mask = prev_task_mask.repeat(2)


        # warm-up learning rate
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        # compute loss
        features, encoded = model(images, return_feat=True)

        # Asym SupCon
        # f1, f2 = torch.split(F.normalize(encoded, dim=1), [bsz, bsz], dim=0)
        # projected_f1 = compute_subspace_projection(f1)
        # projected_f2 = compute_subspace_projection(f2)
        # features = torch.cat([projected_f1.unsqueeze(1), projected_f2.unsqueeze(1)], dim=1)
        
        f1, f2 = torch.split(features, [bsz, bsz], dim=0)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)

        # loss = criterion(features, labels, target_labels=list(range(opt.target_task*opt.cls_per_task, (opt.target_task+1)*opt.cls_per_task)))
        
        # loss = criterion(features, labels, target_labels=list(range((opt.target_task+1)*opt.cls_per_task)), contrast_mode='all')
        
        if opt.target_task == 0: loss = criterion(features, labels, target_labels=list(range((opt.target_task+1)*opt.cls_per_task)), contrast_mode='all')
        # loss = criterion(features, labels, target_labels=list(range(opt.target_task*opt.cls_per_task, (opt.target_task+1)*opt.cls_per_task)), contrast_mode='all')
        
        # IRD (past)
        if opt.target_task > 0:
            with torch.no_grad():
                features_prev_task, encoded_prev_task = model2(images, return_feat=True)  # Used
                # f1_old, f2_old = torch.split(features_prev_task, [bsz, bsz], dim=0)  # Used
                # encoded_old = model2.encoder(images)
                
                # f1_old, f2_old = torch.split(F.normalize(encoded_prev_task, dim=1), [bsz, bsz], dim=0)
                # projected_f1_old = compute_subspace_projection(f1_old)
                # projected_f2_old = compute_subspace_projection(f2_old)
            # features_prev_task = F.normalize(model2.head(encoded_old.detach()), dim=1)
            f1_old, f2_old = torch.split(features_prev_task, [bsz, bsz], dim=0)
            # features_combined_new = torch.cat([projected_f1.unsqueeze(1), projected_f2.unsqueeze(1)], dim=0)
            # features_combined_old = torch.cat([projected_f1_old.unsqueeze(1).detach(), projected_f2_old.unsqueeze(1).detach()], dim=0)
            features_combined_new = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=0)
            features_combined_old = torch.cat([f1_old.unsqueeze(1), f2_old.unsqueeze(1)], dim=0)
            features_combined = torch.cat([features_combined_new, features_combined_old.detach()], dim=1)
            loss_distill = criterion(features_combined, torch.cat((labels, labels)), target_labels=list(range((opt.target_task+1)*opt.cls_per_task)), contrast_mode='one')
            # loss_distill = (-logits2 * torch.log(logits1)).sum(1).mean()
            #closs += opt.distill_power * loss_distill
            loss = loss_distill
            # loss = loss + loss_distill
            # loss /= 2
            
            # loss = (1. - opt.dist_weight)*loss + opt.dist_weight*loss_distill

            distill.update(loss_distill.item(), bsz)

        # loss += sparsity_criterion(encoded, torch.cat((labels, labels)))
        # loss += (torch.norm(encoded, 1) / 2).mean()  # Used
        # loss += FeaturesSparsificationLoss(lfs_normalization='softmax', lfs_loss_fn_touse='entropy')(encoded, torch.cat((labels, labels)))  # Latest, Works Better!
        
        # update metric
        losses.update(loss.item(), bsz)

        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % opt.print_freq == 0 or idx+1 == len(train_loader):
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f} {distill.avg:.3f})'.format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, distill=distill))
            sys.stdout.flush()

    return losses.avg, model2


def main():
    opt = parse_option()

    target_task = opt.target_task

    # build model and criterion
    model, criterion = set_model(opt)
    model2, _ = set_model(opt)
    model2.eval()
    # model2.encoder.eval()
    # model2.head.train()
    
    sparsity_criterion = FeaturesSparsificationLoss()
    
    # build optimizer
    optimizer = set_optimizer(opt, model)
    # optimizer = set_optimizer(opt, model, model2)
    
    replay_indices = None

    if opt.resume_target_task is not None:
        load_file = os.path.join(opt.save_folder, 'last_{policy}_{target_task}.pth'.format(policy=opt.replay_policy ,target_task=opt.resume_target_task))
        model, optimizer = load_model(model, optimizer, load_file)
        if opt.resume_target_task == 0:
            replay_indices = []
        else:
            replay_indices = np.load(
              os.path.join(opt.log_folder, 'replay_indices_{policy}_{target_task}.npy'.format(policy=opt.replay_policy ,target_task=opt.resume_target_task))
            ).tolist()
        print(len(replay_indices))

    # tensorboard
    # logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)
    writer = SummaryWriter(log_dir=opt.tb_folder, flush_secs=2)

    original_epochs = opt.epochs

    if opt.end_task is not None:
        if opt.resume_target_task is not None:
            assert opt.end_task > opt.resume_target_task
        opt.end_task = min(opt.end_task+1, opt.n_cls // opt.cls_per_task)
    else:
        opt.end_task = opt.n_cls // opt.cls_per_task

    for target_task in range(0 if opt.resume_target_task is None else opt.resume_target_task+1, opt.end_task):

        opt.target_task = target_task
        model2 = copy.deepcopy(model)

        print('Start Training current task {}'.format(opt.target_task))

        # acquire replay sample indices
        replay_indices = set_replay_samples(opt, model, prev_indices=replay_indices)

        np.save(
          os.path.join(opt.log_folder, 'replay_indices_{policy}_{target_task}.npy'.format(policy=opt.replay_policy ,target_task=target_task)),
          np.array(replay_indices))

        # build data loader (dynamic: 0109)
        train_loader, subset_indices = set_loader(opt, replay_indices)

        np.save(
          os.path.join(opt.log_folder, 'subset_indices_{policy}_{target_task}.npy'.format(policy=opt.replay_policy ,target_task=target_task)),
          np.array(subset_indices))



        # training routine
        if target_task == 0 and opt.start_epoch is not None:
            opt.epochs = opt.start_epoch
        else:
            opt.epochs = original_epochs

        for epoch in range(1, opt.epochs + 1):

            adjust_learning_rate(opt, optimizer, epoch)

            # train for one epoch
            time1 = time.time()
            loss, model2 = train(train_loader, model, model2, criterion, sparsity_criterion, optimizer, epoch, opt)
            
            mu = .001

            # for curr_param, old_param in zip(model.parameters(), model2.parameters()):
            #     old_param.data = (1. - mu) * old_param.data + mu * curr_param.data
            time2 = time.time()
            print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

            # tensorboard logger
            # logger.log_value('loss_{target_task}'.format(target_task=target_task), loss, epoch)
            # logger.log_value('learning_rate_{target_task}'.format(target_task=target_task), optimizer.param_groups[0]['lr'], epoch)
            
            writer.add_scalar('loss_{target_task}'.format(target_task=target_task), loss, epoch)
            writer.add_scalar('learning_rate_{target_task}'.format(target_task=target_task), optimizer.param_groups[0]['lr'], epoch)

        # save the last model
        save_file = os.path.join(
            opt.save_folder, 'last_{policy}_{target_task}.pth'.format(policy=opt.replay_policy ,target_task=target_task))
        save_model(model, optimizer, opt, opt.epochs, save_file)

if __name__ == '__main__':
    main()
