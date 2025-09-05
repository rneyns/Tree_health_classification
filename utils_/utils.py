# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 14:15:54 2023

@author: Robbe Neyns
"""

import torch
import torch.nn as nn
import numpy as np
import random


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


def distance_loss(x, y):
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(1, -2, x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    dist_sum = dist.sum(1)
    _, index = dist_sum.sort(descending=True)
    return index.long()



import torch
from sklearn.metrics import roc_auc_score, mean_squared_error, confusion_matrix
import numpy as np
from augmentations import embed_data_mask
import torch.nn as nn
import sklearn


def make_default_mask(x):
    mask = np.ones_like(x)
    mask[:, -1] = 0
    return mask


def tag_gen(tag, y):
    return np.repeat(tag, len(y['data']))


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_scheduler(args, optimizer):
    if args.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    elif args.scheduler == 'linear':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                         milestones=[args.epochs // 2.667, args.epochs // 1.6,
                                                                     args.epochs // 1.142], gamma=0.1)
    return scheduler


def imputations_acc_justy(model, dloader, device):
    model.eval()
    m = nn.Softmax(dim=1)
    y_test = torch.empty(0).to(device)
    y_pred = torch.empty(0).to(device)
    prob = torch.empty(0).to(device)
    with torch.no_grad():
        for i, data in enumerate(dloader, 0):
            image, ids, DOY, x_categ, x_cont, y_gts = data[0].to(device), data[1].to(device).type(torch.float32), data[2].to(
                device).type(torch.float32), data[3].to(device).type(torch.float32), data[4].type(torch.float32).to(
                device),data[5].to(device,dtype=torch.long)#,data[6].to(device).type(torch.float32)
            _, x_categ_enc, x_cont_enc, con_mask = embed_data_mask(x_categ, x_cont, model, vision_dset, DOY=DOY)
            reps = model.tab_net.transformer(x_categ_enc, x_cont_enc, con_mask)
            y_reps = reps[:, 0, :]
            y_outs = model.tab_net.mlpfory(y_reps)
            # import ipdb; ipdb.set_trace()
            y_test = torch.cat([y_test, x_categ[:, -1].float()], dim=0)
            y_pred = torch.cat([y_pred, torch.argmax(m(y_outs), dim=1).float()], dim=0)
            prob = torch.cat([prob, m(y_outs)[:, -1].float()], dim=0)

    correct_results_sum = (y_pred == y_test).sum().float()
    acc = correct_results_sum / y_test.shape[0] * 100
    auc = roc_auc_score(y_score=prob.cpu(), y_true=y_test.cpu())
    return acc, auc


def multiclass_acc_justy(model, dloader, device):
    model.eval()
    vision_dset = True
    m = nn.Softmax(dim=1)
    y_test = torch.empty(0).to(device)
    y_pred = torch.empty(0).to(device)
    prob = torch.empty(0).to(device)
    with torch.no_grad():
        for i, data in enumerate(dloader, 0):
            image, ids, DOY, x_categ, x_cont, y_gts = data[0].to(device), data[1].to(device).type(torch.float32), data[2].to(
                device).type(torch.float32), data[3].to(device).type(torch.float32), data[4].type(torch.LongTensor).to(
                device) ,data[5].to(device,dtype=torch.long)#,data[6].to(device).type(torch.float32)
            _, x_categ_enc, x_cont_enc, con_mask = embed_data_mask(x_categ, x_cont, model, vision_dset, DOY=DOY)
            reps = model.tab_net.transformer(x_categ_enc, x_cont_enc, con_mask)
            y_reps = reps[:, 0, :]
            y_outs = model.tab_net.mlpfory(y_reps)
            # import ipdb; ipdb.set_trace()
            y_test = torch.cat([y_test, x_categ[:, -1].float()], dim=0)
            y_pred = torch.cat([y_pred, torch.argmax(m(y_outs), dim=1).float()], dim=0)

    correct_results_sum = (y_pred == y_test).sum().float()
    acc = correct_results_sum / y_test.shape[0] * 100
    return acc, 0


def class_wise_acc(y_pred, y_test, num_classes):
    y_pred_softmax = torch.log_softmax(y_pred, dim=1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim=1)
    correct_pred = (y_pred_tags == y_test).float()
    acc_classwise = []
    total_correct = []
    total_val_batch = []

    for i in range(num_classes):
        correct_pred_class = correct_pred[y_test == i]
        acc_class = correct_pred_class.sum() / len(correct_pred_class)
        acc_classwise.append(acc_class)
        total_correct.append(correct_pred_class.sum().cpu().data.numpy())
        # total_val_batch.append(torch.tensor(len(correct_pred_class), dtype=torch.int8))
        total_val_batch.append(len(correct_pred_class))
        # print('Accuracy of {} : {} / {} = {:.4f} %'.format(i, correct_pred_class.sum() , len(correct_pred_class) , 100 * acc_class))

    return acc_classwise, total_correct, total_val_batch


def class_wise_acc_(model, dloader, device):
    model.eval()
    vision_dset = False
    y_test = torch.empty(0).to(device)
    y_pred = torch.empty(0).to(device)
    y_pred_img = torch.empty(0).to(device)
    y_pred_tab = torch.empty(0).to(device)

    with torch.no_grad():
        for i, data in enumerate(dloader, 0):
            image, ids, DOY, x_categ, x_cont, y_gts = data[0].to(device), data[1].to(device).type(torch.float32), data[2].to(
                device).type(torch.float32), data[3].to(device).type(torch.float32), data[4].type(torch.float32).to(
                device),data[5].to(device, dtype=torch.long)#,data[6].to(device).type(torch.float32)

            _, x_categ_enc, x_cont_enc, con_mask = embed_data_mask(x_categ, x_cont, model.tab_net, False, DOY=DOY)

            a, v, out = model(image, x_categ_enc, x_cont_enc, con_mask)
            # import ipdb; ipdb.set_trace()
            y_test = torch.cat([y_test, y_gts], dim=0)
            y_pred = torch.cat([y_pred, torch.argmax(out, dim=1).float()], dim=0)
            y_pred_img = torch.cat([y_pred_img, torch.argmax(a, dim=1).float()], dim=0)
            y_pred_tab = torch.cat([y_pred_tab, torch.argmax(v, dim=1).float()], dim=0)

    acc_classwise, total_correct, total_val_batch = class_wise_acc(y_pred, y_test, num_classes=5)
    acc_classwise_img, total_correct_img, total_val_batch_img = class_wise_acc(y_pred_img, y_test, num_classes=5)
    acc_classwise_tab, total_correct_tab, total_val_batch_tab = class_wise_acc(y_pred_tab, y_test, num_classes=5)


    # Compute the confusion matrix
    y_pred_softmax = torch.log_softmax(y_pred, dim=1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim=1)
    conf_matrix = confusion_matrix(y_test.cpu().numpy(),
                                   y_pred_tags.cpu().numpy())
    return acc_classwise, acc_classwise_tab, acc_classwise_img, conf_matrix


def classification_scores(model, dloader, device, task, vision_dset):
    model.eval()
    m = nn.Softmax(dim=1)
    y_test = torch.empty(0).to(device)
    y_pred = torch.empty(0).to(device)
    prob = torch.empty(0).to(device)
    with torch.no_grad():
        for i, data in enumerate(dloader, 0):
            image, ids, DOY, x_categ, x_cont, y_gts = data[0].to(device), data[1].to(device).type(torch.float32), data[2].to(
                device).type(torch.float32), data[3].to(device).type(torch.float32), data[4].type(torch.float32).to(
                device) ,data[5].to(device, dtype=torch.long)#,data[6].to(device).type(torch.float32)

            _, x_categ_enc, x_cont_enc, con_mask = embed_data_mask(x_categ, x_cont, model.tab_net, False, DOY=DOY)
            a, v, out = model(image, x_categ_enc, x_cont_enc, con_mask)
            # import ipdb; ipdb.set_trace()
            y_test = torch.cat([y_test, y_gts], dim=0)
            y_pred = torch.cat([y_pred, torch.argmax(out, dim=1).float()], dim=0)
            y_pred_img = torch.cat([y_pred_img, torch.argmax(a, dim=1).float()], dim=0)
            y_pred_tab = torch.cat([y_pred_tab, torch.argmax(v, dim=1).float()], dim=0)

            if task == 'binary':
                prob = torch.cat([prob, m(y_outs)[:, -1].float()], dim=0)

    correct_results_sum = (y_pred == y_test).sum().float()
    correct_results_sum_img = (y_pred_img == y_test).sum().float()
    correct_results_sum_tab = (y_pred_tab == y_test).sum().float()

    acc = correct_results_sum / y_test.shape[0] * 100
    acc_img = correct_results_sum_img / y_test.shape[0] * 100
    acc_tab = correct_results_sum_img / y_test.shape[0] * 100

    kappa = sklearn.metrics.cohen_kappa_score(np.array(y_pred.cpu()), np.array(y_test.cpu()))
    auc = 0
    if task == 'binary':
        auc = roc_auc_score(y_score=prob.cpu(), y_true=y_test.cpu())
    return acc.cpu().numpy(), acc_img.cpu().numpy(), acc_tab.cpu().numpy(), kappa


def mean_sq_error(model, dloader, device, vision_dset):
    model.eval()
    y_test = torch.empty(0).to(device)
    y_pred = torch.empty(0).to(device)
    with torch.no_grad():
        for i, data in enumerate(dloader, 0):
            image, ids, DOY, x_categ, x_cont, y_gts = data[0].to(device), data[1].to(device).type(torch.float32), data[2].to(
                device).type(torch.float32), data[3].to(device).type(torch.float32), data[4].type(torch.float32).to(
                device),data[5].to(device, dtype=torch.long)#,data[6].to(device).type(torch.float32)
            _, x_categ_enc, x_cont_enc, con_mask = embed_data_mask(x_categ, x_cont, model, vision_dset, DOY=DOY)
            reps = model.transformer(x_categ_enc, x_cont_enc, con_mask)
            y_reps = reps[:, 0, :]
            y_outs = model.mlpfory(y_reps)
            y_test = torch.cat([y_test, y_gts.squeeze().float()], dim=0)
            y_pred = torch.cat([y_pred, y_outs], dim=0)
        # import ipdb; ipdb.set_trace()
        rmse = mean_squared_error(y_test.cpu(), y_pred.cpu(), squared=False)
        return rmse