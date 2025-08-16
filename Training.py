"""
training.py

This module contains functions related to the training and validation of the model.
It handles the training loop, loss computation, gradient calculation, and evaluation.

Functions:
    - train_epoch: Performs one training epoch, including forward pass, loss calculation,
      backward pass, and optimization step.
    - valid: Evaluates the model on a validation or test set and returns metrics like accuracy and loss.

Author: Robbe Neyns
Created: Tue Sep 19 15:07:11 2023
"""


# training.py - for training and evaluation logic
from utils_.utils import setup_seed
from augmentations import embed_data_mask
from augmentations import add_noise
import torch
import torch.nn as nn

def prepare_data_embedding(data, args, model, device, cat_mask=None, con_mask=None):
    # x_categ is the the categorical data, with y appended as last feature. x_cont has continuous data. cat_mask is an array of ones same shape as x_categ except for last column(corresponding to y's) set to 0s. con_mask is an array of ones same shape as x_cont.
    image, ids, DOY, x_categ, x_cont, y_gts = data[0].to(device), data[1].to(device), data[2].to(
        device).type(torch.float32), data[3].to(device).type(torch.float32), data[4].type(torch.float32).to(
        device),data[5].to(device).type(torch.float32)#,data[6].to(device).type(torch.float32)
    if args.train_noise_type is not None and args.train_noise_level > 0:
        noise_dict = {
            'noise_type': args.train_noise_type,
            'lambda': args.train_noise_level
        }
        if args.train_noise_type == 'cutmix':
            x_categ, x_cont = add_noise(x_categ, x_cont, noise_params=noise_dict)
        elif args.train_noise_type == 'missing':
            cat_mask, con_mask = add_noise(cat_mask, con_mask, noise_params=noise_dict)
    # We are converting the data to embeddings in the next step
    _, x_categ_enc, x_cont_enc, con_mask = embed_data_mask(x_categ, x_cont, model, False, DOY)
    return image, x_categ_enc, x_cont_enc, con_mask


def train_epoch(args, epoch, model, device, dataloader, optimizer, scheduler, ratio_a, writer=None):
    criterion = nn.CrossEntropyLoss()
    softmax = nn.Softmax(dim=1)

    model.train()
    print("Start training ... ")

    _loss = 0
    _loss_a = 0
    _loss_v = 0
    _a_angle = 0
    _v_angle = 0
    _ratio_a = 0

    for step, data in enumerate(dataloader):
        # Forward pass
        optimizer.zero_grad()
        image, x_categ_enc, x_cont_enc, con_mask = prepare_data_embedding(data, args, model, device)

        a, v, out = model(image, x_categ_enc, x_cont_enc, con_mask)

        # Loss calculation
        loss = criterion(out, label)
        loss_v = criterion(v, label)
        loss_a = criterion(a, label)

        loss.backward()  # Backward pass
        optimizer.step()

        # Metrics tracking
        _loss += loss.item()
        _loss_a += loss_a.item()
        _loss_v += loss_v.item()

        if step % 10 == 0:
            wandb.log({"loss": loss, "loss_img": loss_a, "loss_tab": loss_v})

    if args.optimizer == 'SGD':
        scheduler.step()

    return _loss / len(dataloader), _loss_a / len(dataloader), _loss_v / len(dataloader)
