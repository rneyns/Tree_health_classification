"""
model.py

This module contains functions for initializing and configuring the model.
It includes the model setup for tabular and visual networks, as well as
the optimizer and scheduler configuration.

Functions:
    - initialize_model: Initializes and returns the multimodal model based on the provided configuration.
    - initialize_optimizer_and_scheduler: Sets up the optimizer and learning rate scheduler based on the chosen optimizer.

Author: Robbe Neyns
Created: Tue Sep 19 15:07:11 2023
"""


# model.py - for model and related utilities
from models import SAINT, MM_model, BasicBlock
import torch.optim as optim
import numpy as np

def initialize_model(args, device, cat_dims, con_idxs):
    print("---------length of the continous index array is: ", len(con_idxs), "---------")
    model_tab = SAINT(
    categories = tuple(cat_dims),
    num_continuous = args.timeSteps*4,
    dim = args.embedding_size,
    dim_out = 1,
    depth = args.transformer_depth,
    heads = args.attention_heads,
    attn_dropout = args.attention_dropout,
    ff_dropout = args.ff_dropout,
    mlp_hidden_mults = (4, 2),
    cont_embeddings = args.cont_embeddings,
    attentiontype = args.attentiontype,
    final_mlp_style = args.final_mlp_style,
    y_dim = args.numClasses
    )
    model_tab.to(device)
    model = MM_model(model_tab, BasicBlock, [2, 2, 2, 2], num_classes=args.numClasses, n_dates=args.timeSteps,
                     DEM=False, final_vector_conv=args.numClasses, final_vector_mlp=args.numClasses,
                     batch_size=16, lr_img=1e-6, lr_tab=0.001, dropout=0.3, regularization=0.000)
    return model

def initialize_optimizer_and_scheduler(args, model):
    parameters_tab = []
    parameters_img = []
    for name, param in model.named_parameters():
        if 'img' in name or 'dem' in name:
            parameters_img.append(param)
        else:
            parameters_tab.append(param)

    if args.optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.StepLR(optimizer, args.lr_decay_step, args.lr_decay_ratio)
    elif args.optimizer == 'Adam':
        optimizer = optim.Adam([
            {"params": parameters_tab},
            {"params": parameters_img},
        ], lr=args.learning_rate, betas=(0.9, 0.99), weight_decay=1e-4)
        scheduler = None
    return optimizer, scheduler
