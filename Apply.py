# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 09:23:31 2023

@author: Robbe Neyns
"""

if __name__ == "__main__":
    import os
    import certifi

    # Force Python/torchvision to use certifi's certificate bundle, this way I can download the resnet weights
    os.environ['SSL_CERT_FILE'] = certifi.where()


    import torch
    from torch.utils import model_zoo
    import torchvision.models as models
    import torch.optim as optim

    from get_arguments import get_arguments_apply
    from Dataloader_init import dataloader_init_apply
    from Training import make_predictions
    from Utils import valid
    from Model import initialize_model
    from models import SAINT


    args = get_arguments_apply()

    #setup_seed(args.random_seed)

    if torch.cuda.is_available():
        device = torch.device("cuda")  # default: cuda:0
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")  # Apple Silicon (Metal Performance Shaders)
        print("Using MPS device")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    # 1) Load the data
    applyloader, cat_dims, con_idxs, y_dim, DOY, w0_norm, w1_norm, args = dataloader_init_apply(args)

    # 2) Initialize the model
    model = initialize_model(args, device, cat_dims, con_idxs)

    # 3) Get pretrained weights
    weights = torch.load(args.weights)['model']

    # 4) Load into YOUR model
    load_info = model.load_state_dict(weights)
    print("Missing:", load_info.missing_keys[:10])
    print("Unexpected:", load_info.unexpected_keys[:10])

    vision_dset =  False
    print(y_dim)
    print(args.task)

    model.to(device)

    print(type(model))
    print(hasattr(model, "module"), hasattr(model, "img_net"))

    w_pre = weights["layer1.0.conv1.weight"].flatten()[:5]
    w_now = model.img_net.state_dict()["layer1.0.conv1.weight"].flatten()[:5]
    print("Pretrained sample:", w_pre)
    print("Model sample:", w_now)

    #5) make predictions on the datain the applyloader
    make_predictions(model, applyloader, device)


