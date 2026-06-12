"""
Train_model.py

This script is the entry point for the multimodal model training pipeline.
It orchestrates the entire model setup, training, and evaluation process.

Key Steps:
    1. Argument parsing and setup.
    2. Model and optimizer initialization.
    3. Training loop through multiple epochs.
    4. Evaluation and logging of results.
    5. Saving of best models based on performance metrics.

Author: Robbe Neyns
Created: Tue Sep 19 15:07:11 2023
"""

if __name__ == "__main__":

    import os

    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"



    import torch
    from torch.utils import model_zoo
    import torchvision.models as models
    import torch.optim as optim

    from get_arguments import get_arguments
    from Dataloader_init import dataloader_init
    from Training import train_epoch, train_epoch_tab, train_epoch_img
    from Utils import valid
    from Model import initialize_model
    from models import SAINT

    from utils_.utils import count_parameters, classification_scores, mean_sq_error, class_wise_acc_

    import wandb

    print(f"The benchmark setting is on: {torch.backends.cudnn.benchmark}")
    torch.backends.cudnn.benchmark = False
    print(f"The benchmark setting after change is on: {torch.backends.cudnn.benchmark}")
    args = get_arguments()

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

    # usage
    print("Selected device:", device)

    # Set environment variables to avoid the CUDNN internal error
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False

    # 1) Load the data
    trainloader, validloader, testloader, cat_dims, con_idxs, y_dim, DOY, w0_norm, w1_norm, args = dataloader_init(args)

    # 2) Initialize the model
    model = initialize_model(args, device, cat_dims, con_idxs)

    imagenet_weights = False
    # 3) Get pretrained resnet18
    if imagenet_weights == True:
        weights = models.ResNet18_Weights.DEFAULT
        resnet18_pre = models.resnet18(weights=weights)

        # 4) Load into YOUR model (adjust ".backbone" to your attribute name or remove it if not wrapped)
        state = resnet18_pre.state_dict()
        core = getattr(model, "module", model)  # unwrap DDP/DataParallel if present
        load_target = getattr(core, "img_net", core)  # your attribute
        # If load_target is still a wrapper holding the real resnet in .backbone:
        if hasattr(load_target, "backbone"):
            load_target = load_target.backbone  # <-- descend to the real resnet
        load_info = load_target.load_state_dict(state, strict=False)
        print("Missing:", load_info.missing_keys[:10])
        print("Unexpected:", load_info.unexpected_keys[:10])
        w_pre = state["layer1.0.conv1.weight"].flatten()[:5]
        w_now = model.img_net.state_dict()["layer1.0.conv1.weight"].flatten()[:5]
        print("Pretrained sample:", w_pre)
        print("Model sample:", w_now)
    else:
        #print("train from scratch")
        #weights = torch.load('/Users/robbe_neyns/Documents/Work_local/research/UHI tree health/Data analysis/model_weights_ResNET18_v2.pth')
        #weights = torch.load('/Users/robbe_neyns/Documents/Work_local/research/UHI tree health/Data analysis/model6_epoch.pt')
        weights = torch.load('model_weights_ResNET18_v2.pth')
        load_info = model.img_net.load_state_dict(weights, strict=False)

        print("Missing:", load_info.missing_keys)
        print("Unexpected:", load_info.unexpected_keys)
    vision_dset =  False
    print(y_dim)
    print(args.task)

    model.to(device)

    print(type(model))
    print(hasattr(model, "module"), hasattr(model, "img_net"))

    wandb.login(key='f746dc65fb72b570908ed1dd5b2b780d7e438243')
    wandb.init(
        # set the wandb project where this run will be logged --> maybe change this to a with statement so it gets closed at the end
        project="UHI tree health",
        name=str(args.numClasses),
        dir="/theia/scratch/brussel/104/vsc10421",
        # track hyperparameters and run metadata
        config={
            "final_vector_branch": "num_classes",
            "learning_rate": 1e-5,
            "epochs_max": 500,
            "dropout": 0,
            "weight_loss_branch": 0.4,
            "regularization:": 1e-4,
            "batch_size": args.batch_size,
            "DEM": "False"
        }
    )

    # Separate image branch and rest of model
    img_params = list(model.img_net.parameters())  # adjust if your attribute is different
    other_params = [p for n, p in model.named_parameters()
                    if not n.startswith("img_net.")]

    optimizer = optim.AdamW(
        [
            {"params": other_params, "lr": 1e-3},
            {"params": img_params, "lr": 1e-6},
        ],
        weight_decay=1e-2  # (set if you want)
    )

    optimizer = optim.AdamW(model.parameters(),lr=1e-4, betas=(0.9, 0.99), weight_decay=1e-4)
    scheduler = None

    print("param_groups:", [len(g["params"]) for g in optimizer.param_groups])
    print("LRs:", [g["lr"] for g in optimizer.param_groups])
    trainable = sum(p.requires_grad for p in model.parameters())
    in_opt = sum(p.numel() for g in optimizer.param_groups for p in g["params"])
    print("trainable tensors:", trainable, "  params in opt groups:", in_opt)

    #Freeze the backbone weights to perform sanity check
    # 1) Freeze backbone
    for n, p in model.img_net.named_parameters():
        if n.startswith(("conv0", "fc")):
            p.requires_grad = True
        else:
            p.requires_grad = False

    # Check
    print("Any trainable params:", any(p.requires_grad for p in model.img_net.parameters()))
    print(f"The batch size is: {args.batch_size}")

    if args.train:

        best_acc = 0.0

        ratio_a = torch.Tensor([1]).to(device)

        for epoch in range(args.epochs):


            model.train()

            if epoch == 2:
                # 3) Unfreeze backbone after 4 epochs
                for p in model.parameters():
                    p.requires_grad = True
                print("Backbone unfrozen!")

            train_epoch(args, epoch, model, device, trainloader, optimizer, scheduler, ratio_a=None)

            model.eval()
            with torch.no_grad():
                accuracy, accuracy_img,  accuracy_tab, kappa = classification_scores(model, validloader, device, args.task,
                                                               False)
                test_accuracy, test_accuracy_img, test_accuracy_tab, test_kappa = classification_scores(model, testloader, device,
                                                                              args.task, False)
                acc_classwise, acc_classwise_tab, acc_classwise_img, conf_matrix = class_wise_acc_(model, validloader, device)
                print('[EPOCH %d] VALID ACCURACY: %.3f, VALID IMG: %.3f, VALID TAB: %.3f, VALID KAPPA: %.3f' %
                      (epoch + 1, accuracy, accuracy_img, accuracy_tab, kappa))
                print('[EPOCH %d] TEST ACCURACY: %.3f, TEST IMG: %.3f, TEST TAB: %.3f, TEST KAPPA: %.3f' %
                      (epoch + 1, test_accuracy, test_accuracy_img, test_accuracy_tab, test_kappa))
                print(f"class_wise_accuracies: {acc_classwise}")
                print(f"confusion matrix: {conf_matrix}")
                wandb.log({"accuracy_img": accuracy_img, "accuracy_tab": accuracy_tab, "epoch": epoch})
            #acc, acc_a, acc_v = valid(args, model, device, validloader)


            print('Epoch: {}: '.format(epoch))




            if accuracy > best_acc:
                best_acc = float(accuracy)

                print('Saving model....')
                torch.save(
                    {
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict()
                    },
                    'model_imgnet{}_epoch.pt'.format(args.numClasses))
                print('Saved model!!!')

