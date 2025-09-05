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
    import certifi

    # Force Python/torchvision to use certifi's certificate bundle, this way I can download the resnet weights
    os.environ['SSL_CERT_FILE'] = certifi.where()

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

    # 1) Load the data
    trainloader, validloader, testloader, cat_dims, con_idxs, y_dim, DOY, w0_norm, w1_norm, args = dataloader_init(args)

    # 2) Initialize the model
    model = initialize_model(args, device, cat_dims, con_idxs)

    # 3) Get pretrained resnet18
    weights = models.ResNet18_Weights.DEFAULT
    resnet18_pre = models.resnet18(weights=weights)

    # 4) Load into YOUR model (adjust ".backbone" to your attribute name or remove it if not wrapped)
    load_target = getattr(model, "img_net", model)  # falls back to model if no .backbone
    state = resnet18_pre.state_dict()

    # 5) Actually load and see what happened
    load_info = load_target.load_state_dict(state, strict=False)
    print("Missing keys:", load_info.missing_keys)
    print("Unexpected keys:", load_info.unexpected_keys)

    vision_dset =  False
    print(y_dim)
    print(args.task)

    model.to(device)

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
            {"params": other_params, "lr": 1e-4},
            {"params": img_params, "lr": 1e-6},
        ],
        weight_decay=1e-2  # (set if you want)
    )

    scheduler = None

    print("param_groups:", [len(g["params"]) for g in optimizer.param_groups])
    print("LRs:", [g["lr"] for g in optimizer.param_groups])
    trainable = sum(p.requires_grad for p in model.parameters())
    in_opt = sum(p.numel() for g in optimizer.param_groups for p in g["params"])
    print("trainable tensors:", trainable, "  params in opt groups:", in_opt)

    if args.train:

        best_acc = 0.0

        ratio_a = torch.Tensor([1]).to(device)

        for epoch in range(args.epochs):

            if epoch < 50:

                model.train()

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
                    wandb.log({"acc_tab": accuracy, "accuracy_extra_test": accuracy, 'auroc': auroc, "epoch": epoch})
                #acc, acc_a, acc_v = valid(args, model, device, validloader)


                print('Epoch: {}: '.format(epoch))


            else:

                batch_loss, batch_loss_a, batch_loss_v, a_angle, v_angle, ratio_a = train_epoch(args, epoch, model, device,
                                                                                                trainloader, optimizer,
                                                                                                scheduler, ratio_a)

                acc, acc_a, acc_v = valid(args, model, device, validloader)

                #wandb.log({"Accuracy": acc, "acc_img": acc_a, "acc_tab": acc_v, "epoch": epoch})

                print('epoch: ', epoch, 'loss: ', batch_loss, batch_loss_a, batch_loss_v)
                print('epoch: ', epoch, 'acc: ', acc, 'acc_a: ', acc_a, 'acc_v: ', acc_v)
                print('epoch: ', epoch, 'a_angle: ', a_angle, 'v_angle: ', v_angle)

                if acc > best_acc:
                    if acc > best_acc:
                        best_acc = float(acc)

                    print('Saving model....')
                    torch.save(
                        {
                            'model': model.state_dict(),
                            'optimizer': optimizer.state_dict()
                        },
                        '/theia/scratch/brussel/104/vsc10421/model{}_epoch.pt'.format(args.numClasses))
                    print('Saved model!!!')

