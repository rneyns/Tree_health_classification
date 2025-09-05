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
    from Training import train_epoch, train_epoch_tab
    from Utils import valid
    from Model import initialize_model
    from models import SAINT

    from utils_.utils import count_parameters, classification_scores, mean_sq_error, class_wise_acc_

    import wandb


    args = get_arguments()

    #setup_seed(args.random_seed)
    device = torch.device('cuda:' + str(args.gpu) if args.use_cuda else 'cpu')
    device = torch.device("mps")
    print("Using device: ", device)

    trainloader, validloader, testloader, cat_dims, con_idxs, y_dim, DOY, w0_norm, w1_norm, args = dataloader_init(args)

    # Initialize the models
    model = initialize_model(args, device, cat_dims, con_idxs)

    # Load the pretrained ResNet18 weights
    weights = models.ResNet18_Weights.DEFAULT
    pretrained_resnet = models.resnet18(weights=weights)
    pretrained_dict = pretrained_resnet.state_dict()
    model_dict = model.state_dict()

    from collections import OrderedDict

    prefixed = OrderedDict((f"img_net.{k}", v) for k, v in pretrained_resnet.state_dict().items())
    missing, unexpected = model.load_state_dict(prefixed, strict=False)

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

    optimizer = optim.AdamW(model.parameters(), lr=0.0001)
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

                train_epoch_tab(args, epoch, model.tab_net, device, trainloader, optimizer, scheduler, ratio_a=None)

                model.eval()
                with torch.no_grad():
                    accuracy, auroc, kappa = classification_scores(model.tab_net, validloader, device, args.task,
                                                                   False)
                    test_accuracy, test_auroc, test_kappa = classification_scores(model.tab_net, testloader, device,
                                                                                  args.task, False)
                    acc_classwise, conf_matrix = class_wise_acc_(model.tab_net, validloader, device)
                    print('[EPOCH %d] VALID ACCURACY: %.3f, VALID AUROC: %.3f, VALID KAPPA: %.3f' %
                          (epoch + 1, accuracy, auroc, kappa))
                    print('[EPOCH %d] TEST ACCURACY: %.3f, TEST AUROC: %.3f, TEST KAPPA: %.3f' %
                          (epoch + 1, test_accuracy, test_auroc, test_kappa))
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

