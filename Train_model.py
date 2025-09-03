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

    from utils_.utils import count_parameters, classification_scores, mean_sq_error, class_wise_acc_

    import wandb


    args = get_arguments()

    #setup_seed(args.random_seed)
    device = torch.device('cuda:' + str(args.gpu) if args.use_cuda else 'cpu')
    print("Using device: ", device)

    trainloader, validloader, testloader, cat_dims, con_idxs, y_dim, DOY, w0_norm, w1_norm, args = dataloader_init(args)



    # cat_dims = np.append(np.array([1]),np.array(cat_dims)).astype(int) #Appending 1 for CLS token, this is later used to generate embeddings.
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
    print("Loaded with prefixed keys (img_net.)")
    print("Missing:", missing)
    print("Unexpected:", unexpected[:10])

    model.to(device)

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
    elif args.optimizer == 'AdaGrad':
        optimizer = optim.Adagrad(model.parameters(), lr=args.learning_rate)
        scheduler = None
    elif args.optimizer == 'Adam':
        optimizer = optim.AdamW([
            {"params": parameters_tab},
            {"params": parameters_img},
        ], lr=args.learning_rate,
            betas=(0.9, 0.99),
            weight_decay=1e-4)

        optimizer.param_groups[1]['lr'] = 1e-5
        optimizer.param_groups[0]['lr'] = 1e-4
        scheduler = None

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
    wandb.save('Prototypical_modal_rebalance_and_saint.sh')
    wandb.save('Prototypical_modal_rebalance_and_saint.py')
    wandb.save('multi_modal_model_pytorch.py')
    wandb.save('fusion_modules.py')
    wandb.save('data_prep.py')

    if args.train:

        best_acc = 0.0

        ratio_a = torch.Tensor([1]).to(device)

        # tell wandb what is happening in the model
        wandb.watch(model, log="all", log_freq=10)

        for epoch in range(args.epochs):

            if epoch < 50:
                train_epoch_tab(args, epoch, model, device, trainloader, optimizer, scheduler, ratio_a)
                acc, acc_a, acc_v = valid(args, model, device, validloader)


                print('Epoch: {}: '.format(epoch))
                if epoch % 1 == 0:
                    model.eval()
                    with torch.no_grad():
                        if args.task in ['binary', 'multiclass']:
                            accuracy, auroc, kappa = classification_scores(model, validloader, device, args.task,
                                                                           False)
                            test_accuracy, test_auroc, test_kappa = classification_scores(model, testloader, device,
                                                                                          args.task, False)
                            acc_classwise, conf_matrix = class_wise_acc_(model, validloader, device)
                            print('[EPOCH %d] VALID ACCURACY: %.3f, VALID AUROC: %.3f, VALID KAPPA: %.3f' %
                                  (epoch + 1, accuracy, auroc, kappa))
                            print('[EPOCH %d] TEST ACCURACY: %.3f, TEST AUROC: %.3f, TEST KAPPA: %.3f' %
                                  (epoch + 1, test_accuracy, test_auroc, test_kappa))
                            print(f"class_wise_accuracies: {acc_classwise}")
                            print(f"confusion matrix: {conf_matrix}")
                            wandb.log({"acc_tab": accuracy, "epoch": epoch})

            else:

                batch_loss, batch_loss_a, batch_loss_v, a_angle, v_angle, ratio_a = train_epoch(args, epoch, model, device,
                                                                                                trainloader, optimizer,
                                                                                                scheduler, ratio_a)

                acc, acc_a, acc_v = valid(args, model, device, validloader)

                wandb.log({"Accuracy": acc, "acc_img": acc_a, "acc_tab": acc_v, "epoch": epoch})

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

