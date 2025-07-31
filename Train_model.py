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

from get_arguments import get_arguments

args = get_arguments()
args.use_cuda = torch.cuda.is_available() and not args.no_cuda
vision_dset = args.vision_dset

setup_seed(args.random_seed)

device = torch.device('cuda:' + str(args.gpu) if args.use_cuda else 'cpu')

# initialize the dataloaders
dataset = train_val_test_div_2(args.multiTemp, args.labelHeader)
cat_dims, cat_idxs, con_idxs, X_train, y_train, ids_train, X_valid, y_valid, ids_valid, X_test, y_test, ids_test, train_mean, train_std = data_prep_premade(
    dataset, args.dset_seed, args.task)

continuous_mean_std = np.array([train_mean, train_std]).astype(np.float32)

##### Setting some hyperparams based on inputs and dataset
_, nfeat = X_train['data'].shape
if nfeat > 100:
    args.embedding_size = min(4, args.embedding_size)
    args.batch_size = min(64, args.batch_size)
if args.attentiontype != 'col':
    args.transformer_depth = 1
    args.attention_heads = 4
    args.attention_dropout = 0.8
    args.embedding_size = 16
    if args.optimizer == 'SGD':
        args.ff_dropout = 0.4
        args.lr = 0.01
    else:
        args.ff_dropout = 0.8

# TODO: include oversampling in this version of the script
train_ds = DataSetCatCon(X_train, y_train, ids_train, cat_idxs, args.task, continuous_mean_std)
train_dataloader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)

valid_ds = DataSetCatCon(X_valid, y_valid, ids_valid, cat_idxs, args.task, continuous_mean_std)
valid_dataloader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

test_ds = DataSetCatCon(X_test, y_test, ids_test, cat_idxs, args.task, continuous_mean_std)
test_dataloader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

if args.task == 'regression':
    y_dim = 1
else:
    y_dim = len(np.unique(y_train['data'][:, 0]))
    print(f"y_dim is: {y_dim}")

# cat_dims = np.append(np.array([1]),np.array(cat_dims)).astype(int) #Appending 1 for CLS token, this is later used to generate embeddings.


# Initialize the models
model_tab = SAINT(
    categories=tuple(cat_dims),
    num_continuous=len(con_idxs),
    dim=args.embedding_size,
    dim_out=1,
    depth=args.transformer_depth,
    heads=args.attention_heads,
    attn_dropout=args.attention_dropout,
    ff_dropout=args.ff_dropout,
    mlp_hidden_mults=(4, 2),
    cont_embeddings=args.cont_embeddings,
    attentiontype=args.attentiontype,
    final_mlp_style=args.final_mlp_style,
    y_dim=y_dim
)

model_tab.to(device)
print(f"predset_id: {args.predset_id}")
if args.predset_id is not None:
    # extract the dataset for the pretraining
    print(f"The method of pretraining is {args.pt_tasks}")
    # cat_dims_pre, cat_idxs_pre, con_idxs_pre, X_train_pre, y_train_pre, ids_train_pre, X_valid_pre, y_valid_pre, _, X_test_pre, y_test_pre, _, train_mean_pre, train_std_pre = data_prep(args.predset_id, args.dset_seed ,args.task, datasplit=[1, 0, 0],pretraining=True)
    # continuous_mean_std_pre = np.array([train_mean_pre,train_std_pre]).astype(np.float32)

    if args.pretrain:
        from pretraining import SAINT_pretrain

        model_tab = SAINT_pretrain(model_tab, cat_idxs, X_train, y_train, ids_train, continuous_mean_std, args, device)

model = MM_model(model_tab, BasicBlock, [2, 2, 2, 2], num_classes=args.numClasses, n_dates=args.timeSteps, DEM=args.DEM,
                 final_vector_conv=args.numClasses, final_vector_mlp=args.numClasses, final_vector_dem=args.numClasses,
                 batch_size=16,
                 lr_img=1e-6, lr_tab=0.001, dropout=0.3, regularization=0.000, branch_weight=0.000,
                 fusion=args.fusion_method)
# model.apply(weight_init)
model.load_state_dict(model_zoo.load_url(model_urls['resnet18']), strict=False)

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
    optimizer = optim.Adam([
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
    project="prototypical_modal_rebalance_and_saint",
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

        print('Epoch: {}: '.format(epoch))

        batch_loss, batch_loss_a, batch_loss_v, a_angle, v_angle, ratio_a = train_epoch(args, epoch, model, device,
                                                                                        train_dataloader, optimizer,
                                                                                        scheduler, ratio_a)

        acc, acc_a, acc_v = valid(args, model, device, test_dataloader)

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

if __name__ == "__main__":
    main()