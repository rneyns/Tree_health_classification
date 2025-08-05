"""
Dataloader_init.py

This module contains the functions related to getting the arguments from the user.

Functions:
    - create_DOY: Create the day of the year vector to indicate this as extra info to the embeddings module
    - merge_bands: The reflectances are stored in separate files for each band, so they need to be merged before they can be used in the embeddings module
    - dataloader_init: The initialization of the dataloader
    - prepare_predictloader: A separate function to initialize the dataloader for prediction
    -

Author: Robbe Neyns
Created: Thu Jul 31 10:38:11 2023
"""
import pandas as pd
import os
import numpy as np
from Utils import data_prep, DataSetCatCon, data_prep_premade
from Utils import train_val_test_div_2
from Utils import resample
from torch.utils.data import DataLoader
import torch


def create_DOY(dataset):
    DOY = torch.from_numpy(np.array(
        [[37], [45], [59], [71], [76], [78], [80], [82], [95], [99], [107], [108], [119], [123], [124], [128], [132],
         [153], [160], [163], [165], [168], [187], [188], [195], [208], [223], [235], [243], [246], [248], [249], [257],
         [270], [276], [281], [284], [297], [302], [322]]))

    DOY = DOY.repeat(len(dataset), 1, 1)

    return DOY


def merge_bands(args):
    #The reflection of each band is in a separate csv file. This function merges them into one dataframe in a coherent way
    dfs = []
    for df in os.listdir(args.dset_id):
        dfs.append(pd.read_csv(args.dset_id + "/" + df))
    # Initialize a new DataFrame with the same structure
    df1 = dfs[0]
    dataset = pd.DataFrame(index=df1.index, columns=df1.columns)

    # Populate the new DataFrame with lists from corresponding cells of all DataFrames
    for col in dataset.columns:
        for idx in dataset.index:
            dataset.at[idx, col] = [df.at[idx, col] for df in dfs]  # Create a list of values from all dataframes

    # make sure that the id and label column is not a list
    dataset["essence_cat"] = df1["essence_cat"]
    dataset["id"] = df1["id"]

    # Add this dataframe to the final list
    print(f"dataset has shape: {dataset.shape}")
    return dataset

def dataloader_init(args):
    dataset = merge_bands(args)

    print(f'The fixed train test parameter is {args.fixed_train_test}')

    # Divide in training and validation
    ###################################
    if not opt.fixed_train_test:
        print("-----Dividing the dataset into a training and test set ------")
        dataset = train_val_test_div_2(dataset, "essence_cat")
        print("-----Saving the dataset with the added training-validation division")
        dataset.to_csv("basis" + args.output_name, index=False)

    # Over or undersample
    ###################################
    if args.undersample:
        print("-----Under/oversampling the dataset-----")
        dataset = resample(dataset, sampling="over", num_classes=2, NearMissV=3, seed=2)
        print("-----Saving the undersampled dataset-----")
        dataset.to_csv("post_undersample_check.csv", index=False)
        print("----Number of samples after under/oversamling-----")
        ##### Count the number of rows with value 1
        num_rows_with_1 = (dataset['essence_cat'] == 1).sum()
        ##### Count the number of rows with value 0
        num_rows_with_0 = (dataset['essence_cat'] == 0).sum()

        print("Number of rows with value 1:", num_rows_with_1)
        print("Number of rows with value 0:", num_rows_with_0)

    # Print some information about the dataset to check if everything is ok
    print(f"dataset shape: {dataset.shape}")
    num_continuous = (dataset.shape[1] - 3) * 4
    print(f"number of continous variables: {num_continuous}")

    ##### Count the number of rows with value 1
    num_rows_with_1 = (dataset['essence_cat'] == 1).sum()

    ##### Count the number of rows with value 0
    num_rows_with_0 = (dataset['essence_cat'] == 0).sum()

    print("Number of rows with label 1:", num_rows_with_1)
    print("Number of rows with label 0:", num_rows_with_0)

    #### Defining the weights simply as the inverse frequency of each class and rescaling to the number of classes
    w0 = 1 / (num_rows_with_0 / (num_rows_with_0 + num_rows_with_1))
    w1 = 1 / (num_rows_with_1 / (num_rows_with_0 + num_rows_with_1))
    w0_norm = (w0 / (w0 + w1)) * 2
    w1_norm = (w1 / (w0 + w1)) * 2

    print('---- Initializing the dataloaders ----')

    cat_dims, cat_idxs, con_idxs, X_train, y_train, ids_train, X_valid, y_valid, ids_valid, X_test, y_test, ids_test, train_mean, train_std, DOY_train, DOY_valid = data_prep_premade(
        ds_id=dataset, DOY=DOY, seed=opt.dset_seed, task=opt.task)
    continuous_mean_std = np.array([train_mean, train_std]).astype(np.float32)

    ##### Setting some hyperparams based on inputs and dataset
    _, nfeat, nbands = X_train['data'].shape
    print(f"Number of dates: {nfeat}; and number of bands: {nbands}")

    if nfeat > 100:
        args.embedding_size = min(4, opt.embedding_size)
        # The batch size needs to be at least  to make optimal use of the intersample attention
        args.batchsize = min(64, opt.batchsize)
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

    train_ds = DataSetCatCon(X_train, y_train, DOY_train, ids_train, cat_idxs, args.dtask)#, continuous_mean_std)
    trainloader = DataLoader(train_ds, batch_size=args.batchsize, shuffle=True, num_workers=1)

    valid_ds = DataSetCatCon(X_valid, y_valid, DOY_valid, ids_valid, cat_idxs, args.dtask)#, continuous_mean_std)
    validloader = DataLoader(valid_ds, batch_size=args.batchsize, shuffle=False, num_workers=1)

    test_ds = DataSetCatCon(X_test, y_test, DOY_valid, ids_test, cat_idxs, args.dtask)#, continuous_mean_std)
    testloader = DataLoader(test_ds, batch_size=args.batchsize, shuffle=False, num_workers=1)



    y_dim = len(np.unique(y_train['data'][:, 0]))
    print(y_dim)

    cat_dims = np.append(np.array([1]), np.array(cat_dims)).astype(int)  # Appending 1 for CLS token, this is later used to generate embeddings.

    return trainloader, validloader, testloader, cat_dims, con_idxs, y_dim, DOY, w0_norm, w1_norm, args

def prepare_predictloader(args):
    ### configuring the dataloader for the predict step (needs to happen before the undersampling)
    dataset["Train_test"] = 0
    cat_dims_pre, cat_idxs_pre, con_idxs_pre, X_train_pre, y_train_pre, ids_train_pre, X_valid_pre, y_valid_pre, ids_valid_pre, X_test_pre, y_test_pre, ids_test_pre, train_mean_pre, train_std_pre, DOY_train_pre, DOY_valid_pre = data_prep_premade(
        ds_id=dataset, DOY=DOY_pre, seed=opt.dset_seed, task=opt.task)

    # Create the predictloader
    # continuous_mean_std = np.array([train_mean,train_std]).astype(np.float32)
    ds = DataSetCatCon(X_train_pre, y_train_pre, DOY_train_pre, ids_train_pre, cat_idxs_pre,
                       args.dtask)  # , continuous_mean_std=continuous_mean_std)
    predictloader = DataLoader(ds, batch_size=args.batchsize, shuffle=False, num_workers=1)