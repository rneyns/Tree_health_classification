"""
get_arguments.py

This module contains the functions related to getting the arguments from the user.

Functions:
    - get_arguments: argument parser requesting the necessary hyperparameters from the user

Author: Robbe Neyns
Created: Thu Jul 31 10:38:11 2023
"""

import argparse

def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("-i", "--images", metavar="IMAGES", help="Path to the folder with the orthophoto patches")
    parser.add_argument("-multi", "--multiTemp", metavar="MULTI_TEMPORAL",
                        help="Path to the folder with the multi_temporal_data for each band")
    parser.add_argument("-lh", "--labelHeader", metavar="LABEL_HEADER",
                        help="Name of the column that contains the class label (binary)")
    parser.add_argument("-idh", "--IDHeader", metavar="ID_HEADER",
                        help="Name of the column that contains the associated image ID")
    parser.add_argument("-w", "--imageWidth", default=241, metavar="IMAGE_WIDTH",
                        help="Width of the image patches (assumed to be the same as height)")
    parser.add_argument("-ts", "--timeSteps", default=12, metavar="TIME_STEPS", help="Length of the time-series",
                        type=int)
    parser.add_argument("-nc", "--numClasses", metavar="NUMBER_CLASSES",
                        help="Number of classes that will be predicted by the model", type=int)
    parser.add_argument('--fixed_train_test', action='store_true')
    parser.add_argument('--undersample', action='store_true')
    parser.add_argument('--spatio_temp', action='store_true')
    parser.add_argument('--transfer_learning', action='store_true')
    parser.add_argument('--apply_version', action='store_false')

    parser.add_argument("-ResNetV", "--ResNetV", metavar="VERSION_OF_RESNET",
                        help="It is possible to train the network with ResNet18 (0), ResNet34 (1), ResNet50 (2), ResNet101 (3) and ResNet152 (4). To choose a model, please specify the associated number",
                        type=int, default=1)
    parser.add_argument('--modulation', default='OGM_GE', type=str,
                        choices=['Normal', 'OGM', 'OGM_GE', 'Acc'])
    parser.add_argument('--fusion_method', default='concat', type=str,
                        choices=['sum', 'concat', 'gated', 'film'])
    parser.add_argument('--fps', default=1, type=int, help='Extract how many frames in a second')
    parser.add_argument('--num_frame', default=3, type=int, help='use how many frames for train')

    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--epochs', default=150, type=int)
    parser.add_argument('--embed_dim', default=512, type=int)
    parser.add_argument('--optimizer', default='SGD', type=str)

    parser.add_argument('--learning_rate', default=0.00001, type=float, help='initial learning rate')
    parser.add_argument('--lr_decay_step', default=70, type=int, help='where learning rate decays')
    parser.add_argument('--lr_decay_ratio', default=0.1, type=float, help='decay coefficient')

    parser.add_argument('--modulation_starts', default=0, type=int, help='where modulation begins')
    parser.add_argument('--modulation_ends', default=50, type=int, help='where modulation ends')
    parser.add_argument('--alpha', required=True, type=float, help='alpha in OGM-GE')

    parser.add_argument('--ckpt_path', default='ckpt', type=str, help='path to save trained models')
    parser.add_argument('--train', default=True, help='turn on train mode')


    parser.add_argument('--logs_path', default='logs', type=str, help='path to save tensorboard logs')

    parser.add_argument('--random_seed', default=0, type=int)

    parser.add_argument('--gpu', type=int, default=0)  # gpu
    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA')

    parser.add_argument('--task', required=True, default='clf', type=str,
                        choices=['binary', 'multiclass', 'regression', 'clf'])
    parser.add_argument('--cont_embeddings', default='spatio-temporal', type=str, choices=['MLP', 'spatio-temporal'])
    parser.add_argument('--embedding_size', default=32, type=int)
    parser.add_argument('--transformer_depth', default=6, type=int)
    parser.add_argument('--attention_heads', default=8, type=int)
    parser.add_argument('--attention_dropout', default=0.1, type=float)
    parser.add_argument('--ff_dropout', default=0.1, type=float)
    parser.add_argument('--attentiontype', default='colrow', type=str,
                        choices=['col', 'colrow', 'row', 'justmlp', 'attn', 'attnmlp'])

    # parser.add_argument('--optimizer', default='AdamW', type=str,choices = ['AdamW','Adam','SGD'])
    parser.add_argument('--scheduler', default='cosine', type=str, choices=['cosine', 'linear'])

    parser.add_argument('--lr', default=0.0001, type=float)

    parser.add_argument('--savemodelroot', default='./bestmodels', type=str)
    parser.add_argument('--run_name', default='testrun', type=str)
    parser.add_argument('--set_seed', default=1, type=int)
    parser.add_argument('--dset_seed', default=1, type=int)
    parser.add_argument('--active_log', action='store_true')

    parser.add_argument('--pretrain', default=False)
    parser.add_argument('--pretrain_epochs', default=50, type=int)
    parser.add_argument('--pt_tasks', default=['contrastive', 'denoising'], type=str, nargs='*',
                        choices=['contrastive', 'contrastive_sim', 'denoising'])
    parser.add_argument('--pt_aug', default=[], type=str, nargs='*', choices=['mixup', 'cutmix'])
    parser.add_argument('--pt_aug_lam', default=0.1, type=float)
    parser.add_argument('--mixup_lam', default=0.3, type=float)

    parser.add_argument('--train_noise_type', default=None, type=str, choices=['missing', 'cutmix'])
    parser.add_argument('--train_noise_level', default=0, type=float)

    parser.add_argument('--ssl_samples', default=None, type=int)
    parser.add_argument('--pt_projhead_style', default='diff', type=str, choices=['diff', 'same', 'nohead'])
    parser.add_argument('--nce_temp', default=0.7, type=float)

    parser.add_argument('--lam0', default=0.5, type=float)
    parser.add_argument('--lam1', default=10, type=float)
    parser.add_argument('--lam2', default=1, type=float)
    parser.add_argument('--lam3', default=10, type=float)
    parser.add_argument('--final_mlp_style', default='sep', type=str, choices=['common', 'sep'])

    args = parser.parse_args()
    #
    args.use_cuda = torch.cuda.is_available() and not args.no_cuda

    return args