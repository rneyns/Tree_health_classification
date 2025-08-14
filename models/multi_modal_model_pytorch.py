# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 15:53:06 2023

@author: Robbe Neyns
"""

import math
import torch
import torch.nn as nn

import torchmetrics
from Utils.fusion_modules import SumFusion, ConcatFusion, FiLM, GatedFusion


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class depth_sep_conv(nn.Module):
    def __init__(self,input_size,output_size):
        super(depth_sep_conv,self).__init()
        self.depth_conv = nn.Conv1d(in_channels=input_size, out_channels=input_size, kernel_size=1, groups=input_size)
        self.point_conv = nn.Conv1d(in_channels=input_size, out_channels=output_size, kernel_size=1)
        
    def forward(self,x):
        x = self.depth_conv(x)
        x = self.point_conv(x)
        
        return x
    

class imgClassifier(nn.Module):
    def __init__(self, block, layers, num_classes):
        self.inplanes = 64
        super(imgClassifier,self).__init__()
        
        self.conv0 = nn.Conv2d(4,3,kernel_size=1,stride=1)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7)
        self.fc_pers = nn.Linear(16384, num_classes)

    
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
    
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
    
        return nn.Sequential(*layers)
        
        
    def forward(self, img):
        #For the image data
        img = self.conv0(img)
        img = self.conv1(img)
        img = self.bn1(img)
        img = self.relu(img)

        img = self.layer1(img)
        img = self.layer2(img)
        img = self.layer3(img)
        #img = self.layer4(img)

        img = self.avgpool(img)
        img = img.view(img.size(0), -1)
        img = self.fc_pers(img)
    
        return img
        

class MM_model(nn.Module):
    def __init__(self, tab_net, block, layers, DEM = True, num_classes = 5, final_vector_conv = 5, final_vector_mlp = 5, final_vector_dem = 5, n_dates = 32,
                 mlp_simple = False, lr_img: float = 1e-6, lr_tab: float = 1e-4 , num_workers: int = 4, batch_size: int = 50,dropout = 0.35,regularization=0.01, branch_weight = 0.000, fusion = 'concat'):
        self.inplanes = 64
        self.lr_img = lr_img
        self.lr_tab = lr_tab
        self.dropout = dropout
        self.regularization = regularization
        self.branch_weight = branch_weight
        self.mlp_simple=mlp_simple
        self.dem = DEM
        self.num_workers = num_workers
        self.num_classes=num_classes
        self.batch_size = batch_size
        super(MM_model, self).__init__()
        self.outputs_val=[]
        self.outputs_test=[]
        self.cohenkappa = torchmetrics.classification.CohenKappa(num_classes=int(num_classes))
        self.F1 = torchmetrics.classification.F1Score(num_classes = int(num_classes))
        self.val_outputs = torch.empty(0).to("cuda:0")
        self.val_labels = torch.empty(0).to("cuda:0")
        self.class_weights = torch.Tensor([1.0, 20.0]).double()
        self.criterion = torch.nn.CrossEntropyLoss()

        self.img_net = imgClassifier(block, layers, num_classes)
        self.tab_net = tab_net
                                       
         
        
        #self.ln9_tab = nn.Linear(10,num_classes)
            
        if fusion == 'sum':
            self.fusion_module = SumFusion(input_dim=num_classes, output_dim=num_classes)
        elif fusion == 'concat':
            self.fusion_module = ConcatFusion(input_dim=num_classes*2, output_dim=num_classes)
        elif fusion == 'film':
            self.fusion_module = FiLM(input_dim=num_classes, output_dim=num_classes, x_film=True)
        elif fusion == 'gated':
            self.fusion_module = GatedFusion(input_dim=num_classes, output_dim=num_classes, x_gate=True)
        else:
            raise NotImplementedError('Incorrect fusion method: {}!'.format(fusion))
 
            
     
    def forward(self, img,  x_categ_enc, x_cont_enc, con_mask):
        img = self.img_net(img)
        reps = self.tab_net.transformer(x_categ_enc, x_cont_enc, con_mask)
        y_reps = reps[:,0,:]
        tab = self.tab_net.mlpfory(y_reps)
        
        a, v, out = self.fusion_module(img,tab)

        return  a, v, out 
    
 
