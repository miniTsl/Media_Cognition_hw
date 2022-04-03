#========================================================
#             Media and Cognition
#             Homework 3 Convolutional Neural Network
#             model.py -  Convolutional Neural Networks
#             Student ID:
#             Name:
#             Tsinghua University
#             (C) Copyright 2022s
#========================================================
import torch
import torch.nn as nn

# Import self-implemented Conv2d layer 
from conv import Conv2d
from pool import MaxPool2d
from batchnorm import BatchNorm2d, BatchNorm1d

class CNN(nn.Module):
    def __init__(self, in_channels, num_class, batch_norm=False, p=0.0):
        '''
        Convolutional Neural Networks
        ----------------------
        :param in_channels: channel number of input image
        :param num_class: number of classes for the classification task
        :param batch_norm: whether to use batch normalization in convolutional layers and linear layers
        :param p: dropout ratio of dropout layer which ranges from 0 to 1
        '''
        super(CNN, self).__init__()

        if batch_norm:
            bn1d = BatchNorm1d
            bn2d = BatchNorm2d
        else:
            bn1d = nn.Identity
            bn2d = nn.Identity

        # TODO 2.1: complete a multilayer convolutional neural network with nn.Sequential function.
        # For convolution layers, please use Conv2d you have completed in conv.py
        # input image with size [batch_size, in_channels, img_h, img_w]
        # Network structure:
        #        kernel_size  stride  padding  out_channels
        # conv       5          1        2          32
        # batchnorm
        # relu
        self.conv1 = nn.Sequential(Conv2d(in_channels, 32, 5, 1, 2), bn2d(32), nn.ReLU())
        # conv       5          1        2          64
        # batchnorm
        # relu
        self.conv2 = nn.Sequential(Conv2d(32, 64, 5, 1, 2), bn2d(64), nn.ReLU())
        # maxpool    2          2        0   
        self.pool1 = MaxPool2d(2, 2, 0)
        # conv       3          1        1          64
        # batchnorm
        # relu
        self.conv3 = nn.Sequential(Conv2d(64, 64, 3, 1, 1), bn2d(64), nn.ReLU())
        # conv       3          1        1          128
        # batchnorm
        # relu
        self.conv4 = nn.Sequential(Conv2d(64, 128, 3, 1, 1), bn2d(128), nn.ReLU())
        # maxpool    2          2        0
        self.pool2 = MaxPool2d(2, 2, 0)
        # conv       3          1        1          128
        # batchnorm
        # relu
        # dropout(p), where p is input parameter of dropout ratio
        self.conv5 = nn.Sequential(Conv2d(128, 128, 3, 1, 1), bn2d(128), nn.ReLU(), nn.Dropout(p))
        # end TODO 2.1

        # TODO 2.2: complete a sub-network with two linear layers by using nn.Sequential function
        # Hint: note that the size of input images is (1, 32, 32) by default, what is the size of output of convolution layers?
        # Network structure:
        # linear       (in_features, out_features=256)
        # batchnorm 1D (out_features=256)
        # activation, i.e. nn.ReLu()
        # dropout(p), where p is input parameter of dropout ratio
        # linear    (in_features_output_layer, num_class)
        # TODO 2.1 中各层输出图像的[h,w] 分别为[32,32],[32,32],[16,16],[16,16], [16,16], [8,8], [8,8]
        # 当然总的形状是[batch_size, out_channels = 128, 8, 8]
        self.fc_net = nn.Sequential(
            nn.Linear(128*8*8, 256),
            bn1d(256),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(256, num_class)
        )
        # end TODO 2.2

        self.init_weights()

    def init_weights(self):
        '''
        Initialize weights
        '''
        for l in self.modules():
            if isinstance(l, Conv2d):
                # kaiming_uniform.
                # The initialized values of weights W are sampled from the uniform distribution U(-1/k, 1/k),  where k = (in_channels * kernel_h * kernel_w) ** (1/2)
                nn.init.kaiming_uniform_(l.W.data, a=5**(1/2))
                if l.b is not None:
                    nn.init.constant_(l.b.data, 0)

            if isinstance(l, nn.Linear):
                nn.init.kaiming_uniform_(l.weight.data, a=5**(1/2))
                if l.bias is not None:
                    nn.init.constant_(l.bias.data, 0)


    def forward(self, x, return_features=False):
        '''
        Define the forward function
        :param x: input features with size [batch_size, in_channels, img_h, img_w]
        :return: output features with size [batch_size, num_classes]
        '''
        batch_size = x.shape[0]
        # TODO 2.3: forward process
        # step 1: forward process for convolutional layers, apply residual connection in conv3 and conv5
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        p1 = self.pool1(x2)
        x3 = self.conv3(p1) + p1
        x4 = self.conv4(x3)
        p2 = self.pool2(x4)
        x5 = self.conv5(p2) +p2
        # step 2: using Tensor.view() to flatten the tensor so as to match the size of input of fully connected layers. 
        out = x5.view(batch_size, -1)
        # step 3: forward process for linear layers
        out = self.fc_net(out)
        # end TODO 2.3

        if return_features:
            return x1, x2, x3, x4, x5, out

        return out