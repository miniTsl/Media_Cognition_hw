#========================================================
#             Media and Cognition
#             Homework 2 Multilayer Perceptron
#             losses.py - loss functions
#             Student ID:
#             Name:
#             Tsinghua University
#             (C) Copyright 2022
#========================================================

import torch
import torch.nn.functional as F

'''
In this script we will implement our MSE and Cross Entropy loss functions, including both the forward and backward processes.
More details about customizing a backward process can be found in:
https://pytorch.org/tutorials/beginner/examples_autograd/two_layer_net_custom_function.html
'''


#TODO 1: Complete the CrossEntropyLoss function
class CrossEntropyLoss(torch.autograd.Function):
    '''
    Cross entropy loss function:
        loss = - log sum (q[i]*label[i])
    where
        q_i = softmax(z_i) = exp(z_i) / (exp(z_0) + exp(z_1) + ...)

    However, when z_i has a lager value, exp(z_i) might become infinity.
    So we use stable softmax:
        softmax(z_i) = A exp(z_i) / A (exp(z_0) + exp(z_1) + ...)
    where
        A = exp(-z_max) = exp(-max{z_0, z_1, ...})
    therefore we have
        softmax(z_i) = softmax(z_i - z_max)
    '''

    @staticmethod
    def forward(ctx, logits, label):
        """
        :param logits: logits with shape [batch_size, n_classes], the "z" in the description above
        :param label: groundtruth with shape [batch_size], where 0 <= label[i] < n_classes - 1
        :return: cross entropy loss, averaged by batch_size
        """
        # step 1: calculate softmax(z) using stable softmax method
        # hint: you can use torch.exp(x) to calculate exp(x), and remeber to convert label into one-hot version
        # e.g., if label = [0, 2] and n_classes=4, then the one-hot version is [[1,0,0,0], [0,0,1,0]]
        # 1.1: calculate z_max
        z_max = logits.max(dim = 1, keepdim=True)
        # 1.2: calculate exps = exp(z - z_max)
        exps = torch.exp(logits-z_max.values)
        # 1.3: calculate p = softmax(y - y_max) 
        partition = exps.sum(dim=1, keepdim=True)
        p = exps/partition
        # step 2: convert label into one-hot version by using F.one_hot()
        # e.g., if label = [0, 2] and n_classes=4, then the one-hot version is [[1,0,0,0], [0,0,1,0]]
        # the converted label has shape [batch_size, n_classes]
        label_one_hot = F.one_hot(label, logits.size(1))
        # step 3: calculate cross entropy loss = - log q_i, and averaged by batch
        # save result of softmax and one-hot label in ctx for gradient computation
        loss = -torch.log((p * label_one_hot).sum(1)+1e-9)
        loss = torch.mean(loss)
        ctx.save_for_backward(p, label_one_hot)
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        # step 4: get p and label from ctx and calculate the derivative of loss w.r.t. pred (dL/dz)
        q, label_one_hot = ctx.saved_tensors
        
        # dl/dz的形状和z（logits）的形状相同
        grad_input = grad_output*(q-label_one_hot)
        
        # return None for gradient of label since we do not need to compute dL/dlabel
        return grad_input, None
# End TODO 2