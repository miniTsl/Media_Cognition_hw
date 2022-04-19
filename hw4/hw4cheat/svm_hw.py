# ========================================================
#             Media and Cognition
#             Homework 4 Support Vector Machine
#             svm_hw.py - the implementation of SVM using hinge loss
#             Student ID:
#             Name:
#             Tsinghua University
#             (C) Copyright 2021
# ========================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from libsvm.svmutil import *


class Linear(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, W, b):
        """
        W is of the shape (1, channels), x is of the shape (batch_size, channels) and b is of the shape (1, )
        in our homework, channels refers to the dimension of features, i.e. channels=2 for the input samples
        you may need to multiply x with W^T using torch.matmul()
        the output should be of the shape: (batch_size, 1)
        """
        # TODO: compute the output of the linear function: output=W^T*x + b
        output = torch.matmul(x, W.T) + b   # 注意W和x的形状和输出的形状
        ctx.save_for_backward(x, W, b)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        the grad_output is of the shape (batch_size, 1);
        in this homework, you need to sum grad_W or grad_b across the batch_size axis;
        the shape of grad_W should be (1, channels), you may need torch.reshape() or .view() to modify the shape
        the shape of grad_b should be (1, )
        in pytorch, (1, ) refers to the shape of one-dimensional vector
        you may need torch.reshape() or .view() to modify the shape
        """
        x, W, b = ctx.saved_tensors
        batch, channels = x.shape
        # TODO: compute the grad with respect to W and b: dL/dW, dL/db
        grad_W = torch.sum(grad_output * x, dim=0).reshape(1,-1)
        grad_b = torch.sum(grad_output, dim=0).reshape(1,-1)

        return None, grad_W, grad_b


class Hinge(torch.autograd.Function):

    @staticmethod
    def forward(ctx, output, W, label, C):
        """
        in this homework, the input parameter 'label' is y in the equation for loss calculation
        the input parameter 'output' is the output of the linear layer, i.e. output = W^T*x + b
        you may need F.relu() to implement the max() function.
        """
        C = C.type_as(W)
        # TODO: compute the hinge loss (together with L2 norm for SVM): loss = 0.5*||w||^2 + C*\sum_i{max(0, 1 - y_i*output_i)}
        loss = 0.5*torch.norm(W,2)**2 + C*torch.sum((F.relu(1-label.view(-1,1)*output)))
        ctx.save_for_backward(output, W, label, C)
        return loss

    @staticmethod
    def backward(ctx, grad_loss):
        """
        the input parameter 'grad_loss' refers to the gradient of the final target loss with respect to the output (variable 'loss') of the forward function
        in fact, this should be 1 ?
        the shape of grad_output should be (batch_size, 1) and the shape of grad_W should be (1, channels)
        """
        output, W, label, C = ctx.saved_tensors
        # TODO: compute the grad with respect to the output of the linear function and W: dL/doutput, dL/dW
        grad_output = grad_loss * C * ((1-label.view(-1,1)*output)>0) * (-label.view(-1,1)) 
        grad_W = grad_loss * W
        return grad_output, grad_W, None, None


class SVM_HINGE(nn.Module):

    def __init__(self, in_channels, C):
        """
        the shape of W should be (1, channels) and the shape of b should be (1, )
        you need to use nn.Parameter() to make W and b be trainable parameters, don't forget to set requires_grad=True for self.W and self.b
        please use torch.randn() to initialize W and b
        """
        super().__init__()
        # TODO: define the parameters W and b
        self.W = nn.Parameter(torch.randn(1,in_channels),requires_grad = True)
        self.b = nn.Parameter(torch.randn(1,),requires_grad = True)
        self.C = torch.tensor([[C]], requires_grad=False)

    def forward(self, x, label=None):
        output = Linear.apply(x, self.W, self.b)
        if label is not None:
            loss = Hinge.apply(output, self.W, label, self.C)
        else:
            loss = None
        output = (output > 0.0).type_as(x) * 2.0 - 1.0
        return output, loss


def libsvm(train_file_path, val_file_path, C):
    train_data = np.load(train_file_path)
    train_labels = np.concatenate([np.ones(train_data.shape[0] // 2).astype(np.float32),
                                   -1.0 * np.ones(train_data.shape[0] // 2).astype(np.float32)], axis=0)
    val_data = np.load(val_file_path)
    val_labels = np.concatenate([np.ones(val_data.shape[0] // 2).astype(np.float32),
                                 -1.0 * np.ones(val_data.shape[0] // 2).astype(np.float32)], axis=0)
    m = svm_train(train_labels, train_data, f'-c {C} -t 0')

    sv = m.get_sv_indices()
    sv = [x-1 for x in sv]
    sv_feature = train_data[sv, :]
    sv_labels = train_labels[sv]
    sv_coef = m.get_sv_coef()
    sv_coef = [x[0] for x in sv_coef]

    W = (np.reshape(sv_coef, (-1, 1)) * np.reshape(sv_feature, (-1, 2))).sum(0)
    W = np.reshape(W, (1, 2))
    b = (np.reshape(sv_labels, (-1, )) - (W * np.reshape(sv_feature, (-1, 2))).sum(1)).mean()

    svm_predict(val_labels, val_data, m)
    return W, b, sv