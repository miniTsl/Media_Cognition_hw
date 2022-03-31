#========================================================
#             Media and Cognition
#             Homework 3 Convolutional Neural Network
#             batchnorm.py -  Batch Normalization
#             Student ID:
#             Name:
#             Tsinghua University
#             (C) Copyright 2022
#========================================================

import torch 
import torch.nn as nn
from torch.autograd import Function
from torch.nn import Parameter

'''
In this script we will implement our BatchNorm layer.
For the BatchNorm layer, we will calculate both the forward and backward processes by our own.
More details about customizing a backward process can be found in:
https://pytorch.org/tutorials/beginner/examples_autograd/two_layer_net_custom_function.html
'''

class batchnorm(Function):
    '''
    we will implement the batchnorm function:
    y = gamma * (x - E[x]) / \sqrt{Var[x] + eps} + beta
    as well as its gradient computation process
    '''

    @staticmethod
    def forward(ctx, input, gamma, beta, eps, running_mean, running_var, affine, train_mode):
        '''
        :param ctx: a context object that can be used to stash information for backward computation
        :param x: input features with size [batch_size, num_features]
        :param gamma: weight parameter vector with size [num_features]
        :param beta: bias parameter vector with size [num_features]
        :param eps: a value added to the variance to avoid dividing zero
        :param running_mean: running estimate of data mean with size [num_features] during training, which is used for normalization during evaluation
        :param running_var: running estimate of data variance with size [num_features] during training, which is used for normalization during evaluation
        :param affine: a boolean value that when set to True, this module has learnable parameter vectors
        :param train_mode: a boolean value that when set to True means training, or means evaluation
        '''

        num_features = input.shape[1]
        if train_mode:
            # The mean and variance are calculated per-dimension over the mini-batches during training.
            # Note that the variance is calculated via the biased estimator.
            mean = input.mean(dim=0)
            var = input.var(dim=0, unbiased=False)
        else:
            # During evaluation, if the running statistics are not None, using them as the mean and variance;
            # Or calculate the mean and variance using the same way in training.
            if running_mean is not None and running_var is not None:
                mean = running_mean
                var = running_var
            else:
                mean = input.mean(dim=0)
                var = input.var(dim=0, unbiased=False)

        # Normalize the input
        input_norm = (input - mean.unsqueeze(dim=0)) / (var.unsqueeze(dim=0) + eps)**(1/2)
        # Scale and shift
        output = input_norm * gamma.view(1, num_features) + beta.view(1, num_features)

        # Save variables for backward
        ctx.save_for_backward(mean, (var.unsqueeze(dim=0) + eps)**(1/2), input_norm, gamma, affine)
        return output

    @staticmethod
    def backward(ctx, output_grad):
        '''
        :param ctx: a context object with saved variables
        :param output_grad: dL/dy, with size [batch_size, num_features]
        :return input_grad: dL/dx, with size [batch_size, num_features]
        :return gamma_grad: dL/d(gamma), with size [num_features]
        :return beta_grad: dL/d(beta), with size [num_features]
        :return None: the gradients of unlearnable parameters are None
        '''
        # Load saved variables from ctx
        mean, var, input_norm, gamma, affine = ctx.saved_tensors
        batch_size, num_features = output_grad.shape[0], output_grad.shape[1]

        # If gamma and beta are learnable, calculate their gradients, otherwise, return None.
        if affine:
            gamma_grad = torch.sum(output_grad * input_norm, dim=0)
            beta_grad = torch.sum(output_grad, dim=0)
        else:
            gamma_grad = None
            beta_grad = None
        
        # Calculate the gradient of input.
        grad = output_grad * gamma.unsqueeze(dim=0)
        grad_var = torch.sum(grad * input_norm, dim = 0) * (-1 / 2) / var**2
        grad_mean = - torch.sum(grad, dim=0) / var - 2 * grad_var * torch.sum(input_norm, dim=0) * var / batch_size

        input_grad = grad / var.unsqueeze(dim=0) + 2 * grad_var.unsqueeze(dim=0) * input_norm * var.unsqueeze(dim=0) / batch_size + grad_mean.unsqueeze(dim=0) / batch_size

        return input_grad, gamma_grad, beta_grad, None, None, None, None, None, None

# Define a basic module for batch normalization
class BatchNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        '''
        :param num_features: the number of input feature channels
        :eps: a value added to the variance to avoid dividing zero
        :momentum: the value used for the running_mean and running_var computation. Can be set to None for cumulative moving average. Default: 0.1
        :affine: a boolean value that when set to True, this module has learnable parameter vectors
        :track_running_stats: a boolean value that when set to True, this module tracks the running mean and variance
        '''
        super(BatchNorm, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = torch.Tensor([affine]).type(torch.uint8)
        self.track_running_stats = track_running_stats

        self.gamma = torch.Tensor(num_features)
        self.beta = torch.Tensor(num_features)
        if self.affine:
            # If affine is set to True, gamma and beta are learnable parameters.
            self.gamma = Parameter(self.gamma)
            self.beta = Parameter(self.beta)
        else:
            self.register_buffer('gamma', None)
            self.register_buffer('beta', None)   

        if self.track_running_stats:
            # If track_running_stats is set to True, initialize the running mean and variance.
            self.register_buffer('running_mean', torch.zeros(num_features))
            self.register_buffer('running_var', torch.ones(num_features))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            # Else, set the running mean and variance to None.
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)
            self.register_parameter('num_batches_tracked', None)

        # Initialize model parameters.
        self.reset_parameters()

    def reset_parameters(self):
        self.gamma.data.uniform_()
        self.beta.data.zero_()
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)
            self.num_batches_tracked.zero_()

    def forward(self, x):
        
        raise NotImplementedError

# Define BatchNorm1d as a subclass of BatchNorm
class BatchNorm1d(BatchNorm):
    def forward(self, x):
        '''
        :params x: input with size [batch_size, num_features] or [batch_size, num_features, length]
        '''
        batch_size = x.shape[0]
        # Check whether the input shape is valid.
        assert (len(x.shape) == 2) or (len(x.shape) == 3), "The input for BatchNorm1d must have 2 or 3 dims!"
        assert x.shape[1] == self.num_features, 'Expected the inputs to have {} channels'.format(self.num_features)

        if len(x.shape) == 3:
            # If the input size is [batch_size, num_features, length], reshape to [batch_size*length, num_features]
            x = x.permute(0, 2, 1).contiguous().view(-1, self.num_features)

        if self.track_running_stats and self.training:
            # If track_running_stats is set to True, calculate the running mean and variance when training.
            # Note that the variance is calculated by unbiased estimator.
            self.num_batches_tracked += 1
            self.momentum = self.momentum if self.momentum is not None else 1.0 / float(self.num_batches_tracked)
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * x.detach().mean(dim=0)
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * x.detach().var(dim=0, unbiased=True)
        # Call batchnorm function.
        output = batchnorm.apply(x, self.gamma, self.beta, self.eps, 
                                 self.running_mean, self.running_var, self.affine,  self.training)

        # Recover the size
        if len(x.shape) == 3:
            output = output.view(batch_size, -1, self.num_features).permute(0, 2, 1).contiguous()
        return output

class BatchNorm2d(BatchNorm):
    def forward(self, x):
        '''
        :params x: input with size [batch_size, num_features, height, width]
        '''
        batch_size = x.shape[0]
        # Check whether the input shape is valid.
        assert len(x.shape) == 4, "The input for BatchNorm2d must have 4 dims!"
        assert x.shape[1] == self.num_features, 'Expected the inputs to have {} channels'.format(self.num_features)

        # Reshape input size to [batch_size*height*width, num_features]
        h, w = x.shape[2], x.shape[3]
        x = x.permute(0, 2, 3, 1).contiguous().view(-1, self.num_features)

        if self.track_running_stats and self.training:
            # If track_running_stats is set to True, calculate the running mean and variance when training.
            # Note that the variance is calculated by unbiased estimator.
            self.num_batches_tracked += 1
            self.momentum = self.momentum if self.momentum is not None else 1.0 / float(self.num_batches_tracked)
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * x.detach().mean(dim=0)
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * x.detach().var(dim=0, unbiased=True)

        # Call batchnorm function.
        output = batchnorm.apply(x, self.gamma, self.beta, self.eps, 
                                 self.running_mean, self.running_var, self.affine, self.training)

        # Recover the size
        output = output.view(batch_size, h, w, self.num_features).permute(0, 3, 1, 2).contiguous()

        return output
