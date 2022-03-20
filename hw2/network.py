#========================================================
#             Media and Cognition
#             Homework 2 Multilayer Perceptron
#             network.py - linear layer and MLP network
#             Student ID:
#             Name:
#             Tsinghua University
#             (C) Copyright 2022
#========================================================
import torch
import torch.nn as nn
from activations import Activation

'''
In this script we will implement our Linear layer and MLP network.
For the linear layer, we will calculate both the forward and backward processes by our own.
More details about customizing a backward process can be found in:
https://pytorch.org/tutorials/beginner/examples_autograd/two_layer_net_custom_function.html
'''


class LinearFunction(torch.autograd.Function):
    '''
    we will implement the linear function:
    y = xW^T + b
    as well as its gradient computation process
    '''

    @staticmethod
    def forward(ctx, x, W, b):
        '''
        Input:
        :param ctx: a context object that can be used to stash information for backward computation
        :param x: input features with size [batch_size, input_size]
        :param W: weight matrix with size [output_size, input_size]
        :param b: bias with size [output_size]
        Return:
        y :output features with size [batch_size, output_size]
        '''

        # TODO 1: calculate y = xW^T + b and save results in ctx
        y = torch.matmul(x,W.T)+b
        ctx.save_for_backward(x,W)
        # End TODO 1
        return y

    @staticmethod
    def backward(ctx, grad_output):
        '''
        Input:
        :param ctx: a context object with saved variables
        :param grad_output: dL/dy, with size [batch_size, output_size]
        Return:
        grad_input: dL/dx, with size [batch_size, input_size]
        grad_W: dL/dW, with size [output_size, input_size], summed for data in the batch
        grad_b: dL/db, with size [output_size], summed for data in the batch
        '''

        # TODO 2: get x and W from ctx and calculate the gradients by ctx.saved_variables
        x,W = ctx.saved_tensors
        # calculate dL/dx (grad_input) by using dL/dy (grad_output) and W, eg., dL/dx = dL/dy*W
        # calculate dL/dW (grad_W) by using dL/dy (grad_output) and x
        # calculate dL/db (grad_b) using dL/dy (grad_output)
        # you can use torch.matmul(A, B) to compute matrix product of A and B
        grad_input = torch.matmul(grad_output, W)
        grad_W = torch.matmul(grad_output.T, x)
        grad_b = grad_output.sum(0)
        # End TODO 2
        return grad_input, grad_W, grad_b


class Linear(nn.Module):
    def __init__(self, input_size, output_size):
        '''
        a linear layer we have implemented in the last homework,
        instead of using PyTorch's automatic derivative mechanism,
        this time we use our own LinearFunction implemented above.
        -----------------------------------------------
        :param input_size: dimension of input features
        :param output_size: dimension of output features
        '''
        super(Linear, self).__init__()

        # TODO 3: initialize weights and bias of the linear layer and set W and b trainable parameters
        # hint: you can refer homework 1
        # （0，1）正态分布的W
        W = torch.randn(output_size, input_size)
        b = torch.zeros(output_size)
        self.W = nn.Parameter(W, requires_grad = True)
        self.b = nn.Parameter(b, requires_grad = True)
        # End TODO 3

    def forward(self, x):
        # here we call the LinearFunction we implement above
        return LinearFunction.apply(x, self.W, self.b)


class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, n_layers, act_type):
        '''
        Multilayer Perceptron
        ----------------------
        :param input_size: dimension of input features
        :param output_size: dimension of output features
        :param hidden_size: <a list> containing hidden size for each hidden layer
        :param n_layers: number of layers
        :param act_type: type of activation function <for each hidden layer>, can be none, sigmoid, tanh, or relu
        '''
        # 父类初始化
        super(MLP, self).__init__()

        # total layer number should be hidden layer number + 1 (output layer)
        assert len(hidden_size) + 1 == n_layers, 'total layer number should be hidden layer number + 1'

        # define the activation function by activation function in activations.py
        self.act = Activation(act_type)

        # initialize a list to save layers
        layers = nn.ModuleList()

        if n_layers == 1:
            # if n_layers == 1, MLP degenerates to a Linear layer
            layer = Linear(input_size, output_size)
            # append the layer into layers
            layers.append(layer)
            layers.append(self.act)

        # TODO 4: Finish MLP with at least 2 layers
        else:
            # step 1: initialize the input layer
            layer = Linear(input_size, hidden_size[0])
            # step 2: append the input layer and the activation layer into layers
            layers.append(layer)
            layers.append(self.act)
            # step 3: construct the hidden layers and add it to layers
            for i in range(1, n_layers - 1):
                #initialize a hidden layer and activation layer
                # hint: Noting that the output size of a hidden layer is hidden_size[i], so what is its input size?
                layer = Linear(hidden_size[i-1],hidden_size[i])
                layers.append(layer)
                layers.append(self.act)

            # step 4: initialize the output layer and append the layer into layers
            # hint: what is the output size of the output layer?
            # hint: here we do not need activation layer
            layer = Linear(hidden_size[-1], output_size)
            layers.append(layer)

            # End TODO 4        
        #Use nn.Sequential to get the neural network
        self.net = nn.Sequential(*layers)


    def forward(self, x):
        '''
        Define the forward function
        :param x: input features with size [batch_size, input_size]
        :return: output features with size [batch_size, output_size]
        '''
        out = self.net(x)
        return out
