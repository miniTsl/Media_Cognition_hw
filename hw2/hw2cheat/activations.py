#========================================================
#             Media and Cognition
#             Homework 2 Multilayer Perceptron
#             activations.py - activation functions
#             Student ID:
#             Name:
#             Tsinghua University
#             (C) Copyright 2022
#========================================================
import torch
import torch.nn as nn

'''
In this script we will implement four activation functions, including both their forward and backward processes.
More details about customizing a backward process in PyTorch can be found in:
https://pytorch.org/tutorials/beginner/examples_autograd/two_layer_net_custom_function.html
'''

class Tanh(torch.autograd.Function):
    '''
    Tanh activation function
    y = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
    '''
    # staticmethod of a python class means that we can call the function without initializing an instance of the class
    @staticmethod
    def forward(ctx, x):
        '''
        In the forward pass we receive a Tensor containing the input x and return
        a Tensor containing the output. 
        
        ctx: it is a context object that can be used to save information for backward computation. You can save 
        objects by using ctx.save_for_backward, and get objects by using ctx.saved_tensors

        x: input with arbitrary shape
        '''
        # Please think if we use "y = (exp(x) - exp(-x)) / (exp(x) + exp(-x))", what might happen when x has a large absolute value
        # y = (torch.exp(x) - torch.exp(-x)) / (torch.exp(x) + torch.exp(-x))

        # here we directly use torch.tanh(x) to avoid the problem above
        y = torch.tanh(x)

        # save an variable in ctx
        ctx.save_for_backward(y)

        return y

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.

        grad_output: dL/dy
        grad_input: dL/dx = dL/dy * dy/dx, where y = forward(x)
        """
        # get an variable from ctx
        y, = ctx.saved_tensors

        # chain rule: dL/dx = dL/dy * dy/dx
        # where dL/dy = grad_output, and the dy/dx of tanh function is (1-y^2)!
        grad_input = grad_output * (1 - y ** 2)

        return grad_input


class Identity(torch.autograd.Function):
    '''
    Identity activation function means no activation function
    y = x
    '''
    @staticmethod
    def forward(ctx, x):
        
        y = x
        return y

    @staticmethod
    def backward(ctx, grad_output):
        
        grad_input = grad_output
        return grad_input



class Sigmoid(torch.autograd.Function):
    '''
    Sigmoid activation function
    y = 1 / (1 + exp(-x))
    '''

    @staticmethod
    def forward(ctx, x):

        # use torch.exp(x) to calculate exp(x)
        y = 1 / (1 + torch.exp(-x))

        # here we save y in ctx, in this way we can use y to calculate gradients in backward process
        ctx.save_for_backward(y)
        

        return y

    @staticmethod
    def backward(ctx, grad_output):

        # get y from ctx
        y, = ctx.saved_tensors

        # implement gradient of x (grad_input), grad_input refers to dL/dx
        # chain rule: dL/dx = dL/dy * dy/dx
        # where dL/dy = grad_output, and you have calculated dy/dx of sigmoid function in theoretical problems of the homework!
        grad_input = grad_output * y * (1 - y)

        return grad_input


class ReLU(torch.autograd.Function):
    '''
    ReLU activation function
    y = max{x, 0}
    '''

    @staticmethod
    def forward(ctx, x):

        # set elements less than 0 in x to 0
        # this operation is inplace
        x[x < 0] = 0

        # save an variable in ctx
        ctx.save_for_backward(x)

        # return the output
        return x

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """

        x, = ctx.saved_tensors

        # chain rule: dL/dx = dL/dy * dy/dx
        # where dL/dy = grad_output, and you have calculated dy/dx of relu function in theoretical problems of the homework!
        grad_input = grad_output.clone()
        grad_input[x <= 0] = 0
        return grad_input

# activate function class according to the type
class Activation(nn.Module):
    def __init__(self, type):
        '''
        :param type:  'none', 'sigmoid', 'tanh', or 'relu'
        '''
        super().__init__()

        if type == 'none':
            # to apply a customized activate function, we use Function.apply method
            self.act = Identity.apply
        elif type == 'sigmoid':
            self.act = Sigmoid.apply
        elif type == 'tanh':
            self.act = Tanh.apply
        elif type == 'relu':
            self.act = ReLU.apply
        else:
            print('activation type should be one of [none, sigmoid, tanh, relu]')
            raise NotImplementedError

    def forward(self, x):
        return self.act(x)
