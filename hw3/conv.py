#========================================================
#             Media and Cognition
#             Homework 3 Convolutional Neural Network
#             conv.py -  2D convolutional layer
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
In this script we will implement our Conv2d layer.
For the Conv2d layer, we will implement both the forward and backward processes.
More details about customizing a backward process can be found in:
https://pytorch.org/tutorials/beginner/examples_autograd/two_layer_net_custom_function.html
'''

class Conv2dFunction(Function):
    '''
    we will implement the 2D convolution function as well as its gradient computation process
    '''

    @staticmethod
    def forward(ctx, input, weight, bias, kernel_size, stride, padding):
        '''
        :param ctx: a context object that can be used to stash information for backward computation
        :param input: input feature maps or image with size [batch_size, in_channels, input_h, input_w]
        :param weight: weight parameter vector with size [out_channels, in_channels, kernel_h, kernel_w]
        :param bias: bias parameter vector with size [out_channels]
        :param kernel_size: size of the convolution kernel with format (kernel_h, kernel_w)
        :param stride: stride of convolution with format (stride_h, stride_w)
        :param padding: zero-padding added to both sides of the input with format (padding_h, padding_w)
        :return output: convolution result with size [batch_size, out_channels, out_h, out_w]
        '''
        batch_size, _, input_h, input_w = input.shape
        kernel_h, kernel_w = kernel_size
        padding_h, padding_w = padding
        stride_h, stride_w = stride
        out_channels = weight.shape[0]
        # TODO 1.1: calculate the height and width of output feature maps via input_h/input_w, kernel_h/kernel_w, stride_h/stride_w and padding_h/padding_w.

        # end TODO 1.1

        # TODO 1.2: calculate the results of 2D convolution via img2col and matrix multiplication
        # step 1: adopt torch.nn.functional.unfold function to transfer images into columns. This step is also called img2col.
        # Please refer to the introduction of img2col in our lecture notes for this homework in week 6. 媒体与认知第三次习题课课件.pdf (page 5)
        # More details about unfold function can refer to  https://pytorch.org/docs/stable/generated/torch.nn.Unfold.html?highlight=unfold#torch.nn.Unfold .
        # The output size of unfold function is [batch_size, in_channels*kernel_h*kernel_w, out_h*out_w] (b,c,n)
        

        # step 2: reshape weights and calculate W * x based on the unfolded x_cols, weight_re via torch.matmul()
        # reshape the size of weights from [out_channels, in_channels, kernel_h, kernel_w] to [out_channels, in_channels*kernel_h*kernel_w] (o, c)
        
        # Hint: the size of x_cols is [batch_size, in_channels*kernel_h*kernel_w, out_h*out_w] (b,c,n)
        # the size of weight_re is [out_channels, in_channels*kernel_h*kernel_w] (o,c)
        # the size of output is [batch_size, out_channels, out_h*out_w] (b,o,n)
        

        # step 3: reshape bias and calculate the final result of 2D convolution W * x + b
        # the size of output is [batch_size, out_channels, out_h*out_w] (b,o,n)
        # reshape the size of bias from [out_channels] to [1, out_channels,1] (1,o,1)
        
        # Hint: we expect that the size of result is [batch_size, out_channels, out_h, out_w]
        
        # end TODO 1.2

        # Save the values of related variables for backward computation.
        # ctx is a context object that can be used to stash information for backward computation.
        ctx.save_for_backward(x_cols, weight, torch.LongTensor([input_h, input_w]), 
                              torch.LongTensor(kernel_size), torch.LongTensor(stride), torch.LongTensor(padding))
        return output

    @staticmethod
    def backward(ctx, output_grad):
        '''
        :param ctx: a context object with saved variables
        :param output_grad: dL/dy, with size [batch_size, out_channels, out_h, out_w]
        :return input_grad: dL/dx, with size [batch_size, in_channels, input_h, input_w]
        :return W_grad: dL/d(weight), with size [out_channels, in_channels, kernel_h, kernel_w]
        :return b_grad: dL/d(bias), with size [out_channels]
        :return None: no need to calculate the gradients for the rest unlearnable parameters (kernel_size, stride, padding) in the input parameters of forward function
        '''
        # Load saved variables
        x_cols, weight, input_size, kernel_size, stride, padding = ctx.saved_tensors
        
        kernel_h, kernel_w = kernel_size
        in_channels = weight.shape[1]

        # TODO 1.3: calculate dL/dW, dL/db and dL/dx.
        # step 1: reshape output_grad (dL/dy) and weights before matrix multiplication

        # reshape the size of dL/dy from [batch_size, out_channels, out_h, out_w] to [batch_size, out_channels, out_h*out_w] (b,o,n)

        # reshape the size of weight from [out_channels, in_channels, kernel_h, kernel_w] to [out_channels, in_channels*kernel_h*kernel_w] (o,c)


        # step 2: calculate dL/dW based on the unfolded x_cols and output_grad via torch.matmul().
        # Hint: the size of x_cols is [batch_size, in_channels*kernel_h*kernel_w, out_h*out_w] (b,c,n)
        # the size of output_grad is [batch_size, out_channels, out_h*out_w] (b,o,n)
        # the size of dL/dW (i.e. W_grad) is the same as W_re which is [out_channels, in_channels*kernel_h*kernel_w] (o,c)
        # [batch_size, out_channels, in_channels*kernel_h*kernel_w] (b,o,c)

        # Hint: we expect that the size of dL/dW (i.e. W_grad) is [out_channels, in_channels, kernel_h, kernel_w]
        

        # step 3: calculate dL/db based on output_grad via torch.sum().
        # the size of output_grad is [batch_size, out_channels, out_h*out_w] (b,o,n)
        # the size of dL/db (i.e. b_grad) is [out_channels,]
        

        # step 4: calculate dL/d(x_cols) based on output_grad and weight_re via torch.matmul() and obtain dL/dx via torch.nn.functional.fold
        # Hint: the size of output_grad is [batch_size, out_channels, out_h*out_w] (b,o,n)
        # the size of weight_re be [out_channels, in_channels*kernel_h*kernel_w] (o,c)
        # the size of dL/d(x_cols) (i.e. x_cols_grad) is [batch_size, in_channels*kernel_h*kernel_w, out_h*out_w] (b,c,n)
        

        # Use torch.nn.functional.fold function to transfer the size of x_cols_grad from [batch_size, in_channels*kernel_h*kernel_w, out_h*out_w] to [batch_size, in_channels, input_h, input_w].
        # Note that kernel_size, stride and padding are Tensor here, use tuple(x.numpy()) to transfer them into tuple as parameters for fold function.
        # More details about fold function can refer to  https://pytorch.org/docs/stable/generated/torch.nn.Unfold.html?highlight=unfold#torch.nn.Fold .
        
        # end TODO 1.3
        
        return x_grad, W_grad, b_grad, None, None, None

# Define Conv2d module
class Conv2d(nn.Module):
    '''
    Conv2d layer
    '''
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        '''
        :param in_channels: number of channels in the input feature maps or images
        :param out_channels: number of channels in the output feature maps
        :param kernel_size: size of the convolution kernel (int or tuple)
        :param stride: stride of convolution (int or tuple)
        :param padding: zero-padding added to both sides of the input (int or tuple)
        '''
        super(Conv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        if isinstance(kernel_size, tuple):
            self.kernel_size = kernel_size[:2]
        elif isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            raise TypeError("The type of kernel size must be tuple or int!")

        if isinstance(stride, tuple):
            self.stride = stride[:2]
        elif isinstance(stride, int):
            self.stride = (stride, stride)
        else:
            raise TypeError("The type of stride must be tuple or int!")

        if isinstance(padding, tuple):
            self.padding = padding[:2]
        elif isinstance(padding, int):
            self.padding = (padding, padding)
        else:
            raise TypeError("The type of padding must be tuple or int!")

        # TODO 1.4: initialize weights and bias of the 2D convolution layer and set W and b trainable parameters

        # End TODO 1.4
			
    def forward(self, x):
        # Check whether the input shape is valid.
        assert x.shape[1] == self.in_channels, 'Expected the inputs to have {} channels'.format(self.in_channels)
        # Call conv2d function.
        return Conv2dFunction.apply(x, self.W, self.b, self.kernel_size, self.stride, self.padding)
