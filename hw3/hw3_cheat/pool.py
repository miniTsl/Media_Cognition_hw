#========================================================
#             Media and Cognition
#             Homework 3 Convolutional Neural Network
#             pool.py -  2D maxpool layer
#             Student ID:
#             Name:
#             Tsinghua University
#             (C) Copyright 2022
#========================================================

import torch 
import torch.nn as nn
from torch.autograd import Function

'''
In this script we will implement our MaxPool2d layer.
For the MaxPool2d layer, we will implement both the forward and backward processes.
More details about customizing a backward process can be found in:
https://pytorch.org/tutorials/beginner/examples_autograd/two_layer_net_custom_function.html
'''

class maxpool2d(Function):
    '''
    we will implement the 2D maxpool function as well as its gradient computation process
    '''

    @staticmethod
    def forward(ctx, input, kernel_size, stride, padding):
        '''
        :param ctx: a context object that can be used to stash information for backward computation
        :param input: input feature maps or image with size [batch_size, in_channels, input_h, input_w]
        :param kernel_size: size of the kernel with format (kernel_h, kernel_w)
        :param stride: stride of max-pooling with format (stride_h, stride_w)
        :param padding: padding added to both sides of the input with format (padding_h, padding_w)
        :return output: pooling result with size [batch_size, out_channels, out_h, out_w]
        '''
        batch_size, in_channels, input_h, input_w = input.shape
        kernel_h, kernel_w = kernel_size
        padding_h, padding_w = padding
        stride_h, stride_w = stride
        
        # Calculate the height and width of output feature maps via input_h/input_w, kernel_h/kernel_w, stride_h/stride_w and padding_h/padding_w.
        out_h = (input_h-kernel_h+2*padding_h)//stride_h+1
        out_w = (input_w-kernel_w+2*padding_w)//stride_w+1

        # get a number from a tensor in PyTorch which contains a single value
        out_h = out_h.item()
        out_w = out_w.item()
        kernel_h = kernel_h.item()
        kernel_w = kernel_w.item()

        # max-pooling requires the contents of padding to be -inf instead of zero. The padding size is [pad_left, pad_right, pad_top, pad_down]
        pad_size = (padding_w, padding_w, padding_h, padding_h)
        x_cols = torch.nn.functional.pad(input, pad_size, mode='constant', value=-float('inf'))

        # We will use matrix computation to implement 2D max-pooling, so that we need to adopt torch.nn.functional.unfold function to transfer images into columns. This step is also called im2col.
        # Please refer to the introduction of im2col in our lecture notes for this homework in week 6. 媒体与认知第三次习题课课件.pdf (page 12)

        # Note that kernel_size, stride and padding are Tensor here, use tuple(x.numpy()) to transfer them into tuple as paramters for unfold function.
        # More details about unfold function can refer to  https://pytorch.org/docs/stable/generated/torch.nn.Unfold.html?highlight=unfold#torch.nn.Unfold .
        x_cols = torch.nn.functional.unfold(x_cols, tuple(kernel_size.numpy()), padding=0, stride=tuple(stride.numpy()))
        x_cols = x_cols.view(batch_size, in_channels, kernel_h * kernel_w, out_h * out_w)

        # Compute maximum values and indices of the unfolded x_cols via torch.max(). Indices of maximum values, max_idx, are used to compute gradient in backward process.
        output, max_idx = torch.max(x_cols, dim=2)

        # The size of the current feature maps after calling torch.max() is [batch_size, in_channels, out_h*out_w]
        # we expect that the size of output feature maps is [batch_size, in_channels, out_h, out_w]
        output = output.view(batch_size, in_channels, out_h, out_w)

        # Save the values of related variables for backward computation.
        # ctx is a context object that can be used to stash information for backward computation.
        ctx.save_for_backward(x_cols, max_idx, torch.LongTensor([input_h]), torch.LongTensor([input_w]), kernel_size, stride, padding)
        return output

    @staticmethod
    def backward(ctx, output_grad):
        '''
        :param ctx: a context object with saved variables
        :param output_grad: dL/dy, with size [batch_size, out_channels, out_h, out_w]
        :return input_grad: dL/dx, with size [batch_size, in_channels, input_h, input_w]
        :return None: no need to calculate the gradients for the rest unlearnable parameters (kernel_size, stride, padding) in the input parameters of forward function
        '''
        # Load saved variables
        x_cols, max_idx, input_h, input_w, kernel_size, stride, padding = ctx.saved_tensors
        # get a number from a tensor in PyTorch which contains a single value
        input_h, input_w = input_h.item(), input_w.item()

        batch_size, in_channels, out_h, out_w = output_grad.shape

        # Use torch.meshgrid to obtain the correspond indices of every elements in the output_grad.
        # More details about torch.meshgrid can refer to: https://pytorch.org/docs/stable/generated/torch.meshgrid.html?highlight=meshgrid .
        grad_cols = torch.zeros(x_cols.shape).type_as(x_cols)
        batch_idx, channel_idx, out_idx = torch.meshgrid(torch.arange(batch_size), torch.arange(in_channels), torch.arange(out_h * out_w))

        # Use output_grad, batch_idx, channel_idx, out_idx and max_idx to compute dL/d(x_cols).
        # The sizes of batch_idx, channel_idx, max_idx and out_idx are all [batch_size, in_channels, out_h*out_w], the size of output_grad is [batch_size, in_channels, out_h, out_w]
        # Please refer to lecture notes for this homework (page 22-24) 媒体与认知第三次习题课.pdf
        grad_cols[batch_idx, channel_idx, max_idx, out_idx] = output_grad.view(batch_size, in_channels, -1)

        # Use torch.nn.functional.fold function to transfer the size of gradient [batch_size, in_channels*kernel_h*kernel_w, out_h*out_w] to [batch_size, in_channels, input_h, input_w].
        # Note that kernel_size, stride and padding are Tensor here, use tuple(x.numpy()) to transfer them into tuple as paramters for fold function.
        # More details about fold function can refer to  https://pytorch.org/docs/stable/generated/torch.nn.Unfold.html?highlight=unfold#torch.nn.Fold .
        grad_cols = grad_cols.view(batch_size, -1, out_h * out_w)
        grad = torch.nn.functional.fold(grad_cols, (input_h, input_w), tuple(kernel_size.numpy()), padding=tuple(padding.numpy()), stride=tuple(stride.numpy()))

        return grad, None, None, None

# Define MaxPool2d module
class MaxPool2d(nn.Module):
    '''
    Maxpool2d layer
    '''
    def __init__(self, kernel_size, stride=1, padding=0):
        '''
        :param kernel_size: size of the pooling kernel (int or tuple)
        :param stride: stride of pooling (int or tuple)
        :param padding: padding added to both sides of the input (int or tuple)
        '''
        super(MaxPool2d, self).__init__()

        # Transfer int or tuple to torch.LongTensor
        if isinstance(kernel_size, tuple):
            self.kernel_size = torch.LongTensor(kernel_size[:2])
        elif isinstance(kernel_size, int):
            self.kernel_size = torch.LongTensor((kernel_size, kernel_size))
        else:
            raise TypeError("The type of kernel size must be tuple or int!")

        if isinstance(stride, tuple):
            self.stride = torch.LongTensor(stride[:2])
        elif isinstance(stride, int):
            self.stride = torch.LongTensor((stride, stride))
        else:
            raise TypeError("The type of stride must be tuple or int!")

        if isinstance(padding, tuple):
            self.padding = torch.LongTensor(padding[:2])
        elif isinstance(padding, int):
            self.padding = torch.LongTensor((padding, padding))
        else:
            raise TypeError("The type of padding must be tuple or int!")

    def forward(self, x):
        # Call maxpool2d function
        return maxpool2d.apply(x, self.kernel_size, self.stride, self.padding)
