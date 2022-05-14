# ========================================================
#             Media and Cognition
#             Homework 5 Recurrent Neural Network
#             network.py: define a network which consists of a CNN and an RNN
#             Tsinghua University
#             (C) Copyright 2022
# ========================================================

import torch
import torch.nn as nn


class CRNN(nn.Module):  # a model consists of a CNN, an RNN, and a linear layer
    def __init__(self):
        super(CRNN, self).__init__()

        # step 1: define <self.cnn> for vision feature extraction
        # -- hint: you may use the following classes:
        #          nn.Conv2d(), nn.BatchNorm2d(), nn.ReLU(), nn.MaxPool2d(), and other layers
        #          and you can use nn.Sequential() to stack all layers
        # -- here we give an optional CNN configuration
        # ------------------------------------------------------------------------------------
        # block |           convolution          | batch norm | activation |     max pool
        #   0   | #k=16, ksize=3x3, s=2x2, p=1x1 |    yes     |    ReLU    | ksize=2x2, s=2x2
        #   1   | #k=32, ksize=3x3, s=1x1, p=1x1 |    yes     |    ReLU    | ksize=2x1, s=2x1
        #   2   | #k=48, ksize=3x3, s=1x1, p=1x1 |    yes     |    ReLU    | ksize=2x1, s=2x1
        #   3   | #k=64, ksize=3x3, s=1x1, p=1x1 |    yes     |    ReLU    | ksize=2x1, s=2x1
        #   4   | #k=64, ksize=1x1, s=1x1, p=0   |     no     |    None    |       None
        # ------------------------------------------------------------------------------------
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, 3, 2, 1),  # input channels are rgb
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16,32,3,1,1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((2,1)),
            nn.Conv2d(32,48,3,1,1),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.MaxPool2d((2,1)),
            nn.Conv2d(48,64,3,1,1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2,1)),
            nn.Conv2d(64,64,1)
        )
        # step 2: define self.rnn for sequence modeling
        # -- hint 1: you may use nn.LSTM(), but it is also OK to use nn.RNN() or nn.GRU()
        #            please refer to the following document to look up the usage of nn.LSTM()
        #https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html?highlight=lstm#torch.nn.LSTM
        # -- hint 2: the input_size of RNN should be equal to the output channel numbers of CNN,
        #            which is 64 if you use the optional CNN
        # -- here we give an optional RNN configuration
        #    input_size = 64, hidden_size = 32, num_layers = 1, bidirectional = True
        self.rnn = nn.LSTM(input_size = 64, hidden_size = 32, num_layers = 1, 
                           bidirectional = True)
        # step 3: define self.linear for classification
        # -- hint 1: input dimension of self.linear should be output dimension of RNN,
        #            if you set bidirectional=True in RNN, the output dimension
        #            will be 2 times the hidden_size of RNN.
        #            注意LSTM双向，输出的形状是 hidden_size * 2 = 64！！！
        # -- hint 2: output dimension of self.linear should be number of classes,
        #         in this task we have 38 classes (26 letters, 10 digits, 'blank' and '<unk>')
        self.linear = nn.Linear(64,38)
        # =======================================
        # TODO 1: complete network initialization
        # =======================================
    
    # b=32, c=64, h=1, w=128/4=32, 
    def forward(self, x):
        '''
        :param x: input images with size [b, 3, H, W], b is batch size, 
                    3 refers to the dimension of RGB and H(32) and W(128) 
                    are height and width of the image
        :return logits: logits with size [w, b, 38], w = 32 is the width 
                    of feature maps extracted by CNN
        :return seq_lengths: torch.LongTensor with size [b], equals to [w, w, ..., w]
        '''

        # step 1: apply self.cnn on input "x" and get feature maps "feats"
        feats = self.cnn(x) # [b, c, h, w]
        
        # step 2: compute the length of feature sequence "seq_lengths" for CTC loss
        # -- hint: seq_lengths is [w, w, ..., w] (b times)
        #          you can use feats.size(0) and feats.size(3) to get "b" and "w"
        seq_lengths = torch.LongTensor(feats.size(0) * [feats.size(3)])
        # step 3: transform feature maps into RNN input
        # -- hint: the size of feature maps is [batch_size, channels, h, w]
        #          (h=1 if you use the optional CNN),
        #          but the input size of RNN should be [w, b, c].
        #          you may use functions such as tensor.squeeze(dim) and tensor.permute(*dims)
        #          to change the shape of CNN output.
        feats = feats.squeeze(2).permute(2, 0, 1)
        
        # step 4: apply self.rnn on feature sequences
        # -- hint: the outputs of RNN is "output, h_n" or "output, (h_n, c_n)"
        #          we only need "output" for the subsequent recognition
        output, _= self.rnn(feats)  # [w b c] --> [w b 32*2]
        # step 5: apply self.linear on RNN output to obtain "logits"
        logits = self.linear(output)    # [w b 32*2] --> [w, b, 38]
        # ========================================
        # TODO 2: complete network forward process
        # ========================================

        # return logits and seq_lengths
        return logits, seq_lengths


if __name__ == '__main__':  # make sure the model output has correct size
    x = torch.rand(2, 3, 32, 128)  # you can change this if you use different normalization size
    model = CRNN()
    logits, seq_lengths = model(x)

    assert len(logits.size()) == 3 and logits.size(1) == 2 and logits.size(2) == 38, \
        f"logits should be with size of [width, batch_size, n_class], but got {logits.size()}"

    assert len(seq_lengths.size()) == 1 and seq_lengths.size(0) == 2, \
        f"seq_lengths should be with size of [batch_size], but got {seq_lengths.size()}"

    assert all([seq_len == logits.size(0) for seq_len in seq_lengths]), \
        f"seq_lengths should be equal to batch_size x [width], but width = {logits.size(0)} and seq_lengths = {seq_lengths}"

    print('The output size of model is correct!')
