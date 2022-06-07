# ========================================================
#             Media and Cognition
#             Homework 6 Transformer
#             network.py: define a network consists of a CNN and a Transformer
#             Tsinghua University
#             (C) Copyright 2022
# ========================================================

from matplotlib.transforms import Transform
import torch
import torch.nn as nn
from transformer_utils import TransformerModel


class CNNTransformer(nn.Module):
    def __init__(self):
        super(CNNTransformer, self).__init__()
		
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, 3, 2, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),
            nn.Conv2d(32, 48, 3, 1, 1),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),
            nn.Conv2d(48, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),
            nn.Conv2d(64, 64, 1))

        # define self.transformer for sequence modeling
        # -- hint: you can directly use the TransformerModel defined in transformer_utils.py
        # -- here we give an optional Transformer configuration
        #    d_input = 64, d_model = 32, n_head = 2, num_encoder_layers = 1, num_decoder_layers = 1, dim_feedforward = 32, output_class = 36 + 4, v
		#    d_input is the width of feature maps extracted by CNN: 64
		# =======================================
        # TODO 1: complete the initialization for self.transformer
        # =======================================
        self.transformer = TransformerModel(
            d_input = 64,   # 需要先做一个d_inout到d_model的线性转变
            d_model = 32, 
            n_head = 2, 
            num_encoder_layers = 1, 
            num_decoder_layers = 1 ,
            dim_feedforward = 32,
            output_class = 40,
            max_timestep=30)
        
    def forward(self, x, tgt, tgt_length):
        '''
        :param x: input images with size [b, 3, H, W], b is batch size, H and W are height and width of the image
		:param tgt: target words with size [max_len, b], max_len is the maximum length of target word in this batch
		:param tgt_length: the length of each target words with size [b]
        :return logits: logits with size [max_len, b, 40]
        '''
        feats = self.cnn(x)
        feats = feats.squeeze(2).permute(2, 0, 1)

        # apply self.transformer on feature sequences to get the logits
		# -- hint: you need to input tgt and tgt_length to self.transformer to train the model in parallel
        
        # ========================================
        # TODO 2: complete network forward process
        # ========================================
        logits = self.transformer.forward(feats, tgt, tgt_length) 
        return logits

    def inference(self, x):
        feats = self.cnn(x)
        feats = feats.squeeze(2).permute(2, 0, 1)

        # apply self.transformer on feature sequences to get the logits and predictions
		# -- hint: inference is different from the forward pass
        
        # ========================================
        # TODO 3: complete network inference process
        # ========================================
        preds, logits = self.transformer.inference(feats)
        return preds, logits


if __name__ == '__main__':  # make sure the model output has correct size
    x = torch.rand(2, 3, 32, 128)  # you can change this if you use different normalization size
    tgt = torch.randint(0, 40, [10, 2])
    tgt_length = torch.LongTensor([10, 10])
    model = CNNTransformer()
    logits = model(x, tgt ,tgt_length)
    preds, logits_inf = model.inference(x)

    assert len(logits.size()) == 3 and logits.size(1) == 2 and logits.size(2) == 40,\
        f"logits should be with size of [width, batch_size, n_class], but got {logits.size()}"

    assert len(tgt_length.size()) == 1 and tgt_length.size(0) == 2, \
        f"tgt_length should be with size of [batch_size], but got {tgt_length.size()}"

    assert all([seq_len == logits.size(0) for seq_len in tgt_length]), \
        f"tgt_length should be equal to batch_size x [width], but width = {logits.size(0)} and tgt_length = {tgt_length}"
    
    assert logits_inf.shape[0] == preds.shape[0] - 1, \
        f"the time steps of logits should be equal to max timesteps(30), but got {logits_inf.shape[0]}"
		
    print('The output size of model is correct!')
