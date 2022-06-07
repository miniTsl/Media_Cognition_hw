# ========================================================
#             Media and Cognition
#             Homework 6 Transformer
#             transformer_utils.py: define the Transformer network
#             Tsinghua University
#             (C) Copyright 2022
# ========================================================
import math
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    """Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        :param x: the sequence fed to the positional encoder model with size [b, d_model] 
        :return x_hat: apply positional encoding and dropout to the input x 
        """
	    # apply self.transformer on feature sequences to get the logits
	    # -- hint: firstly add x with self.pe, and then apply self.dropout to it
        # 先添加位置编码然后再做Dropout,只需要用到前32位即可
	    # ========================================
	    # TODO 4: complete positional encoding forward process
	    # ========================================
        x_hat = self.dropout(x + self.pe[:x.size(0), :])
        
        return x_hat


class TransformerModel(nn.Module):
    def __init__(self, 
            d_input,
            d_model, 
            n_head, 
            num_encoder_layers, 
            num_decoder_layers,
            dim_feedforward,
            output_class,
            max_timestep=30):
        """
        :param d_input: the size of the CNN feature
		:param d_model: the size of the transformer model
        :param n_head: the number of heads in multi-head attention adoped in encoder layers and decoder layers
		:param num_encoder_layers: the number of encoder layers
		:param num_decoder_layers: the number of decoder layers
		:param dim_feedforward: the size of feed forward network adoped in encoder layers and decoder layers
		:param output_class: the number of output classes (set to 40 in HW6)
		:param max_timestep: the maximum time step adpted in inference (set to 30 in HW6)
        """

        super(TransformerModel, self).__init__()
        self.char_emb = nn.Embedding(output_class, d_model, padding_idx=2)
        self.pos_emb = PositionalEncoding(d_model)
        self.max_timestep = max_timestep
	
		# define the each part of the transformer model for sequence modeling
	    # you may follow the below steps
        # 1. you should firstly map the input cnn feature to the size of d_model
		# 2. you should define self.encoder with nn.TransformerEncoderLayer and nn.TransformerEncoder
		# 3. you should define self.decoder with nn.TransformerDecoderLayer and nn.TransformerDecoder
		# 4. you should finally map the transformer output feature to the size of output_class
		#-- hint: you can refer to https://pytorch.org/docs/master/nn.html#transformer-layers for more details about how to use transformer layers
		#         you only need to set d_model, nhead, dim_feedforward in nn.TransformerEncoderLayer and nn.TransformerDecoderLayer
		
		#                                ###### NOTICE! ####### 
		#	directly use nn.Transformer introduced in the TA courses will NOT get the scores in this part!
		#                                ###### NOTICE! ####### 
		
		# =======================================
        # TODO 5: complete the initialization for the transformer model
        # =======================================
        self.input_fc = nn.Linear(d_input, d_model) # 先做一个d_input到d_model的线性转变

        encoder_layer = nn.TransformerEncoderLayer(d_model, n_head, dim_feedforward)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)
        decoder_layer = nn.TransformerDecoderLayer(d_model, n_head, dim_feedforward)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)
        
        self.fc = nn.Linear(d_model, output_class)
        
        self.max_timestep = max_timestep
        
    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward_train(self, src, tgt, tgt_lengths):
        src = self.pos_emb(self.input_fc(src))
        tgt = self.pos_emb(self.char_emb(tgt))
        tgt_mask = self._generate_square_subsequent_mask(tgt.size(0)).to(src.device)
        tgt_key_padding_mask = torch.ones((src.size(1), tgt.size(0)), dtype=torch.uint8, device=src.device)
        for i, l in enumerate(tgt_lengths):
            tgt_key_padding_mask[i, :l + 2] = 0
        tgt_key_padding_mask = (tgt_key_padding_mask == 1)

        memory = self.encoder(src)
        output = self.decoder(tgt = tgt, memory = memory, tgt_mask = tgt_mask, tgt_key_padding_mask = tgt_key_padding_mask)
        
        output = self.fc(output)
        
        return output

    def inference(self, src):
        """
        :param src: the encoded image sequence feed to the transformer model with size [time_length, b, d_input] 
        :return tgt: predicted indexes of the characters with the size [max_timestep, b] 
        :return logits: predicted logits with the size [max_timestep - 1, b, output_class] 
        """
        
        # define the each part of the transformer model for sequence modeling
        # you may follow the below steps
        # 1. you should firstly apply input_fc and self.pos_emb to src
        # 2. call self.encoder to get the memory like the forward pass
        # 3. initialize tgt with a tensor filled with zero(the index of <sos>)
        # 4. iteratively feed the logits to the transformer network (use a for loop)
        # 4.1. apply self.pos_emb to tgt, and generate the square subsequent mask for it
        # 4.2. call self.decoder to get the logits
        # 4.3. predict new character with the last logits (logits[-1, ...]) and prolong tgt
		#-- hint: you can imitate the process in func forward_train for some of the steps
        
        # =======================================
        # TODO 6: complete the inference process for the transformer
        # =======================================
        src = self.pos_emb(self.input_fc(src))  # 先过一个线性层，然后做位置编码
        memory = self.encoder(src)  # 前传获得memory
        
        tgt = torch.zeros(self.max_timestep, src.size(1)).long().to(src.device) 
        # 构建目标tgt: [max_timestep, batch_size]，现在元素全是0，即<sos>
        for t in range(1, self.max_timestep):   # 推理阶段是串行的，和RNN类似
            tgt_emb = self.pos_emb(self.char_emb(tgt[:t,:]))    
            # 预测t时刻的输出时，只需要把前0到t-1的tgt送进来就行了，这样可以减小运算量
            tgt_mask = self._generate_square_subsequent_mask(t).to(src.device) 
            # 相应的tgt_mask也只需要t大小
            decoder_output = self.decoder(tgt_emb, memory, tgt_mask = tgt_mask)
            # tgt_mask的大小决定decoder_output的大小
            char_cls = self.fc(decoder_output[-1, ...])
            char_cls = torch.argmax(char_cls, dim = 1)  # 选择预测概率最大元素的index 
            tgt[t, :] = char_cls # 获得第t时刻的预测输出
        logits = self.fc(decoder_output)
        return tgt, logits
    
    def forward(self, src, tgt=None, tgt_lengths=None):
        if self.training:
            return self.forward_train(src, tgt, tgt_lengths)
        else:
            return self.inference(src)