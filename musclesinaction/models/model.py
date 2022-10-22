'''
Neural network architecture description.
'''
import torch
import torch.nn as nn
import torch.optim as optim
import random
import math
import numpy as np

# Internal imports.
import utils


import torch
import torch.nn as nn
import torch.optim as optim
import random
import math
import numpy as np
import pdb

class PositionalEncoding(nn.Module):
    def __init__(self, dim_model, dropout_p, max_len):
        super().__init__()
        # Modified version from: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
        # max_len determines how far the position can have an effect on a token (window)
        
        # Info
        self.dropout = nn.Dropout(dropout_p)
        
        # Encoding - From formula
        pos_encoding = torch.zeros(max_len, dim_model)
        positions_list = torch.arange(0, max_len, dtype=torch.float).view(-1, 1) # 0, 1, 2, 3, 4, 5
        division_term = torch.exp(torch.arange(0, dim_model, 2).float() * (-math.log(10000.0)) / dim_model) # 1000^(2i/dim_model)
        
        # PE(pos, 2i) = sin(pos/1000^(2i/dim_model))
        pos_encoding[:, 0::2] = torch.sin(positions_list * division_term)
        
        # PE(pos, 2i + 1) = cos(pos/1000^(2i/dim_model))
        pos_encoding[:, 1::2] = torch.cos(positions_list * division_term)
        
        # Saving buffer (same as parameter without gradients needed)
        pos_encoding = pos_encoding.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pos_encoding",pos_encoding)
        
    def forward(self, token_embedding: torch.tensor) -> torch.tensor:
        # Residual connection + pos encoding
        return self.dropout(token_embedding + self.pos_encoding[:token_embedding.size(0), :])

class TransformerEnc(nn.Module):
    """
    Model from "A detailed guide to Pytorch's nn.Transformer() module.", by
    Daniel Melchor: https://medium.com/@danielmelchor/a-detailed-guide-to-pytorchs-nn-transformer-module-c80afbc9ffb1
    """
    # Constructor
    def __init__(
        self,
        num_tokens,
        dim_model,
        num_classes,
        num_heads,
        classif,
        num_encoder_layers,
        num_decoder_layers,
        dropout_p,
        device,
        embedding
    ):
        super().__init__()

        # INFO
        self.model_type = "Transformer"
        self.dim_model = dim_model
        self.device = device
        self.classif = classif

        # LAYERS
        self.positional_encoder = PositionalEncoding(
            dim_model=dim_model, dropout_p=dropout_p, max_len=30
        )

        self.embedding = nn.Linear(num_tokens, dim_model) #ALTERNATIVE
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm1d(dim_model) #B,S,D
        
        #self.clipproj = nn.Linear(512,256)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=dim_model, nhead=num_heads)
        self.transformer0 = nn.TransformerEncoder(self.encoder_layer, num_layers=num_encoder_layers)
        
        self.num_classes8 = num_classes
        if not self.classif:
            self.out0 = nn.Linear(dim_model, 1)
            self.out1 = nn.Linear(dim_model, 1)
            self.out2 = nn.Linear(dim_model, 1)
            self.out3 = nn.Linear(dim_model, 1)
            self.out4 = nn.Linear(dim_model, 1)
            self.out5 = nn.Linear(dim_model, 1)
            self.out6 = nn.Linear(dim_model, 1)
            self.out7 = nn.Linear(dim_model, 1)
        else:
            self.out = nn.Linear(dim_model, self.num_classes)
        
    def forward(self, src, src_pad_mask=None, tgt_pad_mask=None):
        # Src size must be (batch_size, src sequence length)
        # Tgt size must be (batch_size, tgt sequence length)

        src = src.float()* math.sqrt(self.dim_model) 
        src = self.embedding(src)
        src = src.permute(0,2,1)
        src = self.bn(src)
        src = src.permute(0,2,1)
        
        src = self.positional_encoder(src)
        
        # We could use the parameter batch_first=True, but our KDL version doesn't support it yet, so we permute
        # to obtain size (sequence length, batch_size, dim_model),
        src = src.permute(1,0,2)#src.permute(1,0,2)

        # Transformer blocks - Out size = (sequence length, batch_size, num_tokens)
        transformer_out = self.transformer0(src,src_key_padding_mask=src_pad_mask)
       
        #transformer_out = self.transformer(src, tgt, tgt_mask=tgt_mask, src_key_padding_mask=src_pad_mask, tgt_key_padding_mask=tgt_pad_mask)
        out0 = self.out0(transformer_out)
        out1 = self.out1(transformer_out)
        out2 = self.out2(transformer_out)
        out3 = self.out3(transformer_out)
        out4 = self.out4(transformer_out)
        out5 = self.out5(transformer_out)
        out6 = self.out6(transformer_out)
        out7 = self.out7(transformer_out)
        out0 = out0.permute(1,2,0)
        out1 = out1.permute(1,2,0)
        out2 = out2.permute(1,2,0)
        out3 = out3.permute(1,2,0)
        out4 = out4.permute(1,2,0)
        out5 = out5.permute(1,2,0)
        out6 = out6.permute(1,2,0)
        out7 = out7.permute(1,2,0)
        
        #out = out.permute(1,2,0)
        if not self.classif:
            out0 = self.relu(out0)
            out1 = self.relu(out1)
            out2 = self.relu(out2)
            out3 = self.relu(out3)
            out4 = self.relu(out4)
            out5 = self.relu(out5)
            out6 = self.relu(out6)
            out7 = self.relu(out7)

        out = torch.cat([out0,out1,out2,out3,out4,out5,out6,out7],dim=1)

        return out  
      
    def get_tgt_mask(self, size) -> torch.tensor:
        # Generates a squeare matrix where the each row allows one word more to be seen
        mask = torch.tril(torch.ones(size, size) == 1) # Lower triangular matrix
        mask = mask.float()
        mask = mask.masked_fill(mask == 0, float('-inf')) # Convert zeros to -inf
        mask = mask.masked_fill(mask == 1, float(0.0)) # Convert ones to 0
        
        # EX for size=5:
        # [[0., -inf, -inf, -inf, -inf],
        #  [0.,   0., -inf, -inf, -inf],
        #  [0.,   0.,   0., -inf, -inf],
        #  [0.,   0.,   0.,   0., -inf],
        #  [0.,   0.,   0.,   0.,   0.]]
        
        return mask
    
    def create_pad_mask(self, matrix: torch.tensor, pad_token: int) -> torch.tensor:
        # If matrix = [1,2,3,0,0,0] where pad_token=0, the result mask is
        # [False, False, False, True, True, True]
        return (matrix == pad_token)