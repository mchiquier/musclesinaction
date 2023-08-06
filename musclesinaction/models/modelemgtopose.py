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

import torch
import torch.nn as nn
import torch.optim as optim
import random
import math
import numpy as np

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
        pos_encoding = pos_encoding.unsqueeze(0)
        self.register_buffer("pos_encoding",pos_encoding)
        
    def forward(self, token_embedding: torch.tensor) -> torch.tensor:
        # Residual connection + pos encoding
        #pdb.set_trace()
        return self.dropout(token_embedding + self.pos_encoding)

class TransformerEnc(nn.Module):
    """
    Model from "A detailed guide to Pytorch's nn.Transformer() module.", by
    Daniel Melchor: https://medium.com/@danielmelchor/a-detailed-guide-to-pytorchs-nn-transformer-module-c80afbc9ffb1
    """
    # Constructor
    def __init__(
        self,
        threed,
        num_tokens,
        dim_model,
        num_classes,
        num_heads,
        classif,
        num_encoder_layers,
        num_decoder_layers,
        dropout_p,
        device,
        embedding,
        step
    ):
        super().__init__()

        # INFO
        self.model_type = "Transformer"
        self.dim_model = dim_model
        self.device = device
        self.classif = classif

        # LAYERS
        self.positional_encoder = PositionalEncoding(dim_model=126, dropout_p=dropout_p, max_len=30)#int(step))

        self.jointdim = nn.Linear(126, 8) #ALTERNATIVE
        self.conv1 = nn.Conv2d(1,126,(8,9),(1,1),(0,4))
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm1d(dim_model) #B,S,D
        self.threed = threed
        
        #self.clipproj = nn.Linear(512,256)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=128, nhead=num_heads)
        self.transformer0 = nn.TransformerEncoder(self.encoder_layer, num_layers=num_encoder_layers)
        
        self.num_classes8 = num_classes
        if not self.classif:
            if self.threed == "True":
                self.out0 = nn.Linear(dim_model, 75)
            else:
                self.out0 = nn.Linear(dim_model, 50)
        else:
            self.out = nn.Linear(dim_model, self.num_classes)
        
    def forward(self, src, condval, src_pad_mask=None, tgt_pad_mask=None):
        # Src size must be (batch_size, src sequence length)
        # Tgt size must be (batch_size, tgt sequence length)

        src = src.float()* math.sqrt(self.dim_model) 

        src = torch.unsqueeze(src,dim=1).permute(0,1,2,3)
        
        src = self.conv1(src)[:,:,0,:].permute(0,2,1)
        
        #newcond = torch.unsqueeze(cond,dim=1).repeat(1,30,2).type(torch.cuda.FloatTensor)
        #pdb.set_trace()
        src = self.positional_encoder(src)
        condition=torch.ones(src.shape[0],src.shape[1],2).type(torch.cuda.FloatTensor)*condval.reshape(condval.shape[0],1,1)
        srccat = torch.cat([src,condition.type(torch.cuda.FloatTensor)],dim=2)
        #newsrc = torch.cat([src,newcond],dim=2)
        
        # We could use the parameter batch_first=True, but our KDL version doesn't support it yet, so we permute
        # to obtain size (sequence length, batch_size, dim_model),
        src = srccat.permute(1,0,2) 

        # Transformer blocks - Out size = (sequence length, batch_size, num_tokens)
        transformer_out = self.transformer0(src,src_key_padding_mask=src_pad_mask)
       
        #transformer_out = self.transformer(src, tgt, tgt_mask=tgt_mask, src_key_padding_mask=src_pad_mask, tgt_key_padding_mask=tgt_pad_mask)
        #pdb.set_trace()
        out0 = self.out0(transformer_out)
        #pdb.set_trace()
        out0 = out0.permute(1,2,0)
        #pdb.set_trace()        
        #out = out.permute(1,2,0)
        #if not self.classif:
        #    out0 = self.relu(out0)
           
        return out0.permute(0,2,1)
      
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