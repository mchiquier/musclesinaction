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
        pos_encoding = pos_encoding.unsqueeze(0) #.transpose(0, 1)
        self.register_buffer("pos_encoding",pos_encoding)
        
    def forward(self, token_embedding: torch.tensor) -> torch.tensor:
        # Residual connection + pos encoding
        return self.dropout(token_embedding + self.pos_encoding[:,:,:])

class MyLayer(torch.nn.Module):
    def __init__(self, dim_model,num_heads):
        """
        In the constructor we instantiate four parameters and assign them as
        member parameters.
        """
        super().__init__()
        self.encoder_layer_temporal = nn.TransformerEncoderLayer(d_model=dim_model, nhead=num_heads, batch_first=True)
        self.encoder_layer_spatial = nn.TransformerEncoderLayer(d_model=dim_model, nhead=num_heads, batch_first=True)
        self.encoder_layer_temporal2 = nn.TransformerEncoderLayer(d_model=dim_model, nhead=num_heads, batch_first=True)
        self.encoder_layer_spatial2 = nn.TransformerEncoderLayer(d_model=dim_model, nhead=num_heads, batch_first=True)
        self.weightlinear1 = nn.Linear(256,2)
        self.softmax1 = nn.Softmax(dim=3)
    def forward(self, x, src_mask,src_key_padding_mask):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        src_spatial = x
        spatial_shape = (src_spatial.shape[0]*src_spatial.shape[1],src_spatial.shape[2],src_spatial.shape[3])
        spatial_shape_old = (src_spatial.shape[0],src_spatial.shape[1],src_spatial.shape[2],src_spatial.shape[3])
        src_temporal = x.permute(0,2,1,3)
        temporal_shape = (src_temporal.shape[0]*src_temporal.shape[1],src_temporal.shape[2],src_temporal.shape[3])
        temporal_shape_old = (src_temporal.shape[0],src_temporal.shape[1],src_temporal.shape[2],src_temporal.shape[3])

        transformer_out_spatial = self.encoder_layer_spatial(src_spatial.reshape(spatial_shape),src_key_padding_mask=src_key_padding_mask)
        transformer_out_spatial = transformer_out_spatial.reshape(spatial_shape_old)
        
        transformer_out_temporal = self.encoder_layer_temporal(src_temporal.reshape(temporal_shape),src_key_padding_mask=src_key_padding_mask)
        transformer_out_temporal = transformer_out_temporal.reshape(temporal_shape_old).permute(0,2,1,3)

        #transformer_out = transformer_out_temporal + transformer_out_spatial #torch.Size([1, 30, 25, 256])
        src_spatial = transformer_out_temporal
        src_temporal = transformer_out_spatial.permute(0,2,1,3)
        transformer_out_spatial = self.encoder_layer_temporal2(src_spatial.reshape(spatial_shape),src_key_padding_mask=src_key_padding_mask)
        transformer_out_spatial = transformer_out_spatial.reshape(spatial_shape_old)
        
        transformer_out_temporal = self.encoder_layer_temporal2(src_temporal.reshape(temporal_shape),src_key_padding_mask=src_key_padding_mask)
        transformer_out_temporal = transformer_out_temporal.reshape(temporal_shape_old).permute(0,2,1,3)

        concat = torch.cat([transformer_out_temporal,transformer_out_spatial],dim=3)
        weights = self.weightlinear1(concat)
        outputweights = self.softmax1(weights)

        transformer_out = (transformer_out_temporal*outputweights[:,:,:,:1]) + (transformer_out_spatial*outputweights[:,:,:,1:2])
        return transformer_out

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
        self.positional_encoder_time = PositionalEncoding(
            dim_model=dim_model, dropout_p=dropout_p, max_len=30
        )

        self.positional_encoder_space = PositionalEncoding(
            dim_model=dim_model, dropout_p=dropout_p, max_len=25
        )

        self.embedding = nn.Linear(2, dim_model) #ALTERNATIVE
        self.relu = nn.ReLU()
        
        #self.clipproj = nn.Linear(512,256)
        """self.encoder_layer_temporal = nn.TransformerEncoderLayer(d_model=dim_model, nhead=num_heads, batch_first=True)
        self.encoder_layer_spatial = nn.TransformerEncoderLayer(d_model=dim_model, nhead=num_heads, batch_first=True)
        self.transformer_spatial = nn.TransformerEncoder(self.encoder_layer_spatial, num_layers=1)
        self.transformer_temporal = nn.TransformerEncoder(self.encoder_layer_temporal, num_layers=1)

        self.transformer_spatial_two = nn.TransformerEncoder(self.encoder_layer_spatial, num_layers=1)
        self.transformer_temporal_two = nn.TransformerEncoder(self.encoder_layer_temporal, num_layers=1)

        self.transformer_spatial_three = nn.TransformerEncoder(self.encoder_layer_spatial, num_layers=1)
        self.transformer_temporal_three = nn.TransformerEncoder(self.encoder_layer_temporal, num_layers=1)

        self.transformer_spatial_four = nn.TransformerEncoder(self.encoder_layer_spatial, num_layers=1)
        self.transformer_temporal_four = nn.TransformerEncoder(self.encoder_layer_temporal, num_layers=1)"""

        self.first = MyLayer(dim_model=dim_model, num_heads=num_heads)
        self.transformer = nn.TransformerEncoder(self.first, num_layers=3)
        
        self.num_classes8 = num_classes
     
        self.out_joints = nn.Linear(25, 8)
        self.out_channel = nn.Linear(128, 1)
        self.weightlinear1 = nn.Linear(512,2)
        self.weightlinear2 = nn.Linear(512,2)
        self.softmax1 = nn.Softmax(dim=3)
        self.softmax2 = nn.Softmax(dim=3)
        self.tanh1 = nn.Tanh()

    def forward(self, src, src_pad_mask=None, tgt_pad_mask=None):
        # Src size must be (batch_size, src sequence length)
        # Tgt size must be (batch_size, tgt sequence length)

        #batch, seq length, dim model
        src = self.embedding(src)
        #print(src.shape)

        src_spatial = src.float()
        spatial_shape = (src_spatial.shape[0]*src_spatial.shape[1],src_spatial.shape[2],src_spatial.shape[3])
        spatial_shape_old = (src_spatial.shape[0],src_spatial.shape[1],src_spatial.shape[2],src_spatial.shape[3])
        src_spatial = self.positional_encoder_space(src_spatial.reshape(spatial_shape))
        src_spatial = src_spatial.reshape(spatial_shape_old)

        src_temporal = src_spatial.permute(0,2,1,3).float()
        temporal_shape = (src_temporal.shape[0]*src_temporal.shape[1],src_temporal.shape[2],src_temporal.shape[3])
        temporal_shape_old = (src_temporal.shape[0],src_temporal.shape[1],src_temporal.shape[2],src_temporal.shape[3])
        src_temporal = self.positional_encoder_time(src_temporal.reshape(temporal_shape))
        src_temporal = src_temporal.reshape(temporal_shape_old)

        src_spatial = src_temporal.permute(0,2,1,3)

        transformer_out = self.transformer(src_spatial,src_key_padding_mask=src_pad_mask)
        """transformer_out_spatial = self.transformer_spatial(src_spatial.reshape(spatial_shape),src_key_padding_mask=src_pad_mask)
        transformer_out_spatial = transformer_out_spatial.reshape(spatial_shape_old)
        
        transformer_out_temporal = self.transformer_temporal(src_temporal.reshape(temporal_shape),src_key_padding_mask=src_pad_mask)
        transformer_out_temporal = transformer_out_temporal.reshape(temporal_shape_old).permute(0,2,1,3)

        #transformer_out = transformer_out_temporal + transformer_out_spatial #torch.Size([1, 30, 25, 256])
        src_spatial = transformer_out_temporal
        src_temporal = transformer_out_spatial.permute(0,2,1,3)
        transformer_out_spatial = self.transformer_spatial_two(src_spatial.reshape(spatial_shape),src_key_padding_mask=src_pad_mask)
        transformer_out_spatial = transformer_out_spatial.reshape(spatial_shape_old)
        
        transformer_out_temporal = self.transformer_temporal_two(src_temporal.reshape(temporal_shape),src_key_padding_mask=src_pad_mask)
        transformer_out_temporal = transformer_out_temporal.reshape(temporal_shape_old).permute(0,2,1,3)

        concat = torch.cat([transformer_out_temporal,transformer_out_spatial],dim=3)
        weights = self.weightlinear1(concat)
        outputweights = self.softmax1(weights)

        transformer_out = (transformer_out_temporal*outputweights[:,:,:,:1]) + (transformer_out_spatial*outputweights[:,:,:,1:2])

        ###### BLOCK 2 ########
        src_spatial = transformer_out
        src_temporal = transformer_out.permute(0,2,1,3)
        transformer_out_spatial = self.transformer_spatial_three(src_spatial.reshape(spatial_shape),src_key_padding_mask=src_pad_mask)
        transformer_out_spatial = transformer_out_spatial.reshape(spatial_shape_old)
        
        transformer_out_temporal = self.transformer_temporal_three(src_temporal.reshape(temporal_shape),src_key_padding_mask=src_pad_mask)
        transformer_out_temporal = transformer_out_temporal.reshape(temporal_shape_old).permute(0,2,1,3)

        src_spatial = transformer_out_temporal
        src_temporal = transformer_out_spatial.permute(0,2,1,3)
        transformer_out_spatial = self.transformer_spatial_four(src_spatial.reshape(spatial_shape),src_key_padding_mask=src_pad_mask)
        transformer_out_spatial = transformer_out_spatial.reshape(spatial_shape_old)
        
        transformer_out_temporal = self.transformer_temporal_four(src_temporal.reshape(temporal_shape),src_key_padding_mask=src_pad_mask)
        transformer_out_temporal = transformer_out_temporal.reshape(temporal_shape_old).permute(0,2,1,3)

        concat = torch.cat([transformer_out_temporal,transformer_out_spatial],dim=3)
        weights = self.weightlinear2(concat)
        outputweights = self.softmax2(weights)

        transformer_out = (transformer_out_temporal*outputweights[:,:,:,:1]) + (transformer_out_spatial*outputweights[:,:,:,1:2])
        """
        #### END #####
        transformer_out = self.out_channel(transformer_out)
        transformer_out = transformer_out.permute(0,1,3,2)
        transformer_out =self.tanh1(transformer_out)
        transformer_out = self.out_joints(transformer_out)
        #src = src.float()* math.sqrt(self.dim_model) 

        out = transformer_out.permute(0,1,3,2)[:,:,:,0]

        out = self.relu(out)

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