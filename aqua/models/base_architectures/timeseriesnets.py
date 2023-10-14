# MIT License

# Copyright (c) 2023 Carnegie Mellon University, Auton Lab

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import torch
from os.path import join
from torch.nn import (Module, Conv1d, BatchNorm1d, 
                      ReLU, Sequential, Softmax, 
                      AvgPool1d, Linear, CrossEntropyLoss,
                      LayerNorm, Flatten, Dropout)
from typing import Tuple, Union, Callable, Optional
from collections import OrderedDict
from .base_utils import (AttentionLayer, 
    FullAttention, EncoderLayer, Encoder, PatchEmbedding)

class ResNet1D(torch.nn.Module):
    """
    Parameters
    ---------- 
    input_shape: tuple
        Shape of the input time series. (f, T) where 
        f is the number of features and T is the number
        of time steps.
    n_classes: int
        Number of classes. 
    n_feature_maps: int
        Number of feature maps to use. 
    name: str
        Model name. Default: "ResNet"
    verbose: int
        Controls verbosity while training and loading the models. 
    load_weights: bool
        If true load the weights of a previously saved model. 
    random_seed: int
        Random seed to instantiate the random number generator. 
    """
    def __init__(self, 
                 input_shape:Tuple[int, int], 
                 n_feature_maps:int=64, 
                 verbose:bool=False, 
                 weight_path:str=None,
                 **kwargs):
        super(ResNet1D, self).__init__()

        self.n_feature_maps = n_feature_maps
        self.input_shape = input_shape # (num_features, num_timesteps)
        self.verbose = verbose

        if weight_path is not None: # Then just load the weights of the model.
            print(f"Loading model at {weight_path}")
            self.model = torch.load(weight_path)

        else:
            self.model = self.build_model()

        if self.verbose:
            print(self)
    
    def build_model(self):
        # BLOCK 1
        self.conv_block_1 = Sequential(
            Conv1d(in_channels=self.input_shape[0], out_channels=self.n_feature_maps, kernel_size=8, padding='same'),
            BatchNorm1d(num_features=self.n_feature_maps),
            ReLU(inplace=True),
            Conv1d(in_channels=self.n_feature_maps, out_channels=self.n_feature_maps, kernel_size=5, padding='same'),
            BatchNorm1d(num_features=self.n_feature_maps),
            ReLU(inplace=True),
            Conv1d(in_channels=self.n_feature_maps, out_channels=self.n_feature_maps, kernel_size=3, padding='same'),
            BatchNorm1d(num_features=self.n_feature_maps)
            )
        
        self.shortcut_block_1 = Sequential(
            Conv1d(in_channels=self.input_shape[0], out_channels=self.n_feature_maps, kernel_size=1, padding='same'),
            BatchNorm1d(num_features=self.n_feature_maps)
            )

        # BLOCK 2
        self.conv_block_2 = Sequential(
            Conv1d(in_channels=self.n_feature_maps, out_channels=self.n_feature_maps * 2, kernel_size=8, padding='same'),
            BatchNorm1d(num_features=self.n_feature_maps * 2),
            ReLU(inplace=True),
            Conv1d(in_channels=self.n_feature_maps * 2, out_channels=self.n_feature_maps * 2, kernel_size=5, padding='same'),
            BatchNorm1d(num_features=self.n_feature_maps * 2),
            ReLU(inplace=True),
            Conv1d(in_channels=self.n_feature_maps * 2, out_channels=self.n_feature_maps * 2, kernel_size=3, padding='same'),
            BatchNorm1d(num_features=self.n_feature_maps * 2)
            )

        # expand channels for the sum
        self.shortcut_block_2 = Sequential(
            Conv1d(in_channels=self.n_feature_maps, out_channels=self.n_feature_maps * 2, kernel_size=1, padding='same'),
            BatchNorm1d(num_features=self.n_feature_maps * 2)
            )

        # BLOCK 3
        self.conv_block_3 = Sequential(
            Conv1d(in_channels=self.n_feature_maps * 2, out_channels=self.n_feature_maps * 2, kernel_size=8, padding='same'),
            BatchNorm1d(num_features=self.n_feature_maps * 2),
            ReLU(inplace=True),
            Conv1d(in_channels=self.n_feature_maps * 2, out_channels=self.n_feature_maps * 2, kernel_size=5, padding='same'),
            BatchNorm1d(num_features=self.n_feature_maps * 2),
            ReLU(inplace=True),
            Conv1d(in_channels=self.n_feature_maps * 2, out_channels=self.n_feature_maps * 2, kernel_size=3, padding='same'),
            BatchNorm1d(num_features=self.n_feature_maps * 2)
            )

        # no need to expand channels because they are equal
        self.shortcut_block_3 = BatchNorm1d(num_features=self.n_feature_maps * 2)

    def forward(self, x):
        output_block_1 = self.conv_block_1(x) + self.shortcut_block_1(x)
        output_block_1 = ReLU()(output_block_1)

        output_block_2 = self.conv_block_2(output_block_1) + self.shortcut_block_2(output_block_1)
        output_block_2 = ReLU()(output_block_2)

        output_block_3 = self.conv_block_3(output_block_2) + self.shortcut_block_3(output_block_2)
        output_block_3 = ReLU()(output_block_3)

        # output_gap_layer = AvgPool1d((self.input_shape[1], 1))(output_block_3).squeeze()
        output_gap_layer = AvgPool1d(kernel_size=output_block_3.shape[2], stride=1)(output_block_3).squeeze()

        # del output_block_1, output_block_2, output_block_3, gap_layer # release memory
        
        return output_gap_layer
    

################# LSTM - FCN ####################
class LSTM(torch.nn.Module):
    
    def __init__(self, input_length, units, dropout):
    
        '''
        Parameters:
        __________________________________
        input_length: int.
            Length of the time series.
        units: list of int.
            The length of the list corresponds to the number of recurrent blocks, the items in the
            list are the number of units of the LSTM layer in each block.
        dropout: float.
            Dropout rate to be applied after each recurrent block.
        '''
        
        super(LSTM, self).__init__()
        
        # check the inputs
        if type(units) != list:
            raise ValueError(f'The number of units should be provided as a list.')
        
        # build the model
        modules = OrderedDict()
        for i in range(len(units)):
            modules[f'LSTM_{i}'] = torch.nn.LSTM(
                input_size=input_length if i == 0 else units[i - 1],
                hidden_size=units[i],
                batch_first=True
            )
            modules[f'Lambda_{i}'] = Lambda(f=lambda x: x[0])
            if i < len(units) - 1:
                modules[f'Dropout_{i}'] = torch.nn.Dropout(p=dropout)
        self.model = torch.nn.Sequential(modules)
    
    def forward(self, x):
        
        '''
        Parameters:
        __________________________________
        x: torch.Tensor.
            Time series, tensor with shape (samples, 1, length) where samples is the number of
            time series and length is the length of the time series.
        '''
        
        return self.model(x)[:, -1, :]


class FCN(torch.nn.Module):
    
    def __init__(self, filters, kernel_sizes):
    
        '''
        Parameters:
        __________________________________
        filters: list of int.
            The length of the list corresponds to the number of convolutional blocks, the items in the
            list are the number of filters (or channels) of the convolutional layer in each block.
        kernel_sizes: list of int.
            The length of the list corresponds to the number of convolutional blocks, the items in the
            list are the kernel sizes of the convolutional layer in each block.
        '''
        
        super(FCN, self).__init__()
        
        # check the inputs
        if len(filters) == len(kernel_sizes)+1:
            blocks = len(filters)-1
        else:
            raise ValueError(f'The number of filters and kernel sizes must be the same.')

        # build the model
        modules = OrderedDict()
        for i in range(blocks):
            modules[f'Conv1d_{i}'] = torch.nn.Conv1d(
                in_channels=filters[i],
                out_channels=filters[i+1],
                kernel_size=(kernel_sizes[i],),
                padding='same'
            )
            modules[f'BatchNorm1d_{i}'] = torch.nn.BatchNorm1d(
                num_features=filters[i+1],
                eps=0.001,
                momentum=0.99
            )
            modules[f'ReLU_{i}'] = torch.nn.ReLU()
        self.model = torch.nn.Sequential(modules)
        
    def forward(self, x):
        
        '''
        Parameters:
        __________________________________
        x: torch.Tensor.
            Time series, tensor with shape (samples, 1, length) where samples is the number of
            time series and length is the length of the time series.
        '''
        
        return torch.mean(self.model(x), dim=-1)


class LSTM_FCN(torch.nn.Module):
    
    def __init__(self, input_length, 
                        units, 
                        dropout, 
                        filters, 
                        kernel_sizes):
        
        '''
        Parameters:
        __________________________________
        input_length: int.
            Length of the time series.
        units: list of int.
            The length of the list corresponds to the number of recurrent blocks, the items in the
            list are the number of units of the LSTM layer in each block.
        dropout: float.
            Dropout rate to be applied after each recurrent block.
        filters: list of int.
            The length of the list corresponds to the number of convolutional blocks, the items in the
            list are the number of filters (or channels) of the convolutional layer in each block.
        kernel_sizes: list of int.
            The length of the list corresponds to the number of convolutional blocks, the items in the
            list are the kernel sizes of the convolutional layer in each block.
        num_classes: int.
            Number of classes.
        '''
        
        super(LSTM_FCN, self).__init__()
        
        self.fcn = FCN(filters, kernel_sizes)
        self.lstm = LSTM(input_length, units, dropout)
        #self.linear = torch.nn.Linear(in_features=filters[-1] + units[-1], out_features=num_classes)
    
    def forward(self, x):
        
        '''
        Parameters:
        __________________________________
        x: torch.Tensor.
            Time series, tensor with shape (samples, length) where samples is the number of time series
            and length is the length of the time series.
        
        Returns:
        __________________________________
        y: torch.Tensor.
            Logits, tensor with shape (samples, num_classes) where samples is the number of time series
            and num_classes is the number of classes.
        '''
        
        #x = torch.reshape(x, (x.shape[0], 1, x.shape[1]))
        y = torch.concat((self.fcn(x), self.lstm(x)), dim=-1)
        
        return y

################# PatchTST ####################

class PatchTST(torch.nn.Module):
    """
    Paper link: https://arxiv.org/pdf/2211.14730.pdf
    """

    def __init__(self, 
                 seq_len:int=512, 
                 patch_len:int=16, 
                 stride:int=8, 
                 d_model:int=512, 
                 dropout:float=0.1,
                 output_attention:bool=False,
                 n_heads:int=8,
                 d_ff:int=2048,
                 activation:str='gelu',
                 e_layers:int=2):
        """
        """
        super().__init__()
        self.seq_len = seq_len
        padding = stride

        # patching and embedding
        self.patch_embedding = PatchEmbedding(d_model, patch_len, 
                                              stride, padding, dropout)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(attention_dropout=dropout, output_attention=output_attention), 
                        d_model, n_heads),
                    d_model, d_ff, dropout=dropout, activation=activation
                ) for l in range(e_layers)
            ],
            norm_layer=LayerNorm(d_model)
        )
   
        self.flatten = Flatten(start_dim=-2)
        self.dropout = Dropout(dropout)
    
    def forward(self, x_enc):
        # Normalization 
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # Patching and Embedding
        x_enc = x_enc.permute(0, 2, 1)
        enc_out, n_vars = self.patch_embedding(x_enc)

        # Encoder
        enc_out, _ = self.encoder(enc_out)
        # z: [batch_size x nvars x patch_num x d_model]
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        # z: [batch_size x nvars x d_model x patch_num]
        enc_out = enc_out.permute(0, 1, 3, 2)

        # Decoder
        output = self.flatten(enc_out)
        output = self.dropout(output)
        output = output.reshape(output.shape[0], -1)
        return output

class Lambda(torch.nn.Module):
    
    def __init__(self, f):
        super(Lambda, self).__init__()
        self.f = f
    
    def forward(self, x):
        return self.f(x)
    
class TimeSeriesNet(Module):
    def __init__(self, model_type, output_dim, in_channels=2, **kwargs) -> None:
        super(TimeSeriesNet, self).__init__()
        self.output_dim = output_dim
        self.in_channels = in_channels
        self.model, self.linear = self.__get_model_type(model_type, **kwargs)

    def __get_model_type(self, model_type, **kwargs):
        if model_type == 'resnet1d':
            return ResNet1D(input_shape=((self.in_channels, 0)),
                            **kwargs), Linear(kwargs['n_feature_maps']*2, self.output_dim)
        elif model_type == 'fcn':
            kwargs['filters'][0] = self.in_channels
            return LSTM_FCN(input_length=kwargs['input_length'],
                            units=kwargs['units'],
                            dropout=kwargs['dropout'],
                            filters=kwargs['filters'],
                            kernel_sizes=kwargs['kernel_sizes']), Linear(in_features=kwargs['filters'][-1] + kwargs['units'][-1], out_features=self.output_dim)
        elif model_type == 'patchtst':
            head_nf = kwargs['d_model'] * int((kwargs['seq_len'] - kwargs['patch_len']) / kwargs['stride'] + 2)
            return PatchTST(seq_len=kwargs['seq_len'], 
                            patch_len=kwargs['patch_len'],
                            stride=kwargs['stride'],
                            d_model=kwargs['d_model'],
                            dropout=kwargs['dropout'],
                            output_attention=kwargs['output_attention'],
                            n_heads=kwargs['n_heads'],
                            d_ff=kwargs['d_ff'],
                            activation=kwargs['activation'],
                            e_layers=kwargs['e_layers']), Linear(head_nf * self.in_channels, self.output_dim) 
        else:
            raise NotImplementedError(f"Given model type: {model_type} is not supported. Currently supported methods are: {'resnet1d'}")
        
    def forward(self, x, return_feats=False, **kwargs):
        feats = self.model(x)
        x = self.linear(feats)

        if x.shape[0] != feats.shape[0]:
            x = x.unsqueeze(0)
        if not return_feats:
            return x
        else:
            return x, feats