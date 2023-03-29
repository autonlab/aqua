import torch
from os.path import join
from torch.nn import Module, Conv1d, BatchNorm1d, ReLU, Sequential, Softmax, AvgPool1d, Linear, CrossEntropyLoss
from typing import Tuple, Union, Callable, Optional

class ResNet1D(Module):
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
        
    def forward(self, x, return_feats=False, **kwargs):
        feats = self.model(x)
        x = self.linear(feats)
        if not return_feats:
            return x
        else:
            return x, feats