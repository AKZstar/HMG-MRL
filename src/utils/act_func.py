import torch.nn as nn


act_dict = {
    'relu': nn.ReLU(),
    'selu': nn.SELU(),
    'prelu': nn.PReLU(),
    'elu': nn.ELU(),
    'lrelu_01': nn.LeakyReLU(negative_slope=0.1),
    'lrelu': nn.LeakyReLU(),
    'lrelu_005': nn.LeakyReLU(negative_slope=0.05),
    'tanh':nn.Tanh()
}