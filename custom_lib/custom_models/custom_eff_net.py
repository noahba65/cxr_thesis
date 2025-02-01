import torch
from torch import nn
from math import ceil
import re



def define_custom_eff_net(efficient_net_config, num_classes, model_name, device):
    """
    Defines and initializes a custom EfficientNet model based on the specified configuration.

    Args:
        efficient_net_config (dict): A dictionary mapping version keys (e.g., "b0", "b1") to tuples 
            that define model parameters. The tuple format must be:
                (width_mult: float, depth_mult: float, resolution: int, dropout_rate: float)
            Example:
                {
                    "b0": (1.0, 1.0, 224, 0.2),
                    "b1": (1.0, 1.1, 240, 0.2),
                    "b2": (1.1, 1.2, 260, 0.3),
                    ...
                }
        num_classes (int): The number of output classes for the model.
        model_name (str): The name of the model, including the version number 
            (e.g., "custom_b0"). The version (e.g., "b0") must match a key in `efficient_net_config`.
        device (torch.device): The device (e.g., "cpu" or "cuda") to which the model will be sent.

    Raises:
        ValueError: If the version extracted from `model_name` does not exist in `efficient_net_config`.

    Returns:
        torch.nn.Module: The initialized EfficientNet model.
    """
    
    # Create list of accepted models
    accepted_models = list(efficient_net_config.keys())

    # Regex to extract the letter and number after "custom_" for the version number
    match = re.search(r"([a-zA-Z]\d+)", model_name)
    version = match.group(1)

    # Check to see if the version is accepted. If not throw an error listing the accepted models
    if version not in accepted_models:
        raise ValueError(f"Version '{version}' is not an accepted version number. Please use a version from the following: {accepted_models}.")

    # Define custom model parameters to be passed to the custom efficientnet function
    width_mult, depth_mult, res, dropout_rate = efficient_net_config[version]

    # Define model with custom efficientnet function 
    model = customEfficientNet(width_mult, depth_mult, dropout_rate, num_classes=num_classes)
    model.to(device)

    return model


class ConvBnAct(nn.Module):
    
    def __init__(self, n_in, n_out, kernel_size = 3, stride = 1, 
                 padding = 0, groups = 1, bn = True, act = True,
                 bias = False
                ):
        
        super(ConvBnAct, self).__init__()
        
        self.conv = nn.Conv2d(n_in, n_out, kernel_size = kernel_size,
                              stride = stride, padding = padding,
                              groups = groups, bias = bias
                             )
        self.batch_norm = nn.BatchNorm2d(n_out) if bn else nn.Identity()
        self.activation = nn.SiLU() if act else nn.Identity()
        
    def forward(self, x):
        
        x = self.conv(x)

        # Skip BatchNorm if spatial size is 1x1
        if x.shape[-1] > 1 and x.shape[-2] > 1:
            x = self.batch_norm(x)
        x = self.activation(x)
        
        return x
    

class SpatialSeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 3, stride=1, padding=0, bias=True,
                 bn = True, act = True):
        super(SpatialSeparableConv2d, self).__init__()

        self.convk1 = nn.Conv2d(in_channels, out_channels, (kernel_size, 1), 
                                stride=(stride, 1), padding=(padding, 0), bias=True)
        self.conv1k = nn.Conv2d(in_channels, out_channels, (1, kernel_size), 
                        stride=(1, stride), padding=(0, padding), bias=True)

        self.batch_norm = nn.BatchNorm2d(out_channels) if bn else nn.Identity()
        self.activation = nn.SiLU() if act else nn.Identity()

    def forward(self, x):
        x = self.convk1(x)
        x = self.conv1k(x)
        x = self.batch_norm(x)
        x = self.activation(x)

        return x
#------------------------------------------------------------------------------

''' Squeeze and Excitation Block '''

class SqueezeExcitation(nn.Module):
    
    def __init__(self, n_in, reduced_dim):
        super(SqueezeExcitation, self).__init__()
        
        
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(n_in, reduced_dim, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(reduced_dim, n_in, kernel_size=1),
            nn.Sigmoid()
        )
       
    def forward(self, x):
        
        y = self.se(x)
        
        return x * y
                                    
#------------------------------------------------------------------------------

''' Stochastic Depth Module'''

class StochasticDepth(nn.Module):
    
    def __init__(self, survival_prob = 0.8):
        super(StochasticDepth, self).__init__()
        
        self.p =  survival_prob
        
    def forward(self, x):
        
        if not self.training:
            return x
        
        binary_tensor = torch.rand(x.shape[0], 1, 1, 1, device=x.device) < self.p
        
        return torch.div(x, self.p) * binary_tensor
        
#-------------------------------------------------------------------------------

''' Residual Bottleneck Block with Expansion Factor = N as defined in Mobilenet-V2 paper
    with Squeeze and Excitation Block and Stochastic Depth. 
'''

class MBConvN(nn.Module):
    
    def __init__(self, n_in, n_out, kernel_size = 3, 
                 stride = 1, expansion_factor = 6,
                 reduction = 4, # Squeeze and Excitation Block
                 survival_prob = 0.8 # Stochastic Depth
                ):
        
        super(MBConvN, self).__init__()
        
        self.skip_connection = (stride == 1 and n_in == n_out) 
        intermediate_channels = int(n_in * expansion_factor)
        padding = (kernel_size - 1)//2
        reduced_dim = int(n_in//reduction)
        
        self.expand = nn.Identity() if (expansion_factor == 1) else ConvBnAct(n_in, intermediate_channels, kernel_size = 1)
        
        self.depthwise_conv = ConvBnAct(intermediate_channels, intermediate_channels,
                                        kernel_size = kernel_size, stride = stride, 
                                        padding = padding, groups = intermediate_channels
                                       )
        
   

        self.conv1k = nn.Conv2d(intermediate_channels, intermediate_channels, (1, kernel_size), 
                                stride=(1, stride), padding=(0, padding), bias=True)

        self.convk1 = nn.Conv2d(intermediate_channels, intermediate_channels, (kernel_size, 1), 
                        stride=(stride, 1), padding=(padding, 0), bias=True)
 
        
        self.se = SqueezeExcitation(intermediate_channels, reduced_dim = reduced_dim)
        self.pointwise_conv = ConvBnAct(intermediate_channels, n_out, 
                                        kernel_size = 1, act = False
                                       )
        self.drop_layers = StochasticDepth(survival_prob = survival_prob)

        
        
    def forward(self, x):
        
        residual = x

        # print("residual", x.shape)

        
        x = self.expand(x)

        # print("self.expand", x.shape)

        x = self.depthwise_conv(x)

        # print("self.depthwise_conv", x.shape)
        
        x = self.convk1(x)
        
        # # print("self.convka", x.shape)

        x = self.conv1k(x)

        # x = self.depthwise_conv(x)

        # # print("self.conv1k", x.shape)

        # x = self.batch_norm(x)

        # x = self.activation(x)

        x = self.se(x)

        # print("self.se", x.shape)

        x = self.pointwise_conv(x)
        
        # print("self.pointwise_conv", x.shape)

        if self.skip_connection:
            x = self.drop_layers(x)
            x += residual
        
        return x
    

#----------------------------------------------------------------------------------------------

'''Efficient-net Class'''

class customEfficientNet(nn.Module):
    
    '''Generic Efficient net class which takes width multiplier, Depth multiplier, and Survival Prob.'''
    
    def __init__(self, width_mult = 1, depth_mult = 1, 
                 dropout_rate = 0.2, num_classes = 1000):
        super(customEfficientNet, self).__init__()
        
        last_channel = ceil(1280 * width_mult)
        self.features = self._feature_extractor(width_mult, depth_mult, last_channel)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(last_channel, num_classes)
        )
        
    def forward(self, x):
        
        x = self.features(x)
        x = self.avgpool(x)
        x = self.classifier(x.view(x.shape[0], -1))
        
        return x
    
        
    def _feature_extractor(self, width_mult, depth_mult, last_channel):
        
        channels = 4*ceil(int(32*width_mult) / 4)
        layers = [ConvBnAct(3, channels, kernel_size = 3, stride = 2, padding = 1)]
        in_channels = channels
        
        kernels = [3, 3, 5, 3, 5, 5, 3]
        expansions = [1, 6, 6, 6, 6, 6, 6]
        num_channels = [16, 24, 40, 80, 112, 192, 320]
        num_layers = [1, 2, 2, 3, 3, 4, 1]
        strides =[1, 2, 2, 2, 1, 2, 1]
        
        # Scale channels and num_layers according to width and depth multipliers.
        scaled_num_channels = [4*ceil(int(c*width_mult) / 4) for c in num_channels]
        scaled_num_layers = [int(d * depth_mult) for d in num_layers]

        
        for i in range(len(scaled_num_channels)):
             
            layers += [MBConvN(in_channels if repeat==0 else scaled_num_channels[i], 
                               scaled_num_channels[i],
                               kernel_size = kernels[i],
                               stride = strides[i] if repeat==0 else 1, 
                               expansion_factor = expansions[i]
                              )
                       for repeat in range(scaled_num_layers[i])
                      ]
            in_channels = scaled_num_channels[i]
        
        layers.append(ConvBnAct(in_channels, last_channel, kernel_size = 1, stride = 1, padding = 0))
    
        return nn.Sequential(*layers)