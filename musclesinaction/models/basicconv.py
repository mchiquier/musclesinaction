
import torch.nn as nn

class BasicConv(nn.Module):
    """BasicBlock 3d block for ResNet3D.
    Args:
        inplanes (int): Number of channels for the input in first conv3d layer.
        planes (int): Number of channels produced by some norm/conv3d layers.
        stride (tuple): Stride is a two element tuple (temporal, spatial). Default: (1, 1).
        downsample (nn.Module | None): Downsample layer. Default: None.
        inflate (bool): Whether to inflate kernel. Default: True.
        conv_cfg (dict): Config dict for convolution layer. Default: 'dict(type='Conv3d')'.
        norm_cfg (dict): Config for norm layers. required keys are 'type'. Default: 'dict(type='BN3d')'.
        act_cfg (dict): Config dict for activation layer. Default: 'dict(type='ReLU')'.
    """
    expansion = 1
    #-B x 1 x 49*2 x 30 —> B x S x 1 x 30 
    def __init__(self, device):
        super().__init__()
        self.device = device

        self.numlayers = 3

        self.conv1 = nn.Conv2d(1,128,(50,9),(1,1),(0,4))
        self.bn1 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128,64,(1,9),(1,1),(0,4))
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64,128,(1,9),(1,1),(0,4))
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128,64,(1,9),(1,1),(0,4))
        self.conv5 = nn.Conv2d(64,8,(1,9),(1,1),(0,4))

        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()
        self.relu4 = nn.ReLU()
        self.relu5 = nn.ReLU()

    def forward(self, x):
        """Defines the computation performed at every call."""

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)

        x = self.conv4(x)
        x = self.relu4(x)

        x = self.conv5(x)
        x = self.relu5(x)

        return x

class OldBasicConv(nn.Module):
    """BasicBlock 3d block for ResNet3D.
    Args:
        inplanes (int): Number of channels for the input in first conv3d layer.
        planes (int): Number of channels produced by some norm/conv3d layers.
        stride (tuple): Stride is a two element tuple (temporal, spatial). Default: (1, 1).
        downsample (nn.Module | None): Downsample layer. Default: None.
        inflate (bool): Whether to inflate kernel. Default: True.
        conv_cfg (dict): Config dict for convolution layer. Default: 'dict(type='Conv3d')'.
        norm_cfg (dict): Config for norm layers. required keys are 'type'. Default: 'dict(type='BN3d')'.
        act_cfg (dict): Config dict for activation layer. Default: 'dict(type='ReLU')'.
    """
    expansion = 1
    #-B x 1 x 49*2 x 30 —> B x S x 1 x 30 
    def __init__(self, device):
        super().__init__()
        self.device = device

        self.numlayers = 3

        self.conv1 = nn.Conv2d(1,32,(13,9),(1,1),(1,4))
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32,64,(13,9),(1,1),(1,4))
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64,128,(13,9),(1,1),(1,4))
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128,64,(13,9),(1,1),(1,4))
        self.conv5 = nn.Conv2d(64,8,(13,9),(1,1),(1,4))

        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()
        self.relu4 = nn.ReLU()
        self.relu5 = nn.ReLU()


    def forward(self, x):
        """Defines the computation performed at every call."""

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)

        x = self.conv4(x)
        x = self.relu4(x)

        x = self.conv5(x)
        x = self.relu5(x)

        return x