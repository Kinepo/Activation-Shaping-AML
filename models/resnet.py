import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class BaseResNet18(nn.Module):
    def __init__(self):
        super(BaseResNet18, self).__init__()
        self.resnet = resnet18(weights=ResNet18_Weights)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 7)

    def forward(self, x):
        return self.resnet(x)

# Multiply A (output of layer) and M and binarize (step 1+2)
def activation_shaping(layer):
    def activation_shaping_hook(module, input, output):
        # attention random M à modifier
        M = torch.randint(0,2,(output.size()[0], output.size()[1]))
        Z = torch.mul(output, M)
        for i in range(Z.size()[0]):
            for j in range(Z.size()[1]):
                if Z[i][j] != 0.0:
                    Z[i][j] = 1.0
        return Z
    
# Attach hook (activation_shaping_hook)
def call_activation_shaping_hook(self):
    layers = {}

    # Get all conv layers
    # à tester et modifier
    for name, layer in self.named_modules():
       if isinstance(layer, nn.Conv2d):
           layers[name] = layer

    # Every 3 convolutions          
    # for i in range (0, len(layers), 3):
           
    # Every convolution
    for layer in layers:
        Z = self.register_forward_hook(activation_shaping(layer))
        
    # pensez à detacher le hook
    return None

class ASHResNet18(nn.Module):
    def __init__(self):
        super(ASHResNet18, self).__init__()
        self.resnet = resnet18(weights=ResNet18_Weights)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 7)

    def forward(self, x):
        call_activation_shaping_hook(self)
        return self.resnet(x)

######################################################
# TODO: either define the Activation Shaping Module as a nn.Module
# class ActivationShapingModule(nn.Module):
# ...
#
# OR as a function that shall be hooked via 'register_forward_hook'
# def activation_shaping_hook(module, input, output):
# ...
#
######################################################
# TODO: modify 'BaseResNet18' including the Activation Shaping Module
# class ASHResNet18(nn.Module):
#    def __init__(self):
#        super(ASHResNet18, self).__init__()
#        ...
#    
#    def forward(self, x):
#        ...
#
######################################################