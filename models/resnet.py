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



class ASHResNet18_DA(nn.Module):
    def __init__(self):
        super(ASHResNet18_DA, self).__init__()
        self.resnet = resnet18(weights=ResNet18_Weights)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 7)

    def point_3(self, targ_x):
        def hook_1(module, input, output):
            global machin
            machin = output

        layer = self.resnet.layer4[0].bn1
        hook = layer.register_forward_hook(hook_1)
        self.resnet(targ_x)
        hook.remove()

        def hook_2(module, input, output):
            M = torch.where(machin > 0, 1.0, 0.0)
            output = torch.where(output > 0, 1.0, 0.0) * M
            hook.remove()
            return output

        hook = layer.register_forward_hook(hook_2)
        # penser à detacher le hook si on part sur du multy layer
        return None

    def forward(self, x, targ_x=None):
        if targ_x is not None:
            self.point_3(targ_x)
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