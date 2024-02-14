import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from globals import CONFIG


class BaseResNet18(nn.Module):
    def __init__(self):
        super(BaseResNet18, self).__init__()
        self.resnet = resnet18(weights=ResNet18_Weights)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 7)

    def forward(self, x):
        return self.resnet(x)


class ASHResNet18(nn.Module):
    def __init__(self):
        super(ASHResNet18, self).__init__()
        self.resnet = resnet18(weights=ResNet18_Weights)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 7)

    def point_2(self):
        def hook_2(module, input, output):
            M = torch.randint(0, 2, output.size(), device=CONFIG.device)
            Z = torch.mul(output, M)
            Z = torch.where(Z > 0, 1.0, 0.0)
            hook.remove()
            return Z

        match CONFIG.num_layer:
            case 1:
                layer = self.resnet.layer1[CONFIG.num_block]
                if CONFIG.num_bn == 1:
                    hook = layer.bn1.register_forward_hook(hook_2)
                else:
                    hook = layer.bn2.register_forward_hook(hook_2)
            case 2:
                layer = self.resnet.layer2[CONFIG.num_block]
                if CONFIG.num_bn == 1:
                    hook = layer.bn1.register_forward_hook(hook_2)
                else:
                    hook = layer.bn2.register_forward_hook(hook_2)
            case 3:
                layer = self.resnet.layer3[CONFIG.num_block]
                if CONFIG.num_bn == 1:
                    hook = layer.bn1.register_forward_hook(hook_2)
                else:
                    hook = layer.bn2.register_forward_hook(hook_2)
            case 4:
                layer = self.resnet.layer4[CONFIG.num_block]
                if CONFIG.num_bn == 1:
                    hook = layer.bn1.register_forward_hook(hook_2)
                else:
                    hook = layer.bn2.register_forward_hook(hook_2)
            case _:
                layer = self.resnet.layer4[0].bn1
                hook = layer.register_forward_hook(hook_2)

        return None

    def forward(self, x):
        self.point_2()
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
