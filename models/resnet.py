import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from globals import CONFIG
import ast


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

        global list_of_source
        list_of_source = []
        hook1 = []
        hook2 = []

        def hook_1(module, input, output):
            list_of_source.append(output.clone().detach())
            hook1[-1].remove()
            hook1.pop()

        def hook_2(module, input, output):
            if CONFIG.experiment in ['ASHResNet18_DA']:
                output = torch.mul(torch.where(output > 0, 1.0, 0.0), torch.where(list_of_source[-1] > 0, 1.0, 0.0))
                list_of_source.pop()
                hook2[-1].remove()
                hook2.pop()
                return output
            elif CONFIG.experiment in ['ASHResNet18_DA_BA1']:
                output = output * list_of_source[-1]
                list_of_source.pop()
                hook2[-1].remove()
                hook2.pop()
                return output
            elif CONFIG.experiment in ['ASHResNet18_DA_BA2']:
                output = output * torch.where(
                    torch.where(list_of_source[-1] > 0, 1.0, 0.0) not in
                    torch.topk(output.flatten(), k=CONFIG.hyper_parameter)[0], 1.0, 0.0)
                list_of_source.pop()
                hook2[-1].remove()
                hook2.pop()
                return output

        for _ in reversed(ast.literal_eval(CONFIG.list_layers)):
            match _[0]:
                case 1:
                    layer = self.resnet.layer1[_[1]]
                    if _[2] == 1:
                        hook1.append(layer.bn1.register_forward_hook(hook_1))
                        hook2.append(layer.bn1.register_forward_hook(hook_2))
                    else:
                        hook1.append(layer.bn2.register_forward_hook(hook_1))
                        hook2.append(layer.bn2.register_forward_hook(hook_2))
                case 2:
                    layer = self.resnet.layer2[_[1]]
                    if _[2] == 1:
                        hook1.append(layer.bn1.register_forward_hook(hook_1))
                        hook2.append(layer.bn1.register_forward_hook(hook_2))
                    else:
                        hook1.append(layer.bn2.register_forward_hook(hook_1))
                        hook2.append(layer.bn2.register_forward_hook(hook_2))
                case 3:
                    layer = self.resnet.layer3[_[1]]
                    if _[2] == 1:
                        hook1.append(layer.bn1.register_forward_hook(hook_1))
                        hook2.append(layer.bn1.register_forward_hook(hook_2))
                    else:
                        hook1.append(layer.bn2.register_forward_hook(hook_1))
                        hook2.append(layer.bn2.register_forward_hook(hook_2))
                case 4:
                    layer = self.resnet.layer4[_[1]]
                    if _[2] == 1:
                        hook1.append(layer.bn1.register_forward_hook(hook_1))
                        hook2.append(layer.bn1.register_forward_hook(hook_2))
                    else:
                        hook1.append(layer.bn2.register_forward_hook(hook_1))
                        hook2.append(layer.bn2.register_forward_hook(hook_2))

        with torch.autocast(device_type=CONFIG.device, enabled=False):
            with torch.no_grad():
                self.resnet(targ_x)

        return None

    def forward(self, x, targ_x=None):
        if targ_x is not None:
            self.point_3(targ_x)
        return self.resnet(x)


class ASHResNet18(nn.Module):
    def __init__(self):
        super(ASHResNet18, self).__init__()
        self.resnet = resnet18(weights=ResNet18_Weights)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 7)

    def point_2(self):
        def hook_2(module, input, output):
            if CONFIG.experiment in ['ASHResNet18']:
                output = torch.mul(torch.where(output > 0, 1.0, 0.0),
                                   torch.bernoulli(torch.full(output.size(), fill_value=CONFIG.random_parameter,
                                                              device=CONFIG.device)))
                hook[-1].remove()
                hook.pop()
                return output
            elif CONFIG.experiment in ['ASHResNet18_BA1']:
                output = output * torch.bernoulli(
                    torch.full(output.size(), fill_value=CONFIG.random_parameter, device=CONFIG.device))
                hook[-1].remove()
                hook.pop()
                return output
            elif CONFIG.experiment in ['ASHResNet18_BA2']:
                output = output * torch.where(
                    torch.bernoulli(
                        torch.full(output.size(), fill_value=CONFIG.random_parameter, device=CONFIG.device)) not in
                    torch.topk(output.flatten(), k=CONFIG.hyper_parameter)[
                        0], 1.0, 0.0)
                hook[-1].remove()
                hook.pop()
                return output

        hook = []
        for _ in reversed(ast.literal_eval(CONFIG.list_layers)):
            match _[0]:
                case 1:
                    layer = self.resnet.layer1[_[1]]
                    if _[2] == 1:
                        hook.append(layer.bn1.register_forward_hook(hook_2))
                    else:
                        hook.append(layer.bn2.register_forward_hook(hook_2))
                case 2:
                    layer = self.resnet.layer2[_[1]]
                    if _[2] == 1:
                        hook.append(layer.bn1.register_forward_hook(hook_2))
                    else:
                        hook.append(layer.bn2.register_forward_hook(hook_2))
                case 3:
                    layer = self.resnet.layer3[_[1]]
                    if _[2] == 1:
                        hook.append(layer.bn1.register_forward_hook(hook_2))
                    else:
                        hook.append(layer.bn2.register_forward_hook(hook_2))
                case 4:
                    layer = self.resnet.layer4[_[1]]
                    if _[2] == 1:
                        hook.append(layer.bn1.register_forward_hook(hook_2))
                    else:
                        hook.append(layer.bn2.register_forward_hook(hook_2))
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
