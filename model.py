import torch
import torchvision
from collections import OrderedDict
from torch import nn
from torchinfo import summary

model = torchvision.models.resnet18()

# print(model)
# print(summary(model,(1, 3, 224, 224)))
# num_fc_ftr = model.fc.in_features
# model.fc = torch.nn.Linear(num_fc_ftr, 100)
# print(summary(model, (1, 3, 224, 224)))


def initialize_weights(model):
    for m in model.modules():
        # 判断是否属于Conv2d
        if isinstance(m, nn.Conv2d):
            torch.nn.init.kaiming_normal_(m.weight.data)
            # 判断是否有偏置
            if m.bias is not None:
                torch.nn.init.constant_(m.bias.data, 0.3)
        elif isinstance(m, nn.Linear):
            torch.nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias.data)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            torch.nn.init.constant_(m.bias.data, 0.1)


# model.apply(initialize_weights)
# print(model.state_dict())
# save_path = "./model/initial_model.pth"
# torch.save(model.state_dict(), save_path)
# print("Successfully save the model at ", save_path)


class IntermediateLayerGetter(nn.ModuleDict):
    """
    Module wrapper that returns intermediate layers from a model
    It has a strong assumption that the modules have been registered
    into the model in the same order as they are used.
    This means that one should **not** reuse the same nn.Module
    twice in the forward if you want this to work.
    Additionally, it is only able to query submodules that are directly
    assigned to the model. So if `model` is passed, `model.feature1` can
    be returned, but not `model.feature1.layer2`.
    Arguments:
        model (nn.Module): model on which we will extract the features
        return_layers (Dict[name, new_name]): a dict containing the names
            of the modules for which the activations will be returned as
            the key of the dict, and the value of the dict is the name
            of the returned activation (which the user can specify)."""
    """https://zhuanlan.zhihu.com/p/341176618"""

    def __init__(self, model, return_layers):
        if not set(return_layers).issubset([name for name, _ in model.named_children()]):
            raise ValueError("return_layers are not present in model")

        orig_return_layers = return_layers
        return_layers = {k: v for k, v in return_layers.items()}
        layers = OrderedDict()
        for name, module in model.named_children():
            layers[name] = module
            if name in return_layers:
                del return_layers[name]
            if not return_layers:
                break

        super(IntermediateLayerGetter, self).__init__(layers)
        self.return_layers = orig_return_layers

    def forward(self, x):
        out = OrderedDict()
        for name, module in self.named_children():
            x = module(x)
            if name in self.return_layers:
                out_name = self.return_layers[name]
                out[out_name] = x
        return out


hidden_output = torchvision.models._utils.IntermediateLayerGetter(model, {'layer1': 'feat1', 'layer2': 'feat2',
                                                                          'layer3': 'feat3', 'layer4': 'feat4'})
out = hidden_output(torch.rand(1, 3, 224, 224))
# if __name__=='__main__':
# print([(k, v.shape) for k, v in out.items()])
# print(out["feat1"])
#  featx is the returned feature map
# ('feat1', torch.Size([1, 64, 56, 56]))
# ('feat2', torch.Size([1, 128, 28, 28]))
# ('feat3', torch.Size([1, 256, 14, 14]))
# ('feat4', torch.Size([1, 512, 7, 7]))]
