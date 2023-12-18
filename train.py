import torch
from loss import total_loss
from torch import nn
from dataset import *
from model import initialize_weights
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

"""Hyper-parameter"""
batch_size = 64
lr = 0.02
weight_decay = 0.05
num_workers = 2
num_epochs = 10

"""Data load"""
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, num_workers=num_workers, shuffle=True,
                                           drop_last=True)

"""Backbone Model load"""
save_path = "./model/initial_model.pth"  # Initialized Resnet18 from model.py
model = torchvision.models.resnet18()
num_fc_ftr = model.fc.in_features
model.fc = torch.nn.Linear(num_fc_ftr, 100)
model.load_state_dict(torch.load(save_path))

"""BottleNeck Block"""


class BottleNeck(nn.Module):
    def __init__(self, in_channel, size, **kwargs):
        super(BottleNeck, self).__init__(**kwargs)
        self.in_channel = in_channel
        self.size = size
        mid_channel = int(self.in_channel / 2)
        self.bottlenet = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channel, out_channels=mid_channel, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(mid_channel),
            nn.ReLU(),
            nn.Conv2d(in_channels=mid_channel, out_channels=mid_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(mid_channel),
            nn.ReLU(),
            nn.Conv2d(in_channels=mid_channel, out_channels=self.in_channel, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.in_channel),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(self.in_channel * self.size * self.size, 100)
        )
        self.bottlenet.apply(initialize_weights)

    def forward(self, x):
        return self.bottlenet(x)


# 模型多输出方法
# https://geek-docs.com/pytorch/pytorch-questions/252_pytorch_how_can_i_make_a_neural_network_that_has_multiple_outputs_using_pytorch.html

"""Ensemble model"""


class Self_Distill(nn.Module):
    def __init__(self, model, **kwargs):
        super(Self_Distill, self).__init__(**kwargs)
        self.model = model
        self.feat1_bottle = BottleNeck(64, 56)
        self.feat2_bottle = BottleNeck(128, 28)
        self.feat3_bottle = BottleNeck(256, 14)
        self.feat4_bottle = BottleNeck(512, 7)

    def forward(self, x):
        hidden_output = torchvision.models._utils.IntermediateLayerGetter(self.model, {'layer1': 'feat1', 'layer2': 'feat2',
                                                             'layer3': 'feat3', 'layer4': 'feat4'})
        feat1 = hidden_output(x)["feat1"]
        feat2 = hidden_output(x)["feat2"]
        feat3 = hidden_output(x)["feat3"]
        feat4 = hidden_output(x)["feat4"]

        final_output = self.model(x)
        feat1 = self.feat1_bottle(feat1)
        feat2 = self.feat2_bottle(feat2)
        feat3 = self.feat3_bottle(feat3)
        feat4 = self.feat4_bottle(feat4)
        return [final_output, feat1, feat2, feat3, feat4]


def try_gpu(i=0):
    """如果存在，则返回gpu(i)，否则返回cpu()"""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


ensemble_model = Self_Distill(model)
# print(ensemble_model)
# ensemble_model.load_state_dict(torch.load("./model/experiment1/ensemble_model6.pth", map_location=try_gpu()))
optimizer = torch.optim.SGD(ensemble_model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)

ensemble_model = ensemble_model.to(device=try_gpu())


def train(epoch, model):
    model.train()
    train_loss = 0
    batch = 1
    writer = SummaryWriter("Visualize Training")
    # 命令行参数 tensorboard --logdir="Visualize Training"
    for data, label in train_loader:
        data, label = data.to(device=try_gpu()), label.to(device=try_gpu())
        optimizer.zero_grad()
        output = model(data)
        loss = total_loss(output, label)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * data.size(0)
        print('Batch: {} \tTraining Loss: {:.6f}'.format(batch, loss.item() * data.size(0)))
        writer.add_scalar("Epoch{} Training Loss".format(epoch), scalar_value=loss.item() * data.size(0), global_step=batch)
        batch += 1

    print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch, train_loss))
    torch.save(ensemble_model.state_dict(), "./model/experiment2/ensemble_model%d.pth" % epoch)


if __name__ == '__main__':
    for i in range(1, num_epochs + 1):
        train(i, ensemble_model)
