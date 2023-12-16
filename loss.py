from torch import nn
import torch.nn.functional as F
"""data:[final_output, feat1, feat2, feat3, feat4]"""


def combine_loss(feat, label, final_feat=None, T = 5, lamda=0.8):
    if final_feat != None:
        loss_h = nn.CrossEntropyLoss(reduction="mean")
        loss_hard = loss_h(feat, label)
        # CrossEntropyLoss 自带Softmax
        feat_pro = F.log_softmax(feat / T, dim = 1)
        final_feat_pro = F.softmax(final_feat / T, dim = 1)
        loss_s = nn.KLDivLoss(reduction="mean")
        loss_soft = loss_s(feat_pro, final_feat_pro)
        # KLDivLoss 不自带Softmax
        sum_loss = (1.0 - lamda) * loss_soft + lamda * loss_hard
    else:
        loss_h = nn.CrossEntropyLoss(reduction="mean")
        sum_loss = loss_h(feat, label)
    return sum_loss


def total_loss(data, label):
    sum_loss = 0
    for i in range(1, 5):
        sum_loss += combine_loss(data[i], label, final_feat=data[0])
    sum_loss += combine_loss(data[0], label)
    return sum_loss
