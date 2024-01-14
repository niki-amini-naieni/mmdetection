import torch
import torch.nn as nn

from mmdet.registry import MODELS
from .utils import weighted_loss

@weighted_loss
def my_loss(pred, target):
    assert pred.size() == target.size() and target.numel() > 0
    box_centers_x_pred = pred[:, 0] + pred[:, 2] / 2
    box_centers_y_pred = pred[:, 1] + pred[:, 3] / 2
    box_centers_x_targ = target[:, 0] + target[:, 2] / 2
    box_centers_y_targ = target[:, 1] + target[:, 3] / 2

    pred = torch.stack([box_centers_x_pred, box_centers_y_pred], dim=-1)
    target = torch.stack([box_centers_x_targ, box_centers_y_targ], dim=-1)
    loss = torch.abs(pred - target)
    return loss

@MODELS.register_module()
class MyLoss(nn.Module):

    def __init__(self, reduction='mean', loss_weight=1.0):
        super(MyLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        assert weight[:, 0] == weight[:, 1] and weight[:, 1] == weight[:, 2] and weight[:, 2] == weight[:, 3]

        weight = weight[:, :2]
        reduction = (
            reduction_override if reduction_override else self.reduction)
        
        loss_bbox = self.loss_weight * my_loss(
            pred, target, weight, reduction=reduction, avg_factor=avg_factor)
        return loss_bbox