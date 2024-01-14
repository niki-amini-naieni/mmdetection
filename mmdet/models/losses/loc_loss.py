import torch
import torch.nn as nn

from mmdet.registry import MODELS
from .utils import weighted_loss

'''
Assumes bbox in format (cx, cy, w, h) as defined below:
def bbox_xyxy_to_cxcywh(bbox: Tensor) -> Tensor:
    """Convert bbox coordinates from (x1, y1, x2, y2) to (cx, cy, w, h).

    Args:
        bbox (Tensor): Shape (n, 4) for bboxes.

    Returns:
        Tensor: Converted bboxes.
    """
    x1, y1, x2, y2 = bbox.split((1, 1, 1, 1), dim=-1)
    bbox_new = [(x1 + x2) / 2, (y1 + y2) / 2, (x2 - x1), (y2 - y1)]
    return torch.cat(bbox_new, dim=-1)
'''

@weighted_loss
def loc_loss(pred, target):
    assert pred.size() == target.size() and target.numel() > 0

    loss = torch.abs(pred[:, :2] - target[:, :2])
    return loss

@MODELS.register_module()
class LocLoss(nn.Module):

    def __init__(self, reduction='mean', loss_weight=1.0):
        super(LocLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        assert reduction_override in (None, 'none', 'mean', 'sum')

        weight = weight[:, :2]
        reduction = (
            reduction_override if reduction_override else self.reduction)
        
        loss_bbox = self.loss_weight * loc_loss(
            pred, target, weight, reduction=reduction, avg_factor=avg_factor)
        return loss_bbox