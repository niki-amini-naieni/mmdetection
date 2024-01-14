import torch
import torch.nn as nn

from mmdet.registry import MODELS
from mmdet.models.task_modules.assigners.match_cost import TASK_UTILS, BaseMatchCost, bbox_xyxy_to_cxcywh, InstanceData, Optional, Tensor, Union
from .utils import weighted_loss

@weighted_loss
def l1_loss(pred: Tensor, target: Tensor) -> Tensor:
    """L1 loss.

    Args:
        pred (Tensor): The prediction.
        target (Tensor): The learning target of the prediction.

    Returns:
        Tensor: Calculated loss
    """
    if target.numel() == 0:
        return pred.sum() * 0

    assert pred.size() == target.size()
    loss = torch.abs(pred - target)
    return loss

@MODELS.register_module()
class LocLoss(nn.Module):
    """L1 loss. **Expects bboxes in cxcywh format**

    Args:
        reduction (str, optional): The method to reduce the loss.
            Options are "none", "mean" and "sum".
        loss_weight (float, optional): The weight of loss.
    """

    def __init__(self,
                 reduction: str = 'mean',
                 loss_weight: float = 1.0) -> None:
        super().__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred: Tensor,
                target: Tensor,
                weight: Optional[Tensor] = None,
                avg_factor: Optional[int] = None,
                reduction_override: Optional[str] = None) -> Tensor:
        """Forward function.

        Args:
            pred (Tensor): The prediction.
            target (Tensor): The learning target of the prediction.
            weight (Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.

        Returns:
            Tensor: Calculated loss
        """
        if weight is not None and not torch.any(weight > 0):
            if pred.dim() == weight.dim() + 1:
                weight = weight.unsqueeze(1)
            return (pred * weight).sum()
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss_bbox = self.loss_weight * l1_loss(
            pred[:, :2], target[:, :2], weight[:, :2], reduction=reduction, avg_factor=avg_factor)
        return loss_bbox
    
@TASK_UTILS.register_module()
class LocCost(BaseMatchCost):
    """BBoxL1Cost.

    Note: ``bboxes`` in ``InstanceData`` passed in is of format 'xyxy'
    and its coordinates are unnormalized.

    Args:
        box_format (str, optional): 'xyxy' for DETR, 'xywh' for Sparse_RCNN.
            Defaults to 'xyxy'.
        weight (Union[float, int]): Cost weight. Defaults to 1.

    Examples:
        >>> from mmdet.models.task_modules.assigners.
        ... match_costs.match_cost import BBoxL1Cost
        >>> import torch
        >>> self = BBoxL1Cost()
        >>> bbox_pred = torch.rand(1, 4)
        >>> gt_bboxes= torch.FloatTensor([[0, 0, 2, 4], [1, 2, 3, 4]])
        >>> factor = torch.tensor([10, 8, 10, 8])
        >>> self(bbox_pred, gt_bboxes, factor)
        tensor([[1.6172, 1.6422]])
    """

    def __init__(self,
                 box_format: str = 'xyxy',
                 weight: Union[float, int] = 1.) -> None:
        super().__init__(weight=weight)
        assert box_format in ['xyxy', 'xywh']
        self.box_format = box_format

    def __call__(self,
                 pred_instances: InstanceData,
                 gt_instances: InstanceData,
                 img_meta: Optional[dict] = None,
                 **kwargs) -> Tensor:
        """Compute match cost.

        Args:
            pred_instances (:obj:`InstanceData`): ``bboxes`` inside is
                predicted boxes with unnormalized coordinate
                (x, y, x, y).
            gt_instances (:obj:`InstanceData`): ``bboxes`` inside is gt
                bboxes with unnormalized coordinate (x, y, x, y).
            img_meta (Optional[dict]): Image information. Defaults to None.

        Returns:
            Tensor: Match Cost matrix of shape (num_preds, num_gts).
        """
        pred_bboxes = pred_instances.bboxes
        gt_bboxes = gt_instances.bboxes

        # convert box format
        if self.box_format == 'xywh':
            gt_bboxes = bbox_xyxy_to_cxcywh(gt_bboxes)
            pred_bboxes = bbox_xyxy_to_cxcywh(pred_bboxes)

        # normalized
        img_h, img_w = img_meta['img_shape']
        factor = gt_bboxes.new_tensor([img_w, img_h, img_w,
                                       img_h]).unsqueeze(0)
        gt_bboxes = gt_bboxes / factor
        pred_bboxes = pred_bboxes / factor

        bbox_cost = torch.cdist(pred_bboxes[:, :2], gt_bboxes[:, :2], p=1)
        return bbox_cost * self.weight