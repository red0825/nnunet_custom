import torch
import torch.nn as nn
import torch.nn.functional as F
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.loss.robust_ce_loss import RobustCrossEntropyLoss
import numpy as np

class WeightedBCELoss2d(nn.Module):
    def __init__(self, **kwargs):
        super(WeightedBCELoss2d, self).__init__()

    def forward(self, logits, labels, weights):
        w = weights.view(-1)
        z = logits.contiguous().view(-1)
        t = labels.view(-1)
        loss = w * z.clamp(min=0) - w * z * t + w * torch.log(1 + torch.exp(-z.abs()))
        loss = loss.sum() / w.sum()
        return loss

class WeightedSoftDiceLoss(nn.Module):
    def __init__(self, **kwargs):
        super(WeightedSoftDiceLoss, self).__init__()

    def forward(self, logits, labels, weights):
        probs = torch.sigmoid(logits)
        num = labels.size(0)
        w = (weights).view(num, -1)
        w2 = w * w
        m1 = (probs).view(num, -1)
        m2 = (labels).view(num, -1)
        intersection = (m1 * m2)
        smooth = 1.
        score = 2. * ((w2 * intersection).sum(1) + smooth) / ((w2 * m1).sum(1) + (w2 * m2).sum(1) + smooth)
        score = 1 - score.sum() / num
        return score

class CustomLoss(nn.Module):
    def __init__(self, kernel_size=3, **kwargs):
        super(CustomLoss, self).__init__()
        self.criterion1 = nn.BCEWithLogitsLoss()
        self.criterion2 = SoftCustomDiceLoss()
        self.criterion3 = nn.CrossEntropyLoss()
        self.bce = WeightedBCELoss2d()
        self.dice = WeightedSoftDiceLoss()
        
        self.kernel_size = kernel_size

    def forward(self, logits, labels):
        
        loss = 0.5 * self.criterion3(logits, labels)
        
        for current_class in labels.unique():
            tmp_loss = 0
            tmp_mask = 1 - (labels != current_class) * 1.0
            
            mask_pool = F.avg_pool2d(labels, kernel_size=self.kernel_size, padding=self.kernel_size // 2, stride=1)
            ind = mask_pool.ge(0.01) * mask_pool.le(0.99)
            ind = ind.float()
            weights = torch.ones(mask_pool.size()).to(device=logits.device)

            w0 = weights.sum()
            weights = weights + ind * 2
            w1 = weights.sum()
            weights = weights / w1 * w0
            
            tmp_loss += (0.5 * self.criterion1(logits[:, current_class, :, :], tmp_mask))
            tmp_loss += (0.2 * self.criterion2(logits[:, current_class, :, :], tmp_mask))
            tmp_loss += (0.2 * (self.bce(logits[:, current_class, :, :], tmp_mask, weights) + self.dice(logits[:, current_class, :, :], tmp_mask, weights)))
            loss += (tmp_loss / len(labels.unique()))

        return loss
    

class SoftCustomDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(SoftCustomDiceLoss, self).__init__()

    def forward(self, logits, targets):
        smooth = 1e-5
        logits = torch.sigmoid(logits)
        iflat = 1 - logits.view(-1)
        tflat = 1 - targets.view(-1)
        intersection = (iflat * tflat).sum()
        
        logits = torch.sigmoid(logits)
        iflat = logits.view(-1)
        tflat = targets.view(-1)
        intersection = (iflat * tflat).sum()

        return 1 - 2 * ((2. * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth))


class nnUNetTrainerCustomLoss(nnUNetTrainer):
    def _build_loss(self):
        assert not self.label_manager.has_regions, "regions not supported by this trainer"
        loss = CustomLoss()

        # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
        # this gives higher resolution outputs more weight in the loss
        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2**i) for i in range(len(deep_supervision_scales))])
            weights[-1] = 0

            # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
            weights = weights / weights.sum()
            # now wrap the loss
            loss = DeepSupervisionWrapper(loss, weights)
        return loss


class nnUNetTrainerCustomLoss_250epochs(nnUNetTrainerCustomLoss):
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        unpack_dataset: bool = True,
        device: torch.device = torch.device("cuda"),
    ):
        """used for debugging plans etc"""
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 250
