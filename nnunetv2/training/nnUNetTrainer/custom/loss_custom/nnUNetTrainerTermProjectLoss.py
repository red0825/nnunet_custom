import torch
import torch.nn as nn
import torch.nn.functional as F
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.loss.robust_ce_loss import RobustCrossEntropyLoss
import numpy as np

class FocalTverskyLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(FocalTverskyLoss, self).__init__()

    def tversky_index(self, y_pred, y_true):
        smooth = 1e-6
        y_true_pos = y_true.reshape(-1)
        y_pred_pos = y_pred.reshape(-1)
        true_pos = (y_true_pos * y_pred_pos).sum()
        false_neg = (y_true_pos * (1 - y_pred_pos)).sum()
        false_pos = ((1 - y_true_pos) * y_pred_pos).sum()
        alpha = 0.7
        return (true_pos + smooth) / (true_pos + alpha * false_neg + (1 - alpha) * false_pos + smooth)

    def forward(self, logits, labels):
        loss = 0.0
        gamma = 0.75
        logits = torch.sigmoid(logits)  # Only apply sigmoid without converting to long
        labels = labels.long()
        
        for current_class in labels.unique():
            tmp_loss = 0
            tmp_mask = (labels == current_class).float()  # Create mask for the current class
            class_pred = logits[:, current_class, :, :]  # Prediction for the current class
            tmp_loss += torch.pow((1 - self.tversky_index(class_pred, tmp_mask)), gamma)
            loss += (tmp_loss / len(labels.unique()))
        
        return loss

class LogCoshDiceLoss(nn.Module):
    def __init__(self, smooth: float = 1e-6, use_softmax: bool = True):
        """
        saves 1.6 GB on Dataset017 3d_lowres
        """
        super(LogCoshDiceLoss, self).__init__()

        self.smooth = smooth
        self.use_softmax = use_softmax

    def forward(self, x, y):
        if self.use_softmax:
            x = F.softmax(x, dim=1)
        
        shp_x, shp_y = x.shape, y.shape

        # make everything shape (b, c)
        axes = list(range(2, len(shp_x)))

        with torch.no_grad():
            if len(shp_x) != len(shp_y):
                y = y.reshape((shp_y[0], 1, *shp_y[1:]))

            if all([i == j for i, j in zip(shp_x, shp_y)]):
                # if this is the case then gt is probably already a one hot encoding
                y_onehot = y
            else:
                gt = y.long()
                y_onehot = torch.zeros(shp_x, device=x.device, dtype=torch.bool)
                y_onehot.scatter_(1, gt, 1)

            sum_gt = y_onehot.sum(axes)

        intersect = (x * y_onehot).sum(axes)
        sum_pred = x.sum(axes)

        dc = (2 * intersect + self.smooth) / (sum_gt + sum_pred + self.smooth)
        dc = dc.mean()
        return torch.log(torch.cosh(1 - dc))

class CustomDice(nn.Module):
    def __init__(self, smooth: float = 1e-6):
        """
        saves 1.6 GB on Dataset017 3d_lowres
        """
        super(CustomDice, self).__init__()

        self.smooth = smooth

    def forward(self, x, y):
        
        shp_x, shp_y = x.shape, y.shape

        # make everything shape (b, c)
        axes = list(range(2, len(shp_x)))

        with torch.no_grad():
            if len(shp_x) != len(shp_y):
                y = y.reshape((shp_y[0], 1, *shp_y[1:]))

            if all([i == j for i, j in zip(shp_x, shp_y)]):
                # if this is the case then gt is probably already a one hot encoding
                y_onehot = y
            else:
                gt = y.long()
                y_onehot = torch.zeros(shp_x, device=x.device, dtype=torch.bool)
                y_onehot.scatter_(1, gt, 1)

            sum_gt = y_onehot.sum(axes)

        intersect = (x * y_onehot).sum(axes)
        sum_pred = x.sum(axes)

        dc = (2 * intersect + self.smooth) / (sum_gt + sum_pred + self.smooth)
        dc = dc.mean()
        return dc

class ExponentialLogarithmicLoss(nn.Module):
    def __init__(self, dice_kwargs, ce_kwargs, weight_ce=0.2, weight_dice=0.8, gamma=0.3, ignore_label=None, use_softmax=True,
                 class_weights=None):

        super(ExponentialLogarithmicLoss, self).__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.ignore_label = ignore_label
        self.class_weights = class_weights
        self.gamma = gamma
        self.use_softmax = use_softmax

        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        self.dc = CustomDice(**dice_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        
        if self.use_softmax:
            net_output = F.softmax(net_output, dim=1)

        target_dice = target
            
        if self.class_weights is not None:
            # Move class weights to the same device as target
            class_weights = self.class_weights.to(target.device)
            weight_map = class_weights[target.long()]
        else:
            weight_map = torch.ones_like(target, dtype=torch.float, device=net_output.device)

        dc_loss = self.dc(net_output, target_dice)
        ce_loss = self.ce(net_output, target[:, 0].long())
        l_dice = torch.mean(torch.pow(-torch.log(dc_loss), self.gamma))   # mean w.r.t to label
        l_cross = torch.mean(torch.mul(weight_map, torch.pow(ce_loss, self.gamma)))
        result = self.weight_ce * l_cross + self.weight_dice * l_dice

        return result

class nnUNetTrainerExponentialLogarithmic(nnUNetTrainer):
    def _build_loss(self):
        assert not self.label_manager.has_regions, "regions not supported by this trainer"
        loss = ExponentialLogarithmicLoss({}, {}, weight_ce=0.2, weight_dice=0.8, gamma=0.3,
                                  class_weights=torch.tensor([0.975, 0.143, 0.128, 0.114]))
        # loss = ExponentialLogarithmicLoss(class_weights=torch.tensor([0.975, 0.143, 0.128, 0.114]))
        deep_supervision_scales = self._get_deep_supervision_scales()

        # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
        # this gives higher resolution outputs more weight in the loss
        weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
        weights[-1] = 0

        # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
        weights = weights / weights.sum()
        # now wrap the loss
        loss = DeepSupervisionWrapper(loss, weights)
        return loss

class nnUNetTrainerFocalTversky(nnUNetTrainer):
    def _build_loss(self):
        assert not self.label_manager.has_regions, "regions not supported by this trainer"
        loss = FocalTverskyLoss()

        deep_supervision_scales = self._get_deep_supervision_scales()

        # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
        # this gives higher resolution outputs more weight in the loss
        weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
        weights[-1] = 0

        # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
        weights = weights / weights.sum()
        # now wrap the loss
        loss = DeepSupervisionWrapper(loss, weights)
        return loss

class nnUNetTrainerLogCoshDice(nnUNetTrainer):
    def _build_loss(self):
        assert not self.label_manager.has_regions, "regions not supported by this trainer"
        loss = LogCoshDiceLoss()

        deep_supervision_scales = self._get_deep_supervision_scales()

        # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
        # this gives higher resolution outputs more weight in the loss
        weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
        weights[-1] = 0

        # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
        weights = weights / weights.sum()
        # now wrap the loss
        loss = DeepSupervisionWrapper(loss, weights)
        return loss

class nnUNetTrainerExponentialLogarithmic1000epochs(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        """used for debugging plans etc"""
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 1000
        self.initial_lr = 1e-2
        self.weight_decay = 3e-5
        
    def _build_loss(self):
        assert not self.label_manager.has_regions, "regions not supported by this trainer"
        loss = ExponentialLogarithmicLoss({}, {}, weight_ce=0.2, weight_dice=0.8, gamma=0.3,
                                  class_weights=torch.tensor([0.975, 0.143, 0.128, 0.114]))
        # loss = ExponentialLogarithmicLoss(class_weights=torch.tensor([0.975, 0.143, 0.128, 0.114]))
        deep_supervision_scales = self._get_deep_supervision_scales()

        # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
        # this gives higher resolution outputs more weight in the loss
        weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
        weights[-1] = 0

        # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
        weights = weights / weights.sum()
        # now wrap the loss
        loss = DeepSupervisionWrapper(loss, weights)
        return loss

class nnUNetTrainerFocalTversky1000epochs(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        """used for debugging plans etc"""
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 1000
        self.initial_lr = 1e-2
        self.weight_decay = 3e-5
        
    def _build_loss(self):
        assert not self.label_manager.has_regions, "regions not supported by this trainer"
        loss = FocalTverskyLoss()

        deep_supervision_scales = self._get_deep_supervision_scales()

        # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
        # this gives higher resolution outputs more weight in the loss
        weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
        weights[-1] = 0

        # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
        weights = weights / weights.sum()
        # now wrap the loss
        loss = DeepSupervisionWrapper(loss, weights)
        return loss

class nnUNetTrainerLogCoshDice1000epochs(nnUNetTrainer):
    
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        """used for debugging plans etc"""
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 1000
        self.initial_lr = 1e-2
        self.weight_decay = 3e-5
        
    def _build_loss(self):
        assert not self.label_manager.has_regions, "regions not supported by this trainer"
        loss = LogCoshDiceLoss()

        deep_supervision_scales = self._get_deep_supervision_scales()

        # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
        # this gives higher resolution outputs more weight in the loss
        weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
        weights[-1] = 0

        # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
        weights = weights / weights.sum()
        # now wrap the loss
        loss = DeepSupervisionWrapper(loss, weights)
        return loss

class nnUNetTrainerExponentialLogarithmic1e1(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        """used for debugging plans etc"""
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 1000
        self.initial_lr = 1e-1
        self.weight_decay = 3e-5
        
    def _build_loss(self):
        assert not self.label_manager.has_regions, "regions not supported by this trainer"
        loss = ExponentialLogarithmicLoss({}, {}, weight_ce=0.2, weight_dice=0.8, gamma=0.3,
                                  class_weights=torch.tensor([0.975, 0.143, 0.128, 0.114]))
        # loss = ExponentialLogarithmicLoss(class_weights=torch.tensor([0.975, 0.143, 0.128, 0.114]))
        deep_supervision_scales = self._get_deep_supervision_scales()

        # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
        # this gives higher resolution outputs more weight in the loss
        weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
        weights[-1] = 0

        # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
        weights = weights / weights.sum()
        # now wrap the loss
        loss = DeepSupervisionWrapper(loss, weights)
        return loss

class nnUNetTrainerExponentialLogarithmic1e3(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        """used for debugging plans etc"""
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 1000
        self.initial_lr = 1e-3
        self.weight_decay = 3e-5
        
    def _build_loss(self):
        assert not self.label_manager.has_regions, "regions not supported by this trainer"
        loss = ExponentialLogarithmicLoss({}, {}, weight_ce=0.2, weight_dice=0.8, gamma=0.3,
                                  class_weights=torch.tensor([0.975, 0.143, 0.128, 0.114]))
        # loss = ExponentialLogarithmicLoss(class_weights=torch.tensor([0.975, 0.143, 0.128, 0.114]))
        deep_supervision_scales = self._get_deep_supervision_scales()

        # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
        # this gives higher resolution outputs more weight in the loss
        weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
        weights[-1] = 0

        # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
        weights = weights / weights.sum()
        # now wrap the loss
        loss = DeepSupervisionWrapper(loss, weights)
        return loss

class nnUNetTrainerExponentialLogarithmic1e4(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        """used for debugging plans etc"""
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 1000
        self.initial_lr = 1e-4
        self.weight_decay = 3e-5
        
    def _build_loss(self):
        assert not self.label_manager.has_regions, "regions not supported by this trainer"
        loss = ExponentialLogarithmicLoss({}, {}, weight_ce=0.2, weight_dice=0.8, gamma=0.3,
                                  class_weights=torch.tensor([0.975, 0.143, 0.128, 0.114]))
        # loss = ExponentialLogarithmicLoss(class_weights=torch.tensor([0.975, 0.143, 0.128, 0.114]))
        deep_supervision_scales = self._get_deep_supervision_scales()

        # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
        # this gives higher resolution outputs more weight in the loss
        weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
        weights[-1] = 0

        # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
        weights = weights / weights.sum()
        # now wrap the loss
        loss = DeepSupervisionWrapper(loss, weights)
        return loss