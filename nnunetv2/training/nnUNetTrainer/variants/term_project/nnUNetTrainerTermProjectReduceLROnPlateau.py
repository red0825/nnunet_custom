import torch
import torch.nn as nn
import torch.nn.functional as F
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.loss.robust_ce_loss import RobustCrossEntropyLoss
from torch.optim.lr_scheduler import _LRScheduler
import numpy as np

from torch import distributed as dist
from typing import List
from nnunetv2.utilities.collate_outputs import collate_outputs

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

class PolyLRScheduler(_LRScheduler):
    def __init__(self, optimizer, initial_lr: float, max_steps: int, exponent: float = 0.9, current_step: int = None):
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.max_steps = max_steps
        self.exponent = exponent
        self.ctr = 0
        super().__init__(optimizer, current_step if current_step is not None else -1, False)

    def step(self, current_step=None):
        if current_step is None or current_step == -1:
            current_step = self.ctr
            self.ctr += 1

        new_lr = self.initial_lr * (1 - current_step / self.max_steps) ** self.exponent
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr

class nnUNetTrainerExpLogReduceLR(nnUNetTrainer):
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
    
    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay,
                                    momentum=0.99, nesterov=True)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
        return optimizer, lr_scheduler
    
    def on_train_epoch_start(self):
        self.network.train()
        self.print_to_log_file('')
        self.print_to_log_file(f'Epoch {self.current_epoch}')
        self.print_to_log_file(
            f"Current learning rate: {np.round(self.optimizer.param_groups[0]['lr'], decimals=5)}")
        # lrs are the same for all workers so we don't need to gather them in case of DDP training
        self.logger.log('lrs', self.optimizer.param_groups[0]['lr'], self.current_epoch)
        
    def on_train_epoch_end(self, train_outputs: List[dict]):
        outputs = collate_outputs(train_outputs)

        if self.is_ddp:
            losses_tr = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(losses_tr, outputs['loss'])
            loss_here = np.vstack(losses_tr).mean()
        else:
            loss_here = np.mean(outputs['loss'])

        self.logger.log('train_losses', loss_here, self.current_epoch)
        
    def on_validation_epoch_end(self, val_outputs: List[dict]):
        outputs_collated = collate_outputs(val_outputs)
        tp = np.sum(outputs_collated['tp_hard'], 0)
        fp = np.sum(outputs_collated['fp_hard'], 0)
        fn = np.sum(outputs_collated['fn_hard'], 0)

        if self.is_ddp:
            world_size = dist.get_world_size()

            tps = [None for _ in range(world_size)]
            dist.all_gather_object(tps, tp)
            tp = np.vstack([i[None] for i in tps]).sum(0)

            fps = [None for _ in range(world_size)]
            dist.all_gather_object(fps, fp)
            fp = np.vstack([i[None] for i in fps]).sum(0)

            fns = [None for _ in range(world_size)]
            dist.all_gather_object(fns, fn)
            fn = np.vstack([i[None] for i in fns]).sum(0)

            losses_val = [None for _ in range(world_size)]
            dist.all_gather_object(losses_val, outputs_collated['loss'])
            loss_here = np.vstack(losses_val).mean()
        else:
            loss_here = np.mean(outputs_collated['loss'])

        global_dc_per_class = [i for i in [2 * i / (2 * i + j + k) for i, j, k in
                                           zip(tp, fp, fn)]]
        mean_fg_dice = np.nanmean(global_dc_per_class)
        self.logger.log('mean_fg_dice', mean_fg_dice, self.current_epoch)
        self.logger.log('dice_per_class_or_region', global_dc_per_class, self.current_epoch)
        self.logger.log('val_losses', loss_here, self.current_epoch)
        self.lr_scheduler.step(loss_here)

class nnUNetTrainerExpLogReduceLRAdam(nnUNetTrainerExpLogReduceLR):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        """used for debugging plans etc"""
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 1000
        self.initial_lr = 1e-3
        self.weight_decay = 3e-5

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.network.parameters(),
                          lr=self.initial_lr,
                          weight_decay=self.weight_decay,
                          amsgrad=True)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
        return optimizer, lr_scheduler