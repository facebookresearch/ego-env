#  Copyright (c) Meta Platforms, Inc. and affiliates.
#  All rights reserved.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.

import copy
import torch
import torch.nn.functional as F
import torch.optim
import pytorch_lightning as pl
from einops import rearrange
from sklearn.metrics import average_precision_score
from tabulate import tabulate
import pandas as pd
import numpy as np
import warnings

from habitat.core.registry import registry

from data.constants import OBJ_VOCAB
from .data import dataset
from .models import env_model

class LocalStatePrediction(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        print (cfg)
        self.cfg = cfg
        self.net = registry._get_impl('model', self.cfg.MODEL.NAME)(self.cfg)

    def aggregate_step_outputs(self, step_outputs, mode='ddp'):
        outputs = {}
        for key in step_outputs[0].keys():
            agg = torch.stack if step_outputs[0][key].ndim==0 else torch.cat
            output = agg([x[key] for x in step_outputs], 0)
            if mode == 'ddp':
                outputs[key] = torch.cat(self.all_gather(output).unbind())
            else:
                outputs[key] = output
        return outputs

    def training_step(self, batch, idx):
        preds = self.net(batch)
        loss, _ = self.calculate_loss(preds, batch)
        self.log('train_loss', loss, on_epoch=True)
        return loss

    def validation_step(self, batch, idx):
        preds = self.net(batch)
        raise NotImplementedError
    
    def validation_epoch_end(self, outputs):
        outputs = self.aggregate_step_outputs(outputs)
        raise NotImplementedError

    def setup(self, stage):
        dset = registry._get_impl('dataset', self.cfg.DATA.TASK)(self.cfg)
        self.trainset = copy.deepcopy(dset).set_mode('train')
        self.valset = copy.deepcopy(dset).set_mode('val')

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.trainset,
            batch_size=self.cfg.OPTIM.BATCH_SIZE,
            shuffle=True,
            num_workers=self.cfg.OPTIM.WORKERS,
            persistent_workers=True,
            drop_last=True,
            collate_fn=self.trainset.collate_fn
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.valset,
            batch_size=self.cfg.OPTIM.BATCH_SIZE,
            shuffle=False,
            num_workers=self.cfg.OPTIM.WORKERS,
            persistent_workers=True,
            drop_last=True,
            collate_fn=self.valset.collate_fn
        )

    def test_dataloader(self):
        return self.val_dataloader()

    def configure_optimizers(self):
        params = list(filter(lambda p: p.requires_grad, self.parameters()))
        print('%d params to optimize'%len(params))

        optimizer = torch.optim.Adam(params, self.cfg.OPTIM.LR, weight_decay=self.cfg.OPTIM.WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.cfg.OPTIM.MILESTONES, gamma=0.1)
        return [optimizer], [scheduler]


#---------------------------------------------------------------------------------------------------#
# Pose embedding task
#---------------------------------------------------------------------------------------------------#

@registry._register_impl(_type='task', to_register=None, name='PoseEmbedding')
class PoseEmbedding(LocalStatePrediction):


    def calculate_loss(self, preds, batch):

        R_pred = rearrange(preds[:, :, :, 0:5], 'b s r d -> b d s r')
        theta_pred = rearrange(preds[:, :, :, 5:], 'b s r d -> b d s r')

        R_loss = F.cross_entropy(R_pred, batch['rel_R_disc'], ignore_index=-1)
        theta_loss = F.cross_entropy(theta_pred, batch['rel_theta_disc'], ignore_index=-1)

        loss = R_loss + theta_loss

        loss_info = {
            'R_loss': R_loss,
            'theta_loss': theta_loss,
        }

        return loss, loss_info


    def validation_step(self, batch, idx):
        preds = self.net(batch)
        loss, loss_info = self.calculate_loss(preds, batch)
        self.log('val_loss', loss)    

        for key in loss_info:
            self.log(key, loss_info[key])

        return {
            'preds': preds,
            'R': batch['rel_R_disc'],
            'theta': batch['rel_theta_disc'],
        }

    def validation_epoch_end(self, outputs):
        outputs = self.aggregate_step_outputs(outputs)

        preds = outputs['preds']
        R = outputs['R']
        theta = outputs['theta']

        R_pred = preds[:, :, :, 0:5].argmax(3)
        theta_pred = preds[:, :, :, 5:].argmax(3)

        R_acc = (R_pred == R)[R != -1].float().mean()
        theta_acc = (theta_pred == theta)[theta != -1].float().mean()

        stats = {'R_acc': R_acc, 'theta_acc': theta_acc}

        for key, value in stats.items():
            self.log(key, value, rank_zero_only=True)

        if self.trainer.is_global_zero:
            stats_table = pd.DataFrame(data=[stats])
            print(tabulate(stats_table, headers='keys', tablefmt='psql', showindex=False))

#---------------------------------------------------------------------------------------------------#
# Cardinal object state
#---------------------------------------------------------------------------------------------------#

def AP_score(preds, labels):
    stats = {}
    for idx in range(labels.shape[1]): # each direction in ['→', '↓', '←']
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            AP = average_precision_score(labels[:, idx], preds[:, idx])
        stats[f'val_{idx}_AP'] = AP
    stats['val_AP'] = np.mean(list(stats.values()))
    return stats

@registry._register_impl(_type='task', to_register=None, name='CardinalObjectState')
class CardinalObjectStateTask(LocalStatePrediction):
    def __init__(self, cfg):
        cfg.defrost()
        cfg.MODEL.NUM_CLASSES = len(OBJ_VOCAB[cfg.DATA.SRC])
        cfg.freeze()
        super().__init__(cfg)

    def calculate_loss(self, preds, batch):

        obj_labels, dist_preds = preds

        obj_loss = F.binary_cross_entropy_with_logits(
                obj_labels,
                batch['labels'],
                reduction='none'
        ).mean()

        dist_preds = rearrange(dist_preds, 'b t d (o c) -> b c t d o', o=self.cfg.MODEL.NUM_CLASSES, c=self.cfg.MODEL.NUM_DIST_BUCKETS)
        dist_loss = F.cross_entropy(dist_preds, batch['dists'], ignore_index=-1)
    
        loss = obj_loss + self.cfg.DATA.MULTITASK_WT*dist_loss
        loss_info = {'obj_loss': obj_loss, 'dist_loss': dist_loss}

        return loss, loss_info


    def validation_step(self, batch, idx):
        preds = self.net(batch)
        loss, loss_info = self.calculate_loss(preds, batch)
        self.log('val_loss', loss)

        return {
            'obj_preds': preds[0],
            'dist_preds': preds[1],
            'labels': batch['labels'],
            'dists': batch['dists'],
            **loss_info
        }

    def validation_epoch_end(self, outputs):
        outputs = self.aggregate_step_outputs(outputs)

        preds = outputs['obj_preds'] # (N, T, 4, #obj)
        labels = outputs['labels'] # (N, T, 4, #obj)
        dists = outputs['dists']

        # (N, T, 4, #obj) --> (N', 4, #obj)
        preds = rearrange(preds, 'n t d o -> (n t) d o')
        labels = rearrange(labels, 'n t d o -> (n t) d o')
        stats = AP_score(preds.cpu(), labels.cpu())

        # add aux losses for logging
        loss_keys = [key for key in outputs.keys() if key.endswith('_loss')]
        stats.update({k: outputs[k].mean().item() for k in loss_keys})

        preds = rearrange(outputs['dist_preds'], 'n t d (o c) -> n c t d o', o=self.cfg.MODEL.NUM_CLASSES, c=self.cfg.MODEL.NUM_DIST_BUCKETS)
        preds = preds.argmax(1) # (N, T, 4, #obj)

        mask = (dists != -1)
        dist_acc = (preds[mask] == dists[mask]).float().mean()
        stats['dist_acc'] = dist_acc

        for key, value in stats.items():
            self.log(key, value, rank_zero_only=True)

        if self.trainer.is_global_zero:
            stats.update({'val_loss': self.trainer.callback_metrics['val_loss'].item()})
            stats_table = pd.DataFrame(data=[stats])
            print(tabulate(stats_table, headers='keys', tablefmt='psql', showindex=False))

