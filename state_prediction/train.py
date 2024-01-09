#  Copyright (c) Meta Platforms, Inc. and affiliates.
#  All rights reserved.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.

import os
import glob
import torch
import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.plugins import DDPPlugin

from habitat.core.registry import registry

from .config.defaults import get_config
from .task import LocalStatePrediction

torch.autograd.set_detect_anomaly(True)

def train(cfg):

    # adjust batch size for ddp
    cfg.defrost()
    cfg.OPTIM.BATCH_SIZE = cfg.OPTIM.BATCH_SIZE // cfg.GPUS
    cfg.freeze()

    task = registry._get_impl('task', cfg.DATA.TASK)(cfg)

    callbacks = []
    checkpoint_callback = ModelCheckpoint(
        filename='{epoch:02d}-{val_loss:.2E}',
        save_top_k=1,
        monitor='val_loss',
        verbose=True,
        save_last=True,
    )
    callbacks.append(checkpoint_callback)
    callbacks.append(LearningRateMonitor(logging_interval='step'))

    # handle pre-emption
    checkpoints = glob.glob(f'{cfg.CHECKPOINT_DIR}/lightning_logs/version_*/checkpoints/*.ckpt')
    latest_checkpoint = None
    if len(checkpoints) > 0 and not cfg.DEBUG:
        latest_checkpoint = max(checkpoints, key=os.path.getctime)
        print (f'Resuming from {latest_checkpoint}')

    trainer = pl.Trainer(
        callbacks=callbacks,
        default_root_dir=cfg.CHECKPOINT_DIR,
        gpus=cfg.GPUS,
        max_epochs=cfg.OPTIM.MAX_EPOCHS,
        accelerator="gpu",
        strategy=DDPPlugin(find_unused_parameters=False),
        resume_from_checkpoint=latest_checkpoint,
        check_val_every_n_epoch=cfg.OPTIM.EVAL_VAL_EVERY,
    )

    trainer.fit(task)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=None, help='config yaml for experiment')
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER, help="Modify config options from command line")
    args = parser.parse_args()

    cfg = get_config(args.config, args.opts)
    os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)
    torch.save(cfg, f'{cfg.CHECKPOINT_DIR}/cfg.pth')
    train(cfg)
