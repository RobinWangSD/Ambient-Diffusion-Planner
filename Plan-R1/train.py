from argparse import ArgumentParser

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, WandbLogger

from datamodules import NuplanDataModule
from model import PlanR1
from utils import load_config

import os
import wandb

CURRENT_FILE_PATH = os.path.realpath(__file__)
PROJECT_ROOT = os.path.dirname(CURRENT_FILE_PATH)
os.environ["PROJECT_ROOT"] = PROJECT_ROOT

if __name__ == '__main__':
    pl.seed_everything(1024, workers=True)

    parser = ArgumentParser()
    parser.add_argument('--config', type=str, default='config/train/pred.yaml')
    args = parser.parse_args()
    config = load_config(args.config)

    # Load the model from a checkpoint if provided
    if config['trainer']['ckpt_path']:
        print(f"Loading model from checkpoint: {config['trainer']['ckpt_path']}")
        model = PlanR1.load_from_checkpoint(config['trainer']['ckpt_path'], **config['model'])
    else:
        print("No checkpoint path provided, initializing a new model.")
        model = PlanR1(**config['model'])
    
    datamodule = NuplanDataModule(**config['datamodule'])
    model_checkpoint = ModelCheckpoint(**config['trainer']['ckpt'])
    lr_monitor = LearningRateMonitor(**config['trainer']['lr_monitor'])

    use_wandb = config.get("use_wandb", True)
    if use_wandb:
        wandb.login(key="caad08df59bfd0cb22f3613849ad66faeb65d4b0")
        # Build run name from config
        run_name = (
            f"plan_r1-{config['model']['mode']}"
            f"-agents{config['datamodule'].get('max_agents', 'NA')}"
            f"-bs{config['datamodule']['train_batch_size']}"
            f"-lr{config['model']['lr']}"
            f"-dim{config['model']['hidden_dim']}"
            f"-layers{config['model']['num_attn_layers']}"
        )
        logger = WandbLogger(
            name=run_name,
            project='ambient_diffusion_planner',
            entity='luw015',
            log_model=True,
            dir=config['trainer']['csv_logger']['save_dir'],
        )
    else:
        logger = CSVLogger(**config['trainer']['csv_logger'])
        # csv_logger = CSVLogger(**config['trainer']['csv_logger'])

    trainer = pl.Trainer(
        strategy=config['trainer']['strategy'],
        devices=config['trainer']['devices'],
        accelerator=config['trainer']['accelerator'],
        callbacks=[model_checkpoint, lr_monitor],
        max_epochs=config['trainer']['max_epochs'],
        logger=logger,
    )

    trainer.fit(model, datamodule)