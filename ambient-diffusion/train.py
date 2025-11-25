import os
import torch
import argparse
from typing import Any, Dict, Optional

import wandb
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger as PLTensorBoardLogger

from datamodules import NuplanDataModule
from models import DiffusionPredictor 

from utils.train_utils import set_seed

os.environ['NCCL_TIMEOUT'] = '600'

def boolean(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_args():
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--name', type=str, help="log name (default: 'ambient-diffusion')", default='ambient-diffusion')
    parser.add_argument('--save_dir', type=str, help='save dir for model ckpt', default='.')

    # Data
    parser.add_argument('--root', type=str, help='path to dataset root', default='../nuplan')
    parser.add_argument('--train_metadata', type=str, help='path to trainining metadata', default=None)
    parser.add_argument('--val_metadata', type=str, help='path to validation metadata', default=None)
    parser.add_argument('--train_batch_size', type=int, help='training batch size', default=8)
    parser.add_argument('--val_batch_size', type=int, help='validation batch size', default=16)
    parser.add_argument('--num_historical_steps', type=int, default=21, help='Number of historical timesteps to include (default: 20)')
    parser.add_argument('--num_future_steps', type=int, default=80, help='Number of future timesteps to predict (default: 80)')
    parser.add_argument('--max_agents', type=int, default=64, help='Maximum number of agents to consider (default: 64)')
    parser.add_argument('--shuffle', action='store_true', default=False, help='Whether to shuffle the dataset (default: True)')

    # DataLoader parameters
    parser.add_argument('--num_workers', type=int, default=8, help='Number of data loading workers (default: 8)')
    parser.add_argument('--pin_memory', action='store_true', default=False, help='Whether to pin memory for CUDA (default: False)')
    parser.add_argument('--persistent_workers', action='store_true', default=False, help='Whether to use persistent workers (default: False)')

    # Training
    parser.add_argument('--seed', type=int, help='fix random seed', default=18)
    parser.add_argument('--train_epochs', type=int, help='epochs of training', default=500)
    parser.add_argument('--learning_rate', type=float, help='learning rate (default: 5e-4)', default=5e-4)
    parser.add_argument('--weight_decay', type=float, help='weight decay for optimizer (default: 0.0001)', default=0.0001)
    parser.add_argument('--warmup_epochs', type=int, help='number of warmup epochs (default: 4)', default=4)
    parser.add_argument('--logger', default='wandb', type=str, choices=['none', 'qualcomm', 'wandb', 'tensorboard'])
    parser.add_argument('--accelerator', default='gpu', type=str, help='accelerator type')
    parser.add_argument('--devices', default=-1, type=int, help='number of devices to use (-1 for all)')
    parser.add_argument('--strategy', default='ddp', type=str, help='distributed strategy')

    # Model
    parser.add_argument('--ckpt_path', type=str, help='path to load model weights', default=None)
    parser.add_argument('--hidden_dim', type=int, default=128, help='Hidden dimension size (default: 128)')
    parser.add_argument('--num_heads', type=int, default=8, help='Number of attention heads (default: 8)')
    parser.add_argument('--num_hops', type=int, default=4, help='Number of hops for map encoder graph propagation (default: 4)')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate (default: 0.1)')
    parser.add_argument('--agent_time_span', type=int, default=10, help='Agent encoder temporal window span')
    parser.add_argument('--agent_radius', type=float, default=60.0, help='Agent encoder neighbor radius')
    parser.add_argument('--agent_polygon_radius', type=float, default=50.0, help='Agent to map radius')
    parser.add_argument('--agent_num_attn_layers', type=int, default=3, help='Agent encoder attention layers')
    parser.add_argument('--agent_num_heads', type=int, default=8, help='Agent encoder attention heads')
    parser.add_argument('--agent_dropout', type=float, default=0.1, help='Agent encoder dropout')
    parser.add_argument('--diffuser_num_layers', type=int, default=3, help='Number of diffuser attention layers')
    parser.add_argument('--diffuser_num_heads', type=int, default=8, help='Number of diffuser attention heads')
    parser.add_argument('--diffuser_dropout', type=float, default=0.1, help='Diffuser dropout rate')
    parser.add_argument('--diffuser_temporal_span', type=int, default=6, help='Temporal attention span for diffuser')
    parser.add_argument('--diffuser_agent_radius', type=float, default=60.0, help='Diffuser agent-agent radius')
    parser.add_argument('--diffuser_polygon_radius', type=float, default=50.0, help='Diffuser map radius')
    parser.add_argument('--diffuser_segment_length', type=int, default=80, help='Diffuser segment length')
    parser.add_argument('--diffuser_segment_overlap', type=int, default=0, help='Diffuser segment overlap')
    parser.add_argument('--diffuser_normalize_segments', action='store_true', default=False, help='Normalize diffuser segment to the starting points')

    args = parser.parse_args()

    return args 

def main():
    args = get_args()

    torch.set_float32_matmul_precision('high')

    # TODO: change training_name 
    training_name = '{}'.format(
        args.name,
    )
    save_path = f'{args.save_dir}/training_log/{args.name}/'
    os.makedirs(save_path, exist_ok=True)

    # Save args
    args_dict = vars(args)
    # TODO: check if we need to add state normalizer
    # args_dict = {k: v if not isinstance(v, (StateNormalizer, ObservationNormalizer)) else v.to_dict() for k, v in args_dict.items() }

    from mmengine.fileio import dump
    dump(args_dict, os.path.join(save_path, 'args.json'), file_format='json', indent=4)
    
    # Set seed
    set_seed(args.seed)

    # Initialize model weights
    model_params = {
        'num_historical_steps': args.num_historical_steps,
        'num_future_steps': args.num_future_steps,
        'hidden_dim': args.hidden_dim,
        'num_heads': args.num_heads,
        'num_hops': args.num_hops,
        'dropout': args.dropout,
        'agent_time_span': args.agent_time_span,
        'agent_radius': args.agent_radius,
        'agent_polygon_radius': args.agent_polygon_radius,
        'agent_num_attn_layers': args.agent_num_attn_layers,
        'agent_num_heads': args.agent_num_heads,
        'agent_dropout': args.agent_dropout,
        'diffuser_num_layers': args.diffuser_num_layers,
        'diffuser_num_heads': args.diffuser_num_heads,
        'diffuser_dropout': args.diffuser_dropout,
        'diffuser_temporal_span': args.diffuser_temporal_span,
        'diffuser_agent_radius': args.diffuser_agent_radius,
        'diffuser_polygon_radius': args.diffuser_polygon_radius,
        'diffuser_segment_length': args.diffuser_segment_length,
        'diffuser_segment_overlap': args.diffuser_segment_overlap,
        'diffuser_normalize_segments': args.diffuser_normalize_segments,
        'lr': args.learning_rate,
        'weight_decay': args.weight_decay,
        'warmup_epochs': args.warmup_epochs
    }

    # TODO: Load from checkpoint path 
    if args.ckpt_path is not None:
        print(f'Loading model from checkpoint: {args.ckpt_path}')
        model = None
    else:
        print('Initializing with new weights ...')
        model = DiffusionPredictor(**model_params)
    
    # Initialize datamodule
    datamodule = NuplanDataModule(
        root = args.root,
        train_metadata_path = args.train_metadata,
        val_metadata_path = args.val_metadata,
        train_batch_size = args.train_batch_size,
        val_batch_size = args.val_batch_size,
        num_historical_steps = args.num_historical_steps,
        num_future_steps=args.num_future_steps,
        max_agents=args.max_agents,
        shuffle=args.shuffle,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers,
    )


    segment_info = f"seg{args.diffuser_segment_length}o{args.diffuser_segment_overlap}n{args.diffuser_normalize_segments}"
    model_info = f"ma{args.max_agents}_ts{args.agent_time_span}_hd{args.hidden_dim}_heads{args.num_heads}_layers{args.diffuser_num_layers}"
    logger_name = f"{args.name}_{segment_info}_{model_info}"
    # Setup logging
    logger = None
    if args.logger == 'none':
        return
    elif args.logger in ['wandb', 'qualcomm']:
        # Login
        if args.logger == 'qualcomm':
            wandb.login(key='local-340ff373668e9a4ebcfa60a206a40ff0f2b75eef', host='https://server.auto-wandb.qualcomm.com/')
        else:
            wandb.login(key='caad08df59bfd0cb22f3613849ad66faeb65d4b0')
        # Create logger
        logger = WandbLogger(
            project='AmbientDiffusion',
            name=logger_name,
            entity='luw015',
            log_model=True,
            save_dir=save_path,
            config=vars(args)
        )
    elif args.logger == 'tensorboard':
        logger = PLTensorBoardLogger(save_dir=save_path, name=logger_name)

    # Setup callbacks
    training_config = {
        'args': args_dict,
        'model_params': model_params,
        'logger': args.logger,
        'checkpoint_dir': save_path,
    }
    dump(training_config, os.path.join(save_path, 'training_config.json'), file_format='json', indent=4)
    checkpoint_callback = ModelCheckpoint(
        dirpath=save_path,
        filename=None,
        monitor=None,
        mode='min',
        save_top_k=-1,  # save all checkpoints 
        every_n_train_steps=10000,
        save_last=True,
        verbose=True,
    )
    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    callbacks = [checkpoint_callback, lr_monitor]

    # Initialize trainer
    trainer = pl.Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        strategy='ddp_find_unused_parameters_true' if args.strategy=='ddp' else args.strategy,
        max_epochs=args.train_epochs,
        logger=logger,
        callbacks=callbacks,
        gradient_clip_val=5.0,  # Gradient clipping
        gradient_clip_algorithm='norm',
        log_every_n_steps=50,
        enable_checkpointing=True,
        enable_progress_bar=True,
        enable_model_summary=True,
        check_val_every_n_epoch=1,
    )

    trainer.fit(model, datamodule)


if __name__ == "__main__":
    main()
