import os
import torch
import argparse
from typing import Any, Dict, Optional
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger as PLTensorBoardLogger
from torch import optim
from timm.utils import ModelEma
from torch.utils.data import DataLoader
import wandb
from datetime import datetime
import json

from diffusion_planner.model.factorized_diffusion_planner import Factorized_Diffusion_Planner
from diffusion_planner.utils.train_utils import set_seed
from diffusion_planner.utils.normalizer import ObservationNormalizer, StateNormalizer
from diffusion_planner.utils.lr_schedule import CosineAnnealingWarmUpRestarts
from diffusion_planner.utils.data_augmentation import StatePerturbation
from diffusion_planner.utils.dataset import DiffusionPlannerData
from diffusion_planner.utils.batch_aware_dataset import DiffusionPlannerDataDistributed, BatchAwareDistributedSampler
from diffusion_planner.loss import diffusion_loss_factorized_func

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

class DiffusionPlannerLightningModule(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        # Convert args to dict for save_hyperparameters, excluding complex objects
        args_dict = {}
        for key, value in vars(args).items():
            if not isinstance(value, (StateNormalizer, ObservationNormalizer)):
                args_dict[key] = value
        self.save_hyperparameters(args_dict)
        
        self.args = args
        # Initialize model
        self.model = Factorized_Diffusion_Planner(args)
        # Initialize EMA if requested
        if args.use_ema:
            self.model_ema = ModelEma(
                self.model,
                decay=0.999,
                device=args.device,
            )
        else:
            self.model_ema = None
        # Initialize data augmentation
        self.aug = StatePerturbation(
            augment_prob=args.augment_prob,
            device=args.device,
        ) if args.use_data_augment else None
        # Store normalization parameters
        self.observation_normalizer = args.observation_normalizer
        self.state_normalizer = args.state_normalizer

    def forward(self, inputs):
        return self.model(inputs)

    def _process_batch(self, batch, apply_augmentation=False):
        """Common batch processing logic for train and val steps."""
        # Prepare data
        inputs = {
            'ego_current_state': batch[0],
            'neighbor_agents_past': batch[2],
            'lanes': batch[4],
            'lanes_speed_limit': batch[5],
            'lanes_has_speed_limit': batch[6],
            'route_lanes': batch[7],
            'route_lanes_speed_limit': batch[8],
            'route_lanes_has_speed_limit': batch[9],
            'static_objects': batch[10]
        }
        ego_future = batch[1]
        neighbors_future = batch[3]
        # Apply data augmentation only if requested
        if apply_augmentation and self.aug is not None:
            inputs, ego_future, neighbors_future = self.aug(inputs, ego_future, neighbors_future)
        # Convert heading to cos/sin representation
        ego_future = torch.cat([
            ego_future[..., :2],
            torch.stack([
                ego_future[..., 2].cos(),
                ego_future[..., 2].sin()
            ], dim=-1),
        ], dim=-1)
        mask = torch.sum(torch.ne(neighbors_future[..., :3], 0), dim=-1) == 0
        neighbors_future = torch.cat([
            neighbors_future[..., :2],
            torch.stack([
                neighbors_future[..., 2].cos(),
                neighbors_future[..., 2].sin()
            ], dim=-1),
        ], dim=-1)
        neighbors_future[mask] = 0.
        # Normalize inputs
        inputs = self.observation_normalizer(inputs)
        return inputs, ego_future, neighbors_future, mask

    def training_step(self, batch, batch_idx):
        inputs, ego_future, neighbors_future, mask = self._process_batch(
            batch, apply_augmentation=True
        )
        # Compute loss
        loss = {}
        loss, _ = diffusion_loss_factorized_func(
            self.model,
            inputs,
            self.model.sde.marginal_prob,
            (ego_future, neighbors_future, mask),
            self.state_normalizer,
            loss,
            self.args.diffusion_model_type
        )
        # Combine losses with separate alpha weights
        ego_weight = getattr(self.args, 'alpha_ego_loss', self.args.alpha_planning_loss)
        neighbor_weight = getattr(self.args, 'alpha_neighbor_loss', 1.0)
        
        total_loss = neighbor_weight * loss['neighbor_prediction_loss'] + ego_weight * loss['ego_planning_loss']
        loss['loss'] = total_loss
        # Log losses
        self.log('train_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_ego_loss', loss['ego_planning_loss'], on_step=True, on_epoch=True)
        self.log('train_neighbor_loss', loss['neighbor_prediction_loss'], on_step=True, on_epoch=True)
        self.log('train_ego_weight', ego_weight, on_step=True, on_epoch=True)
        self.log('train_neighbor_weight', neighbor_weight, on_step=True, on_epoch=True)
        return total_loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if self.model_ema is not None:
            self.model_ema.update(self.model)

    def on_save_checkpoint(self, checkpoint):
        if self.model_ema is not None:
            ema_state_dict = {}
            for key, value in self.model_ema.ema.state_dict().items():
                ema_state_dict[f'model_ema.ema.{key}'] = value
            # Add EMA weights to the checkpoint
            checkpoint['state_dict'].update(ema_state_dict)
            # Add metadata
            checkpoint['ema_enabled'] = True
        else:
            checkpoint['ema_enabled'] = False
        return checkpoint
    
    def on_load_checkpoint(self, checkpoint):
        """Called when loading checkpoint"""
        # Extract EMA weights from checkpoint
        ema_state_dict = {}
        keys_to_remove = []
        for key in list(checkpoint['state_dict'].keys()):
            if key.startswith('model_ema.ema.'):
                new_key = key.replace('model_ema.ema.', '')
                ema_state_dict[new_key] = checkpoint['state_dict'][key]
                keys_to_remove.append(key)
        # Remove EMA keys from state_dict to avoid warnings
        for key in keys_to_remove:
            del checkpoint['state_dict'][key]
        # Load EMA weights if model_ema exists and we have EMA weights
        if self.model_ema is not None and ema_state_dict:
            # Load the EMA state dict
            self.model_ema.ema.load_state_dict(ema_state_dict)
            print(f"Loaded EMA weights: {len(ema_state_dict)} parameters")
        elif self.model_ema is not None and checkpoint.get('ema_enabled', False):
            print("Warning: EMA was enabled during training but no EMA weights found in checkpoint")

    def validation_step(self, batch, batch_idx):
        """Validation step - no gradients, no augmentation, use EMA if available."""
        # Process batch without augmentation
        inputs, ego_future, neighbors_future, mask = self._process_batch(
            batch, apply_augmentation=False
        )
        # Use EMA model if available for validation
        if self.model_ema is not None:
            model_to_use = self.model_ema.ema
        else:
            model_to_use = self.model
        # Compute loss
        loss = {}
        loss, _ = diffusion_loss_factorized_func(
            model_to_use,
            inputs,
            self.model.sde.marginal_prob,
            (ego_future, neighbors_future, mask),
            self.state_normalizer,
            loss,
            self.args.diffusion_model_type
        )
        # Combine losses with separate alpha weights
        ego_weight = getattr(self.args, 'alpha_ego_loss', self.args.alpha_planning_loss)
        neighbor_weight = getattr(self.args, 'alpha_neighbor_loss', 1.0)
        
        total_loss = neighbor_weight * loss['neighbor_prediction_loss'] + ego_weight * loss['ego_planning_loss']
        # Log validation losses
        self.log('val_loss', total_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('val_ego_loss', loss['ego_planning_loss'], on_step=False, on_epoch=True, sync_dist=True)
        self.log('val_neighbor_loss', loss['neighbor_prediction_loss'], on_step=False, on_epoch=True, sync_dist=True)
        return total_loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.args.learning_rate)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
                'frequency': 1,
            }
        }

class DiffusionPlannerDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args

    def setup(self, stage: str):
        if stage == "fit" or stage is None:
            # Training dataset
            self.train_dataset = DiffusionPlannerDataDistributed(
                data_dir=self.args.train_set,
                mapping_pkl=self.args.train_mapping_pkl,
                past_neighbor_num=self.args.agent_num,
                predicted_neighbor_num=self.args.predicted_neighbor_num,
                future_len=self.args.future_len,
                filter_prefix=getattr(self.args, 'train_prefix', None),
                max_files=int(self.args.max_files),
            )
            # Validation dataset
            if hasattr(self.args, 'val_set') and self.args.val_set is not None:
                self.val_dataset = DiffusionPlannerDataDistributed(
                    data_dir=self.args.val_set,
                    mapping_pkl=self.args.val_mapping_pkl,
                    past_neighbor_num=self.args.agent_num,
                    predicted_neighbor_num=self.args.predicted_neighbor_num,
                    future_len=self.args.future_len,
                    filter_prefix=getattr(self.args, 'val_prefix', None),
                    max_files=int(self.args.max_val_files) if hasattr(self.args, 'max_val_files') else 10000,
                )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            pin_memory=self.args.pin_mem,
            drop_last=True,
            shuffle=True,
        )

    def val_dataloader(self):
        if hasattr(self, 'val_dataset'):
            return DataLoader(
                self.val_dataset,
                batch_size=self.args.batch_size,
                num_workers=self.args.num_workers,
                pin_memory=self.args.pin_mem,
                drop_last=False,
                shuffle=False,
            )
        return None

def load_args_from_checkpoint(checkpoint_path, args_path=None):
    """Load original training arguments from checkpoint directory or specified path"""
    if args_path is not None:
        # Use explicitly provided args path
        if not os.path.exists(args_path):
            raise FileNotFoundError(f"Specified args file not found: {args_path}")
        args_json_path = args_path
        print(f"Loading original training config from: {args_json_path}")
    else:
        # Look for args.json in the same directory as checkpoint or parent directories
        checkpoint_dir = os.path.dirname(checkpoint_path)
        
        # Try different locations for args.json
        possible_paths = [
            os.path.join(checkpoint_dir, 'args.json'),
            os.path.join(os.path.dirname(checkpoint_dir), 'args.json'),
            os.path.join(os.path.dirname(os.path.dirname(checkpoint_dir)), 'args.json'),
        ]
        
        args_json_path = None
        for path in possible_paths:
            if os.path.exists(path):
                args_json_path = path
                break
        
        if args_json_path is None:
            raise FileNotFoundError(f"Could not find args.json near checkpoint {checkpoint_path}")
        
        print(f"Loading original training config from: {args_json_path}")
    
    with open(args_json_path, 'r') as f:
        args_dict = json.load(f)
    
    # Convert back to argparse.Namespace (compatible with PyTorch Lightning)
    import argparse
    args = argparse.Namespace()
    
    # First set all regular attributes
    for key, value in args_dict.items():
        if key not in ['state_normalizer', 'observation_normalizer']:
            setattr(args, key, value)
    
    # Then reconstruct normalizers using from_json (requires normalization_file_path)
    if 'state_normalizer' in args_dict or 'observation_normalizer' in args_dict:
        # Make sure normalization_file_path is available
        if not hasattr(args, 'normalization_file_path'):
            args.normalization_file_path = args_dict.get('normalization_file_path', 'normalization.json')
        
        args.state_normalizer = StateNormalizer.from_json(args)
        args.observation_normalizer = ObservationNormalizer.from_json(args)
    
    return args

def get_finetune_args():
    """Get minimal finetuning arguments - just checkpoint and epochs"""
    parser = argparse.ArgumentParser(description='Finetuning Diffusion Planner')
    
    # Required arguments
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help='Path to the pretrained checkpoint')
    parser.add_argument('--args_path', type=str, default=None,
                        help='Path to args.json file (optional - will search near checkpoint if not specified)')
    parser.add_argument('--train_epochs', type=int, default=50,
                        help='Number of finetuning epochs')
    parser.add_argument('--train_mapping_pkl', type=str, default=None,
                        help='Path to new training data mapping pickle file (optional - uses original if not specified)')
    
    # Loss weighting arguments
    parser.add_argument('--alpha_ego_loss', type=float, default=None,
                        help='Weight for ego planning loss (default: use original alpha_planning_loss)')
    parser.add_argument('--alpha_neighbor_loss', type=float, default=None,
                        help='Weight for neighbor prediction loss (default: 1.0)')

    parser.add_argument('--learning_rate', type=float, default=None,
                        help='learning rate')
    
    return parser.parse_args()

def main():
    # Get finetuning arguments (just checkpoint path and epochs)
    finetune_args = get_finetune_args()
    
    # Load original training configuration
    args = load_args_from_checkpoint(finetune_args.checkpoint_path, finetune_args.args_path)
    
    # Override epochs and set finetuning defaults
    args.train_epochs = finetune_args.train_epochs
    args.learning_rate = finetune_args.learning_rate  # Lower learning rate for finetuning
    args.warm_up_epoch = 2  # Shorter warmup for finetuning
    
    # Set separate alpha weights if specified
    if finetune_args.alpha_ego_loss is not None:
        args.alpha_ego_loss = finetune_args.alpha_ego_loss
    if finetune_args.alpha_neighbor_loss is not None:
        args.alpha_neighbor_loss = finetune_args.alpha_neighbor_loss
    args.name = f"{args.name}/ft-lr_{args.learning_rate}-ego_{args.alpha_ego_loss}-nei_{args.alpha_neighbor_loss}"
    
    # Override training mapping if specified
    if finetune_args.train_mapping_pkl is not None:
        args.train_mapping_pkl = finetune_args.train_mapping_pkl
    
    print(f"Starting finetuning from checkpoint: {finetune_args.checkpoint_path}")
    print(f"Original experiment: {args.name.replace('_finetune', '')}")
    print(f"Finetuning experiment: {args.name}")
    # print(f"Learning rate reduced by 10x for finetuning: {args.learning_rate}")
    print(f"Training epochs: {args.train_epochs}")
    
    # Print loss weights
    ego_weight = getattr(args, 'alpha_ego_loss', args.alpha_planning_loss)
    neighbor_weight = getattr(args, 'alpha_neighbor_loss', 1.0)
    print(f"Loss weights - Ego: {ego_weight}, Neighbor: {neighbor_weight}")
    
    torch.set_float32_matmul_precision('high')
    
    # Create save directory
    time = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    save_path = f"{args.save_dir}/training_log/{args.name}/{time}/"
    os.makedirs(save_path, exist_ok=True)
    
    # Save args
    args_dict = vars(args)
    args_dict = {k: v if not isinstance(v, (StateNormalizer, ObservationNormalizer)) else v.to_dict() 
                for k, v in args_dict.items()}
    
    from mmengine.fileio import dump
    dump(args_dict, os.path.join(save_path, 'args.json'), file_format='json', indent=4)
    
    # Set seed
    set_seed(args.seed)
    
    # Initialize data module
    data_module = DiffusionPlannerDataModule(args)
    
    # Initialize model
    model = DiffusionPlannerLightningModule(args)
    
    # Setup logging
    # logger = None
    # if args.logger == 'wandb':
    #     wandb.login(key='caad08df59bfd0cb22f3613849ad66faeb65d4b0')
    #     logger = WandbLogger(
    #         project='FactorizedDiffusionPlanner_Finetune',
    #         name=args.name,
    #         save_dir=save_path,
    #         config=vars(args),
    #         notes=f"Finetuning from {finetune_args.checkpoint_path}"
    #     )
    # elif args.logger == 'tensorboard':
    #     logger = PLTensorBoardLogger(save_dir=save_path, name=args.name)
    wandb.login(key='local-340ff373668e9a4ebcfa60a206a40ff0f2b75eef', host='https://server.auto-wandb.qualcomm.com/')

    # Create logger
    logger = WandbLogger(
        project='FactorizedDiffusionPlanner_Finetune',
        name=args.name,
        save_dir=save_path,
        config=vars(args),
        notes=f"Finetuning from {finetune_args.checkpoint_path}",
    )
    
    # Setup callbacks
    callbacks = []
    
    # Model checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{args.save_dir}/checkpoints/{args.name}/",
        filename='{epoch}-{step}-{val_loss:.4f}' if hasattr(args, 'val_set') and args.val_set else '{epoch}-{step}-{train_loss:.4f}',
        monitor='val_loss' if hasattr(args, 'val_set') and args.val_set else 'train_loss',
        mode='min',
        save_top_k=-1,
        every_n_epochs=1,  # Save after each epoch
        save_last=True,
        verbose=True,
    )
    checkpoint_callback.CHECKPOINT_EQUALS_CHAR = '_'
    callbacks.append(checkpoint_callback)
    
    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    callbacks.append(lr_monitor)
    
    # Initialize trainer
    trainer = pl.Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        strategy='ddp_find_unused_parameters_true' if args.strategy=='ddp' else args.strategy,
        max_epochs=args.train_epochs,
        logger=logger,
        callbacks=callbacks,
        gradient_clip_val=5.0,
        gradient_clip_algorithm='norm',
        log_every_n_steps=50,
        enable_checkpointing=True,
        enable_progress_bar=True,
        enable_model_summary=True,
        check_val_every_n_epoch=1,
    )

    # Load the checkpoint first to get the model weights
    checkpoint = torch.load(finetune_args.checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'], strict=False)

    # load ema
    ema_state_dict = {}
    keys_to_remove = []
    
    for key in list(checkpoint['state_dict'].keys()):
        if key.startswith('model_ema.ema.'):
            new_key = key.replace('model_ema.ema.', '')
            ema_state_dict[new_key] = checkpoint['state_dict'][key]
            keys_to_remove.append(key)
    
    # Remove EMA keys from state_dict to avoid warnings
    for key in keys_to_remove:
        del checkpoint['state_dict'][key]
    model.model_ema.ema.load_state_dict(ema_state_dict)
    print(f"Loaded EMA weights: {len(ema_state_dict)} parameters")

    # Start training without loading optimizer state
    trainer.fit(model, data_module) 
    # # Start finetuning from checkpoint
    # trainer.fit(model, data_module, ckpt_path=finetune_args.checkpoint_path)

if __name__ == "__main__":
    main()