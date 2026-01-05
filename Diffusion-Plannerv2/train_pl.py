import os
import torch
import argparse
from typing import Any, Dict, Optional
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from torch import optim
from timm.utils import ModelEma
from torch.utils.data import DataLoader

import wandb
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger as PLTensorBoardLogger

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
        self.save_hyperparameters(args)
        self.args = args
        
        # Initialize model
        self.model = Factorized_Diffusion_Planner(args)
        
        # Initialize EMA if requested
        self.model_ema = ModelEma(
            self.model,
            decay=0.999,
            device=None,  # move to the correct device later
        ) if args.use_ema else None
        
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

    # ADD: Helper method to process batch (extract common logic)
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
        
        # Combine losses
        total_loss = loss['neighbor_prediction_loss'] + self.args.alpha_planning_loss * loss['ego_planning_loss']
        loss['loss'] = total_loss
        
        # Log losses
        self.log('train_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_ego_loss', loss['ego_planning_loss'], on_step=True, on_epoch=True)
        self.log('train_neighbor_loss', loss['neighbor_prediction_loss'], on_step=True, on_epoch=True)
        
        return total_loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if self.model_ema is not None:
            self.model_ema.update(self.model)
    
    def on_fit_start(self):
        """Ensure EMA weights live on the local device for this rank."""
        if self.model_ema is not None:
            current_device = torch.device(self.device)
            if getattr(self.model_ema, "device", None) != current_device:
                self.model_ema.ema.to(device=current_device)
                self.model_ema.device = current_device

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
        """
        Called when loading checkpoint
        """
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
        
        # Combine losses
        total_loss = loss['neighbor_prediction_loss'] + self.args.alpha_planning_loss * loss['ego_planning_loss']
        
        # Log validation losses
        self.log('val_loss', total_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('val_ego_loss', loss['ego_planning_loss'], on_step=False, on_epoch=True, sync_dist=True)
        self.log('val_neighbor_loss', loss['neighbor_prediction_loss'], on_step=False, on_epoch=True, sync_dist=True)
        
        return total_loss
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.args.learning_rate)
        
        scheduler = CosineAnnealingWarmUpRestarts(
            optimizer, 
            self.args.train_epochs, 
            self.args.warm_up_epoch
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
                'frequency': 1,
            }
        }
    
    # def on_before_optimizer_step(self, optimizer):
    #     # Gradient clipping
    #     torch.nn.utils.clip_grad_norm_(self.parameters(), 5)


class DiffusionPlannerDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args

    def setup(self, stage: str):
        if stage == "fit" or stage is None:
            train_metadata = getattr(self.args, 'train_metadata', None)
            val_metadata = getattr(self.args, 'val_metadata', None) or train_metadata
            metadata_train_dirs = getattr(self.args, 'metadata_train_dirs', None)
            metadata_val_dirs = getattr(self.args, 'metadata_val_dirs', None)

            self.train_dataset = DiffusionPlannerDataDistributed(
                data_dir=self.args.train_set,
                mapping_pkl=None if train_metadata else self.args.train_mapping_pkl,
                past_neighbor_num=self.args.agent_num,
                predicted_neighbor_num=self.args.predicted_neighbor_num,
                future_len=self.args.future_len,
                filter_prefix=getattr(self.args, 'train_prefix', None),
                max_files=int(self.args.max_files),
                metadata_path=train_metadata,
                data_split='train',
                allowed_dirs=metadata_train_dirs,
            )
            
            # ADD: Validation dataset
            if val_metadata or getattr(self.args, 'val_mapping_pkl', None) or getattr(self.args, 'val_set', None):
                self.val_dataset = DiffusionPlannerDataDistributed(
                    data_dir=self.args.val_set if self.args.val_set is not None else self.args.train_set,
                    mapping_pkl=None if val_metadata else self.args.val_mapping_pkl,
                    past_neighbor_num=self.args.agent_num,
                    predicted_neighbor_num=self.args.predicted_neighbor_num,
                    future_len=self.args.future_len,
                    filter_prefix=getattr(self.args, 'val_prefix', None),
                    max_files=int(self.args.max_val_files) if hasattr(self.args, 'max_val_files') else 10000,
                    metadata_path=val_metadata,
                    data_split='val',
                    allowed_dirs=metadata_val_dirs,
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
                batch_size=self.args.batch_size,  # Could use larger batch for val
                num_workers=self.args.num_workers,
                pin_memory=self.args.pin_mem,
                drop_last=False,  # Don't drop last batch for validation
                shuffle=False,  # No shuffling for validation
            )
        return None

def get_args():
    # Arguments - same as original
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--name', type=str, help='log name (default: "diffusion-planner-training")', default="diffusion-planner-training")
    parser.add_argument('--save_dir', type=str, help='save dir for model ckpt', default=".")

    # Data
    parser.add_argument('--train_set', type=str, help='path to train data (used for relative paths)', default=None)
    parser.add_argument('--train_mapping_pkl', type=str, required=False, default=None,
                        help='Path to the pickle file containing file mappings (optional when using metadata)')
    parser.add_argument('--train_prefix', type=str, default=None,
                        help='Optional prefix to filter training files (e.g., "train_")')
    parser.add_argument('--val_set', type=str, help='path to validation data (used for relative paths)', default=None)
    parser.add_argument('--val_mapping_pkl', type=str, required=False, default=None,
                        help='Path to the pickle file containing file mappings (optional when using metadata)')
    parser.add_argument('--val_prefix', type=str, default=None,
                        help='Optional prefix to filter validation files')
    parser.add_argument('--max_val_files', type=int, help='maximum number of validation samples', 
                        default=10000)
    parser.add_argument('--val_check_interval_steps', type=int, default=10000,
                        help='How many training steps to run validation')
    parser.add_argument('--train_metadata', type=str, default=None,
                        help='Path to metadata JSON containing file_index entries for training split')
    parser.add_argument('--val_metadata', type=str, default=None,
                        help='Path to metadata JSON for validation split (defaults to train_metadata when omitted)')
    parser.add_argument('--metadata_train_dirs', type=str, nargs='+', default=None,
                        help='Optional list of metadata directories to include for training (defaults to NuPlan train dirs)')
    parser.add_argument('--metadata_val_dirs', type=str, nargs='+', default=None,
                        help='Optional list of metadata directories to include for validation (defaults to ["val"])')

    parser.add_argument('--use_batch_aware', action='store_true', default=True,
                        help='Use batch-aware sampling for better I/O performance')
    parser.add_argument('--max_files', type=int, help='maximum number of training samples', default=5e5)

    parser.add_argument('--future_len', type=int, help='number of time point', default=80)
    parser.add_argument('--time_len', type=int, help='number of time point', default=21)

    parser.add_argument('--action_len', type=int, help='smoothing factor for action', default=1)
    parser.add_argument('--action_type', type=str, help='number of time point', default='traj')

    parser.add_argument('--agent_state_dim', type=int, help='past state dim for agents', default=11)
    parser.add_argument('--agent_num', type=int, help='number of agents', default=32)

    parser.add_argument('--static_objects_state_dim', type=int, help='state dim for static objects', default=10)
    parser.add_argument('--static_objects_num', type=int, help='number of static objects', default=5)

    parser.add_argument('--lane_len', type=int, help='number of lane point', default=20)
    parser.add_argument('--lane_state_dim', type=int, help='state dim for lane point', default=12)
    parser.add_argument('--lane_num', type=int, help='number of lanes', default=70)

    parser.add_argument('--route_len', type=int, help='number of route lane point', default=20)
    parser.add_argument('--route_state_dim', type=int, help='state dim for route lane point', default=12)
    parser.add_argument('--route_num', type=int, help='number of route lanes', default=25)

    # DataLoader parameters
    parser.add_argument('--augment_prob', type=float, help='augmentation probability', default=0.5)
    parser.add_argument('--normalization_file_path', default='normalization.json', help='filepath of normalizaiton.json', type=str)
    parser.add_argument('--use_data_augment', default=True, type=boolean)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--pin-mem', action='store_true', help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem', help='')
    parser.set_defaults(pin_mem=True)

    # Training
    parser.add_argument('--seed', type=int, help='fix random seed', default=3407)
    parser.add_argument('--train_epochs', type=int, help='epochs of training', default=500)
    parser.add_argument('--save_utd', type=int, help='save frequency', default=20)
    parser.add_argument('--batch_size', type=int, help='batch size (default: 2048)', default=2048)
    parser.add_argument('--learning_rate', type=float, help='learning rate (default: 5e-4)', default=5e-4)
    parser.add_argument('--warm_up_epoch', type=int, help='number of warm up', default=5)
    parser.add_argument('--encoder_drop_path_rate', type=float, help='encoder drop out rate', default=0.1)
    parser.add_argument('--decoder_drop_path_rate', type=float, help='decoder drop out rate', default=0.1)

    parser.add_argument('--alpha_planning_loss', type=float, help='coefficient of planning loss (default: 1.0)', default=1.0)

    parser.add_argument('--device', type=str, help='run on which device (default: cuda)', default='cuda')
    parser.add_argument('--use_ema', default=True, type=boolean)

    # Model
    parser.add_argument('--encoder_depth', type=int, help='number of encoding layers', default=3)
    parser.add_argument('--decoder_depth', type=int, help='number of decoding layers', default=3)
    parser.add_argument('--num_heads', type=int, help='number of multi-head', default=6)
    parser.add_argument('--hidden_dim', type=int, help='hidden dimension', default=192)
    parser.add_argument('--diffusion_model_type', type=str, help='type of diffusion model [x_start, score]', choices=['score', 'x_start'], default='x_start')

    # decoder
    parser.add_argument('--predicted_neighbor_num', type=int, help='number of neighbor agents to predict', default=10)
    parser.add_argument('--resume_model_path', type=str, help='path to resume model', default=None)
    parser.add_argument('--use_chunking', default=False, type=boolean, help='use chunking or not')
    parser.add_argument('--chunk_size', type=int, help='chunk size', default=1)
    parser.add_argument('--chunk_overlap', type=int, help='overlap area between consecutive chunks', default=1)
    parser.add_argument('--decoder_agent_attn_mask', default=False, type=boolean, help='use agent valid mask as attention mask (True) or key padding mask (False)')
    parser.add_argument('--if_factorized', default=False, type=boolean, help='use factorized decoder or not')
    parser.add_argument('--use_causal_attn', default=True, type=boolean, help='use causal mask in temporal attention or not')
    parser.add_argument('--use_agent_validity_in_temporal', default=False, type=boolean, help='use agent valid mask in temporal attention or not')
    parser.add_argument('--use_chunk_t_embed', default=False, type=boolean, help='add chunk time positional embedding or not')
    parser.add_argument('--ego_separate', default=False, type=boolean, help='separate ego chunk embedding with other agents')
    parser.add_argument('--key_padding', default=False, type=boolean, help='use key padding mask or additive attention mask in factorized attention')
    parser.add_argument('--pad_left', default=False, type=boolean, help='padding left or right to be zero')
    parser.add_argument('--pad_history', default=False, type=boolean, help='fill padding with history')
    parser.add_argument('--v2', default=False, type=boolean, help='use v2 final layer')
    parser.add_argument('--residual_emb', default=False, type=boolean, help='residual layer of time embedding')


    parser.add_argument('--logger', default='wandb', type=str, choices=['none', 'qualcomm', 'wandb', 'tensorboard'])
    parser.add_argument('--notes', default='', type=str)

    # Lightning specific arguments
    parser.add_argument('--accelerator', default='gpu', type=str, help='accelerator type')
    parser.add_argument('--devices', default=-1, type=int, help='number of devices to use (-1 for all)')
    parser.add_argument('--strategy', default='ddp', type=str, help='distributed strategy')

    args = parser.parse_args()

    args.state_normalizer = StateNormalizer.from_json(args)
    args.observation_normalizer = ObservationNormalizer.from_json(args)
    return args


def main():
    args = get_args()
    
    torch.set_float32_matmul_precision('high')

    from datetime import datetime
    time = datetime.now()
    time = time.strftime("%Y-%m-%d-%H:%M:%S")

    save_path = f"{args.save_dir}/training_log/{args.name}/{time}/"
    os.makedirs(save_path, exist_ok=True)
    # Save args
    args_dict = vars(args)
    args_dict = {k: v if not isinstance(v, (StateNormalizer, ObservationNormalizer)) else v.to_dict() for k, v in args_dict.items() }

    from mmengine.fileio import dump
    dump(args_dict, os.path.join(save_path, 'args.json'), file_format='json', indent=4)
    
    # Set seed
    set_seed(args.seed)
    
    # Initialize data module
    data_module = DiffusionPlannerDataModule(args)
    
    # Initialize model
    model = DiffusionPlannerLightningModule(args)
    
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
            project='FactorizedDiffusionPlanner3',
            name=args.name,
            save_dir=save_path,
            config=vars(args)
        )
        
    elif args.logger == 'tensorboard':
        logger = PLTensorBoardLogger(save_dir=save_path, name=args.name)
    
    # Setup callbacks
    callbacks = []
    has_val_data = bool(args.val_metadata or args.train_metadata or args.val_mapping_pkl or args.val_set)
    
    # Model checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{args.save_dir}/checkpoints/{args.name}/",
        filename='{epoch}-{step}-{val_loss:.4f}' if has_val_data else '{step}-{train_loss:.4f}',
        monitor='val_loss' if has_val_data else 'train_loss',
        mode='min',
        save_top_k=-1, 
        every_n_train_steps=10000,
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
        gradient_clip_val=5.0,  # Gradient clipping
        gradient_clip_algorithm='norm',
        log_every_n_steps=50,
        enable_checkpointing=True,
        enable_progress_bar=True,
        enable_model_summary=True,
        check_val_every_n_epoch=1,
    )
    
    # Resume from checkpoint if specified
    ckpt_path = None
    if args.resume_model_path is not None:
        ckpt_path = args.resume_model_path
    
    # Start training
    trainer.fit(model, data_module, ckpt_path=ckpt_path)


if __name__ == "__main__":
    main()
