import os
from torch.utils.tensorboard import SummaryWriter
import wandb
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger as PLTensorBoardLogger


class UnifiedLogger:
    """Unified logger that supports multiple backends"""
    
    def __init__(self, args, run_name, save_path, rank=0, wandb_id=None):
        self.rank = rank
        self.logger = None
        self.tb_writer = None
        
        if rank != 0:
            return
            
        if args.logger == 'none':
            return
            
        elif args.logger in ['wandb', 'qualcomm']:
            # Login
            if args.logger == 'qualcomm':
                wandb.login(key='local-340ff373668e9a4ebcfa60a206a40ff0f2b75eef', host='https://server.auto-wandb.qualcomm.com/')
            else:
                wandb.login(key='caad08df59bfd0cb22f3613849ad66faeb65d4b0')
            
            # Create logger
            self.logger = WandbLogger(
                project='Diffusion-Planner',
                name=run_name,
                save_dir=save_path,
                id=wandb_id,
                config=vars(args)
            )
            
        elif args.logger == 'tensorboard':
            self.logger = PLTensorBoardLogger(save_dir=save_path, name=run_name)
            
        else:  # Original tensorboard with wandb sync
            os.environ["WANDB_MODE"] = "online" if getattr(args, 'use_wandb', True) else "offline"
            wandb.init(
                project='Diffusion-Planner',
                name=run_name,
                notes=getattr(args, 'notes', ''),
                id=wandb_id,
                sync_tensorboard=True,
                dir=save_path
            )
            self.tb_writer = SummaryWriter(log_dir=f'{save_path}/tb')
    
    def log_metrics(self, metrics, step, epoch=None):
        if self.rank != 0:
            return
            
        if self.logger:  # WandB or PL TensorBoard
            if hasattr(self.logger, 'log_metrics'):
                # Add epoch to metrics if provided
                if epoch is not None:
                    metrics = metrics.copy()  # Don't modify original dict
                    metrics['epoch'] = epoch
                self.logger.log_metrics(metrics, step=step)
            else:
                for k, v in metrics.items():
                    self.logger.experiment.add_scalar(k, v, step)
        elif self.tb_writer:  # Original TensorBoard
            for k, v in metrics.items():
                self.tb_writer.add_scalar(k, v, step)
    
    @property
    def id(self):
        """Get the WandB run ID if available"""
        if self.rank != 0:
            return None
            
        if self.logger and hasattr(self.logger, 'experiment'):
            # For WandbLogger
            return self.logger.experiment.id
        elif hasattr(self, 'tb_writer') and self.logger is None:
            # For original tensorboard with wandb sync
            import wandb
            if wandb.run:
                return wandb.run.id
        return None
    
    def finish(self):
        if self.rank != 0:
            return
            
        if self.tb_writer:
            self.tb_writer.close()
            wandb.finish()
        elif hasattr(self.logger, 'finalize'):
            # PyTorch Lightning loggers require a status argument
            self.logger.finalize("success")
        elif self.logger and hasattr(self.logger, 'experiment'):
            # For WandbLogger, finish the experiment
            if hasattr(self.logger.experiment, 'finish'):
                self.logger.experiment.finish()