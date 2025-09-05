import pytorch_lightning as pl
import torch

torch.set_float32_matmul_precision('medium')
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from models import build_model
from datasets import build_dataset
from utils.utils import set_seed, find_latest_checkpoint
from pytorch_lightning.callbacks import ModelCheckpoint  # Import ModelCheckpoint
import hydra
from omegaconf import OmegaConf
import os


@hydra.main(version_base=None, config_path="configs", config_name="config")
def train(cfg):
    set_seed(cfg.seed)
    OmegaConf.set_struct(cfg, False)  # Open the struct
    cfg = OmegaConf.merge(cfg, cfg.method)

    model = build_model(cfg)

    train_set = build_dataset(cfg)
    val_set = build_dataset(cfg, val=True)

    if cfg.process_cache_data_only:
        print("Finished processing cached data. Not execute training.")
        return

    train_batch_size = max(cfg.method['train_batch_size'] // len(cfg.devices),  1)
    eval_batch_size = max(cfg.method['eval_batch_size'] // len(cfg.devices), 1)

    call_backs = []
    if 'flow' not in cfg.method['model_name']:
        exp_name = "{}-bs_{}-lr_{}-na_{}".format(
        cfg.exp_name, 
        cfg.method['train_batch_size'],
        cfg.method['learning_rate'],
        cfg.method['max_num_agents'],
    )
    else:
        exp_name = "{}-bs_{}-lr_{}-noa_{}-nma_{}-na_{}-depth_{}-mlp_ratio_{}-chunk_size_{}-enc_layer_{}-train_type_{}-action_type_{}-loss_{}-supervise_agent_type_{}".format(
            cfg.exp_name, 
            cfg.method['train_batch_size'],
            cfg.method['learning_rate'],
            cfg.method['num_observed_agents'],
            cfg.method['num_modeled_agents'],
            cfg.method['max_num_agents'],
            cfg.method['dit_depth'],
            cfg.method['mlp_ratio'],
            cfg.method['chunk_size'],
            cfg.method['num_encoder_layers'],
            cfg.method['train_type'],
            cfg.method['action_type'],
            cfg.method['supervise_loss_type'],
            cfg.method['supervise_agent_type'],
    )

    checkpoint_callback = ModelCheckpoint(
        monitor=f"val/{cfg.method['supervise_loss_type']}-{cfg.method['action_type']}-fm_loss",  # Replace with your validation metric
        filename='{epoch}-{val/fm_loss:.2f}',
        save_top_k=1,
        mode='min',  # 'min' for loss/error, 'max' for accuracy
        dirpath=f'{cfg.save_ckpt_path}/{exp_name}'
    )
    print(f'model is saved to {cfg.save_ckpt_path}/{exp_name}')
    call_backs.append(checkpoint_callback)

    train_loader = DataLoader(
        train_set, batch_size=train_batch_size, num_workers=cfg.load_num_workers, drop_last=False,
        collate_fn=train_set.collate_fn)

    val_loader = DataLoader(
        val_set, batch_size=eval_batch_size, num_workers=cfg.load_num_workers, shuffle=False, drop_last=False,
        collate_fn=train_set.collate_fn)


    if cfg.logger == 'none' or cfg.debug:
        logger = None
    elif cfg.logger == 'qualcomm':
        import wandb
        from pytorch_lightning.loggers import WandbLogger
        apikey = 'local-340ff373668e9a4ebcfa60a206a40ff0f2b75eef'
        host = 'https://server.auto-wandb.qualcomm.com/'
        wandb.login(key=apikey, host=host)
        logger = WandbLogger(project='fm_planning', save_dir=cfg.logger_save, name=exp_name)
    elif cfg.logger == 'wandb':
        import wandb
        from pytorch_lightning.loggers import WandbLogger
        wandb.login(key='caad08df59bfd0cb22f3613849ad66faeb65d4b0')
        logger = WandbLogger(project='fm_planning', save_dir=cfg.logger_save, name=exp_name)
    elif cfg.logger == 'tensorboard':
        from pytorch_lightning import loggers as pl_loggers
        logger = pl_loggers.TensorBoardLogger(
        save_dir=cfg.logger_save, name=exp_name
        )
    

    trainer = pl.Trainer(
        max_epochs=cfg.method.max_epochs,
        # logger=None if cfg.debug else WandbLogger(project="unitraj", name=cfg.exp_name, id=cfg.exp_name),
        logger=logger,
        devices=1 if cfg.debug else cfg.devices,
        gradient_clip_val=cfg.method.grad_clip_norm,
        # accumulate_grad_batches=cfg.method.Trainer.accumulate_grad_batches,
        accelerator="cpu" if cfg.debug else "gpu",
        profiler="simple",
        strategy="auto" if cfg.debug else "ddp",
        callbacks=call_backs
    )

    # automatically resume training
    if cfg.ckpt_path is None and not cfg.debug:
        # Pattern to match all .ckpt files in the base_path recursively
        search_pattern = os.path.join('./unitraj', exp_name, '**', '*.ckpt')
        cfg.ckpt_path = find_latest_checkpoint(search_pattern)

    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=cfg.ckpt_path)


if __name__ == '__main__':
    train()
