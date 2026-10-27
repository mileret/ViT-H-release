import os
import sys
from datetime import timedelta
from pathlib import Path


# Running `python tools/train.py` puts only `tools/` on sys.path.  Add the
# repository root explicitly so top-level project modules such as
# `canonicalize_motion.py` are importable without requiring PYTHONPATH.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
# Make every project-relative path stable even when train.py is launched from
# another working directory or through a scheduler wrapper.
os.chdir(REPO_ROOT)


import torch


_torch_load = torch.load
torch.load = lambda *a, **kw: _torch_load(*a, **{"weights_only": False, **kw})

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks.checkpoint import Checkpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy

from hmr4d.utils.pylogger import Log
from hmr4d.configs import register_store_offline
from hmr4d.utils.vis.rich_logger import print_cfg
from hmr4d.utils.net_utils import load_pretrained_model, get_resume_ckpt_path


def get_callbacks(cfg: DictConfig) -> list:
    """Parse and instantiate all the callbacks in the config."""
    if not hasattr(cfg, "callbacks") or cfg.callbacks is None:
        return None
    # Handle special callbacks
    enable_checkpointing = cfg.pl_trainer.get("enable_checkpointing", True)
    # Instantiate all the callbacks
    callbacks = []
    for callback in cfg.callbacks.values():
        if callback is not None:
            cb = hydra.utils.instantiate(callback, _recursive_=False)
            # skip when disable checkpointing and the callback is Checkpoint
            if not enable_checkpointing and isinstance(cb, Checkpoint):
                continue
            else:
                callbacks.append(cb)
    return callbacks


def train(cfg: DictConfig) -> None:
    """Train/Test"""
    Log.info(f"[Exp Name]: {cfg.exp_name}")
    if cfg.task == "fit":
        Log.info(f"[GPU x Batch] = {cfg.pl_trainer.devices} x {cfg.data.loader_opts.train.batch_size}")
    if cfg.seed is not None:
        pl.seed_everything(cfg.seed)

    # Initialize W&B before constructing the model/data module or entering
    # Trainer.fit(), because Lightning's subprocess DDP launcher otherwise
    # starts every local rank before rank 0 discovers an authentication or
    # entity-permission error.  Keeping this on local rank 0 prevents spawned
    # child ranks from creating duplicate W&B runs.
    logger = hydra.utils.instantiate(cfg.logger, _recursive_=False)
    local_rank = os.environ.get("LOCAL_RANK")
    if isinstance(logger, WandbLogger) and local_rank in (None, "0"):
        Log.info("[Logger] Preflight W&B initialization before model/data/DDP setup")
        _ = logger.experiment

    # preparation
    datamodule: pl.LightningDataModule = hydra.utils.instantiate(cfg.data, _recursive_=False)
    model: pl.LightningModule = hydra.utils.instantiate(cfg.model, _recursive_=False)
    if cfg.ckpt_path is not None:
        load_pretrained_model(model, cfg.ckpt_path)

    # PL callbacks
    callbacks = get_callbacks(cfg)
    has_ckpt_cb = any([isinstance(cb, Checkpoint) for cb in callbacks])
    if not has_ckpt_cb and cfg.pl_trainer.get("enable_checkpointing", True):
        Log.warning("No checkpoint-callback found. Disabling PL auto checkpointing.")
        cfg.pl_trainer = {**cfg.pl_trainer, "enable_checkpointing": False}
    # PL-Trainer
    if cfg.task == "test":
        Log.info("Test mode forces full-precision.")
        cfg.pl_trainer = {**cfg.pl_trainer, "precision": 32}
    trainer_kwargs = OmegaConf.to_container(cfg.pl_trainer, resolve=True)
    ddp_timeout_raw = os.environ.get("VITH_DDP_TIMEOUT_SECONDS")
    if ddp_timeout_raw is not None:
        ddp_timeout_seconds = int(ddp_timeout_raw)
        if ddp_timeout_seconds <= 0:
            raise ValueError(
                "VITH_DDP_TIMEOUT_SECONDS must be positive, got "
                f"{ddp_timeout_seconds}."
            )
        strategy_name = trainer_kwargs.get("strategy")
        ddp_strategy_kwargs = {
            "ddp": {},
            "ddp_find_unused_parameters_true": {"find_unused_parameters": True},
            "ddp_find_unused_parameters_false": {"find_unused_parameters": False},
        }
        if strategy_name in ddp_strategy_kwargs:
            trainer_kwargs["strategy"] = DDPStrategy(
                timeout=timedelta(seconds=ddp_timeout_seconds),
                **ddp_strategy_kwargs[strategy_name],
            )
            Log.info(
                f"[DDP] Process-group timeout={ddp_timeout_seconds}s "
                f"(strategy={strategy_name})"
            )
        else:
            Log.warning(
                "VITH_DDP_TIMEOUT_SECONDS is set but strategy is not a "
                f"supported DDP string: {strategy_name!r}; keeping it unchanged."
            )

    trainer = pl.Trainer(
        accelerator="gpu",
        logger=logger if logger is not None else False,
        callbacks=callbacks,
        **trainer_kwargs,
    )

    if cfg.task == "fit":
        resume_path = None
        if cfg.resume_mode is not None:
            resume_path = get_resume_ckpt_path(cfg.resume_mode, ckpt_dir=cfg.callbacks.model_checkpoint.dirpath)
            Log.info(f"Resume training from {resume_path}")
        Log.info("Start Fitiing...")
        # Pass the datamodule itself so Lightning launches all DDP ranks and
        # initializes the process group before DataModule.setup() performs the
        # expensive PTH-backed dataset construction on each rank.
        trainer.fit(model, datamodule=datamodule, ckpt_path=resume_path)
    elif cfg.task == "test":
        Log.info("Start Testing...")
        trainer.test(model, datamodule=datamodule)
    else:
        raise ValueError(f"Unknown task: {cfg.task}")

    Log.info("End of script.")


@hydra.main(version_base="1.3", config_path="../hmr4d/configs", config_name="train")
def main(cfg) -> None:
    print_cfg(cfg, use_rich=True)
    train(cfg)


if __name__ == "__main__":
    register_store_offline()
    main()
