import shutil
from pathlib import Path
from weakref import proxy

import torch
import numpy as np
import orbax.checkpoint
from flax.training import orbax_utils
import lightning.pytorch as pl

import userfm.utils


log = userfm.utils.getLoggerByFilename(__file__)


class LogStats(pl.callbacks.Callback):
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.log('train_loss', outputs['loss'], batch_size=len(batch), on_epoch=True, prog_bar=True)
        self.log('train_loss_ema', outputs['loss_ema'], batch_size=len(batch), on_epoch=True, prog_bar=True)
        for k, v in outputs['monitors'].items():
            if isinstance(k, str):
                self.log(k, v, batch_size=len(batch), on_epoch=True, prog_bar=True)
            else:
                self.log(k.__class__.__name__, v, batch_size=len(batch), on_epoch=True, prog_bar=True)
        for k, v in outputs['monitors_ema'].items():
            if isinstance(k, str):
                self.log(f'{k}_ema', v, batch_size=len(batch), on_epoch=True, prog_bar=True)
            else:
                self.log(f'{k.__class__.__name__}_ema', v, batch_size=len(batch), on_epoch=True, prog_bar=True)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.log('loss_val', outputs['loss_val'], batch_size=len(batch), on_epoch=True, prog_bar=True)


class LogPredictStats(pl.callbacks.Callback):
    def __init__(self, save_constraint_values=False):
        super().__init__()
        self.log_constraint_values = save_constraint_values

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        outputs = pl_module._predict_state
        sample_constraint_values = outputs['event_constraint'].constraint(outputs['batch'])
        sample_constraint_values = torch.tensor(np.asarray(sample_constraint_values), dtype=torch.float32)
        satisfactory_sample_count = outputs['event_constraint'].satisfies_constraint(outputs['batch']).sum()
        # log.info('Satisfactory sample count (batch=%d): %d', batch_idx, satisfactory_sample_count.item())
        satisfactory_sample_count = torch.tensor(satisfactory_sample_count.item(), dtype=torch.float32)
        to_log = {}
        if self.log_constraint_values:
            to_log = {f'sample_constraint_{i}': c for i, c in enumerate(sample_constraint_values)}
        to_log['satisfactory_sample_count'] = satisfactory_sample_count
        self.log_dict(to_log, batch_size=1, on_step=True, on_epoch=False)


class ModelCheckpoint(pl.callbacks.ModelCheckpoint):
    CHECKPOINT_EQUALS_CHAR = '_'
    FILE_EXTENSION = ''

    @staticmethod
    def get_checkpoint_directories(filepath):
        filepath = Path(filepath)
        return filepath, filepath.parent/f'{filepath.stem}_ema{filepath.suffix}'

    def _save_checkpoint(self, trainer: "pl.Trainer", filepath: str) -> None:
        params = trainer.lightning_module.params
        params_ema = trainer.lightning_module.params_ema
        orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        for ckpt, directory in zip((params, params_ema), self.get_checkpoint_directories(filepath)):
            save_args = orbax_utils.save_args_from_target(ckpt)
            orbax_checkpointer.save(directory, ckpt, save_args=save_args, force=True)

        self._last_global_step_saved = trainer.global_step
        self._last_checkpoint_saved = filepath

        # notify loggers
        if trainer.is_global_zero:
            for logger in trainer.loggers:
                logger.after_save_checkpoint(proxy(self))

    def _remove_checkpoint(self, trainer: "pl.Trainer", filepath: str) -> None:
        """Calls the strategy to remove the checkpoint file."""
        for directory in self.get_checkpoint_directories(filepath):
            shutil.rmtree(directory)

    @staticmethod
    def _link_checkpoint(trainer: "pl.Trainer", filepath: str, linkpath: str) -> None:
        linkpath = Path(linkpath)
        linkpath_ema = linkpath.parent/f'{linkpath.stem}_ema{linkpath.suffix}'
        for directory, lp in zip(ModelCheckpoint.get_checkpoint_directories(filepath), (linkpath, linkpath_ema)):
            pl.callbacks.ModelCheckpoint._link_checkpoint(trainer, directory, lp)
