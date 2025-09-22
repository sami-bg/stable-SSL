import os
from pathlib import Path
from typing import Optional

import lightning as pl
from loguru import logger as logging


def configure_checkpointing(trainer: pl.Trainer, ckpt_path: Optional[Path]) -> None:
    """Analyzes user configuration for checkpointing and ensures it's set up correctly.

    This function is designed to handle four primary user scenarios by inspecting
    the state of the Trainer's callbacks and the `ckpt_path` provided to the Manager.
    It provides informative logs for each case and can add a `ModelCheckpoint`
    callback as a safety net if needed.

    Args:
        trainer: The PyTorch Lightning Trainer instance whose callbacks will be checked and
                 potentially modified.
        ckpt_path: The checkpoint path provided to the Manager, which indicates the user's
                   intent to resume from or save to a specific file.
    """
    logging.info("\t● 📞📞📞 CHECKPOINTING SETUP 📞📞📞")

    # This flag checks if the user *explicitly* added any ModelCheckpoint
    # instance in their configuration. It runs before Lightning's potential
    # default callback is added.
    is_mc_explicitly_configured = any(
        isinstance(cb, pl.pytorch.callbacks.ModelCheckpoint) for cb in trainer.callbacks
    )

    # This flag checks if any of the *explicitly added* callbacks are configured
    # to save to the directory containing the specific path the Manager cares about.
    is_manager_path_handled_by_callback = False
    is_slurm_job = "SLURM_JOB_ID" in os.environ

    if is_mc_explicitly_configured and ckpt_path:
        for callback in trainer.callbacks:
            if isinstance(callback, pl.pytorch.callbacks.ModelCheckpoint):
                # manually resolve the directory path the callback will use.
                resolved_dirpath = Path(
                    callback._ModelCheckpoint__resolve_ckpt_dir(trainer)
                ).resolve()
                # instead of comparing filenames, which are templates, we compare the parent directory.
                # if the user's ckpt_path is inside the callback's save directory, we
                # can be confident their configuration is aligned.
                if ckpt_path.parent == resolved_dirpath:
                    is_manager_path_handled_by_callback = True
                    break

    # Case 1: Intentional ckpt_path, correct callback passed in - do nothing
    if ckpt_path is not None and is_manager_path_handled_by_callback:
        logging.info(
            f"\t\t Checkpoint: `manager.ckpt_path` ({ckpt_path}) is set and a matching `ModelCheckpoint` callback was found to be saving to the same directory."
        )
        if is_slurm_job:
            logging.info(
                "\t\t This setup is ready for SLURM preemption and requeueing."
            )

    # Case 2: Intentional ckpt_path, but no callback found - assume the user forgot and add a callback
    elif ckpt_path is not None and not is_manager_path_handled_by_callback:
        logging.warning(
            f"\t\t Checkpoint mismatch: `manager.ckpt_path` ({ckpt_path}) was provided, but no matching `ModelCheckpoint` callback was found."
        )
        logging.info(
            "\t\t Automatically creating a `ModelCheckpoint` to save to the specified path to prevent data loss."
        )

        saver = pl.pytorch.callbacks.ModelCheckpoint(
            dirpath=str(ckpt_path.parent),
            filename=ckpt_path.with_suffix("").name,
            save_last=False,  # be explicit, last.ckpt is a special case
            save_on_train_epoch_end=True,
            verbose=True,
            enable_version_counter=False,
        )
        trainer.callbacks.append(saver)
        logging.info(
            "\t\t - Automatic `ModelCheckpoint` callback has been added to the trainer."
        )

    # Case 3: No checkpoint, but with ModelCheckpoint callback - assume we are training from scratch.
    elif ckpt_path is None and is_mc_explicitly_configured:
        logging.info(
            "\t\t Checkpointing: A user-defined `ModelCheckpoint` callback was found. It will be used for saving checkpoints."
        )
        logging.info(
            "\t\t The `Manager` will not manage resuming from a specific path as `manager.ckpt_path` was not provided."
        )
        if is_slurm_job:
            logging.warning(
                "\t\t SLURM WARNING: Since `manager.ckpt_path` is not set, this job will restart from scratch if requeued, even though checkpoints are being saved elsewhere."
            )

    # Case 4: No checkpoint and no ModelCheckpoint callback - assume we are training without saving checkpoints
    elif ckpt_path is None and not is_mc_explicitly_configured:
        logging.info(
            "\t\t No Checkpointing: No `manager.ckpt_path` was provided and no `ModelCheckpoint` callback was found."
        )
        logging.info("\t\t The model will not be saved during this run.")
        if is_slurm_job:
            logging.error(
                "\t\t CRITICAL SLURM WARNING: This job will lose all progress if it is preempted or requeued. It is highly recommended to configure checkpointing."
            )
