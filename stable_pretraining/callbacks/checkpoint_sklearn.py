from typing import Optional

import numpy as np
from lightning.pytorch import Callback, LightningModule, Trainer
from loguru import logger as logging
from tabulate import tabulate
from lightning.pytorch.loggers import WandbLogger

from typing import Any, Dict
import lightning.pytorch as pl
from loguru import logger

from .. import SKLEARN_AVAILABLE

if SKLEARN_AVAILABLE:
    from sklearn.base import ClassifierMixin, RegressorMixin
else:
    ClassifierMixin = None
    RegressorMixin = None


class SklearnCheckpoint(Callback):
    """Callback for saving and loading sklearn models in PyTorch Lightning checkpoints.

    This callback automatically detects sklearn models (Regressors and Classifiers)
    attached to the Lightning module and handles their serialization/deserialization
    during checkpoint save/load operations. This is necessary because sklearn models
    are not natively supported by PyTorch's checkpoint system.

    The callback will:
    1. Automatically discover sklearn models attached to the Lightning module
    2. Save them to the checkpoint dictionary during checkpoint saving
    3. Restore them from the checkpoint during checkpoint loading
    4. Log information about discovered sklearn modules during setup

    Note:
        - Only attributes that are sklearn RegressorMixin or ClassifierMixin instances are saved
        - Private attributes (starting with '_') are ignored
        - The callback will raise an error if a sklearn model name conflicts with existing checkpoint keys
    """

    def setup(
        self, trainer: Trainer, pl_module: LightningModule, stage: Optional[str] = None
    ) -> None:
        sklearn_modules = _get_sklearn_modules(pl_module)
        stats = []
        for name, module in sklearn_modules.items():
            stats.append((name, module.__str__(), type(module)))
        headers = ["Module", "Name", "Type"]
        logging.info("Setting up SklearnCheckpoint callback!")
        logging.info("Sklearn Modules:")
        logging.info(f"\n{tabulate(stats, headers, tablefmt='heavy_outline')}")

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        logging.info("Checking for non PyTorch modules to save... 🔧")
        modules = _get_sklearn_modules(pl_module)
        for name, module in modules.items():
            if name in checkpoint:
                raise RuntimeError(
                    f"Can't pickle {name}, already present in checkpoint"
                )
            checkpoint[name] = module
            logging.info(f"Saving non PyTorch system: {name} 🔧")

    def on_load_checkpoint(self, trainer, pl_module, checkpoint):
        logging.info("Checking for non PyTorch modules to load... 🔧")
        if not SKLEARN_AVAILABLE:
            return
        for name, item in checkpoint.items():
            if isinstance(item, RegressorMixin) or isinstance(item, ClassifierMixin):
                setattr(pl_module, name, item)
                logging.info(f"Loading non PyTorch system: {name} 🔧")


def _contains_sklearn_module(item):
    if not SKLEARN_AVAILABLE:
        return False
    if isinstance(item, RegressorMixin) or isinstance(item, ClassifierMixin):
        return True
    if isinstance(item, list):
        return np.any([_contains_sklearn_module(m) for m in item])
    if isinstance(item, dict):
        return np.any([_contains_sklearn_module(m) for m in item.values()])
    return False


def _get_sklearn_modules(module):
    modules = dict()
    for name, item in vars(module).items():
        if name[0] == "_":
            continue
        item = getattr(module, name)
        if _contains_sklearn_module(item):
            modules[name] = item
    return modules


class StrictCheckpointCallback(Callback):
    """A PyTorch Lightning callback that controls strict checkpoint loading behavior.

    This callback allows you to load checkpoints with mismatched keys by setting
    `strict=False`, which is useful when:
    - Fine-tuning a model with a different architecture
    - Adding or removing layers from a pre-trained model
    - Loading partial weights from a checkpoint

    When `strict=False`, the callback will:
    1. Filter out parameters that don't exist in the current model
    2. Skip parameters with shape mismatches
    3. Clear optimizer states to prevent conflicts
    4. Provide detailed logging of all actions taken

    Args:
        strict (bool): Whether to enforce strict checkpoint loading.
            - If True: All keys must match exactly (default PyTorch Lightning behavior)
            - If False: Missing or mismatched keys are allowed and logged

    Example:
        ```python
        from lightning.pytorch import Trainer

        # Create callback with strict=False
        callback = StrictCheckpointCallback(strict=False)

        # Use with Trainer
        trainer = Trainer(callbacks=[callback])
        trainer.fit(model, ckpt_path="path/to/checkpoint.ckpt")
        ```

    Note:
        When using `strict=False`, optimizer states are automatically cleared
        to prevent shape mismatches during training resumption.
    """

    def __init__(self, strict: bool = True):
        """Initialize the StrictCheckpointCallback.

        Args:
            strict (bool): Whether to enforce strict checkpoint loading. Defaults to True.
        """
        super().__init__()
        self.strict = strict

        logger.info(f"StrictCheckpointCallback initialized with strict={self.strict}")
        if not self.strict:
            logger.warning(
                "Strict mode is disabled. Checkpoint loading will be lenient and "
                "may skip mismatched parameters."
            )

    def on_load_checkpoint(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        checkpoint: Dict[str, Any],
    ) -> None:
        """Called when loading a checkpoint.

        Args:
            trainer: The PyTorch Lightning Trainer instance
            pl_module: The LightningModule being trained
            checkpoint: The checkpoint dictionary being loaded
        """
        if self.strict:
            logger.info(
                "Strict mode enabled - using default checkpoint loading behavior"
            )
            return

        logger.info("=" * 80)
        logger.info("StrictCheckpointCallback: Processing checkpoint with strict=False")
        logger.info("=" * 80)

        if "state_dict" not in checkpoint:
            logger.warning("No 'state_dict' found in checkpoint. Skipping processing.")
            return

        checkpoint_state_dict = checkpoint["state_dict"]
        model_state_dict = pl_module.state_dict()

        logger.info(f"Checkpoint contains {len(checkpoint_state_dict)} parameters")
        logger.info(f"Current model contains {len(model_state_dict)} parameters")

        # Track statistics
        matched_keys = []
        missing_in_checkpoint = []
        missing_in_model = []
        shape_mismatches = []

        # Check for missing keys in checkpoint
        for key in model_state_dict.keys():
            if key not in checkpoint_state_dict:
                missing_in_checkpoint.append(key)
                logger.warning(f"⚠️  Parameter missing in checkpoint: '{key}'")

        # Check for extra keys and shape mismatches
        filtered_state_dict = {}
        for key, value in checkpoint_state_dict.items():
            if key not in model_state_dict:
                missing_in_model.append(key)
                logger.warning(
                    f"⚠️  Parameter in checkpoint but not in model: '{key}' - SKIPPING"
                )
            elif value.shape != model_state_dict[key].shape:
                shape_mismatches.append(
                    {
                        "key": key,
                        "checkpoint_shape": value.shape,
                        "model_shape": model_state_dict[key].shape,
                    }
                )
                logger.error(
                    f"❌ Shape mismatch for '{key}': "
                    f"checkpoint={value.shape}, model={model_state_dict[key].shape} - SKIPPING"
                )
            else:
                filtered_state_dict[key] = value
                matched_keys.append(key)

        # Update checkpoint with filtered state dict
        checkpoint["state_dict"] = filtered_state_dict

        # Clear optimizer states if there were any mismatches
        if missing_in_model or shape_mismatches or missing_in_checkpoint:
            if "optimizer_states" in checkpoint:
                logger.warning(
                    "🗑️  Clearing optimizer states due to parameter mismatches. "
                    "Training will restart optimizer from scratch."
                )
                checkpoint.pop("optimizer_states", None)

            if "lr_schedulers" in checkpoint:
                logger.warning(
                    "🗑️  Clearing learning rate scheduler states due to parameter mismatches."
                )
                checkpoint.pop("lr_schedulers", None)

        # Print summary
        logger.info("-" * 80)
        logger.success(f"✅ Successfully matched parameters: {len(matched_keys)}")

        if missing_in_checkpoint:
            logger.warning(
                f"⚠️  Parameters missing in checkpoint: {len(missing_in_checkpoint)}"
            )
            if len(missing_in_checkpoint) <= 10:
                for key in missing_in_checkpoint:
                    logger.debug(f"    - {key}")
            else:
                logger.debug(f"    (showing first 10 of {len(missing_in_checkpoint)})")
                for key in missing_in_checkpoint[:10]:
                    logger.debug(f"    - {key}")

        if missing_in_model:
            logger.warning(
                f"⚠️  Extra parameters in checkpoint (not in model): {len(missing_in_model)}"
            )
            if len(missing_in_model) <= 10:
                for key in missing_in_model:
                    logger.debug(f"    - {key}")
            else:
                logger.debug(f"    (showing first 10 of {len(missing_in_model)})")
                for key in missing_in_model[:10]:
                    logger.debug(f"    - {key}")

        if shape_mismatches:
            logger.error(f"❌ Shape mismatches found: {len(shape_mismatches)}")
            for mismatch in shape_mismatches[:10]:
                logger.debug(
                    f"    - {mismatch['key']}: "
                    f"{mismatch['checkpoint_shape']} → {mismatch['model_shape']}"
                )
            if len(shape_mismatches) > 10:
                logger.debug(f"    ... and {len(shape_mismatches) - 10} more")

        # Calculate loading percentage
        total_model_params = len(model_state_dict)
        loaded_percentage = (
            (len(matched_keys) / total_model_params * 100)
            if total_model_params > 0
            else 0
        )

        logger.info(
            f"📊 Checkpoint loading coverage: {loaded_percentage:.2f}% ({len(matched_keys)}/{total_model_params})"
        )

        logger.info("=" * 80)
        logger.success("StrictCheckpointCallback: Checkpoint processing complete")
        logger.info("=" * 80)


class WandbCheckpoint(Callback):
    """Callback for saving and loading sklearn models in PyTorch Lightning checkpoints.

    This callback automatically detects sklearn models (Regressors and Classifiers)
    attached to the Lightning module and handles their serialization/deserialization
    during checkpoint save/load operations. This is necessary because sklearn models
    are not natively supported by PyTorch's checkpoint system.

    The callback will:
    1. Automatically discover sklearn models attached to the Lightning module
    2. Save them to the checkpoint dictionary during checkpoint saving
    3. Restore them from the checkpoint during checkpoint loading
    4. Log information about discovered sklearn modules during setup

    Note:
        - Only attributes that are sklearn RegressorMixin or ClassifierMixin instances are saved
        - Private attributes (starting with '_') are ignored
        - The callback will raise an error if a sklearn model name conflicts with existing checkpoint keys
    """

    def setup(
        self, trainer: Trainer, pl_module: LightningModule, stage: Optional[str] = None
    ) -> None:
        logging.info("Setting up WandbCheckpoint callback!")

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        logging.info("Checking for Wandb params to save... 🔧")
        if isinstance(trainer.logger, WandbLogger):
            checkpoint["wandb"] = {"id": trainer.logger.version}
            # checkpoint["wandb_checkpoint_name"] = trainer.logger._checkpoint_name
            logging.info(f"Saving Wandb params {checkpoint['wandb']}")
        logging.info("Checking for Wandb params to save... Done!")

    def on_load_checkpoint(self, trainer, pl_module, checkpoint):
        logging.info("Checking for Wandb init params... 🔧")
        if "wandb" in checkpoint:
            logging.info("Wandb info in checkpoint!!! Restoring same run... 🔧")
            if not hasattr(trainer, "logger"):
                logging.warning("Expected Trainer to have a logger, leaving...")
                return
            elif not isinstance(trainer.logger, WandbLogger):
                logging.warning(
                    f"Expected WandbLogger, got {trainer.logger}, leaving..."
                )
                return
            else:
                logging.info("Trainer has a WandbLogger!")
            import wandb

            if wandb.run is None and trainer.global_rank > 0:
                logging.info(
                    "Run not initialized yet, skipping since this is a slave process!"
                )
                return
            logging.info(
                f"Deleting current run {wandb.run.entity}/{wandb.run.project}/{wandb.run.id}... 🔧"
            )
            api = wandb.Api()
            run = api.run(f"{wandb.run.entity}/{wandb.run.project}/{wandb.run.id}")
            wandb.finish()
            run.delete()
            trainer.logger._experiment = None
            wandb_id = checkpoint["wandb"]["id"]
            trainer.logger._wandb_init["id"] = wandb_id
            trainer.logger._id = wandb_id
            # to reset the run
            trainer.logger.experiment
            logging.info(
                f"New run {wandb.run.entity}/{wandb.run.project}/{wandb.run.id}... 🔧"
            )

            # trainer.logger._wandb_init = wandb_init
            # trainer.logger._project = trainer.logger._wandb_init.get("project")
            # trainer.logger._save_dir = trainer.logger._wandb_init.get("dir")
            # trainer.logger._name = trainer.logger._wandb_init.get("name")
            # trainer.logger._checkpoint_name = checkpoint["wandb_checkpoint_name"]
            # logging.info("Updated Wandb parameters: ")
            # logging.info(f"\t- project={trainer.logger._project}")
            # logging.info(f"\t- _save_dir={trainer.logger._save_dir}")
            # logging.info(f"\t- name={trainer.logger._name}")
            # logging.info(f"\t- id={trainer.logger._id}")
