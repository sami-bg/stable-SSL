from dataclasses import dataclass, field
from typing import Optional
import warnings
import logging


@dataclass
class GeneralConfig:
    """
    Configuration for general training parameters.

    Parameters:
    -----------
    seed : int, optional
        Random seed for reproducibility. Default is None.
    eval_only : bool, optional
        Whether to only evaluate the model without training. Default is False.
    epochs : int, optional
        Number of epochs to train the model. Default is 10.
    max_steps : int, optional
        Maximum number of steps per epoch. Default is -1.
    eval_each_epoch : bool, optional
        Whether to evaluate the model at the end of each epoch. Default is False.
    batch_size : int, optional
        Batch size for training. Default is 2048.
    metric : str, optional
        Metric to use in the embedding space. Default is "cosine
    """

    seed: Optional[int] = None
    eval_only: bool = False
    epochs: int = 10
    max_steps: int = -1
    eval_each_epoch: bool = False
    batch_size: int = 2048
    metric: str = "cosine"


@dataclass
class ModelConfig:
    """
    Configuration for the model architecture parameters.

    Parameters:
    -----------
    architecture : str
        Neural network architecture to use (e.g., "resnet18"). Default is "resnet18".
    sync_batchnorm : bool, optional
        Whether to use synchronized batch normalization. Default is False.
    memory_format : str, optional
        Memory format for tensors (e.g., "channels_last"). Default is "channels_last".
    """

    architecture: str = "resnet18"
    sync_batchnorm: bool = False
    memory_format: str = "channels_last"


@dataclass
class OptimConfig:
    """
    Configuration for the optimizer used for training the model.

    Parameters:
    -----------
    optimizer : str
        Type of optimizer to use (e.g., "AdamW", "RMSprop", "SGD", "LARS"). Default is "AdamW".
    learning_rate : float
        Learning rate for the optimizer. Default is 1e-3.
    weight_decay : float
        Weight decay for the optimizer. Default is 0.
    momentum : float
        Momentum for the optimizer. Default is None.
    beta1 : float
        Beta1 parameter for the AdamW optimizer. Default is 0.9.
    beta2 : float
        Beta2 parameter for the AdamW optimizer. Default is 0.999.
    grad_max_norm : float, optional
        Maximum norm for gradient clipping. Default is None.
    """

    optimizer: str = "AdamW"
    learning_rate: float = 1e-3
    weight_decay: float = 0
    momentum: float = None
    beta1: float = 0.9
    beta2: float = 0.999
    grad_max_norm: Optional[float] = None

    def __post_init__(self):
        if self.optimizer not in ["AdamW", "RMSprop", "SGD", "LARS"]:
            raise ValueError(
                f"Invalid optimizer: {self.optimizer}. Must be one of "
                "'AdamW', 'RMSprop', 'SGD', 'LARS'."
            )

        # Ensure parameters are provided appropriately based on the optimizer
        if (
            self.optimizer == "SGD"
            or self.optimizer == "RMSprop"
            or self.optimizer == "LARS"
        ):
            if self.beta1 is not None or self.beta2 is not None:
                warnings.warn(
                    f"The optimizer {self.optimizer} does not use the beta1 "
                    f"or beta2 parameters. Here beta1 = {self.beta1} and "
                    f"beta2 = {self.beta2} are thus ignored."
                )
        elif self.optimizer == "AdamW":
            if self.momentum is not None:
                warnings.warn(
                    "The AdamW optimizer does not use the momentum parameter. "
                    f"Here momentum = {self.momentum} is thus ignored. "
                )


@dataclass
class HardwareConfig:
    """
    Hardware configuration for training.

    Parameters:
    -----------
    float16 : bool, optional
        Whether to use mixed precision (float16) for training. Default is False.
    gpu : int, optional
        GPU device ID to use for training. Default is 0.
    world_size : int, optional
        Number of processes participating in distributed training. Default is 1.
    port : int, optional
        Port number for distributed training. Default is None.
    workers: int, optional
        Number of workers for data loading. Default is 4.
    """

    float16: bool = False
    gpu: int = 0
    world_size: int = 1
    port: Optional[int] = None
    workers: int = 4


@dataclass
class LogConfig:
    """
    Configuration for logging and checkpointing during training.

    Parameters:
    -----------
    folder : str, optional
        Path to the folder where logs and checkpoints will be saved.
        Default is the current directory.
    add_version : bool, optional
        Whether to append a version number to the folder path. Default is False.
    load_from : str, optional
        Path to a checkpoint from which to load the model, optimizer, and scheduler. Default is None.
    log_level : int, optional
        Logging level (e.g., logging.INFO). Default is logging.INFO.
    checkpoint_frequency : int, optional
        Frequency of saving checkpoints (in terms of epochs). Default is 1.
    save_final_model : bool, optional
        Whether to save the final trained model. Default is False.
    final_model_name : str, optional
        Name for the final saved model. Default is "final_model".
    """

    folder: str = "."
    add_version: bool = False
    load_from: str = "ckpt"
    log_level: int = logging.INFO
    checkpoint_frequency: int = 1
    save_final_model: bool = False
    final_model_name: str = "final_model"


@dataclass
class SSLConfig:
    """
    Global configuration for training a Self-Supervised Learning (SSL) model.

    Parameters:
    -----------
    general : GeneralConfig
        General training parameters.
    optim : OptimConfig
        Optimizer configuration.
    model : ModelConfig
        Model architecture configuration.
    hardware : HardwareConfig
        Hardware configuration.
    log : LogConfig
        Logging and checkpointing configuration
    """

    general: GeneralConfig = field(default_factory=GeneralConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    log: LogConfig = field(default_factory=LogConfig)
