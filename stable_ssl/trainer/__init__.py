from .joint_embedding.simclr import SimCLR
from .joint_embedding.base import SSLTrainer
from .supervised import Supervised
from .base import Trainer

__all__ = ["Trainer", "SSLTrainer", "Supervised", "SimCLR"]
