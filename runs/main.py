import hydra
from omegaconf import DictConfig, OmegaConf

from stable_ssl.config import (
    TrainerConfig,
    OptimConfig,
    ArchitectureConfig,
    HardwareConfig,
    LogConfig,
)
from stable_ssl.trainer import Trainer
from stable_ssl.ssl_modules import SimCLR


@hydra.main(config_path="inputs", config_name="config")
def main(cfg: DictConfig):

    # Convert hydra config file to dictionary
    cfg_dict = OmegaConf.to_object(cfg)

    # Create the input for trainer
    args = TrainerConfig(
        optim=OptimConfig(**cfg_dict.get("optim", {})),
        architecture=ArchitectureConfig(**cfg_dict.get("architecture", {})),
        hardware=HardwareConfig(**cfg_dict.get("hardware", {})),
        log=LogConfig(**cfg_dict.get("log", {})),
    )

    # Create a trainer object
    trainer = SimCLR(args)
    trainer()


if __name__ == "__main__":
    main()
