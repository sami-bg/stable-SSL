import hydra
from omegaconf import DictConfig, OmegaConf

from config import (
    SSLConfig,
    GeneralConfig,
    OptimConfig,
    ModelConfig,
    HardwareConfig,
    LogConfig,
)
from trainer import Trainer
from simclr import SimCLR


@hydra.main(config_path="inputs", config_name="config")
def main(cfg: DictConfig):

    # Convert hydra config file to dictionary
    cfg_dict = OmegaConf.to_object(cfg)

    # Create the input for trainer
    args = SSLConfig(
        general=GeneralConfig(**cfg_dict.get("general", {})),
        optim=OptimConfig(**cfg_dict.get("optim", {})),
        model=ModelConfig(**cfg_dict.get("model", {})),
        hardware=HardwareConfig(**cfg_dict.get("hardware", {})),
        log=LogConfig(**cfg_dict.get("log", {})),
    )

    # Create a trainer object
    trainer = SimCLR(args)
    trainer()


if __name__ == "__main__":
    main()
