import hydra
import os
from hydra import utils
from omegaconf import DictConfig, OmegaConf

from stable_ssl.config import (
    TrainerConfig,
    OptimConfig,
    ModelConfig,
    HardwareConfig,
    LogConfig,
    DataConfig,
)
from stable_ssl.ssl_modules import SimCLR, AutoCLR
from stable_ssl.supervised import Supervised

model_dict = {
    "SimCLR": SimCLR,
    "Supervised": Supervised,
    "AutoCLR": AutoCLR,
}


@hydra.main(config_path="configs")
def main(cfg: DictConfig):

    # Convert hydra config file to dictionary
    cfg_dict = OmegaConf.to_object(cfg)

    # Create the input for trainer
    args = TrainerConfig(
        data=DataConfig(**cfg_dict.get("data", {})),
        optim=OptimConfig(**cfg_dict.get("optim", {})),
        model=ModelConfig(**cfg_dict.get("model", {})),
        hardware=HardwareConfig(**cfg_dict.get("hardware", {})),
        log=LogConfig(**cfg_dict.get("log", {})),
    )

    args.data.data_dir = os.path.join(
        utils.get_original_cwd(), args.data.data_dir, args.data.dataset
    )

    print("--- Arguments ---")
    print(args)

    # Create a trainer object
    trainer = model_dict[args.model.model](args)
    trainer()


if __name__ == "__main__":
    main()
