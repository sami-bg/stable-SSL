import hydra
from omegaconf import DictConfig, OmegaConf

from config import GlobalConfig, ArchitectureConfig, TrainingConfig, OptimizerConfig
from trainer import Trainer


@hydra.main(config_path="inputs", config_name="config")
def main(cfg: DictConfig):

    # Convert DictConfig to a plain dictionary
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    # Efficiently create a Config dataclass instance
    config = GlobalConfig(
        architecture=ArchitectureConfig(**cfg_dict.get("architecture", {})),
        training=TrainingConfig(**cfg_dict.get("training", {})),
        optimizer=OptimizerConfig(**cfg_dict.get("optimizer", {})),
    )

    # Access different parts of the configuration
    print(config.architecture)
    print(config.training)
    print(config.optimizer)


if __name__ == "__main__":
    main()
