import hydra
from omegaconf import DictConfig
from pathlib import Path

import stable_ssl
from stable_ssl.ssl_modules import SimCLR
from stable_ssl.supervised import Supervised

model_dict = {
    "SimCLR": SimCLR,
    "Supervised": Supervised,
}


@hydra.main(
    version_base=None,
    config_path=str(Path(stable_ssl.__file__).parent.parent / "runs" / "configs"),
)
def main(cfg: DictConfig):

    print("--- CFG ---")
    print(cfg)

    args = stable_ssl.get_args(cfg)

    print("--- Arguments ---")
    print(args)

    trainer = model_dict[args.model.model](args)
    trainer()


if __name__ == "__main__":
    main()
