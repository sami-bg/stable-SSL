# Stable-SSL: Made by researchers for researchers

<center>
<img src="./assets/logo.jpg" alt="ssl logo" width="200"/>
</center>


Stable-SSL's philosophy relies on two core dogmas:
- *Lightweight enough to quickly iterate on research ideas*
- *Stable enough to quickly iterate on research ideas*
- *Fast enough to quickly iterate on research ideas*

## Architecture

Stable-SSL provides all the boilerplate to quickly get started doing AI research, with a focus on Self Supervised Learning (SSL) albeit other applicatins can certainly build upon Stable-SSL. In short, we provide a `BaseModel` class that calls the following methods (in order):
```
1. INITIALIZATION PHASE:
  - seed_everything()
  - initialize_modules()
  - initialize_optimizer()
  - initialize_scheduler()
  - load_checkpoint()

2. TRAIN/EVAL PHASE:
  - before_train_epoch()
  - for batch in train_loader:
    - before_train_step()
    - train_step(batch)
    - after_train_step()
  - after_train_epoch()
```
While the organization is related to the one e.g. provided by PytorchLightning, the goal here is to greatly reduce the codebase complexity without sacrificing performances. Think of PytorchLightning as industry driven (abstracting everything away) while Stable-SSL is academia driven (bringing everything in front of the user).


## Installation

The library is not yet available on PyPI. You can install it from the source code, as follows.

```bash
pip install -e .
```
Or you can also run:

```bash
pip install git+https://github.com/rbalestr-lab/stable-SSL
```

## How to launch experiments

The file `main.py` to launch experiments is located in the `runs/` folder.

The default parameters are given in the `sable_ssl/config.py` file.
The parameters are structured in the following groups : data, model, hardware, log, optim.


#### Using default config files

You can use default config files that are located in `runs/configs`. To do so, simply specify the config file with the `--config-name` command as follows:

```bash
python3 train.py --config-name=simclr_cifar10_sgd --config-path configs/
```

#### Setting params in command line

You can modify/add parameters of the config file by adding `++group.variable=value` as follows 

```bash
python3 main.py --config-name=simclr_cifar10_sgd ++optim.lr=2
```

**Remark**: If `group.variable` is already in the config file you can use `group.variable=value` and if it is not you can use `+group.variable=value`. The `++` command handles both cases that's why I would recommend using it.

#### Multi-run

To launch multiple runs, add `-m` and specify the multiple values to try as `++group.variable=value1,value2,value3`. For instance:

```bash
python3 main.py --config-name=simclr_cifar10_sgd -m ++optim.lr=2,5,10
```

#### Slurm 

To launch on slurm simply add `hydra/launcher=submitit_slurm` in the command line for instance:

```bash
python3 main.py hydra/launcher=submitit_slurm hydra.launcher.timeout_min=3
```

**Remark**: All the parameters of the slurm `hydra.launcher` are given [here](https://github.com/facebookresearch/hydra/blob/main/plugins/hydra_submitit_launcher/hydra_plugins/hydra_submitit_launcher/config.py) (similar to submitit).

Or to specify the slurm launcher you can add in the config file:

```yaml
defaults:
  - override hydra/launcher: submitit_slurm
```