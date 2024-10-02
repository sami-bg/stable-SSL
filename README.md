# Stable-SSL: Made by researchers for researchers

<center>

### Stable | Minimalist | Fast *to quickly iterate on your research ideas*

<img src="./assets/logo.jpg" alt="ssl logo" width="200"/>

<sub>[Credit: Imagen3]</sub>

We achieve that by taking the best--and only the best--from the most eponymous AI libraries: PytorchLightning, VISSL, Wandb, Hydra, Submitit.

</center>

## Minimal Examples

### The few things you will need to implement for your `Trainer`

At the very least, you need to implement three methods: 
- `initialize_modules`: this method initialized whatever model and parameter to use for training/inference
- `forward`: that method that will be doing the prediction, e.g., for classification it will be p(y|x)
- `compute_loss`: that method should return a scalar value used for backpropagation/training. 

### Write and Read your logs (Wandb or jsonl)
We support the Weights and Biases API for logging as well as jsonlines (text).

- **Logging values**: you can directly use `self.log({"loss": 0.001, "lr": 1})` which will add an entry or row in Wandb or the text file. If you want to log many different things are once, it can be easier to ``pack'' your log commits, as in 
  ```
  self.log({"loss": 0.001}, commit=False)
  ...
  self.log({"lr": 1})
  ````
  `stable-SSL` will automaticall pack those two and commit the logs.

- **Reading logs (Wandb):**
  ```
  from stable_ssl import reader

  # single run
  config, df = reader.wandb_run(
      ENTITY_NAME, PROJECT_NAME, RUN_NAME
  )

  # single project (multiple runs)
  configs, dfs = reader.wandb_project(ENTITY_NAME, PROJECT_NAME)
  ```
- **Reading logs (jsonl):**
  ```
  from stable_ssl import reader

  # single run
  config, df = reader.jsonl_run(
      FOLDER_NAME, RUN_NAME
  )
  # single project (multiple runs)
  configs, dfs = reader.jsonl_project(FOLDER_NAME)
  ```

## Design

Stable-SSL provides all the boilerplate to quickly get started doing AI research, with a focus on Self Supervised Learning (SSL) albeit other applications can certainly build upon Stable-SSL. In short, we provide a `BaseModel` class that calls the following methods (in order):
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

### I want to pass my own hyper-parameters!

There are two options based if you leverage the Hydra framework or not.
<table border="0">
 <tr>
    <td><u><b style="font-size:10px">With Hydra</b></u></td>
    <td><u><b style="font-size:10px">Without Hydra</b></u></td>
 </tr>
 <tr>
    <td>Simply pass your custom argument when calling the function as `++my_argument=2` and you can retreive anywhere in the `Trainer` with `self.config.my_argument`. If you don't even use the Trainer, you can directly get the value of the parameter in the script like that

    ```
    @hydra.main(version_base=None)
    def main(cfg: DictConfig):

        args = ssl.get_args(cfg)
        args.my_argument

    ```

    </td>

    <td>You can directly pass to the `Trainer` whatever custom argument you might have as
    
    ```
    @hydra.main(version_base=None)
    def main(cfg: DictConfig):

        args = ssl.get_args(cfg)
        trainer = MyCustomSupervised(args, root="~/data", my_argument=2)
    ```

    and anywhere inside the `Trainer` instance you will have access to `self.config.my_argument`.

    </td>
 </tr>
</table>


#### Using default config files

You can use default config files that are located in `runs/configs`. To do so, simply specify the config file with the `--config-name` command as follows:

```bash
python3 train.py --config-name=simclr_cifar10_sgd --config-path configs/
```


### Classification case

- **How is the accuracy calculated?** the predictions are assumed to tbe the output of the forward method, then this is fed into a few metrics along with `self.data[1]` which is assumed to encode the labels

#### Setting params in command line

You can modify/add parameters of the config file by adding `++group.variable=value` as follows 

```bash
python3 main.py --config-name=simclr_cifar10_sgd ++optim.lr=2
# same but with SLURM
python3 main.py --config-name=simclr_cifar10_sgd ++optim.epochs=4 ++optim.lr=1 hydra/launcher=submitit_slurm hydra.launcher.timeout_min=1800 hydra.launcher.cpus_per_task=4 hydra.launcher.gpus_per_task=1 hydra.launcher.partition=gpu-he
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