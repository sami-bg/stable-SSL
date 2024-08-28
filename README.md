# Stable-SSL

## Installation

The library is not yet available on PyPI. You can install it from the source code, as follows.

```python
pip install -e .
```
Or you can also run:

```python
pip install git+https://github.com/rbalestr-lab/stable-SSL
```

## How to launch experiments

The file `main.py` to launch experiments is located in the `runs/` folder.

The default parameters are given in the `sable_ssl/config.py` file.
The parameters are structured in the following groups : data, model, hardware, log, optim.


#### Using default config files

You can use default config files that are located in `runs/configs`. To do so, simply specify the config file with the `--config-name` command as follows:

```python
python3 runs/main.py --config-name=simclr_cifar10_sgd
```

#### Setting params in command line

You can modify/add parameters of the config file by adding `++group.variable=value` as follows 

```python
python3 runs/main.py --config-name=simclr_cifar10_sgd ++optim.lr=2
```

**Remark**: If `group.variable` is already in the config file you can use `group.variable=value` and if it is not you can use `+group.variable=value`. The `++` command handles both cases that's why I would recommend using it.

#### Multi-run

To launch multiple runs, add `-m` and specify the multiple values to try as `++group.variable=value1,value2,value3`. For instance:

```python
python3 runs/main.py --config-name=simclr_cifar10_sgd -m ++optim.lr=2,5,10
```
