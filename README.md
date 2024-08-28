# Advanced-SSL

### Installation

The library is not yet available on PyPI. You can install it from the source code, as follows.

```python
pip install -e .
```
Or you can also run:

```python
pip install git+https://github.com/rbalestr-lab/stable-SSL
```

### Single-run

```python
python3 run_exp.py --config-name=simclr_cifar10_sgd
```

# Append new param


### Multi-run

```python
python3 run_exp.py -m ++db.optim.batch_size=64,128
```
