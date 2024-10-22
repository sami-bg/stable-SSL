.. _getting_started:

.. currentmodule:: stable_ssl

.. automodule:: stable_ssl
   :no-members:
   :no-inherited-members:


Getting Started
================

Write and Read your logs (Wandb or JSON)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. _logs:

- **Loggers**: We support the `Weights and Biases <https://wandb.ai/site>`_ and `jsonlines <https://jsonlines.readthedocs.io/en/latest/>`_ for logging. For the Wandb, you will need to use the following tags: ``log.entity`` (optional), ``log.project`` (optional), ``log.run`` (optional). They are all optional since Wandb handles its own exceptions if those are not passed by users. For jsonlines, the ``log.folder`` / ``log.name`` is where the logs will be dumped. Both are also optional. ``log.folder`` will be set to ``./logs`` and ``log.name`` will be set to ``%Y%m%d_%H%M%S.%f`` of the call. References: ``stable_ssl.configs.LogConfig``, ``stable_ssl.configs.WandbConfig``.

- **Logging values**: we have a unified logging framework regardless of the logger you employ. You can directly use ``self.log({"loss": 0.001, "lr": 1})`` which will add an entry or row in Wandb or the text file. If you want to log many different things at once, it can be easier to "pack" your log commits, as in:

  .. code-block:: python

     self.log({"loss": 0.001}, commit=False)
     ...
     self.log({"lr": 1})

  `stable-SSL` will automatically pack those two and commit the logs.

- **Reading logs (Wandb):**

  .. code-block:: python

     from stable_ssl import reader

     # single run
     config, df = reader.wandb_run(
         ENTITY_NAME, PROJECT_NAME, RUN_NAME
     )

     # single project (multiple runs)
     configs, dfs = reader.wandb_project(ENTITY_NAME, PROJECT_NAME)

- **Reading logs (jsonl):**

  .. code-block:: python

     from stable_ssl import reader

     # single run
     config, df = reader.jsonl_run(
         FOLDER_NAME, RUN_NAME
     )
     # single project (multiple runs)
     configs, dfs = reader.jsonl_project(FOLDER_NAME)

- **Reading logs (json+CLI):**

  .. code-block:: bash

     python cli/plot_metric.py --path PATH --metric eval/epoch/acc1 --savefig ./test.png --hparams model.name,optim.lr


Installation
------------

To install stable-SSL, run in the terminal:

.. code-block:: shell

   pip install stable-ssl