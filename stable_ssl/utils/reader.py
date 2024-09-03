import wandb
import pandas as pd
import numpy as np
from tqdm import tqdm


def wandb_project(entity, project):
    api = wandb.Api()
    runs = api.runs(f"{entity}/{project}")
    configs = []
    dfs = []
    for run in runs:
        config, df = wandb_run(entity, project, run.id)
        configs.append(config)
        dfs.append(df)
    config = pd.DataFrame(configs)
    return config, dfs


def wandb_run(entity, project, run_id):
    api = wandb.Api()
    run = api.run(f"{entity}/{project}/{run_id}")
    summary = run.summary
    df = pd.DataFrame(
        index=range(run.lastHistoryStep),
        columns=[
            name
            for name in summary.keys()
            if name[0] != "_" and np.isscalar(summary[name])
        ]
        + ["_runtime", "_timestamp"],
    )
    df.index.name = "step"
    for row_idx, row in tqdm(
        enumerate(run.scan_history()),
        total=run.lastHistoryStep,
        desc=f"Downloading run: {run.name}",
    ):
        df.update(pd.DataFrame([row], index=[row_idx]))
    return run.config, df
