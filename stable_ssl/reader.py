import wandb
import pandas as pd
import numpy as np
from multiprocessing import Pool
from tqdm import tqdm


def wandb_project(entity, project, max_steps=-1):
    api = wandb.Api()
    runs = api.runs(f"{entity}/{project}")
    configs = []
    dfs = []
    with Pool(5) as p:
        results = list(
            tqdm(
                p.imap(
                    _wandb_run_packed,
                    [(entity, project, r.id, max_steps) for r in runs],
                ),
                total=len(runs),
                desc=f"Downloading project: {project}",
            )
        )
    configs, dfs = zip(*results)
    config = pd.DataFrame(configs)
    return config, dfs


def _wandb_run_packed(args):
    return wandb_run(*args)


def wandb_run(entity, project, run_id, max_steps=-1):
    api = wandb.Api()
    run = api.run(f"{entity}/{project}/{run_id}")

    if max_steps == -1:
        max_steps = run.lastHistoryStep
        min_step = 0
    else:
        min_step = run.lastHistoryStep - max_steps

    summary = run.summary
    # extract names that are not hidden
    columns = [k for k, v in summary.items() if k[0] != "_" and np.isscalar(v)]
    # add back the runtime and timestamp and this is useful to users
    columns += ["_runtime", "_timestamp"]
    df = pd.DataFrame(index=range(max_steps), columns=columns)
    df.index.name = "step"
    for row_idx, row in tqdm(
        enumerate(run.scan_history(min_step=min_step)),
        total=max_steps,
        desc=f"Downloading run: {run.name}",
    ):
        df.update(pd.DataFrame([row], index=[row_idx]))
    return run.config, df
