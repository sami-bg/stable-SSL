import argparse
from pathlib import Path
import stable_ssl as ssl
import matplotlib.pyplot as plt


def parse_rules(v):
    rules = []
    for rule in v.split("&"):
        rules.append(rule.split("="))
    return rules


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=Path, required=True)
    parser.add_argument("--savefig", type=Path)
    parser.add_argument("--filters", type=parse_rules, default=[])
    parser.add_argument("--hparams", type=lambda x: x.split(","), default=[])
    parser.add_argument("--metric", type=str, required=True)
    args = parser.parse_args()

    configs, values = ssl.reader.jsonl_project(args.path)

    for (index, conf), ts in zip(configs.iterrows(), values):
        for rule in args.filters:
            if conf[rule[0]] != rule[1]:
                continue
        ts = [v[args.metric] for v in ts if args.metric in v]
        print(conf)
        p = [f"{name}: {conf[name]}" for name in args.hparams]
        plt.plot(ts, label=", ".join(p))
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.savefig)
    plt.close()
