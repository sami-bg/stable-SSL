import stable_ssl as ssl

data = ssl.utils.reader.read_wandb_run("excap", "single_dataset_sequential", "p67ng6bq")
print(data)