from . import config
from .train import train_experiment


if __name__ == "__main__":
    for name in config.DEFAULT_EXPERIMENT_ORDER:
        train_experiment(config.EXPERIMENTS[name])
