from jepa_stage1_clip_ablation import config
from jepa_stage1_clip_ablation.train import train_experiment


if __name__ == "__main__":
    for name in config.DEFAULT_EXPERIMENT_ORDER:
        train_experiment(config.EXPERIMENTS[name])
