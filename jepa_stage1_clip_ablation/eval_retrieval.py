import json

import numpy as np
import torch
from torch.utils.data import DataLoader

from . import config
from .dataset import CADJEPADataset, build_collate_fn
from .model import create_model
from .utils import (
    compute_retrieval_metrics,
    normalize_embeddings,
    read_jsonl,
    write_csv,
    write_json,
)


@torch.no_grad()
def eval_experiment(exp_cfg):
    from transformers import CLIPProcessor

    checkpoint_path = exp_cfg["output_dir"] / "checkpoints" / "best.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Best checkpoint not found: {checkpoint_path}")

    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    processor = CLIPProcessor.from_pretrained(config.CLIP_MODEL_NAME)
    dataset = CADJEPADataset(config.TEST_MANIFEST, processor=processor)
    loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        collate_fn=build_collate_fn(processor),
    )
    manifest_rows = read_jsonl(config.TEST_MANIFEST)

    model = create_model(exp_cfg).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.eval()

    all_pred = []
    all_target = []
    for batch in loader:
        outputs = model(
            before_pixel_values=batch["before_pixel_values"].to(device),
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            highlight_pixel_values=batch["highlight_pixel_values"].to(device),
        )
        all_pred.append(outputs["z_pred"].cpu())
        all_target.append(outputs["z_target"].cpu())

    z_pred = torch.cat(all_pred, dim=0)
    z_target = torch.cat(all_target, dim=0)
    similarity = normalize_embeddings(z_pred) @ normalize_embeddings(z_target).T
    metrics, ranks = compute_retrieval_metrics(similarity)
    sim_np = similarity.numpy()

    details = []
    for idx, row in enumerate(manifest_rows):
        best_idx = int(np.argmax(sim_np[idx]))
        details.append(
            {
                "sample_id": row["sample_id"],
                "rank": ranks[idx],
                "top1": int(ranks[idx] <= 1),
                "top5": int(ranks[idx] <= 5),
                "top10": int(ranks[idx] <= 10),
                "positive_similarity": float(sim_np[idx, idx]),
                "best_match_sample_id": manifest_rows[best_idx]["sample_id"],
                "best_match_similarity": float(sim_np[idx, best_idx]),
                "instruction": row["instruction"],
                "before_image": row["before_image"],
                "highlight_image": row["highlight_image"],
            }
        )

    write_json(exp_cfg["output_dir"] / "eval" / "test_retrieval_summary.json", metrics)
    write_csv(
        exp_cfg["output_dir"] / "eval" / "test_retrieval_details.csv",
        details,
        [
            "sample_id",
            "rank",
            "top1",
            "top5",
            "top10",
            "positive_similarity",
            "best_match_sample_id",
            "best_match_similarity",
            "instruction",
            "before_image",
            "highlight_image",
        ],
    )
    np.save(exp_cfg["output_dir"] / "eval" / "test_similarity_matrix.npy", sim_np)

    return metrics, checkpoint.get("epoch", 0)


def compare_experiments():
    rows = []
    for exp_cfg in [config.EXP_A, config.EXP_B]:
        metrics, best_epoch = eval_experiment(exp_cfg)
        rows.append(
            {
                "experiment_name": exp_cfg["name"],
                "top1": metrics["top1"],
                "top5": metrics["top5"],
                "top10": metrics["top10"],
                "mrr": metrics["mrr"],
                "mean_rank": metrics["mean_rank"],
                "median_rank": metrics["median_rank"],
                "best_epoch": best_epoch,
                "best_checkpoint": str(exp_cfg["output_dir"] / "checkpoints" / "best.pt"),
            }
        )
    write_csv(
        config.COMPARE_CSV,
        rows,
        [
            "experiment_name",
            "top1",
            "top5",
            "top10",
            "mrr",
            "mean_rank",
            "median_rank",
            "best_epoch",
            "best_checkpoint",
        ],
    )


if __name__ == "__main__":
    compare_experiments()
