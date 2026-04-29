import numpy as np
import torch
from torch.utils.data import DataLoader

from . import config
from .dataset import CADJEPADataset, build_collate_fn
from .model import create_model
from .train import _autocast_context
from .utils import (
    compute_retrieval_metrics,
    ensure_dir,
    normalize_embeddings,
    read_jsonl,
    write_csv,
    write_json,
)


@torch.no_grad()
def eval_experiment(exp_cfg):
    from transformers import AutoImageProcessor, AutoTokenizer

    checkpoint_path = exp_cfg["best_checkpoint"]
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Best checkpoint not found: {checkpoint_path}")

    ensure_dir(exp_cfg["eval_dir"])
    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    image_processor = AutoImageProcessor.from_pretrained(exp_cfg["vision_model_name"])
    tokenizer = AutoTokenizer.from_pretrained(exp_cfg["text_model_name"])
    dataset = CADJEPADataset(config.TEST_MANIFEST)
    loader = DataLoader(
        dataset,
        batch_size=exp_cfg["batch_size"],
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        collate_fn=build_collate_fn(image_processor, tokenizer, image_size=exp_cfg["image_size"]),
        pin_memory=True,
    )
    manifest_rows = read_jsonl(config.TEST_MANIFEST)
    manifest_by_id = {row["sample_id"]: row for row in manifest_rows}

    model = create_model(exp_cfg).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.eval()

    all_pred = []
    all_target = []
    valid_rows = []
    skipped_bad_samples = 0
    skipped_empty_batches = 0
    for batch in loader:
        if batch.get("bad_samples"):
            skipped_bad_samples += len(batch["bad_samples"])
        if batch.get("skip_batch"):
            skipped_empty_batches += 1
            continue
        with _autocast_context(device, exp_cfg["precision"]):
            outputs = model(
                before_pixel_values=batch["before_pixel_values"].to(device),
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                highlight_pixel_values=batch["highlight_pixel_values"].to(device),
        )
        all_pred.append(outputs["z_pred"].float().cpu())
        all_target.append(outputs["z_target"].float().cpu())
        valid_rows.extend([manifest_by_id[sample_id] for sample_id in batch["sample_ids"]])

    if skipped_bad_samples:
        print(
            f"[bad_sample] eval skipped_bad_samples={skipped_bad_samples} "
            f"skipped_empty_batches={skipped_empty_batches}",
            flush=True,
        )
    if not all_pred:
        raise RuntimeError("No valid evaluation batches were produced. Check image paths and bad samples.")

    z_pred = torch.cat(all_pred, dim=0)
    z_target = torch.cat(all_target, dim=0)
    similarity = normalize_embeddings(z_pred) @ normalize_embeddings(z_target).T
    metrics, ranks = compute_retrieval_metrics(similarity)
    sim_np = similarity.numpy()

    details = []
    for idx, row in enumerate(valid_rows):
        best_idx = int(np.argmax(sim_np[idx]))
        details.append(
            {
                "sample_id": row["sample_id"],
                "rank": ranks[idx],
                "top1": int(ranks[idx] <= 1),
                "top5": int(ranks[idx] <= 5),
                "top10": int(ranks[idx] <= 10),
                "positive_similarity": float(sim_np[idx, idx]),
                "best_match_sample_id": valid_rows[best_idx]["sample_id"],
                "best_match_similarity": float(sim_np[idx, best_idx]),
                "instruction": row["instruction"],
                "before_image": row["before_image"],
                "highlight_image": row["highlight_image"],
            }
        )

    write_json(exp_cfg["eval_summary"], metrics)
    write_csv(
        exp_cfg["eval_details"],
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
    np.save(exp_cfg["similarity_matrix"], sim_np)

    return metrics, checkpoint.get("epoch", 0)


def compare_experiments(experiment_names=None):
    rows = []
    names = experiment_names or config.DEFAULT_EXPERIMENT_ORDER
    for name in names:
        exp_cfg = config.EXPERIMENTS[name]
        if not exp_cfg["best_checkpoint"].exists():
            print(f"skip {name}: missing {exp_cfg['best_checkpoint']}")
            continue
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
                "best_checkpoint": str(exp_cfg["best_checkpoint"]),
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
