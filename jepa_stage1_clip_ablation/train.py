from contextlib import nullcontext

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader

from . import config
from .dataset import CADJEPADataset, build_collate_fn
from .model import create_model
from .utils import (
    compute_retrieval_metrics,
    ensure_dir,
    normalize_embeddings,
    set_seed,
    write_csv,
    write_json,
)


def _json_ready(payload):
    if isinstance(payload, dict):
        return {key: _json_ready(value) for key, value in payload.items()}
    if isinstance(payload, list):
        return [_json_ready(value) for value in payload]
    if hasattr(payload, "as_posix"):
        return payload.as_posix()
    return payload


def _autocast_context(device, precision: str):
    if device.type != "cuda" or precision == "fp32":
        return nullcontext()
    if precision == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    if precision == "fp16":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    raise ValueError(f"Unsupported precision: {precision}")


def _build_optimizer(model, exp_cfg):
    groups = [
        {"params": model.source_image_proj.parameters(), "lr": exp_cfg["learning_rate_image_proj"]},
        {"params": model.text_proj.parameters(), "lr": exp_cfg["learning_rate_text_proj"]},
        {"params": model.fusion.parameters(), "lr": exp_cfg["learning_rate_head"]},
        {"params": model.predictor.parameters(), "lr": exp_cfg["learning_rate_head"]},
    ]
    if exp_cfg["train_source_lora"]:
        lora_params = [
            param for name, param in model.source_vision.named_parameters()
            if param.requires_grad and "lora_" in name
        ]
        if lora_params:
            groups.insert(0, {"params": lora_params, "lr": exp_cfg["learning_rate_adapter"]})
    return AdamW(groups, weight_decay=config.WEIGHT_DECAY)


def _compute_losses(z_pred, z_target):
    z_target_detached = z_target.detach()
    loss_cos = 1 - F.cosine_similarity(z_pred, z_target_detached).mean()
    loss_mse = F.mse_loss(z_pred, z_target_detached)
    logits = normalize_embeddings(z_pred) @ normalize_embeddings(z_target_detached).T / config.TEMPERATURE
    labels = torch.arange(z_pred.size(0), device=z_pred.device)
    loss_contrastive = F.cross_entropy(logits, labels)
    loss = (
        config.LAMBDA_COS * loss_cos
        + config.LAMBDA_MSE * loss_mse
        + config.LAMBDA_CONTRASTIVE * loss_contrastive
    )
    return loss, loss_cos, loss_mse, loss_contrastive


@torch.no_grad()
def _evaluate_retrieval(model, loader, device, precision: str):
    model.eval()
    all_pred = []
    all_target = []
    for batch in loader:
        with _autocast_context(device, precision):
            outputs = model(
                before_pixel_values=batch["before_pixel_values"].to(device),
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                highlight_pixel_values=batch["highlight_pixel_values"].to(device),
            )
        all_pred.append(outputs["z_pred"].float().cpu())
        all_target.append(outputs["z_target"].float().cpu())
    similarity = normalize_embeddings(torch.cat(all_pred, dim=0)) @ normalize_embeddings(torch.cat(all_target, dim=0)).T
    metrics, _ = compute_retrieval_metrics(similarity)
    return metrics


def _build_loaders(exp_cfg, image_processor, tokenizer):
    train_dataset = CADJEPADataset(config.TRAIN_MANIFEST)
    test_dataset = CADJEPADataset(config.TEST_MANIFEST)
    collate_fn = build_collate_fn(image_processor, tokenizer, image_size=exp_cfg["image_size"])

    train_loader = DataLoader(
        train_dataset,
        batch_size=exp_cfg["batch_size"],
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=exp_cfg["batch_size"],
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    return train_loader, test_loader


def train_experiment(exp_cfg):
    from transformers import AutoImageProcessor, AutoTokenizer

    if not config.TRAIN_MANIFEST.exists() or not config.TEST_MANIFEST.exists():
        raise FileNotFoundError("Manifest files are missing. Run build_manifest.py first.")

    set_seed(config.SEED)
    for key in ["checkpoint_dir", "log_dir", "eval_dir"]:
        ensure_dir(exp_cfg[key])
    write_json(exp_cfg["config_snapshot"], _json_ready(exp_cfg))

    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    image_processor = AutoImageProcessor.from_pretrained(exp_cfg["vision_model_name"])
    tokenizer = AutoTokenizer.from_pretrained(exp_cfg["text_model_name"])
    train_loader, test_loader = _build_loaders(exp_cfg, image_processor, tokenizer)

    model = create_model(exp_cfg).to(device)
    optimizer = _build_optimizer(model, exp_cfg)

    best_top5 = float("-inf")
    best_epoch = 0
    logs = []

    for epoch in range(1, config.NUM_EPOCHS + 1):
        model.train()
        running = {"loss": 0.0, "cos": 0.0, "mse": 0.0, "contrastive": 0.0, "steps": 0}
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            with _autocast_context(device, exp_cfg["precision"]):
                outputs = model(
                    before_pixel_values=batch["before_pixel_values"].to(device),
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                    highlight_pixel_values=batch["highlight_pixel_values"].to(device),
                )
                loss, loss_cos, loss_mse, loss_contrastive = _compute_losses(
                    outputs["z_pred"],
                    outputs["z_target"],
                )
            loss.backward()
            optimizer.step()

            running["loss"] += loss.item()
            running["cos"] += loss_cos.item()
            running["mse"] += loss_mse.item()
            running["contrastive"] += loss_contrastive.item()
            running["steps"] += 1

        metrics = _evaluate_retrieval(model, test_loader, device, exp_cfg["precision"])
        row = {
            "experiment_name": exp_cfg["name"],
            "epoch": epoch,
            "train_loss": running["loss"] / max(running["steps"], 1),
            "train_loss_cos": running["cos"] / max(running["steps"], 1),
            "train_loss_mse": running["mse"] / max(running["steps"], 1),
            "train_loss_contrastive": running["contrastive"] / max(running["steps"], 1),
            "test_top1": metrics["top1"],
            "test_top5": metrics["top5"],
            "test_top10": metrics["top10"],
            "test_mrr": metrics["mrr"],
            "test_mean_rank": metrics["mean_rank"],
            "test_median_rank": metrics["median_rank"],
        }
        logs.append(row)

        checkpoint = {
            "epoch": epoch,
            "exp_cfg": _json_ready(exp_cfg),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
        }
        torch.save(checkpoint, exp_cfg["latest_checkpoint"])
        if metrics["top5"] > best_top5:
            best_top5 = metrics["top5"]
            best_epoch = epoch
            torch.save(checkpoint, exp_cfg["best_checkpoint"])

        print(
            f"[{exp_cfg['name']}] epoch={epoch} loss={row['train_loss']:.4f} "
            f"top1={metrics['top1']:.4f} top5={metrics['top5']:.4f}"
        )

    write_csv(
        exp_cfg["train_log"],
        logs,
        [
            "experiment_name",
            "epoch",
            "train_loss",
            "train_loss_cos",
            "train_loss_mse",
            "train_loss_contrastive",
            "test_top1",
            "test_top5",
            "test_top10",
            "test_mrr",
            "test_mean_rank",
            "test_median_rank",
        ],
    )
    print(f"best epoch for {exp_cfg['name']}: {best_epoch}")
