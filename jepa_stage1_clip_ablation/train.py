import time
from contextlib import contextmanager, nullcontext

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

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


def _log_stage(message: str) -> None:
    if config.STAGE_LOG_ENABLED:
        print(f"[stage] {message}", flush=True)


@contextmanager
def _stage(message: str):
    _log_stage(f"START {message}")
    started_at = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - started_at
        _log_stage(f"DONE {message} ({elapsed:.1f}s)")


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
    if hasattr(model, "modality_embed"):
        groups.append({"params": [model.modality_embed], "lr": exp_cfg["learning_rate_head"]})
    if hasattr(model, "fusion_norm"):
        groups.append({"params": model.fusion_norm.parameters(), "lr": exp_cfg["learning_rate_head"]})
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
    _log_stage("loading train/test manifests into datasets")
    train_dataset = CADJEPADataset(config.TRAIN_MANIFEST)
    test_dataset = CADJEPADataset(config.TEST_MANIFEST)
    _log_stage(f"dataset sizes train={len(train_dataset)} test={len(test_dataset)}")
    collate_fn = build_collate_fn(image_processor, tokenizer, image_size=exp_cfg["image_size"])

    loader_kwargs = {
        "num_workers": config.NUM_WORKERS,
        "collate_fn": collate_fn,
        "pin_memory": True,
    }
    if config.NUM_WORKERS > 0:
        loader_kwargs["prefetch_factor"] = config.PREFETCH_FACTOR
        loader_kwargs["persistent_workers"] = config.PERSISTENT_WORKERS

    train_loader = DataLoader(
        train_dataset,
        batch_size=exp_cfg["batch_size"],
        shuffle=True,
        **loader_kwargs,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=exp_cfg["batch_size"],
        shuffle=False,
        **loader_kwargs,
    )
    _log_stage(
        "created dataloaders "
        f"batch_size={exp_cfg['batch_size']} num_workers={config.NUM_WORKERS} "
        f"prefetch_factor={config.PREFETCH_FACTOR if config.NUM_WORKERS > 0 else 0} "
        f"persistent_workers={config.PERSISTENT_WORKERS if config.NUM_WORKERS > 0 else False}"
    )
    return train_loader, test_loader


def _gpu_memory_gb(device) -> float:
    if device.type != "cuda":
        return 0.0
    return float(torch.cuda.max_memory_allocated(device) / (1024 ** 3))


def _learning_rate(optimizer) -> float:
    return float(optimizer.param_groups[0]["lr"])


def train_experiment(exp_cfg):
    from transformers import AutoImageProcessor, AutoTokenizer

    _log_stage(f"experiment={exp_cfg['name']}")
    if not config.TRAIN_MANIFEST.exists() or not config.TEST_MANIFEST.exists():
        raise FileNotFoundError("Manifest files are missing. Run build_manifest.py first.")

    with _stage("initialize seed and output directories"):
        set_seed(config.SEED)
        for key in ["checkpoint_dir", "log_dir", "eval_dir"]:
            ensure_dir(exp_cfg[key])
        write_json(exp_cfg["config_snapshot"], _json_ready(exp_cfg))

    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    _log_stage(f"device={device} cuda_available={torch.cuda.is_available()} precision={exp_cfg['precision']}")
    if device.type == "cuda":
        _log_stage(f"gpu_name={torch.cuda.get_device_name(device)}")

    with _stage(f"load image processor {exp_cfg['vision_model_name']}"):
        image_processor = AutoImageProcessor.from_pretrained(exp_cfg["vision_model_name"])
    with _stage(f"load tokenizer {exp_cfg['text_model_name']}"):
        tokenizer = AutoTokenizer.from_pretrained(exp_cfg["text_model_name"])
    with _stage("build datasets and dataloaders"):
        train_loader, test_loader = _build_loaders(exp_cfg, image_processor, tokenizer)

    with _stage("load source/target/text models and create trainable heads"):
        model = create_model(exp_cfg)
    with _stage("move model to device"):
        model = model.to(device)
    with _stage("build optimizer"):
        optimizer = _build_optimizer(model, exp_cfg)

    best_top5 = float("-inf")
    best_epoch = 0
    logs = []
    step_logs = []

    for epoch in range(1, config.NUM_EPOCHS + 1):
        _log_stage(f"START epoch {epoch}/{config.NUM_EPOCHS}")
        model.train()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        running = {"loss": 0.0, "cos": 0.0, "mse": 0.0, "contrastive": 0.0, "steps": 0}
        progress = tqdm(
            train_loader,
            desc=f"{exp_cfg['name']} epoch {epoch}/{config.NUM_EPOCHS}",
            dynamic_ncols=True,
            leave=True,
        )
        for step, batch in enumerate(progress, start=1):
            if epoch == 1 and step == 1:
                _log_stage("first batch loaded; CPU image decode/resize/tokenization warmup is complete")
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

            avg_loss = running["loss"] / running["steps"]
            progress.set_postfix(
                loss=f"{loss.item():.4f}",
                avg=f"{avg_loss:.4f}",
                cos=f"{loss_cos.item():.4f}",
                mse=f"{loss_mse.item():.4f}",
                ctr=f"{loss_contrastive.item():.4f}",
                mem=f"{_gpu_memory_gb(device):.1f}G",
            )

            if step == 1 or step % config.LOG_EVERY_STEPS == 0 or step == len(train_loader):
                step_row = {
                    "experiment_name": exp_cfg["name"],
                    "epoch": epoch,
                    "step": step,
                    "global_step": (epoch - 1) * len(train_loader) + step,
                    "lr": _learning_rate(optimizer),
                    "loss": loss.item(),
                    "loss_cos": loss_cos.item(),
                    "loss_mse": loss_mse.item(),
                    "loss_contrastive": loss_contrastive.item(),
                    "avg_loss": avg_loss,
                    "gpu_peak_memory_gb": _gpu_memory_gb(device),
                }
                step_logs.append(step_row)
                print(
                    f"[{exp_cfg['name']}] epoch={epoch} step={step}/{len(train_loader)} "
                    f"loss={step_row['loss']:.4f} avg_loss={step_row['avg_loss']:.4f} "
                    f"cos={step_row['loss_cos']:.4f} mse={step_row['loss_mse']:.4f} "
                    f"contrastive={step_row['loss_contrastive']:.4f} "
                    f"mem={step_row['gpu_peak_memory_gb']:.1f}GB"
                )

        with _stage(f"evaluate retrieval after epoch {epoch}"):
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

        with _stage(f"write logs after epoch {epoch}"):
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
            write_csv(
                exp_cfg["train_step_log"],
                step_logs,
                [
                    "experiment_name",
                    "epoch",
                    "step",
                    "global_step",
                    "lr",
                    "loss",
                    "loss_cos",
                    "loss_mse",
                    "loss_contrastive",
                    "avg_loss",
                    "gpu_peak_memory_gb",
                ],
            )
        _log_stage(f"DONE epoch {epoch}/{config.NUM_EPOCHS}")

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
    write_csv(
        exp_cfg["train_step_log"],
        step_logs,
        [
            "experiment_name",
            "epoch",
            "step",
            "global_step",
            "lr",
            "loss",
            "loss_cos",
            "loss_mse",
            "loss_contrastive",
            "avg_loss",
            "gpu_peak_memory_gb",
        ],
    )
    print(f"best epoch for {exp_cfg['name']}: {best_epoch}")
