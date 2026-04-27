import csv
import json
import random
import struct
import subprocess
import zlib
from pathlib import Path
from typing import Iterable

import numpy as np

from . import config


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def set_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def create_empty_before_image(path: Path, image_size: int) -> Path:
    ensure_dir(path.parent)
    if not path.exists():
        path.write_bytes(_make_png_bytes(image_size, image_size, (255, 255, 255)))
    return path


def _png_chunk(chunk_type: bytes, data: bytes) -> bytes:
    return (
        struct.pack("!I", len(data))
        + chunk_type
        + data
        + struct.pack("!I", zlib.crc32(chunk_type + data) & 0xFFFFFFFF)
    )


def _make_png_bytes(width: int, height: int, rgb: tuple[int, int, int]) -> bytes:
    signature = b"\x89PNG\r\n\x1a\n"
    ihdr = _png_chunk(
        b"IHDR",
        struct.pack("!IIBBBBB", width, height, 8, 2, 0, 0, 0),
    )
    row = b"\x00" + bytes(rgb) * width
    raw = row * height
    idat = _png_chunk(b"IDAT", zlib.compress(raw))
    iend = _png_chunk(b"IEND", b"")
    return signature + ihdr + idat + iend


def parse_split_item(item: str) -> tuple[str, int]:
    group_index, step_part = item.split("/")
    step_id = int(step_part.replace("step", ""))
    return group_index, step_id


def _run_command(cmd: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        check=True,
        capture_output=True,
        text=True,
    )


def run_ssh_python(script: str) -> str:
    result = _run_command(["ssh", config.REMOTE_HOST, "python3", "-c", script])
    return result.stdout


def scp_remote_file(remote_path: str, local_path: Path) -> Path:
    ensure_dir(local_path.parent)
    _run_command(["scp", f"{config.REMOTE_HOST}:{remote_path}", str(local_path)])
    return local_path


def resolve_image_path(image_path: str, cache_remote: bool = True) -> Path:
    path = Path(image_path)
    if path.exists():
        return path
    if not config.ALLOW_REMOTE_DATASET_FETCH:
        raise FileNotFoundError(f"Image path not accessible locally: {image_path}")
    if not cache_remote:
        raise FileNotFoundError(f"Remote fetch disabled for image path: {image_path}")
    local_path = config.REMOTE_CACHE_DIR / image_path.lstrip("/")
    if local_path.exists():
        return local_path
    return scp_remote_file(image_path, local_path)


def load_pil_image(image_path: str, cache_remote: bool = True):
    resolved = resolve_image_path(image_path, cache_remote=cache_remote)
    try:
        from PIL import Image
    except ModuleNotFoundError:
        return resolved.read_bytes()
    with Image.open(resolved) as img:
        return img.convert("RGB")


def normalize_embeddings(tensor):
    import torch

    return torch.nn.functional.normalize(tensor, dim=-1)


def compute_retrieval_metrics(similarity) -> tuple[dict, list[int]]:
    import torch

    ranks = []
    total = similarity.size(0)
    for idx in range(total):
        order = torch.argsort(similarity[idx], descending=True)
        rank = int((order == idx).nonzero(as_tuple=False)[0].item()) + 1
        ranks.append(rank)

    metrics = {
        "top1": float(np.mean([rank <= 1 for rank in ranks])),
        "top5": float(np.mean([rank <= 5 for rank in ranks])),
        "top10": float(np.mean([rank <= 10 for rank in ranks])),
        "mrr": float(np.mean([1.0 / rank for rank in ranks])),
        "mean_rank": float(np.mean(ranks)),
        "median_rank": float(np.median(ranks)),
    }
    return metrics, ranks


def _read_split_payload() -> dict:
    split_path = config.LOCAL_SPLIT_CACHE if config.LOCAL_SPLIT_CACHE.exists() else config.SPLIT_PATH
    with split_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _build_prompt_lookup() -> dict:
    prompt_path = config.LOCAL_PROMPT_CACHE if config.LOCAL_PROMPT_CACHE.exists() else config.PROMPT_CSV_PATH
    prompt_lookup = {}
    with prompt_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            op = (row.get("op") or "").strip()
            if op in config.SKIP_OPS:
                continue
            raw_key = (row.get("group_index") or "").strip()
            if not raw_key:
                continue
            if "/step" in raw_key:
                group_index, step_part = raw_key.split("/")
                step_id = int(step_part.replace("step", ""))
            else:
                group_index = raw_key
                step_id = int(row["step_id"])
            prompt_lookup[(group_index, step_id)] = {
                "instruction": (row.get("prompt_text") or "").strip(),
                "op": op,
            }
    return prompt_lookup


def _load_inventory() -> set[str]:
    inventory = set()
    if config.LOCAL_IMAGE_INVENTORY.exists():
        with config.LOCAL_IMAGE_INVENTORY.open("r", encoding="utf-8") as f:
            inventory = {line.strip() for line in f if line.strip()}
    return inventory


def _path_exists_in_inventory(path: Path, inventory: set[str]) -> bool:
    return str(path) in inventory or path.exists()


def build_manifest_records(split_name: str) -> tuple[list[dict], dict]:
    split_payload = _read_split_payload()
    prompt_lookup = _build_prompt_lookup()
    inventory = _load_inventory()

    records = []
    stats = {
        "input_count": len(split_payload.get(split_name, [])),
        "valid_count": 0,
        "skipped_missing_prompt": 0,
        "skipped_missing_highlight": 0,
        "skipped_missing_before_non_step0": 0,
        "used_empty_before": 0,
        "warnings": [],
    }

    for item in split_payload.get(split_name, []):
        group_index, step_id = parse_split_item(item)
        prompt = prompt_lookup.get((group_index, step_id))
        if prompt is None:
            stats["skipped_missing_prompt"] += 1
            stats["warnings"].append(f"missing_prompt\t{item}")
            continue

        image_dir = config.IMAGE_ROOT / group_index / f"step{step_id}"
        before_image = image_dir / config.BEFORE_IMAGE_NAME
        highlight_image = image_dir / config.HIGHLIGHT_IMAGE_NAME
        before_missing = not _path_exists_in_inventory(before_image, inventory)
        highlight_exists = _path_exists_in_inventory(highlight_image, inventory)

        if not highlight_exists:
            stats["skipped_missing_highlight"] += 1
            stats["warnings"].append(f"missing_highlight\t{item}")
            continue

        use_empty_before = False
        if before_missing:
            if step_id == 0:
                use_empty_before = True
                stats["used_empty_before"] += 1
            else:
                stats["skipped_missing_before_non_step0"] += 1
                stats["warnings"].append(f"missing_before\t{item}")
                continue

        records.append(
            {
                "sample_id": f"{group_index}_step{step_id}",
                "group_index": group_index,
                "step_id": step_id,
                "split": split_name,
                "instruction": prompt["instruction"],
                "before_image": str(before_image),
                "highlight_image": str(highlight_image),
                "use_empty_before": use_empty_before,
            }
        )

    stats["valid_count"] = len(records)
    return records, stats


def manifest_has_required_fields(row: dict) -> bool:
    required = {
        "sample_id",
        "group_index",
        "step_id",
        "split",
        "instruction",
        "before_image",
        "highlight_image",
    }
    return required.issubset(row.keys()) and all(row[key] not in (None, "") for key in required)


def detect_duplicate_sample_ids(rows: list[dict]) -> list[str]:
    seen = set()
    duplicates = []
    for row in rows:
        sample_id = row["sample_id"]
        if sample_id in seen:
            duplicates.append(sample_id)
        seen.add(sample_id)
    return duplicates


def summarize_check(name: str, payload: dict) -> str:
    parts = [f"{key}={value}" for key, value in payload.items() if key != "warnings"]
    if payload.get("warnings"):
        parts.append(f"warnings={len(payload['warnings'])}")
    return f"{name}: " + ", ".join(parts)
