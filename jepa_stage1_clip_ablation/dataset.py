from collections.abc import Callable
from pathlib import Path

try:
    from torch.utils.data import Dataset
except ModuleNotFoundError:
    class Dataset:  # type: ignore[override]
        pass

from . import config
from .utils import load_pil_image, read_jsonl


def resolve_manifest_image_path(image_path: str) -> str:
    path = Path(image_path)
    if path.name == config.EMPTY_BEFORE_IMAGE.name:
        return str(config.EMPTY_BEFORE_IMAGE)
    return image_path


class CADJEPADataset(Dataset):
    def __init__(self, manifest_path, cache_remote_images: bool = True):
        self.manifest_path = manifest_path
        self.cache_remote_images = cache_remote_images
        self.samples = read_jsonl(manifest_path)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        row = self.samples[idx]
        before_image_path = resolve_manifest_image_path(row["before_image"])
        highlight_image_path = resolve_manifest_image_path(row["highlight_image"])
        before_image = load_pil_image(before_image_path, cache_remote=self.cache_remote_images)
        highlight_image = load_pil_image(highlight_image_path, cache_remote=self.cache_remote_images)
        return {
            "sample_id": row["sample_id"],
            "before_image": before_image,
            "highlight_image": highlight_image,
            "instruction": row["instruction"],
            "before_image_path": before_image_path,
            "highlight_image_path": highlight_image_path,
        }


def build_collate_fn(image_processor, tokenizer, image_size: int) -> Callable:
    def collate_fn(batch):
        before_images = [item["before_image"] for item in batch]
        highlight_images = [item["highlight_image"] for item in batch]
        instructions = [item["instruction"] for item in batch]

        before_inputs = image_processor(
            images=before_images,
            return_tensors="pt",
            size={"height": image_size, "width": image_size},
        )
        target_inputs = image_processor(
            images=highlight_images,
            return_tensors="pt",
            size={"height": image_size, "width": image_size},
        )
        text_inputs = tokenizer(
            instructions,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )

        return {
            "sample_ids": [item["sample_id"] for item in batch],
            "before_pixel_values": before_inputs["pixel_values"],
            "highlight_pixel_values": target_inputs["pixel_values"],
            "input_ids": text_inputs["input_ids"],
            "attention_mask": text_inputs["attention_mask"],
            "instructions": instructions,
            "before_image_paths": [item["before_image_path"] for item in batch],
            "highlight_image_paths": [item["highlight_image_path"] for item in batch],
        }

    return collate_fn


def basic_collate(batch):
    return {
        "sample_ids": [item["sample_id"] for item in batch],
        "before_images": [item["before_image"] for item in batch],
        "highlight_images": [item["highlight_image"] for item in batch],
        "instructions": [item["instruction"] for item in batch],
        "before_image_paths": [item["before_image_path"] for item in batch],
        "highlight_image_paths": [item["highlight_image_path"] for item in batch],
    }
