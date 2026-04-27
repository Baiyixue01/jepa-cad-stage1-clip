from collections.abc import Callable

try:
    from torch.utils.data import Dataset
except ModuleNotFoundError:
    class Dataset:  # type: ignore[override]
        pass

from .utils import load_pil_image, read_jsonl


class CADJEPADataset(Dataset):
    def __init__(self, manifest_path, processor=None, cache_remote_images: bool = True):
        self.manifest_path = manifest_path
        self.processor = processor
        self.cache_remote_images = cache_remote_images
        self.samples = read_jsonl(manifest_path)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        row = self.samples[idx]
        before_image = load_pil_image(row["before_image"], cache_remote=self.cache_remote_images)
        highlight_image = load_pil_image(row["highlight_image"], cache_remote=self.cache_remote_images)
        return {
            "sample_id": row["sample_id"],
            "before_image": before_image,
            "highlight_image": highlight_image,
            "instruction": row["instruction"],
            "before_image_path": row["before_image"],
            "highlight_image_path": row["highlight_image"],
        }


def build_collate_fn(processor) -> Callable:
    def collate_fn(batch):
        before_images = [item["before_image"] for item in batch]
        highlight_images = [item["highlight_image"] for item in batch]
        instructions = [item["instruction"] for item in batch]

        batch_inputs = processor(
            text=instructions,
            images=before_images,
            return_tensors="pt",
            padding=True,
        )
        target_inputs = processor(images=highlight_images, return_tensors="pt")

        return {
            "sample_ids": [item["sample_id"] for item in batch],
            "before_pixel_values": batch_inputs["pixel_values"],
            "input_ids": batch_inputs["input_ids"],
            "attention_mask": batch_inputs["attention_mask"],
            "highlight_pixel_values": target_inputs["pixel_values"],
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
