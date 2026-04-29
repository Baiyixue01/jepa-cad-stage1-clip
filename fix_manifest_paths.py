from jepa_stage1_clip_ablation.config import TEST_MANIFEST, TRAIN_MANIFEST
from jepa_stage1_clip_ablation.utils import read_jsonl, write_jsonl


def fix_manifest(path):
    rows = read_jsonl(path)
    changed = 0
    for row in rows:
        before_image = row.get("before_image", "")
        if before_image.endswith("/empty_before.png"):
            row["before_image"] = "empty_before.png"
            changed += 1
    write_jsonl(path, rows)
    print(f"{path}: changed={changed} total={len(rows)}")


if __name__ == "__main__":
    fix_manifest(TRAIN_MANIFEST)
    fix_manifest(TEST_MANIFEST)
