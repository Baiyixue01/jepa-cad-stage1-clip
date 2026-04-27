from . import config
from .utils import (
    build_manifest_records,
    create_empty_before_image,
    detect_duplicate_sample_ids,
    ensure_dir,
    manifest_has_required_fields,
    summarize_check,
    write_jsonl,
)


def finalize_records(records: list[dict]) -> list[dict]:
    finalized = []
    for row in records:
        item = dict(row)
        if item.pop("use_empty_before", False):
            item["before_image"] = str(config.EMPTY_BEFORE_IMAGE)
        finalized.append(item)
    return finalized


def build_manifest() -> tuple[list[dict], list[dict]]:
    ensure_dir(config.OUTPUT_ROOT)
    ensure_dir(config.MANIFEST_DIR)
    create_empty_before_image(config.EMPTY_BEFORE_IMAGE, config.IMAGE_SIZE)

    train_records_raw, train_stats = build_manifest_records("train")
    test_records_raw, test_stats = build_manifest_records("test")

    train_records = finalize_records(train_records_raw)
    test_records = finalize_records(test_records_raw)

    duplicate_ids = detect_duplicate_sample_ids(train_records + test_records)
    if duplicate_ids:
        print(f"warning: duplicate sample ids detected: {duplicate_ids[:10]}")

    invalid = [row["sample_id"] for row in train_records + test_records if not manifest_has_required_fields(row)]
    if invalid:
        raise ValueError(f"manifest contains rows with missing fields: {invalid[:10]}")

    write_jsonl(config.TRAIN_MANIFEST, train_records)
    write_jsonl(config.TEST_MANIFEST, test_records)

    print(summarize_check("train", train_stats))
    print(summarize_check("test", test_stats))
    if train_stats.get("warnings"):
        print("train warnings sample:")
        for line in train_stats["warnings"][:10]:
            print(line)
    if test_stats.get("warnings"):
        print("test warnings sample:")
        for line in test_stats["warnings"][:10]:
            print(line)
    print(f"saved train manifest: {config.TRAIN_MANIFEST}")
    print(f"saved test manifest: {config.TEST_MANIFEST}")
    print(f"empty before image: {config.EMPTY_BEFORE_IMAGE}")

    return train_records, test_records


if __name__ == "__main__":
    build_manifest()
