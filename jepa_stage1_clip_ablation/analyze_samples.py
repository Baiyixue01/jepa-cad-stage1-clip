import argparse
import csv
import json
from pathlib import Path

import numpy as np

from . import config
from .utils import ensure_dir, read_jsonl, write_csv, write_json


DEFAULT_TOP_K_MATCHES = 5
DEFAULT_NUM_SAMPLES = 50


def _read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _safe_float(value: str | int | float | None, default: float = 0.0) -> float:
    if value in (None, ""):
        return default
    return float(value)


def _safe_int(value: str | int | None, default: int = 0) -> int:
    if value in (None, ""):
        return default
    return int(value)


def _load_valid_manifest_by_eval_order(details: list[dict]) -> dict[str, dict]:
    manifest_by_id = {row["sample_id"]: row for row in read_jsonl(config.TEST_MANIFEST)}
    return {row["sample_id"]: manifest_by_id.get(row["sample_id"], {}) for row in details}


def _rank_sample_rows(details: list[dict], similarity: np.ndarray | None) -> list[dict]:
    rows = []
    valid_manifest = _load_valid_manifest_by_eval_order(details)
    sample_ids = [row["sample_id"] for row in details]

    for idx, row in enumerate(details):
        positive_similarity = _safe_float(row.get("positive_similarity"))
        best_match_similarity = _safe_float(row.get("best_match_similarity"))
        rank = _safe_int(row.get("rank"))

        if similarity is not None and idx < similarity.shape[0]:
            order = np.argsort(similarity[idx])[::-1]
            negative_order = [candidate_idx for candidate_idx in order if candidate_idx != idx]
            hardest_negative_idx = int(negative_order[0]) if negative_order else idx
            hardest_negative_similarity = float(similarity[idx, hardest_negative_idx])
        else:
            hardest_negative_idx = None
            hardest_negative_similarity = best_match_similarity if rank > 1 else float("nan")

        if hardest_negative_idx is None:
            hardest_negative_sample_id = ""
        else:
            hardest_negative_sample_id = sample_ids[hardest_negative_idx]

        margin = positive_similarity - hardest_negative_similarity
        manifest_row = valid_manifest.get(row["sample_id"], {})
        rows.append(
            {
                "sample_id": row["sample_id"],
                "rank": rank,
                "top1": _safe_int(row.get("top1")),
                "top5": _safe_int(row.get("top5")),
                "top10": _safe_int(row.get("top10")),
                "positive_similarity": positive_similarity,
                "hardest_negative_similarity": hardest_negative_similarity,
                "margin_vs_hardest_negative": margin,
                "best_match_sample_id": row.get("best_match_sample_id", ""),
                "best_match_similarity": best_match_similarity,
                "hardest_negative_sample_id": hardest_negative_sample_id,
                "instruction": row.get("instruction", ""),
                "before_image": row.get("before_image", ""),
                "highlight_image": row.get("highlight_image", ""),
                "op": manifest_row.get("op", ""),
            }
        )
    return rows


def _write_match_candidates(
    path: Path,
    details: list[dict],
    similarity: np.ndarray,
    selected_rows: list[dict],
    top_k: int,
) -> None:
    sample_ids = [row["sample_id"] for row in details]
    details_by_id = {row["sample_id"]: row for row in details}
    selected_ids = {row["sample_id"] for row in selected_rows}
    rows = []

    for query_idx, query_row in enumerate(details):
        query_sample_id = query_row["sample_id"]
        if query_sample_id not in selected_ids:
            continue
        order = np.argsort(similarity[query_idx])[::-1][:top_k]
        for candidate_rank, candidate_idx in enumerate(order, start=1):
            candidate_sample_id = sample_ids[int(candidate_idx)]
            candidate_row = details_by_id[candidate_sample_id]
            rows.append(
                {
                    "query_sample_id": query_sample_id,
                    "query_rank": _safe_int(query_row.get("rank")),
                    "query_instruction": query_row.get("instruction", ""),
                    "candidate_rank": candidate_rank,
                    "candidate_sample_id": candidate_sample_id,
                    "is_ground_truth": int(query_idx == int(candidate_idx)),
                    "similarity": float(similarity[query_idx, candidate_idx]),
                    "candidate_instruction": candidate_row.get("instruction", ""),
                    "candidate_highlight_image": candidate_row.get("highlight_image", ""),
                }
            )

    write_csv(
        path,
        rows,
        [
            "query_sample_id",
            "query_rank",
            "query_instruction",
            "candidate_rank",
            "candidate_sample_id",
            "is_ground_truth",
            "similarity",
            "candidate_instruction",
            "candidate_highlight_image",
        ],
    )


def analyze_experiment(
    exp_cfg: dict,
    num_samples: int = DEFAULT_NUM_SAMPLES,
    top_k_matches: int = DEFAULT_TOP_K_MATCHES,
) -> dict:
    details_path = exp_cfg["eval_details"]
    similarity_path = exp_cfg["similarity_matrix"]
    if not details_path.exists():
        raise FileNotFoundError(f"Missing eval details: {details_path}")

    details = _read_csv(details_path)
    similarity = np.load(similarity_path) if similarity_path.exists() else None
    sample_rows = _rank_sample_rows(details, similarity)

    output_dir = ensure_dir(exp_cfg["eval_dir"] / "sample_analysis")
    all_samples_path = output_dir / "all_samples_ranked.csv"
    good_samples_path = output_dir / "good_samples.csv"
    bad_samples_path = output_dir / "bad_samples.csv"
    hard_match_path = output_dir / "bad_sample_top_matches.csv"
    summary_path = output_dir / "summary.json"

    fieldnames = [
        "sample_id",
        "rank",
        "top1",
        "top5",
        "top10",
        "positive_similarity",
        "hardest_negative_similarity",
        "margin_vs_hardest_negative",
        "best_match_sample_id",
        "best_match_similarity",
        "hardest_negative_sample_id",
        "instruction",
        "before_image",
        "highlight_image",
        "op",
    ]
    write_csv(all_samples_path, sample_rows, fieldnames)

    good_rows = sorted(
        sample_rows,
        key=lambda row: (
            row["rank"],
            -row["margin_vs_hardest_negative"],
            -row["positive_similarity"],
        ),
    )[:num_samples]
    bad_rows = sorted(
        sample_rows,
        key=lambda row: (
            -row["rank"],
            row["margin_vs_hardest_negative"],
            row["positive_similarity"],
        ),
    )[:num_samples]

    write_csv(good_samples_path, good_rows, fieldnames)
    write_csv(bad_samples_path, bad_rows, fieldnames)
    if similarity is not None:
        _write_match_candidates(hard_match_path, details, similarity, bad_rows, top_k_matches)

    ranks = [row["rank"] for row in sample_rows]
    margins = [row["margin_vs_hardest_negative"] for row in sample_rows]
    positives = [row["positive_similarity"] for row in sample_rows]
    summary = {
        "experiment_name": exp_cfg["name"],
        "num_eval_samples": len(sample_rows),
        "num_good_samples": len(good_rows),
        "num_bad_samples": len(bad_rows),
        "rank_min": min(ranks) if ranks else None,
        "rank_max": max(ranks) if ranks else None,
        "rank_mean": float(np.mean(ranks)) if ranks else None,
        "margin_mean": float(np.mean(margins)) if margins else None,
        "margin_min": float(np.min(margins)) if margins else None,
        "margin_max": float(np.max(margins)) if margins else None,
        "positive_similarity_mean": float(np.mean(positives)) if positives else None,
        "outputs": {
            "all_samples": str(all_samples_path),
            "good_samples": str(good_samples_path),
            "bad_samples": str(bad_samples_path),
            "bad_sample_top_matches": str(hard_match_path) if similarity is not None else "",
        },
    }
    write_json(summary_path, summary)
    return summary


def _select_experiments(args: argparse.Namespace) -> list[dict]:
    if args.all:
        names = [
            name
            for name in config.EXPERIMENTS
            if name != config.EXP_D["name"]
        ]
    elif args.experiment:
        names = args.experiment
    else:
        names = [config.EXP_REG_GIANT_TRANSFORMER["name"]]

    unknown = [name for name in names if name not in config.EXPERIMENTS]
    if unknown:
        raise KeyError(f"Unknown experiments: {unknown}")
    return [config.EXPERIMENTS[name] for name in names]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find strong and weak retrieval samples from evaluation outputs."
    )
    parser.add_argument(
        "--experiment",
        action="append",
        help="Experiment name. Can be repeated. Defaults to experiment E.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Analyze all registered experiments except experiment D.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=DEFAULT_NUM_SAMPLES,
        help="Number of good and bad samples to export per experiment.",
    )
    parser.add_argument(
        "--top-k-matches",
        type=int,
        default=DEFAULT_TOP_K_MATCHES,
        help="Number of nearest target candidates to export for each bad sample.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for exp_cfg in _select_experiments(args):
        summary = analyze_experiment(
            exp_cfg,
            num_samples=args.num_samples,
            top_k_matches=args.top_k_matches,
        )
        print(
            f"{summary['experiment_name']}: "
            f"samples={summary['num_eval_samples']} "
            f"rank_mean={summary['rank_mean']:.4f} "
            f"margin_mean={summary['margin_mean']:.4f} "
            f"out={exp_cfg['eval_dir'] / 'sample_analysis'}",
            flush=True,
        )


if __name__ == "__main__":
    main()
