from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = PROJECT_ROOT / "outputs"
MANIFEST_DIR = OUTPUT_ROOT / "manifests"
CACHE_DIR = OUTPUT_ROOT / "cache"
REMOTE_CACHE_DIR = CACHE_DIR / "remote_images"
LOCAL_SPLIT_CACHE = CACHE_DIR / "split_result.json"
LOCAL_PROMPT_CACHE = CACHE_DIR / "prompt.csv"
LOCAL_IMAGE_INVENTORY = CACHE_DIR / "image_inventory.txt"

REMOTE_HOST = "lab1"

# =====================
# Remote data paths
# =====================
SPLIT_PATH = Path("/home/baiyixue/project/op-cad/data/split_result.json")
PROMPT_CSV_PATH = Path("/home/baiyixue/project/op-cad/data/prompt.csv")
IMAGE_ROOT = Path("/data/baiyixue/CAD/op_orientated_render_data")

# =====================
# Local outputs
# =====================
TRAIN_MANIFEST = MANIFEST_DIR / "train_manifest.jsonl"
TEST_MANIFEST = MANIFEST_DIR / "test_manifest.jsonl"
EMPTY_BEFORE_IMAGE = OUTPUT_ROOT / "empty_before.png"
COMPARE_CSV = OUTPUT_ROOT / "compare_experiments.csv"

# =====================
# Image file names
# =====================
BEFORE_IMAGE_NAME = "previous_model_grid.png"
HIGHLIGHT_IMAGE_NAME = "location.png"

# =====================
# Shared model / training defaults
# =====================
TEXT_MODEL_NAME = "openai/clip-vit-base-patch16"
TEXT_EMBED_DIM = 512
IMAGE_SIZE = 518
FUSION_DIM = 1024

BATCH_SIZE = 64
NUM_EPOCHS = 30
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 4
DEVICE = "cuda"
PRECISION = "bf16"  # one of: "fp32", "fp16", "bf16"
TEMPERATURE = 0.07
LAMBDA_COS = 1.0
LAMBDA_MSE = 0.5
LAMBDA_CONTRASTIVE = 1.0
SEED = 42
SAVE_EVERY_EPOCH = True

# =====================
# Manifest / data behavior
# =====================
ALLOW_REMOTE_DATASET_FETCH = True
DATASET_CACHE_REMOTE_IMAGES = True
SKIP_OPS = {"chamfer_fillet"}


def experiment_paths(name: str) -> dict:
    output_dir = OUTPUT_ROOT / name
    return {
        "output_dir": output_dir,
        "checkpoint_dir": output_dir / "checkpoints",
        "log_dir": output_dir / "logs",
        "eval_dir": output_dir / "eval",
        "latest_checkpoint": output_dir / "checkpoints" / "latest.pt",
        "best_checkpoint": output_dir / "checkpoints" / "best.pt",
        "train_log": output_dir / "logs" / "train_log.csv",
        "config_snapshot": output_dir / "logs" / "experiment_config.json",
        "eval_summary": output_dir / "eval" / "test_retrieval_summary.json",
        "eval_details": output_dir / "eval" / "test_retrieval_details.csv",
        "similarity_matrix": output_dir / "eval" / "test_similarity_matrix.npy",
    }


def make_exp(
    name: str,
    vision_model_name: str,
    vision_embed_dim: int,
    target_vision_model_name: str | None = None,
    target_embed_dim: int | None = None,
    train_source_lora: bool = False,
    text_model_name: str = TEXT_MODEL_NAME,
    text_embed_dim: int = TEXT_EMBED_DIM,
    image_size: int = IMAGE_SIZE,
    fusion_dim: int = FUSION_DIM,
    batch_size: int = BATCH_SIZE,
    precision: str = PRECISION,
    learning_rate_head: float = 1e-4,
    learning_rate_adapter: float = 1e-5,
    learning_rate_text_proj: float = 1e-4,
    learning_rate_image_proj: float = 1e-4,
    fusion_arch: str = "mlp",
    fusion_layers: int = 2,
    fusion_heads: int = 8,
    fusion_dropout: float = 0.0,
    predictor_arch: str = "mlp",
    lora_rank: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_target_modules: list[str] | None = None,
    notes: str = "",
) -> dict:
    target_name = target_vision_model_name or vision_model_name
    target_dim = target_embed_dim or vision_embed_dim
    return {
        "name": name,
        "vision_model_name": vision_model_name,
        "vision_embed_dim": vision_embed_dim,
        "target_vision_model_name": target_name,
        "target_embed_dim": target_dim,
        "text_model_name": text_model_name,
        "text_embed_dim": text_embed_dim,
        "image_size": image_size,
        "fusion_dim": fusion_dim,
        "batch_size": batch_size,
        "precision": precision,
        "freeze_source_vision": not train_source_lora,
        "freeze_target_vision": True,
        "freeze_text_encoder": True,
        "train_source_lora": train_source_lora,
        "learning_rate_head": learning_rate_head,
        "learning_rate_adapter": learning_rate_adapter,
        "learning_rate_text_proj": learning_rate_text_proj,
        "learning_rate_image_proj": learning_rate_image_proj,
        "fusion_arch": fusion_arch,
        "fusion_layers": fusion_layers,
        "fusion_heads": fusion_heads,
        "fusion_dropout": fusion_dropout,
        "predictor_arch": predictor_arch,
        "lora_rank": lora_rank,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "lora_target_modules": lora_target_modules or ["query", "key", "value"],
        "notes": notes,
        **experiment_paths(name),
    }


# =====================
# Experiment A: DINOv2-large frozen
# =====================
EXP_A = make_exp(
    name="exp_a_dinov2_large_frozen_signal",
    vision_model_name="facebook/dinov2-large",
    vision_embed_dim=1024,
    notes="Frozen DINOv2-large source/target plus frozen CLIP text. Trains projections, fusion, predictor.",
)

# =====================
# Experiment B: DINOv2-giant frozen
# =====================
EXP_B = make_exp(
    name="exp_b_dinov2_giant_frozen_backbone",
    vision_model_name="facebook/dinov2-giant",
    vision_embed_dim=1536,
    notes="Frozen DINOv2-giant source/target. Same trainable head as experiment A.",
)

# =====================
# Experiment C: source DINO LoRA / adapter
# =====================
EXP_C_LARGE = make_exp(
    name="exp_c_dinov2_large_source_lora",
    vision_model_name="facebook/dinov2-large",
    vision_embed_dim=1024,
    train_source_lora=True,
    notes="DINOv2-large source encoder uses LoRA; target encoder stays frozen.",
)

EXP_C_GIANT = make_exp(
    name="exp_c_dinov2_giant_source_lora",
    vision_model_name="facebook/dinov2-giant",
    vision_embed_dim=1536,
    train_source_lora=True,
    notes="DINOv2-giant source encoder uses LoRA; target encoder stays frozen.",
)

# =====================
# Experiment D: stronger frozen target space
# =====================
# Replace this with the exact local/Hugging Face DINOv3-7B checkpoint available
# on the training machine before running experiment D.
DINO_V3_7B_MODEL_NAME = "REPLACE_WITH_DINOV3_7B_MODEL_OR_LOCAL_PATH"
DINO_V3_7B_EMBED_DIM = 4096

EXP_D = make_exp(
    name="exp_d_dinov2_large_to_dinov3_7b_target",
    vision_model_name="facebook/dinov2-large",
    vision_embed_dim=1024,
    target_vision_model_name=DINO_V3_7B_MODEL_NAME,
    target_embed_dim=DINO_V3_7B_EMBED_DIM,
    fusion_dim=1024,
    notes="DINOv2-large source predicts a frozen stronger DINOv3 target embedding space.",
)

# =====================
# Experiment E: DINOv2-with-registers-giant + Fusion Transformer
# =====================
EXP_REG_GIANT_TRANSFORMER = make_exp(
    name="exp_e_dinov2_registers_giant_fusion_transformer",
    vision_model_name="facebook/dinov2-with-registers-giant",
    vision_embed_dim=1536,
    fusion_arch="transformer",
    fusion_layers=4,
    fusion_heads=8,
    fusion_dropout=0.0,
    predictor_arch="mlp",
    notes=(
        "Frozen DINOv2-with-registers-giant source/target plus frozen CLIP text. "
        "Trains image/text projections, 4-layer Fusion Transformer, and MLP predictor."
    ),
)

EXPERIMENTS = {
    EXP_A["name"]: EXP_A,
    EXP_B["name"]: EXP_B,
    EXP_C_LARGE["name"]: EXP_C_LARGE,
    EXP_C_GIANT["name"]: EXP_C_GIANT,
    EXP_D["name"]: EXP_D,
    EXP_REG_GIANT_TRANSFORMER["name"]: EXP_REG_GIANT_TRANSFORMER,
}

DEFAULT_EXPERIMENT_ORDER = [
    EXP_A["name"],
    EXP_B["name"],
    EXP_C_LARGE["name"],
    EXP_C_GIANT["name"],
    EXP_REG_GIANT_TRANSFORMER["name"],
]
