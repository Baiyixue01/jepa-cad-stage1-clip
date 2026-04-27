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
COMPARE_CSV = OUTPUT_ROOT / "compare_exp_a_b.csv"

# =====================
# Image file names
# =====================
BEFORE_IMAGE_NAME = "previous_model_grid.png"
HIGHLIGHT_IMAGE_NAME = "location.png"

# =====================
# CLIP config
# =====================
CLIP_MODEL_NAME = "openai/clip-vit-base-patch16"
IMAGE_SIZE = 224
EMBED_DIM = 512

# =====================
# Training config
# =====================
BATCH_SIZE = 64
NUM_EPOCHS = 30
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 4
DEVICE = "cuda"
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

# =====================
# Experiment A
# =====================
EXP_A = {
    "name": "exp_a_freeze_clip_predictor",
    "output_dir": OUTPUT_ROOT / "exp_a_freeze_clip_predictor",
    "freeze_clip_image_encoder": True,
    "freeze_clip_text_encoder": True,
    "freeze_target_encoder": True,
    "use_lora": False,
    "learning_rate_predictor": 1e-4,
    "learning_rate_clip": 0.0,
    "lora_rank": 0,
    "lora_alpha": 0,
    "lora_dropout": 0.0,
}

# =====================
# Experiment B
# =====================
EXP_B = {
    "name": "exp_b_lora_clip_image_text",
    "output_dir": OUTPUT_ROOT / "exp_b_lora_clip_image_text",
    "freeze_clip_image_encoder": False,
    "freeze_clip_text_encoder": False,
    "freeze_target_encoder": True,
    "use_lora": True,
    "learning_rate_predictor": 1e-4,
    "learning_rate_clip": 1e-5,
    "lora_rank": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.05,
}
