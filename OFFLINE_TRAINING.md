# Offline Training Preparation

This project can run on a machine without internet access, but the models, data, and Python environment must be prepared in advance.

## 1. Required Models

For experiments A/B/C, prepare these Hugging Face model directories:

```bash
huggingface-cli download facebook/dinov2-large --local-dir /path/to/models/dinov2-large
huggingface-cli download facebook/dinov2-giant --local-dir /path/to/models/dinov2-giant
huggingface-cli download openai/clip-vit-base-patch16 --local-dir /path/to/models/clip-vit-base-patch16
```

Experiment D also needs a real DINOv3 or DINOv3-7B checkpoint. Replace this placeholder in `jepa_stage1_clip_ablation/config.py` before running D:

```python
DINO_V3_7B_MODEL_NAME = "REPLACE_WITH_DINOV3_7B_MODEL_OR_LOCAL_PATH"
DINO_V3_7B_EMBED_DIM = 4096
```

Set local model paths in `config.py`, for example:

```python
TEXT_MODEL_NAME = "/path/to/models/clip-vit-base-patch16"

EXP_A["vision_model_name"] = "/path/to/models/dinov2-large"
EXP_A["target_vision_model_name"] = "/path/to/models/dinov2-large"

EXP_B["vision_model_name"] = "/path/to/models/dinov2-giant"
EXP_B["target_vision_model_name"] = "/path/to/models/dinov2-giant"
```

Apply the same path update to `EXP_C_LARGE`, `EXP_C_GIANT`, and `EXP_D` as needed.

## 2. Offline Environment

If the training machine cannot install packages from the internet, build the conda environment on an online machine first:

```bash
conda env create -f environment.yml
conda activate jepa-cad-stage1-clip
conda install -c conda-forge conda-pack
conda-pack -n jepa-cad-stage1-clip -o jepa-cad-stage1-clip.tar.gz
```

Move `jepa-cad-stage1-clip.tar.gz` to the training machine and unpack it:

```bash
mkdir -p /path/to/envs/jepa-cad-stage1-clip
tar -xzf jepa-cad-stage1-clip.tar.gz -C /path/to/envs/jepa-cad-stage1-clip
/path/to/envs/jepa-cad-stage1-clip/bin/conda-unpack
source /path/to/envs/jepa-cad-stage1-clip/bin/activate
```

Set offline flags before training:

```bash
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
```

## 3. Required Data

The repository includes processed manifests and `empty_before.png`, but it does not include the rendered CAD images.

The training machine must be able to read paths like:

```text
/data/baiyixue/CAD/op_orientated_render_data/{group_index}/step{step_id}/previous_model_grid.png
/data/baiyixue/CAD/op_orientated_render_data/{group_index}/step{step_id}/location.png
```

Use one of these options:

- Mount or copy the image dataset to the same `/data/baiyixue/CAD/op_orientated_render_data` path.
- Edit the manifest files under `jepa_stage1_clip_ablation/outputs/manifests/` to point to the new image root.
- Re-run `build_manifest.py` after updating `IMAGE_ROOT`, if `split_result.json`, `prompt.csv`, and the image inventory are available.

## 4. Training Commands

Single experiment:

```bash
python run_exp_a.py
python run_exp_b.py
python run_exp_c_large.py
python run_exp_c_giant.py
python run_exp_d.py
```

All configured experiments:

```bash
python run_all_experiments.py
```

Evaluation:

```bash
python eval_retrieval.py
```

## 5. Output Paths

Each experiment writes to:

```text
jepa_stage1_clip_ablation/outputs/<experiment_name>/
```

Important files:

```text
checkpoints/latest.pt
checkpoints/best.pt
logs/train_log.csv
logs/experiment_config.json
eval/test_retrieval_summary.json
eval/test_retrieval_details.csv
eval/test_similarity_matrix.npy
```

The comparison summary is written to:

```text
jepa_stage1_clip_ablation/outputs/compare_experiments.csv
```
