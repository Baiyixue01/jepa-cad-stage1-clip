# JEPA Stage-1 DINO Ablation

这个目录实现 JEPA 风格训练：用 `before image + instruction` 预测 CAD 操作 `highlight image` 的 frozen vision embedding，并用 retrieval 指标评估预测质量。

## Conda 环境

仓库根目录提供了 [environment.yml](/Users/baiyixue/project/jepa-cad/stage1_train/environment.yml)。

```bash
conda env create -f environment.yml
conda activate jepa-cad-stage1-clip
```

如果目标机器 CUDA 版本不是 12.1，请先把 `environment.yml` 里的 `pytorch-cuda=12.1` 改成匹配版本，或者删除这一行改用 CPU 环境。

## 数据处理逻辑

- 已生成的 manifest 位于 `jepa_stage1_clip_ablation/outputs/manifests/`。
- `build_manifest.py` 会过滤 `prompt.csv` 中 `op == chamfer_fillet` 的样本。
- 对 `step0` 且缺少 `previous_model_grid.png` 的样本，使用 `outputs/empty_before.png`。
- 训练机必须能访问 manifest 中的图片路径，默认是 `/data/baiyixue/CAD/op_orientated_render_data/...`。

## 实验设置

- 实验 A：`DINOv2-large` source/target frozen，CLIP text frozen，训练 `image_proj + text_proj + fusion + predictor`。
- 实验 B：`DINOv2-giant` source/target frozen，训练层与 A 相同，用于比较更强视觉 backbone。
- 实验 C-large：`DINOv2-large` source LoRA，target frozen，训练 LoRA 和 prediction head。
- 实验 C-giant：`DINOv2-giant` source LoRA，target frozen，训练 LoRA 和 prediction head。
- 实验 D：source 默认 `DINOv2-large`，target 预留为 DINOv3/7B frozen embedding 空间。运行前必须在 `config.py` 里把 `DINO_V3_7B_MODEL_NAME` 改成真实本地路径或模型名。

## 离线训练

无网训练机需要提前准备模型、数据和 Python 环境。详细步骤见 [OFFLINE_TRAINING.md](/Users/baiyixue/project/jepa-cad/stage1_train/OFFLINE_TRAINING.md)。

最少需要准备这些模型：

```bash
huggingface-cli download facebook/dinov2-large --local-dir /path/to/models/dinov2-large
huggingface-cli download facebook/dinov2-giant --local-dir /path/to/models/dinov2-giant
huggingface-cli download openai/clip-vit-base-patch16 --local-dir /path/to/models/clip-vit-base-patch16
```

然后在 [jepa_stage1_clip_ablation/config.py](/Users/baiyixue/project/jepa-cad/stage1_train/jepa_stage1_clip_ablation/config.py) 把模型名改成本地路径。

## 运行

构建 manifest：

```bash
python build_manifest.py
```

训练单个实验：

```bash
python run_exp_a.py
python run_exp_b.py
python run_exp_c_large.py
python run_exp_c_giant.py
python run_exp_d.py
```

按配置顺序全部运行：

```bash
python run_all_experiments.py
```

评估 retrieval：

```bash
python eval_retrieval.py
```

## 输出位置

- Manifest: `jepa_stage1_clip_ablation/outputs/manifests/`
- 实验输出: `jepa_stage1_clip_ablation/outputs/<experiment_name>/`
- Checkpoint: `jepa_stage1_clip_ablation/outputs/<experiment_name>/checkpoints/`
- 训练日志: `jepa_stage1_clip_ablation/outputs/<experiment_name>/logs/train_log.csv`
- 实验配置快照: `jepa_stage1_clip_ablation/outputs/<experiment_name>/logs/experiment_config.json`
- Retrieval 结果: `jepa_stage1_clip_ablation/outputs/<experiment_name>/eval/`
- 汇总结果: `jepa_stage1_clip_ablation/outputs/compare_experiments.csv`
