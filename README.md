# JEPA Stage-1 CLIP Ablation

这个目录实现两个 JEPA 风格实验，用 `before image + instruction` 预测 CAD 操作 `highlight image` 的 CLIP embedding，并通过 retrieval 指标评估预测质量。

## Conda 环境

仓库根目录提供了 [environment.yml](/Users/baiyixue/project/jepa-cad/stage1_train/environment.yml)。

创建环境：

```bash
conda env create -f environment.yml
conda activate jepa-cad-stage1-clip
```

如果目标机器 CUDA 版本不是 12.1，请先把 `environment.yml` 里的 `pytorch-cuda=12.1` 改成匹配版本，或者删除这一行改用 CPU 环境。

## 项目目标

- 输入：`previous_model_grid.png` 和文本指令。
- 输出：`location.png` 对应的目标 embedding。
- 对比实验 A 与实验 B，观察冻结 CLIP 与 LoRA 适配的效果差异。

## 数据处理逻辑

- `build_manifest.py` 通过 `ssh lab1` 读取远端 `split_result.json`、`prompt.csv` 和图像目录。
- 自动过滤 `prompt.csv` 中 `op == chamfer_fillet` 的样本。
- 对 `step0` 且缺少 `previous_model_grid.png` 的样本，自动替换为本地生成的 `outputs/empty_before.png`。
- 其余缺少 prompt 或目标图的样本会跳过并打印 warning。
- 生成的 manifest 位于 `jepa_stage1_clip_ablation/outputs/manifests/`。

## 实验 A

- 冻结 CLIP image encoder。
- 冻结 CLIP text encoder。
- 冻结 target CLIP image encoder。
- 只训练 `fusion` 和 `predictor`。

运行：

```bash
python -m jepa_stage1_clip_ablation.run_exp_a
```

## 实验 B

- 对 CLIP image encoder 和 text encoder 注入 LoRA。
- target CLIP image encoder 保持冻结。
- 训练 LoRA 参数、`fusion` 和 `predictor`。

运行：

```bash
python -m jepa_stage1_clip_ablation.run_exp_b
```

## CLIP_MODEL_NAME 配置

在 [jepa_stage1_clip_ablation/config.py](/Users/baiyixue/project/jepa-cad/stage1_train/jepa_stage1_clip_ablation/config.py) 修改 `CLIP_MODEL_NAME`。如果当前环境无法联网，可以替换为本地 CLIP 模型目录。

## 构建 Manifest

```bash
python -m jepa_stage1_clip_ablation.build_manifest
```

## Retrieval 评估

```bash
python -m jepa_stage1_clip_ablation.eval_retrieval
```

## 输出位置

- Manifest: `jepa_stage1_clip_ablation/outputs/manifests/`
- 实验 A: `jepa_stage1_clip_ablation/outputs/exp_a_freeze_clip_predictor/`
- 实验 B: `jepa_stage1_clip_ablation/outputs/exp_b_lora_clip_image_text/`
- 实验对比: `jepa_stage1_clip_ablation/outputs/compare_exp_a_b.csv`
