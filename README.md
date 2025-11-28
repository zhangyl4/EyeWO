## Eyes Wide Open: NeurIPS 2025

**Paper title**: *Eyes Wide Open: [Put Full Paper Title Here]*  
**Conference**: NeurIPS 2025  
**This repository** is the **official implementation** of the NeurIPS 2025 paper “Eyes Wide Open”, including training / inference code, configs, and scripts to reproduce the main results.

> 简短中文说明：本仓库是 NeurIPS 2025 论文 **Eyes Wide Open** 的官方开源实现，包含训练、推理与评测脚本，方便复现实验结果。

### Overview

- **Project name**: Eyes Wide Open (video-language multimodal model / online understanding framework)  
- **Main features**:
  - Train and evaluate the Eyes Wide Open model;
  - Support online / streaming video understanding scenarios;
  - Provide evaluation scripts for ESTP, Ego4D and other benchmarks;
  - Integrate data and scripts for LiveCC / EyeWO2 style datasets (code kept, but most raw data is **not** distributed with this repo).

Some scripts in this repo are adapted from existing open-source projects (e.g., LiveCC).  
Their original code is kept mainly under `baseline/` and related folders; please refer to each subdirectory for the corresponding license.

### Repository Structure

- `engine/`, `models/`, `train.py`: core model definitions and training entrypoints;
- `data/estp`, `data/preprocess`: ESTP-related preprocessing and dataset loading (other `data/*` directories are ignored by `.gitignore` to keep the repo small);
- `scripts/estp`: training / evaluation scripts for ESTP and related tasks (other subfolders under `scripts/` are ignored in `.gitignore`);
- `EyeWO2/`, `livecc/`, `livecc_eyewo/`: EyeWO2 / LiveCC-style data and script extensions (these folders are ignored by default; sync them separately if needed);
- `baseline/`: third-party baselines and related works (ignored from the main git history to avoid a huge repo).

### Dependencies & Environment

We recommend **Python 3.10+** and **CUDA 12+**.  
Basic dependencies can be installed following `env.sh`, which roughly includes:

- `transformers==4.48.3`
- `accelerate`
- `deepspeed==0.15.4`
- `peft`
- `flash-attn`
- `moviepy`, `decord`
- data processing utilities such as `spacy`, `sentencepiece`, etc.

You are encouraged to turn this into a `requirements.txt` or `environment.yml` for one-click setup.

### Datasets & ModelScope / HuggingFace Links

This repo relies on several public or to-be-opened datasets / data collections.  
Please fill in or update the links below when your datasets are publicly available.

- **ESTP-IT** (instruction tuning dataset)
  - **ModelScope dataset repo**: `zhangyl9/ESTP-IT`  
  - Example upload script: see `datasets/upload_hf.py` (using `modelscope.hub.api.HubApi` to upload the whole `datasets/` directory).

- **ESTP-Bench / ESTP_Bench**
  - **HuggingFace dataset**: `zhangyl9/ESTP-Bench`  
  - Example upload code is also shown (commented) in `datasets/upload_hf.py` using `HfApi`.

- **EyeWO2 data** (for online / streaming video QA, captioning, etc.)
  - Related JSONL files are under `EyeWO2/data/`, e.g.:
    - `llava_video_178k_with_seeks_sample_valid.jsonl`
    - `cof_qwen2vl.jsonl`
    - `etbench_qwen2vl_timestamp.jsonl`
  - We recommend hosting them on ModelScope or HuggingFace. Example placeholders:
    - ModelScope: `modelscope://your_org/EyeWO2`
    - HuggingFace: `https://huggingface.co/datasets/your_org/EyeWO2`

- **LiveCC-style pretraining and instruction-tuning data**
  - Configuration and details can be found in `baseline/livecc/README.md`;
  - If you open-source Eyes Wide Open–specific LiveCC data, please add the links here, e.g.:
    - ModelScope: `modelscope://your_org/EyesWideOpen-LiveCC`
    - HuggingFace: `https://huggingface.co/datasets/your_org/EyesWideOpen-LiveCC`

> **Note**: `your_org` and some URLs above are placeholders. Please replace them with the actual public dataset repositories before making the repo public.

### Quick Start: Training & Inference

#### Environment setup

```bash
git clone https://github.com/your_org/eyes-wide-open.git
cd eyes-wide-open

# Install basic dependencies (example)
bash env.sh
```

#### Train on ESTP-style tasks

Use the scripts under `scripts/estp` (names may differ from the example below):

```bash
cd scripts/estp
bash beacon_livel_h_stage2_livebase.sh   # Example; replace with the script you actually use
```

#### Inference / evaluation

```bash
cd /2024233235/videollm-online
python evaluate.py \
  --config configs/your_config.yaml \
  --output_dir outputs/demo
```

Depending on the task (e.g., Ego4D QA, ESTP-Bench, coin benchmarks), you may instead use
`distributed_evaluate_qaego4d_videollmeyewo.py`, `distributed_evaluate_qaego4d_videollmonline.py`,
or other task-specific scripts.

### Relation to the NeurIPS 2025 Paper

- **Main model**: this repo contains the implementation of the Eyes Wide Open video-language model described in the NeurIPS 2025 paper;
- **Datasets**: ESTP-IT, ESTP-Bench, EyeWO2, LiveCC-derived data, etc., correspond to the training and evaluation datasets in the paper;
- **Scripts**: `scripts/estp`, `distributed_evaluate_*.py` and related files reproduce the main experiments (Ego4D, ESTP, online captioning / QA, and more).

### Citation

If you find Eyes Wide Open or this repo useful in your research, please cite our paper (BibTeX placeholder below; update it once the camera-ready version is available):

```bibtex
@inproceedings{eyeswideopen2025,
  title     = {Eyes Wide Open: [Full Title Here]},
  author    = {First Author and Second Author and Others},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2025}
}
```

### License

- We recommend using **Apache-2.0** or **MIT** for the main codebase  
  (please choose one, modify this section accordingly, and add a `LICENSE` file at the repo root);
- Third-party code under directories such as `baseline/` and `livecc/` must follow their **original licenses**.

