## Eyes Wide Open: NeurIPS 2025

**Paper title**: *Eyes Wide Open: Ego Proactive Video-LLM for Streaming Video*  
**Conference**: NeurIPS 2025  
**This repository** is the **official implementation** of the NeurIPS 2025 paper “Eyes Wide Open”, including training / inference code, configs, and scripts to reproduce the main results.

### Overview

- **Project name**: Eyes Wide Open (video-language multimodal model / online understanding framework)  
- **Main features**:
  - Train and evaluate the Eyes Wide Open model;
  - Support online / streaming video understanding scenarios;
  - Provide evaluation scripts for ESTP, OvObench and other benchmarks;

Some scripts in this repo are adapted from existing open-source projects (e.g., VideoLLM-online, Streamingbench, LiveCC).  

### Repository Structure

- `engine/`, `models/`, `train.py`: core model definitions and training entrypoints;
- `data/estp`, `data/preprocess`: ESTP-related preprocessing and dataset loading (other `data/*` directories are ignored by `.gitignore` to keep the repo small);
- `scripts/estp`: training / evaluation scripts for ESTP and related tasks (other subfolders under `scripts/` are ignored in `.gitignore`);
- `livecc/`, `livecc_eyewo/`: LiveCC-style data and script extensions (these folders are ignored by default; sync them separately if needed);
- `baseline/`: third-party baselines and related works (ignored from the main git history to avoid a huge repo).

### Dependencies & Environment

We adopt the environment setup from **[showlab/videollm-online: VideoLLM-online – Online Video Large Language Model for Streaming Video (CVPR 2024)](https://githubovideollm-online/)** as our primary configuration. Please refer to `env.sh` in that repository for the basic setup.

For offline multimodal large language model (MLLM) experiments, we use Hugging Face Transformers and only require the standard LLaVA environment.

For other baselines, please follow the official implementations for environment setup.

### Datasets and Model Weight Links

This repo relies on several public or to-be-opened datasets / data collections.  
Please fill in or update the links below when your datasets are publicly available.

- **datasets**

  - **ESTP-IT** (instruction tuning dataset): **ModelScope dataset repo**: `zhangyl9/ESTP-IT`  

  - **ESTP-Bench**(evaluation data and script):  **ModelScope dataset**: `zhangyl9/ESTP-Bench`  
  - **2FPS Original Ego4D Video:**  **ModelScope dataset**: `zhangyl9/ESTP_origin_video`  

- **Model Weight**

  - 

### Quick Start: Training & Inference

#### Environment setup

```bash
git clone https://github.com/your_org/eyes-wide-open.git
cd eyes-wide-open

# Install basic dependencies (example)
bash env.sh
```

#### Train on ESTP tasks

Use the scripts under `scripts/estp` (names may differ from the example below):

```bash
bash scripts/estp/beacon_livel_h_stage3.5_livebase_cqa.sh   # Example; replace with the script you actually use
# for pre 1 epoch, set add_random_high_res_ratio as 0, then using evaluate_wVsionEncoder.py to get inference result, after that, using data/estp/livechat.py HighResInsertor to construct final training dataset.
```

#### Inference / evaluation

```bash
# for ESTP task, refer to eval_estp.sh
# for ovobench and qaego4d,
distributed_evaluate_ovobench_videollmeyewo.py
EXPORT ONLINE=1 # oneline or not
torchrun --standalone --nproc_per_node=8 distributed_evaluate_qaego4d_videollmeyewo.py
# we provide ours result in evaluation/
```


### TODO

- ESTP-Gen
- LiveCC-EyeWO

### Acknowledge

We thank the open-source contributions of **VideoLLM-Online**, **StreamingBench**, and **Ego4D**.

We also gratefully acknowledge [Zhiyi Wang](https://zhiyi.vision/), [Dingyou Wang](https://openreview.net/profile?id=~Dingyou_Wang1), and **Sihang Zhuang** for their valuable assistance with data collection.

### Citation

If you find Eyes Wide Open or this repo useful in your research, please cite our paper (BibTeX placeholder below; update it once the camera-ready version is available):

```bibtex
@article{zhang2025eyes,
  title={Eyes wide open: Ego proactive video-llm for streaming video},
  author={Zhang, Yulin and Shi, Cheng and Wang, Yang and Yang, Sibei},
  journal={arXiv preprint arXiv:2510.14560},
  year={2025}
}
```

### License

- We recommend using **Apache-2.0** or **MIT** for the main codebase  
  (please choose one, modify this section accordingly, and add a `LICENSE` file at the repo root);
- Third-party code under directories such as `baseline/` and `livecc/` must follow their **original licenses**.

