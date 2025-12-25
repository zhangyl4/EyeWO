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


### News

We are currently conducting liveCC-EyeWO extended training, aiming to further enhance the model into an even stronger multimodal large language model, comparable to Qwen or LLaVA. If you have relevant experience or encounter any issues, we welcome discussions and collaboration.

2024-12-25: Released data and model weights featured in the paper.

### Dependencies & Environment

We adopt the environment setup from **[videollm-online (CVPR 2024)](https://githubovideollm-online/)** as our primary configuration. Please refer to `env.sh` in that repository for the basic setup.

For offline multimodal large language model (MLLM) experiments, we use Hugging Face Transformers and only require the standard LLaVA environment.

For other baselines, please follow the official implementations for environment setup.

### Datasets and Model Weight Links

This repo relies on several public or to-be-opened datasets / data collections.  
Please fill in or update the links below when your datasets are publicly available.

- **datasets**

  - **ESTP-IT** (instruction tuning and origin catpion dataset): **ModelScope dataset repo**: `zhangyl9/ESTP-IT`  

  - **ESTP-Bench**(evaluation data and script):  **ModelScope dataset**: `zhangyl9/ESTP-Bench`  
  - **2FPS Original Ego4D Video:**  **ModelScope dataset**: `zhangyl9/ESTP_origin_video`  

- **Model Weight**
  - **VideoLLM-EyeWO join training version **:  **ModelScope model**: `zhangyl9/VideoLLM-EyeWO`  

### Quick Start: Training & Inference

#### Environment setup

```bash
git clone https://github.com/your_org/eyes-wide-open.git
cd eyes-wide-open

# Install basic dependencies (example)
bash env.sh
```

#### Initialize Model Weights

1. **Download Backbone Models**  
   - Download LLaMA3: `meta-llama/Meta-Llama-3-8B-Instruct`
   - Download SigLIP: `google/siglip-large-patch16-384`

2. **Download VideoLLM-Online LoRA Adapters**  
   - Obtain from: `chenjoya/videollm-online-8b-v1plus`

3. **Merge LoRA into Backbone**  
   - Run the merging script:  
     ```bash
     ./merge_lora.sh
     ```

4. **Extract Multimodal Projector Weights**  
   - Use the provided script to extract projector weights:  
     ```bash
     python extract_projector.py
     ```
   - This will generate the `mm_projector.bin` file needed for initialization.


#### Train on ESTP tasks

1. **Download the ESTP-IT Dataset**  
   - Obtain the dataset from the ModelScope repository: [`zhangyl9/ESTP-IT`](https://www.modelscope.cn/datasets?query=zhangyl9/ESTP-IT)  
   - Extract (untar) the dataset into the `./datasets` directory.

2. **Start Training**  
   - Refer to the configuration options and default values in `./models/arguments_live.py` to customize your training as needed.

**Tip:**  
For ease of reproduction, you can use the provided VideoLLM-Online initial weights and perform single-stage training (starting directly from stage 2). This will yield results comparable to those reported in the paper and serves as a strong baseline for future research or development.

**Usage:**  
Training scripts are provided under the `scripts/estp` directory (script names may vary; adapt as needed). For example:

```bash
bash scripts/estp/beacon_livel_h_stage3.5_livebase_cqa.sh  # Example script – replace with your chosen script
```

- If performing pre-training for 1 epoch, set `add_random_high_res_ratio` to `0`.
- After this, use `evaluate_wVsionEncoder.py` for inference to obtain results.
- Next, apply `data/estp/livechat.py` with the `HighResInsertor` to construct the final training dataset.


#### Inference / evaluation

##### ESTP Benchmark

1. **Prepare Models and Weights**
   - Construct the pretrained VideoLLM-Online model.
   - Download the model weights for VideoLLM-EyeWo.

2. **Download ESTPbench**
   - Obtain the ESTPbench dataset and place it in the `./data` directory.

3. **Run Evaluation Script**
   - To reproduce ESTP task results, you can use the following example script (see `eval_estp.sh` for details):
   ```bash
   # ESTP evaluation example
   export CUDA_VISIBLE_DEVICES=4,5,6,7
   python /2022233235/videollm-online/eval_estp_batch.py  \
       --data_file /2022233235/videollm-online/data/estp_dataset/estp_bench_sq.json \
       --model_name EWO \
       --llm_pretrained /2022233235/.cache/huggingface/hub/models--videollm-online-8b-v1plus/ \
       --pretrain_mm_mlp_adapter /2022233235/.cache/huggingface/hub/models--videollm-online-8b-v1plus/mm_projector.bin \
       --resume_from_checkpoint outputs/ego4d_ESTPSQA/beaconlivel_h_stage2_livebase_all \
       --add_type fusion \
       --add_vision_pretrained facebook/dinov2-large \
       --benchmark_name ESTP_singleQ_benchmark \
       --eval_mode frame_by_frame \
       --output_file /2022233235/videollm-online/data/estp_dataset/estpSqa_ours/LivebaseStage2_v4.json \
       --device cuda:0 \
       --master_port 2280
   ```

##### QAEgo4D and OVO-Bench Evaluation

1. **Download Datasets**
   - Download the QAEgo4D-MC-test dataset from HuggingFace: [`Becomebright/QAEgo4D-MC-test`](https://huggingface.co/Becomebright/QAEgo4D-MC-test)
   - Download OVO-Bench from HuggingFace: [`JoeLeelyf/OVO-Bench`](https://huggingface.co/JoeLeelyf/OVO-Bench)

2. **Run Evaluation Scripts**
   - To evaluate on OVO-Bench and QAEgo4D, use the following commands:
   ```bash
   # OVO-Bench evaluation
   torchrun --standalone --nproc_per_node=8 distributed_evaluate_ovobench_videollmeyewo.py

   # (Optional) Set ONLINE mode; 1 for online, 0 for offline
   export ONLINE=1

   # QAEgo4D evaluation
   torchrun --standalone --nproc_per_node=8 distributed_evaluate_qaego4d_videollmeyewo.py
   ```

> *Note: Our evaluation results are provided in the `evaluation/` directory.*




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

