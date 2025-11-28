import json, os
import argparse
from dataclasses import dataclass, field

import torch
import torch.multiprocessing as mp
import torch.distributed as dist

from typing import Optional, List   

####### DATASET CONFIG #######
@dataclass
class ESTP_singleQ_benchmark_config:
    video_root: str = '/2022233235/videollm-online/full_scale_2fps_max384'

@dataclass
class ESTP_contextualQ_benchmark_config:
    video_root: str = '/2022233235/videollm-online/full_scale_2fps_max384'

####### MODEL CONFIG #######
# offline model
@dataclass
class MiniCPMV_config:
    frame_fps: int = 1

@dataclass
class LLaVAOneVision_config:
    frame_fps: int = 1
    max_frames_num: int = 32
    
@dataclass
class LLaVANextVideo7_config:
    frame_fps: int = 1
    max_frames_num: int = 32

@dataclass
class InternVL_config:
    frame_fps: int = 1
    max_frames_num: int = 32
    
@dataclass
class Qwen2VL_config:
    frame_fps: int = 1
    max_frames_num: int = 32

@dataclass
class VILA_config:
    frame_fps: int = 1
    max_frames_num: int = 16


# streaming detector
@dataclass
class EgoVLP_config:
    backbone_name: str = "egovlp_base"
    classification_layer_name: str = "cosine_similarity"
    temporal_pooling_name: str = "identity"
    n_frames: int = 60
    frame_sample_rate: int = 1
    checkpoint_path: str = None
    model_name: str = "encode_pool_classify"
    backbone_lr: float = -1
    min_backbone_lr: float = -1
    task_name: str = "ESTP_singleQ_benchmark"
    eval_mode: str = "sdqes"
    temporal_pool_backbone: bool = False
    classification_input_dim: int = 512
    norm_mean: tuple[float] = (0.485, 0.456, 0.406)
    norm_std: tuple[float] = (0.229, 0.224, 0.225)
    spatial_size: int = 224

@dataclass
class CLIP_config:
    backbone_name: str = "clip_ViT-B/16"
    classification_layer_name: str = "cosine_similarity"
    temporal_pooling_name: str = "identity"
    n_frames: int = 60
    frame_sample_rate: int = 1
    checkpoint_path: str = None
    model_name: str = "encode_pool_classify"
    backbone_lr: float = -1
    min_backbone_lr: float = -1
    task_name: str = "ESTP_singleQ_benchmark"
    eval_mode: str = "sdqes"
    temporal_pool_backbone: bool = False
    classification_input_dim: int = 512
    norm_mean: tuple[float] = (0.48145466, 0.4578275, 0.40821073)
    norm_std: tuple[float] = (0.26862954, 0.26130258, 0.27577711)
    spatial_size: int = 224
    backbone_freeze: bool = False

@dataclass
class Lavila_config:
    backbone_name: str = "lavila_base"
    classification_layer_name: str = "cosine_similarity"
    temporal_pooling_name: str = "identity"
    n_frames: int = 60
    frame_sample_rate: int = 1
    checkpoint_path: str = None
    model_name: str = "encode_pool_classify"
    backbone_lr: float = -1
    min_backbone_lr: float = -1
    task_name: str = "ESTP_singleQ_benchmark"
    eval_mode: str = "sdqes"
    temporal_pool_backbone: bool = False
    classification_input_dim: int = 512
    norm_mean: tuple[float] = (0.42315351, 0.45603911, 0.40661616)
    norm_std: tuple[float] = (0.26758021, 0.26028187, 0.27469986)
    spatial_size: int = 224


# offline grounding model
@dataclass
class TimeChat_config:
    cfg_path: str = "/2022233235/videollm-online/baseline/TimeChat/eval_configs/timechat.yaml"
    num_beams: int = 1
    temperature: float = 1.0
    frame_fps: int = 2
    max_frames_num: int = 96
    height: int = 224
    width: int = 224
    options: List[str] = field(default_factory=list)

# online model
@dataclass
class VideollmOnline_config:
    frame_fps: int = 2
    resume_from_checkpoint: str = "chenjoya/videollm-online-8b-v1plus"
    llm_pretrained: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    frame_token_interval_threshold: Optional[float] = None
    
@dataclass
class MMDuet_config:
    llm_pretrained: str = "lmms-lab/llava-onevision-qwen2-7b-ov"
    bf16: bool = True
    lora_pretrained: str = "/root/MMDuet/output/mmduet/"
    stream_end_prob_threshold: float = 0.5
    frame_fps: int = 2
    max_num_frames: int = 400
    stream_end_prob_threshold_high: float = 0.5
    score_heads: str = "informative_score,relevance_score"
    remove_assistant_turns: bool = True
    attn_implementation: str = "sdpa"
    frame_resolution: int = 384
    

@dataclass
class EWO_config:
    # Required model configuration parameters
    resume_from_checkpoint: str = "outputs/ego4d_ESTPSQA/beaconlivel_h_ct_stage2_smoothing_random_ratio0.01"
    pretrain_mm_mlp_adapter: str = "outputs/ego_caption_train/livel_h_stage1_3_7/mm_projector.bin"
    live_version: str = "beaconlivel_h"
    finetune_modules: str = "beacon_embed_tokens connnetor"
    llm_pretrained: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    enable_beacon: bool = True
    skip_first: bool = True
    beacon_window: int = 720
    beacon_stride: int = 720
    beacon_attn: str = "full-coverage"
    beacon_attend_prev: bool = True
    beacon_sink_size: int = 0
    beacon_ratio: Optional[List[int]] = (72, 60, 48)
    beacon_ratio_mix: Optional[str] = "step-random"
    beacon_pos: Optional[str] = "interleave"
    beacon_param: Optional[List[str]] = ("q", "k", "v")
    compress_turn: Optional[int] = 1
    low_vision_encoder: bool = True
    add_vision_pretrained: str = ""
    add_type: str = 'fusion' # 'fusion' or 'dual'
    frame_token_interval_threshold: Optional[float] = None
    frame_token_interval_threshold_high: Optional[float] = None

@dataclass
class Qwen2VL_streaming_config:
    model_path: str = '/2022233235/videollm-online/livecc/outputs/livecc_sft_24k480x100_llava178k_sample_lr1e-5/checkpoint-853'
    streaming_eos_base_threshold: float = 0.90
    streaming_eos_threshold_step: float = 0

@dataclass
class Qwen2VL_EyeWO_config:
    pretrained_model_name_or_path: str = 'chenjoya/LiveCC-7B-Instruct'
    resume_from_checkpoint: str = '/2022233235/videollm-online/livecc_eyewo/outputs/livecc_eyewo_sft_24k768x100_lora_lr_sqacqa_balance_v51e-4/checkpoint-1170'
    downsample_ratio: int = 2
    
    enable_beacon: bool = True
    beacon_window: int = 1024
    beacon_stride: int = 1024
    beacon_attn: str = 'full-coverage'
    beacon_ratio: list[int] = (16, 32, 64)
    beacon_ratio_mix: str = 'step-random'
    beacon_param: list[str] = ('q', 'k', 'v')
    beacon_embed_init: str = "eos"
    beacon_sink_size: int = 0
    beacon_attend_prev: bool = True
    beacon_pos: str = 'interleave'
    beacon_parallel_window: int = 1
    beacon_accum: bool = True
    beacon_cache: str = None
    beacon_avg_init: bool = False
    beacon_avg: bool = False
    beacon_self_occurrence: bool = False
    return_all_logits: bool = False
    skip_first: bool = False
    compress_turn: int = 1
    is_smoothing: bool = False
    infer_ct: bool = True

def overwrite_config(args, config):
    if args.fbf_fps is not None:
        assert args.fbf_fps <= 2, "Frame by frame fps must be less than or equal to 2"
        config.frame_fps = args.fbf_fps
    if args.resume_from_checkpoint is not None:
        config.resume_from_checkpoint = args.resume_from_checkpoint
    if args.llm_pretrained is not None:
        config.llm_pretrained = args.llm_pretrained
    if args.pretrain_mm_mlp_adapter is not None:
        config.pretrain_mm_mlp_adapter = args.pretrain_mm_mlp_adapter
    if args.frame_token_interval_threshold is not None:
        config.frame_token_interval_threshold = args.frame_token_interval_threshold

    print(config)
        
    return config

def initialize_benchmark_and_model(args, local_data):
    ####### BENCHMARK #######
    if args.benchmark_name == "ESTP_singleQ_benchmark":
        from data.estp_dataset.benchmark.estp import ESTP_singleQ_benchmark
        benchmark = ESTP_singleQ_benchmark(local_data, config=ESTP_singleQ_benchmark_config)
    if args.benchmark_name == "ESTP_contextualQ_benchmark":
        from data.estp_dataset.benchmark.estp import ESTP_contextualQ_benchmark
        benchmark = ESTP_contextualQ_benchmark(local_data, config=ESTP_contextualQ_benchmark_config)
    
    ##########################
    
    ####### MODEL ############
    if args.model_name == "MiniCPMV":
        from data.estp_dataset.model.MiniCPMV import MiniCPMV
        config = MiniCPMV_config()
        config = overwrite_config(args, config)
        model = MiniCPMV(device=args.device, config=config)
    elif args.model_name == "LLaVAOneVision":
        from data.estp_dataset.model.LLaVAOneVision import LLaVAOneVision
        config = LLaVAOneVision_config()
        config = overwrite_config(args, config)
        model = LLaVAOneVision(device=args.device, config=config)
    elif args.model_name == "LLaVANextVideo7B":
        from data.estp_dataset.model.LLaVANextVideo32 import LLaVANextVideo7
        config = LLaVANextVideo7_config()
        config = overwrite_config(args, config)
        model = LLaVANextVideo7(device=args.device, config=config)
    elif args.model_name == "InternVLV28":
        from data.estp_dataset.model.InternVL import InternVL
        config = InternVL_config()
        config = overwrite_config(args, config)
        model = InternVL(device=args.device, config=config)
    elif args.model_name == "Qwen2VL":
        from data.estp_dataset.model.Qwen2VL import Qwen2VL
        config = Qwen2VL_config()
        config = overwrite_config(args, config)
        model = Qwen2VL(device=args.device, config=config) 
    elif args.model_name == "VILA":
        from data.estp_dataset.model.VILA import VILA
        config = VILA_config()
        config = overwrite_config(args, config)
        model = VILA(device=args.device, config=config)
    elif args.model_name == "EgoVLP":
        from data.estp_dataset.model.EgoVLP import EgoVLP
        config = EgoVLP_config()
        config = overwrite_config(args, config)
        model = EgoVLP(device=args.device, config=config)
    elif args.model_name == "CLIP":
        from data.estp_dataset.model.CLIP import CLIP
        config = CLIP_config()
        config = overwrite_config(args, config)
        model = CLIP(device=args.device, config=config)
    elif args.model_name == "Lavila":
        from data.estp_dataset.model.Lavila import Lavila
        config = Lavila_config()
        config = overwrite_config(args, config)
        model = Lavila(device=args.device, config=config)
    elif args.model_name == "TimeChat":
        from data.estp_dataset.model.TimeChat import TimeChat
        config = TimeChat_config()
        config = overwrite_config(args, config)
        model = TimeChat(device=args.device, config=config)
    elif args.model_name == "VideollmOnline":
        from data.estp_dataset.model.VideollmOnline import VideollmOnline
        config = VideollmOnline_config()
        config = overwrite_config(args, config)
        model = VideollmOnline(device=args.device, config=config)
    elif args.model_name == "MMDuet":
        from data.estp_dataset.model.MMDuet import MMDuet
        config = MMDuet_config()
        config = overwrite_config(args, config)
        model = MMDuet(device=args.device, config=config)
    elif args.model_name == "EWO":
        from data.estp_dataset.model.EWO import EWO
        config = EWO_config(resume_from_checkpoint=args.resume_from_checkpoint,
                            pretrain_mm_mlp_adapter=args.pretrain_mm_mlp_adapter,
                            llm_pretrained=args.llm_pretrained,
                            add_vision_pretrained=args.add_vision_pretrained,
                            add_type=args.add_type,
                            frame_token_interval_threshold=args.frame_token_interval_threshold,
                            frame_token_interval_threshold_high=args.frame_token_interval_threshold_high)
        model = EWO(device=args.device, config=config)
    elif args.model_name == "Qwen2VL_streaming":
        from data.estp_dataset.model.Qwen2VL_stream import Qwen2VL_streaming
        config = Qwen2VL_streaming_config()
        config = overwrite_config(args, config)
        model = Qwen2VL_streaming(device=args.device, config=config)
    elif args.model_name == "Qwen2VL_EyeWO":
        from data.estp_dataset.model.Qwen2VL_EyeWO import Qwen2VL_EyeWO
        config = Qwen2VL_EyeWO_config()
        config = overwrite_config(args, config)
        model = Qwen2VL_EyeWO(device=args.device, config=config)
    else:
        raise ValueError(f"Model {args.model_name} not found")
    return benchmark, model

def main_worker(rank, world_size, args):
    # 设置当前进程使用的 GPU
    torch.cuda.set_device(rank)
    args.device = f"cuda:{rank}"
    
    # 初始化分布式环境（此处采用 NCCL 后端，适用于 GPU 之间的通信）
    dist.init_process_group(backend="nccl", init_method="env://", rank=rank, world_size=world_size)
    
    # 加载数据
    with open(args.data_file, "r") as f:
        data = json.load(f)
    
    # 将数据分片（此处假设 data 为列表；如果是字典，需要根据实际情况修改分片方式）
    if isinstance(data, list):
        local_data = data[rank::world_size]
    elif isinstance(data, dict):
        keys = list(data.keys())
        local_keys = keys[rank::world_size]
        local_data = {k: data[k] for k in local_keys}
    else:
        local_data = data  # 未知结构则不分片

    benchmark, model = initialize_benchmark_and_model(args, local_data)
    
    ######################
    
    # 每个进程将评估结果写入各自的临时文件（例如：output_file.part0, output_file.part1, ...）
    local_output_file = f"{args.output_file}.part{rank}"
    benchmark.eval(local_data, model, local_output_file, args.eval_mode)
    

def main(args):
    # 若只检测到单卡，直接走单卡逻辑
    if torch.cuda.device_count() < 2:
        args.device = "cuda:0"
        with open(args.data_file, "r") as f:
            data = json.load(f)
    
        benchmark, model = initialize_benchmark_and_model(args, data)
        benchmark.eval(data, model, args.output_file, args.eval_mode)
    else:
        # 多卡运行：设置必要的环境变量（MASTER_ADDR 与 MASTER_PORT）
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = str(args.master_port)
        world_size = torch.cuda.device_count()
        mp.spawn(main_worker, args=(world_size, args), nprocs=world_size, join=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", type=str, required=True, help="Path to the data file")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model")
    parser.add_argument("--benchmark_name", type=str, required=True, help="Name of the benchmark")
    parser.add_argument("--output_file", type=str, required=True, help="Path to the output file")
    parser.add_argument("--eval_mode", type=str, default="frame_by_frame", help="Evaluation mode: frame_by_frame or grounding")
    parser.add_argument("--device", type=str, default="auto", help="Device to run the model on")
    parser.add_argument("--master_port", type=int, default=2958, help="Master port")
    
    # model args
    parser.add_argument("--pretrain_mm_mlp_adapter", type=str, default=None, help="Path to the mm_projector file")
    parser.add_argument("--llm_pretrained", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct", help="Path to the llm pretrained file")
    parser.add_argument("--add_vision_pretrained", type=str, default=None, help="Path to the vision pretrained file")
    parser.add_argument("--add_type", type=str, default=None, help="Type of the model: fusion or dual")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path to the checkpoint file")
    parser.add_argument("--frame_token_interval_threshold", type=float, default=None, help="Frame token interval threshold")
    parser.add_argument("--frame_token_interval_threshold_high", type=float, default=None, help="Frame token interval threshold for high resolution")
    parser.add_argument("--fbf_fps", type=float, default=None, help="Frame by frame fps")
    
    args = parser.parse_args()
    main(args)
