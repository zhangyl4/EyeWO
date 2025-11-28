from dataclasses import dataclass, field
from transformers import TrainingArguments

from typing import Dict, Optional, Sequence, List

@dataclass
class ModelArguments:
    enable_beacon: bool = field(
        default=True,
        metadata={'help': 'Enable activation beacon?'}
    )
    beacon_window: Optional[int] = field(
        default=1440,
        metadata={'help': 'The initial sliding window size.'}
    )
    beacon_stride: Optional[int] = field(
        default=1440,
        metadata={'help': 'The stride of the sliding window.'}
    )
    beacon_attn: Optional[str] = field(
        default='full-coverage',
        metadata={'help': 'How to assign attention masks of beacon tokens? {segmentation, step-expansion, full-coverage}'}
    )
    beacon_ratio: Optional[List[int]] = field(
        default=(2,4,8),
        metadata={'help': 'Condensing ratios for beacons.'}
    )
    beacon_ratio_mix: Optional[str] = field(
        default='step-random',
        metadata={'help': 'How to determine the beacon_ratio for each input. {step-random, instance-random, adapt-x}'}
    )
    beacon_param: Optional[List[str]] = field(
        default=(),
        metadata={'help': 'The introduced parameters for beacon.'}
    )
    beacon_embed_init: str = field(
        default="eos",
        metadata={'help': 'Initialize beacon embedding from eos/bos embedding.'}
    )
    beacon_sink_size: Optional[int] = field(
        default=0,
        metadata={'help': 'The number of activations that are always kept in the head of the sequence according to StreamingLLM.'}
    )
    beacon_attend_prev: Optional[bool] = field(
        default=True,
        metadata={'help': 'Can beacon tokens attend to previous beacon tokens?'}
    )
    beacon_pos: Optional[str] = field(
        default='interleave',
        metadata={'help': 'Where to put beacon tokens? {append, interleave}'}
    )
    beacon_parallel_window: Optional[int] = field(
        default=1,
        metadata={'help': 'How many windows to run in parallel?'}
    )
    beacon_accum: Optional[bool] = field(
        default=True,
        metadata={'help': 'Can beacon tokens attend to previous beacon tokens?'}
    )
    beacon_cache: Optional[str] = field(
        default=None,
        metadata={'help': 'beacon token KV cache'}
    )
    beacon_avg_init: Optional[bool] = field(
        default=False,
        metadata={'help': 'Initialize beacon embedding from preview input token features.'}
    )
    beacon_avg: Optional[bool] = field(
        default=False,
        metadata={'help': 'beacon token kv cache is average of all previous tokens'}
    )
    beacon_self_occurrence: Optional[bool] = field(
        default=False,
        metadata={'help': 'whether beacon token features are used for average'}
    )
    
    return_all_logits: bool = False
    skip_first: bool = False
    compress_turn: Optional[int] = None
    
    
    # reponse args
    is_smoothing: bool = False
    
    # build args
    low_vision_encoder: bool = False
    high_vision_encoder: bool = True
    
    # vision args
    add_vision_pretrained: str = None
    add_type: str = 'no' # 'fusion' or 'dual'
    
    # sample strategy
    max_frame_clip_mode_model: Optional[str] = None # 'uniform' or 'query'
    
    
@dataclass
class DataArguments:
    root:str = None
    anno_path: str = None
    max_num_frames: int = 1200 # 1h, 2fps, 7200 frames
    learn_reponse: bool = True
    group_by_stride: Optional[str] = None
    sort_by_stride: Optional[str] = None
    
    # training arguments
    add_random_high_res_ratio: Optional[str] = "" # 0_1 high first all
    data_repeat_num: Optional[int] = 1
    mode: Optional[str] = None
    max_frame_clip_mode_data: Optional[str] = 'last' # 'uniform'
    
    # evaluate arguments
    eval_time_diff_late: bool = False
    force_rep: bool = False
    force_rep_para1:float = 1
    force_rep_para2:float = 1
    
    # config path for multi-dataset
    config_path: Optional[str] = None
    
    
@dataclass
class LiveTrainingArguments(TrainingArguments, ModelArguments, DataArguments):
    live_version: str = 'live1+'
    system_prompt: str = (
        "A multimodal AI assistant is helping users with some activities."
        " Below is their conversation, interleaved with the list of video frames received by the assistant."
    )
    train_datasets: list[str] = None
    eval_datasets: list[str] = None
    stream_loss_weight: float = 1.0
    llm_pretrained: str = 'meta-llama/Meta-Llama-3-8B-Instruct'
    vision_pretrained: str = 'google/siglip-large-patch16-384'
    # LLM LoRA parameters
    lora_modules: str = "model.*(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)|lm_head$"
    lora_r: int = 128
    lora_alpha: int = 256
    # Vision LoRA parameters
    enable_vision_lora: bool = False
    # vision_lora_modules: str = "encoder.*(q_proj|k_proj|v_proj)" # siglip lora
    vision_lora_modules: str = "encoder.*(query|key|value)" # dinov2 lora
    vision_lora_r: int = 32
    vision_lora_alpha: int = 64
    # Other parameters
    finetune_modules: list[str] = field(default_factory=lambda: ['connector'])
    only_modules_to_ft: list[str] = field(default_factory=lambda: [], metadata={'help': 'if not empty, only finetune these modules'})
    adapter_model: Optional[str] = None
    pretrain_mm_mlp_adapter: Optional[str] = None
    frame_fps: int = 2 # for training. inference can be 10
    frame_token_cls: bool = None
    frame_token_pooled: list[int] = None
    frame_resolution: int = 384
    frame_token_interval: str  = None
    frame_token_interval_threshold: Optional[float] = None
    frame_token_interval_threshold_high: Optional[float] = None
    augmentation: bool = False
    attn_implementation: str = 'flash_attention_2'
    output_dir: str = 'outputs/debug'

@dataclass
class LiveOneTrainingArguments(LiveTrainingArguments):
    live_version: str = 'live1'
    frame_token_cls: bool = True
    frame_num_tokens: int = 1
    frame_token_interval: str  = ','
    embed_mark: str = '2fps_max384_1'
    max_num_frames: int = 550 # 1h, 2fps, 7200 frames

@dataclass
class LiveOneTrainingArgumentsMamba(LiveOneTrainingArguments):
    live_version: str = 'live1_mamba'
    finetune_modules: list[str] = field(default_factory=lambda: ['connector', "loss_vison_head"])
    # frame_num_tokens: int = 2
    # frame_token_pooled: list[int] = field(default_factory=lambda: [3,3])
    # embed_mark: str = '2fps_max384_1+3x3'

@dataclass
class LiveOneTrainingArgumentsMambaFt(LiveOneTrainingArgumentsMamba):
    finetune_modules: list[str] = field(default_factory=lambda: ['connector.out_proj'])
    adapter_model:str  = "outputs/ego4d_narration_train/live1_mamba"

@dataclass
class LiveOnePlusTrainingArguments(LiveTrainingArguments):
    live_version: str = 'live1+'
    frame_token_cls: bool = True
    frame_token_pooled: list[int] = field(default_factory=lambda: [3,3])
    frame_num_tokens: int = 10 # 1+3x3
    embed_mark: str = '2fps_max384_1+3x3'
    frame_token_interval: str = ','
    max_num_frames: int = 1200 # 10min, 2fps, 1200 frames

@dataclass
class LiveOnePlusTrainingArgumentsA40(LiveTrainingArguments):
    live_version: str = 'live1+'
    frame_token_cls: bool = True
    frame_token_pooled: list[int] = field(default_factory=lambda: [3,3])
    frame_num_tokens: int = 10 # 1+3x3
    embed_mark: str = '2fps_max384_1+3x3'
    frame_token_interval: str = ','
    max_num_frames: int = 900 # 10min, 2fps, 1200 frames

@dataclass
class LiveOneOnePlusTrainingArguments(LiveTrainingArguments):
    live_version: str = 'live1_1+'
    frame_token_cls: bool = True
    frame_token_pooled_high: list[int] = field(default_factory=lambda: [3,3])
    frame_num_tokens: int = 1 # 1
    frame_num_tokens_high: int = 10 # 1+3x3
    embed_mark: str = '2fps_max384_1'
    embed_mark_high: str = '2fps_max384_1+3x3'
    frame_token_interval: str = ','
    high_frame_token_interval: str = '.'
    high_v_placeholder:str = '<hv>'
    max_num_frames: int = 550 # 10min, 2fps, 1200 frames

# # @dataclass
# class LiveLowHighTrainingArguments(LiveTrainingArguments):
#     live_version: str = 'livel_h'
#     frame_token_cls: bool = True
#     frame_token_pooled: list[int] = field(default_factory=lambda: [2,2]) # None
#     # frame_token_pooled = None
#     frame_token_pooled_high: list[int] = field(default_factory=lambda: [7,7])
#     frame_num_tokens: int = 1+2*2 # 1
#     # frame_num_tokens:int = 1
#     frame_num_tokens_high: int = 1+7*7 # 1+10x10
#     embed_mark: str = '2fps_max384_1+2*2'
#     # embed_mark: str = '2fps_max384_1'
#     embed_mark_high: str = '2fps_max384_1+7*7'
#     frame_token_interval: str = ','
#     high_frame_token_interval: str = '.'
#     high_v_placeholder:str = '<hv>'
#     max_num_frames: int = 1200 # 10min, 2fps, 1200 frames
    
@dataclass
class LiveLowHighTrainingArguments(LiveTrainingArguments):
    live_version: str = 'livel_h'
    frame_token_cls: bool = True
    frame_token_pooled: list[int] = field(default_factory=lambda: [3,3]) # None
    frame_token_pooled_high: list[int] = field(default_factory=lambda: [7,7])
    frame_num_tokens: int = 1+3*3 # 1
    frame_num_tokens_high: int = 1+7*7 # 1+10x10
    embed_mark: str = '2fps_max384_1+3x3'
    embed_mark_high: str = '2fps_max384_1+7x7'
    frame_token_interval: str = ','
    high_frame_token_interval: str = '.'
    high_v_placeholder:str = '<hv>'
    frame_token_interval_threshold: Optional[float] = None
    max_num_frames: int = 1200 # 10min, 2fps, 1200 frames
    
# @dataclass
# class LiveLowHighTrainingArguments(LiveTrainingArguments):
#     live_version: str = 'livel_h'
#     frame_token_cls: bool = True
#     frame_token_pooled: list[int] = field(default_factory=lambda: [2,2]) # None
#     # frame_token_pooled = None
#     frame_token_pooled_high: list[int] = field(default_factory=lambda: [4,4])
#     frame_num_tokens: int = 2+2*2*2 # 1
#     frame_num_tokens_high: int = 2+4*4*2 # 1+10x10
#     embed_mark: str = '2fps_max384_1+2x2'
#     embed_mark_high: str = '2fps_max384_1+4x4'
#     frame_token_interval: str = ','
#     high_frame_token_interval: str = '.'
#     high_v_placeholder:str = '<hv>'
#     max_num_frames: int = 1200 # 10min, 2fps, 1200 frames

def get_args_class(live_version: str):
    if live_version == 'live1':
        return LiveOneTrainingArguments
    elif live_version == 'live1_no_frame_interval' or live_version == 'live1_threshold':
        return LiveOneTrainingArguments
    elif 'live1_mamba' in live_version:
        if 'ft' in live_version:
            return LiveOneTrainingArgumentsMambaFt
        else:
            return LiveOneTrainingArgumentsMamba
    elif live_version == 'live1+':
        return LiveOnePlusTrainingArguments
    elif live_version == 'live1+a40':
        return LiveOnePlusTrainingArgumentsA40
    elif 'live1_1+' in live_version:
        return LiveOneOnePlusTrainingArguments
    elif 'livel_h' in live_version:
        return LiveLowHighTrainingArguments
    raise NotImplementedError
