import json, os, torch, functools, tqdm, random, sys
import numpy as np
import decord
from torch.utils.data import Dataset
from dataclasses import dataclass
from typing import Optional, List

from dataclasses import asdict

from models import build_model_and_tokenizer, set_args_highres
from data import build_concat_train_dataset, build_eval_dataset_dict, get_data_collator, get_compute_metrics_dict
from engine import TrainerWithGenToEval
from transformers import Trainer, AutoProcessor, HfArgumentParser, TrainingArguments, AutoConfig, logging, TrainerCallback

@dataclass
class EWOConfig:
    """Fixed configuration for EWO model inference"""
    
    # Required model configuration parameters
    resume_from_checkpoint: str = "/2022233235/videollm-online/outputs/ego4d_ESTPSQA/beaconlivel_h_stage2_livebase_wodino"
    pretrain_mm_mlp_adapter: str = "/2022233235/.cache/huggingface/hub/models--videollm-online-8b-v1plus/mm_projector.bin"
    live_version: str = "beaconlivel_h"
    finetune_modules: str = "beacon_embed_tokens connnetor"
    llm_pretrained: str = "/2022233235/.cache/huggingface/hub/models--videollm-online-8b-v1plus/"
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
    compress_turn: Optional[int] = 2
    low_vision_encoder: bool = True
    # add_vision_pretrained: str = "facebook/dinov2-large"
    # add_type: str = 'fusion' # 'fusion' or 'dual'
    frame_token_interval_threshold: Optional[float] = None
    frame_token_interval_threshold_high: Optional[float] = None


logger = logging.get_logger(__name__)

def _read_may1fps_video_decord(ele: dict):
    """read video using decord.VideoReader. can handle more cases compared to _read_video_decord.

    Args:
        ele (dict): a dict contains the configuration of video.
        support keys:
            - video: the path of video. support "file://", "http://", "https://" and local path.
            - video_start: the start time of video.
            - video_end: the end time of video.
    Returns:
        torch.Tensor: the video tensor with shape (T, C, H, W).
        sample_fps
        clip_pts if return_pts=True
    """
    video_path = ele["video"]
    if os.path.exists(video_path):
        vr = decord.VideoReader(video_path, num_threads=2)
    elif ele['remote_loader'] is not None:
        vr = decord.VideoReader(ele['remote_loader'](video_path), num_threads=2)
    else:
        raise ValueError(f'video_path {video_path} not found')
    video_start = ele.get('video_start', None)
    video_end = ele.get('video_end', None)
    video_fps = vr.get_avg_fps()
    clip_idxs, clip_pts = None, None
    if video_start is not None or video_end is not None:
        vr.get_frame_timestamp(0)
        video_pts = vr._frame_pts[:,1]
        video_start = video_pts[0] if not video_start else video_start
        video_end = video_pts[-1] if not video_end else video_end
        video_start = min(max(video_pts[0], video_start), video_pts[-1])
        video_end = min(max(video_pts[0], video_end), video_pts[-1])
        video_end = max(video_start + 1, video_end)
        clip_idxs = ((video_start <= video_pts) & (video_pts <= video_end)).nonzero()[0]
        total_frames = len(clip_idxs)
    else:
        total_frames = len(vr)
    total_frames_for_smart_nframes = total_frames
    video_fps_for_smart_nframes = video_fps
    if total_frames < 2:
        total_frames_for_smart_nframes = 2
    if video_fps < FPS:
        total_frames_for_smart_nframes = int(total_frames * FPS / video_fps)
        video_fps_for_smart_nframes = FPS
    nframes = smart_nframes(ele, total_frames=total_frames_for_smart_nframes, video_fps=video_fps_for_smart_nframes) 
    nframes_idxs = np.linspace(0, total_frames - 1, nframes).round().astype(int)
    clip_idxs = nframes_idxs if clip_idxs is None else clip_idxs[nframes_idxs]
    clip = vr.get_batch(clip_idxs).permute(0, 3, 1, 2)  # Convert to TCHW format
    sample_fps = len(clip_idxs) / max(total_frames, 1e-6) * video_fps
    return clip, sample_fps

def save_function_print(function: callable, save_path: str, *args, **kwargs):
    original_stdout = sys.stdout
    try:
        with open(save_path, 'w') as f:
            sys.stdout = f  
            function(*args, **kwargs)          
    finally:
        sys.stdout = original_stdout 

import os, torch
import numpy as np
import decord # NOTE: import decord should be after torch, otherwise seg fault
from transformers import logging
from torchvision import transforms

os.environ['FORCE_QWENVL_VIDEO_READER'] = 'decord+'
os.environ['VIDEO_MAX_PIXELS'] = str(int(os.environ.get('VIDEO_MAX_PIXELS', 24576 * 28 * 28))) # increase this for streaming. 24576 * 28 * 28 = 19267584
import qwen_vl_utils.vision_process
qwen_vl_utils.vision_process.VIDEO_MIN_PIXELS = int(os.environ.get('VIDEO_MIN_PIXELS', 100 * 28 * 28)) # follow qwen2vl paper
qwen_vl_utils.vision_process.FPS_MAX_FRAMES = int(os.environ.get('FPS_MAX_FRAMES', 480)) # decrease this for efficiency 
from qwen_vl_utils.vision_process import (
    FORCE_QWENVL_VIDEO_READER, VIDEO_TOTAL_PIXELS, FPS_MAX_FRAMES, VIDEO_MIN_PIXELS, VIDEO_MAX_PIXELS, FRAME_FACTOR, IMAGE_FACTOR, FPS,
    smart_nframes, smart_resize
)

# os.environ['FORCE_QWENVL_VIDEO_READER'] = 'decord+'
# os.environ['VIDEO_MAX_PIXELS'] = str(int(16384 * 28 * 28)) # increase this for streaming. 24576 * 28 * 28 = 19267584
# import qwen_vl_utils.vision_process
# qwen_vl_utils.vision_process.VIDEO_MIN_PIXELS = int(os.environ.get('VIDEO_MIN_PIXELS', 128 * 28 * 28)) # follow qwen2vl paper
# qwen_vl_utils.vision_process.FPS_MAX_FRAMES = int(os.environ.get('FPS_MAX_FRAMES', 768)) # decrease this for efficiency 
# from qwen_vl_utils.vision_process import (
#     FORCE_QWENVL_VIDEO_READER, VIDEO_TOTAL_PIXELS, FPS_MAX_FRAMES, VIDEO_MIN_PIXELS, VIDEO_MAX_PIXELS, FRAME_FACTOR, IMAGE_FACTOR, FPS,
#     smart_nframes, smart_resize
# )


def _spatial_resize_video(video: torch.Tensor): # 
    video = transforms.functional.resize(
        video,
        [384, 384],
        interpolation=transforms.InterpolationMode.BICUBIC,
        antialias=True,
    ).float() # need float?
    return video


class OvoBenchMCQDataset(Dataset):
    def __init__(self, remote_loader, path, question_prefix, question_postfix, answer_prefix, sample: int = None, 
                 tokenizer=None, frame_fps=4, max_num_frames=1200, add_random_high_res_ratio=None):
        lines = open(path).readlines()
        if sample is not None:
            random.seed(42)
            lines = random.sample(lines, sample)
        self.datums = [json.loads(line) for line in tqdm.tqdm(lines, desc='load datums')]
        if isinstance(self.datums[0], str):
            self.datums = [json.loads(datum) for datum in tqdm.tqdm(self.datums, desc='load datumsx2')]
        # self.datums = [datum for datum in self.datums if datum['task'] not in ['REC', 'SSR', 'CRR']]
        self.src_video_dir = os.path.dirname("/2022233235/.cache/huggingface/hub/datasets--JoeLeelyf--OVO-Bench/snapshots/fec29e3385747b5642d995370143ba92d2819bd2/src_videos/")
        self.question_prefix = question_prefix
        self.question_postfix = question_postfix
        self.answer_prefix = answer_prefix
        self.remote_loader = remote_loader
        self.tokenizer = tokenizer
        self.frame_fps = frame_fps
        self.max_num_frames = max_num_frames
        self.add_random_high_res_ratio = add_random_high_res_ratio
        
    def __len__(self):
        return len(self.datums)

    def _add_stream_and_high_res(self, conversation, high_res_times, num_frames):
        """Add stream and stream_high roles to conversation following benchmarks_high.py format"""
        if self.add_random_high_res_ratio is None:
            conversation.extend([
                {'role': 'stream', 'num_frames': num_frames, 'learn': False},
                {"role": "stream_high", 'num_frames': 1, 'learn': False},
            ])
            high_res_times.append(num_frames - 1)  # Last frame as high res
        else:
            if isinstance(self.add_random_high_res_ratio, str):
                add_random_high_res_ratio = float(self.add_random_high_res_ratio)
            else:
                add_random_high_res_ratio = self.add_random_high_res_ratio
                
            if 0 < add_random_high_res_ratio <= 1:
                # Calculate number of high res frames to insert
                num_high_res = max(1, int(num_frames * add_random_high_res_ratio))
                # Generate random positions to insert high res frames
                high_res_positions = sorted(random.sample(range(num_frames), num_high_res))
                
                current_pos = 0
                for pos in high_res_positions:
                    if pos == 0:
                        continue
                    if pos > current_pos:
                        # Add regular frames up to this position
                        conversation.append({
                            'role': 'stream', 
                            'num_frames': pos - current_pos,
                            'learn': False
                        })
                    # Add high res frame
                    conversation.append({"role": "stream_high", 'num_frames': 1, 'learn': False})
                    high_res_times.append(pos)
                    current_pos = pos
                
                # Add remaining regular frames if any
                if current_pos < num_frames:
                    conversation.append({
                        'role': 'stream',
                        'num_frames': num_frames - current_pos,
                        'learn': False
                    })
                    # Add last stream_high
                    conversation.append({"role": "stream_high", 'num_frames': 1, 'learn': False})  
                    high_res_times.append(num_frames - 1)
            else:
                # Fallback to default behavior
                conversation.extend([
                    {'role': 'stream', 'num_frames': num_frames, 'learn': False},
                    {"role": "stream_high", 'num_frames': 1, 'learn': False},
                ])
                high_res_times.append(num_frames - 1)

    def __getitem__(self, i):
        datum = self.datums[i]
        
        # Prepare query
        if datum['task'] in ['REC', 'SSR', 'CRR']:
            query = datum['question']
            video, _ = _read_may1fps_video_decord({
                'video': os.path.join(self.src_video_dir, datum['video']), 
                'video_end': datum['video_end'],
                'remote_loader': self.remote_loader
            })
        else:
            query = self.question_prefix + datum['question'] + '\n' + '\n'.join(datum['options']) + self.question_postfix
            video, _ = _read_may1fps_video_decord({
                'video': os.path.join(self.src_video_dir, datum['video']), 
                # 'video_start': datum['video_end'] - 20, 
                'video_end': datum['video_end'], 
                # 'fps': 4, 
                'remote_loader': self.remote_loader
            })
        
        video = _spatial_resize_video(video)
        num_frames = video.shape[0]
        
        # Create conversation in stream format
        conversation = []
        high_res_times = []
        
        # Add stream and high res following benchmarks_high.py format
        self._add_stream_and_high_res(conversation, high_res_times, num_frames)
        conversation.append({"role": "user", "content": query})
        
        # Add system prompt and format conversation
        system_prompt: str = (
        "A multimodal AI assistant is helping users with some activities."
        " Below is their conversation, interleaved with the list of video frames received by the assistant."
        )
        conversation = [{"role": "system", "content": system_prompt}] + conversation # TODO
        
        # Apply chat template
        if self.tokenizer:
            text = self.tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
            text = text + self.answer_prefix
            learn_ranges = []  # No learning ranges for evaluation
        else:
            text = query + self.answer_prefix
            learn_ranges = []
        # Prepare frames data
        frames = video  # Regular resolution frames
        if high_res_times:
            high_frames = video[high_res_times]
        else:
            # Create empty tensor with correct shape
            high_frames = torch.empty(0, *video.shape[1:])
        all_high_frames = video  # All frames at high resolution for this simple case
        
        # Evaluation kwargs
        evaluation_kwargs = {
            # 'evaluator': 'generate', 
            # 'max_new_tokens': 512, 
            # 'do_sample': False, 
            # 'use_cache': True, 
            # 'temperature': 1.0, 
            # 'top_p': 1.0
        }
        
        response_clip = [(num_frames-1, num_frames)]  # Response clipping info
        
        return text, frames, high_frames, all_high_frames, learn_ranges, i, evaluation_kwargs, response_clip

    def data_collator_high(self, batch, **kwargs):
        """Data collator following data_collator.py format"""
        batch_data = list(zip(*batch))
        batch_text, batch_frames, batch_high_frames, batch_high_frames_all, batch_learn_ranges, batch_sample_idx, batch_evaluation_kwargs, batch_response_clip = batch_data
        
        # Tokenize texts
        if self.tokenizer:
            batch_tokenized = self.tokenizer(batch_text, return_offsets_mapping=True, add_special_tokens=False, return_tensors="pt", padding=True)
            # For evaluation, we don't need labels, so we can skip label processing
            batch_tokenized.pop('offset_mapping', None)
        else:
            # Fallback if no tokenizer
            batch_tokenized = {'input_ids': torch.tensor([[0] * 100] * len(batch_text))}  # Dummy
        
        # Process frames - handle empty tensors properly
        batch_tokenized['frames'] = torch.cat(batch_frames) if all(f.numel() > 0 for f in batch_frames) else torch.empty(0)
        
        # Handle high frames - may be empty
        non_empty_high_frames = [f for f in batch_high_frames if f.numel() > 0]
        if non_empty_high_frames:
            batch_tokenized['high_frames'] = torch.cat(non_empty_high_frames)
        else:
            batch_tokenized['high_frames'] = torch.empty(0)
            
        batch_tokenized['high_frames_all'] = torch.cat(batch_high_frames_all) if all(f.numel() > 0 for f in batch_high_frames_all) else torch.empty(0)
        batch_tokenized['sample_idxs'] = torch.tensor(batch_sample_idx)
        
        if batch_evaluation_kwargs[0]:
            batch_tokenized['evaluation_kwargs'] = batch_evaluation_kwargs[0]
        
        if batch_response_clip:
            batch_tokenized['response_clip'] = torch.tensor(batch_response_clip)
        
        return batch_tokenized

def preprocess_logits_for_metrics(logits, labels, strict_option_ids): 
    return torch.stack([logit[(logit[:, 0] != -100).nonzero().squeeze()[-1], strict_option_ids] for logit in logits]).argmax(dim=-1)

def mcq_predict(
    model, 
    tokenizer,
    benchmark_path: str, 
    options: list[str], 
    remote_loader: callable,
    question_prefix: str = '', 
    question_postfix: str = '\nPlease select the correct answer.', 
    answer_prefix: str = 'Answer:', 
    abcd_previous_str: str = ': ',
    use_liger_kernel: bool = True,
    per_device_eval_batch_size: int = 2,
    dataloader_num_workers: int = 4,
    frame_fps: int = 4,
    max_num_frames: int = 1200,
    add_random_high_res_ratio=None,
):
    strict_option_ids = [tokenizer(f'{abcd_previous_str}{_}').input_ids[-1] for _ in options] 
    dataset = OvoBenchMCQDataset(
        remote_loader, benchmark_path, 
        question_prefix=question_prefix, 
        question_postfix=question_postfix, 
        answer_prefix=answer_prefix,
        tokenizer=tokenizer,
        frame_fps=frame_fps,
        max_num_frames=max_num_frames,
        add_random_high_res_ratio=add_random_high_res_ratio
    )
    
    # Use the dataset's own data collator
    data_collator = functools.partial(dataset.data_collator_high, tokenizer=tokenizer)
    
    trainer = TrainerWithGenToEval(
        model=model, 
        args=TrainingArguments(
            output_dir='outputs/', do_predict=True, 
            per_device_eval_batch_size=per_device_eval_batch_size, 
            dataloader_num_workers=dataloader_num_workers, 
            report_to='none', use_liger_kernel=use_liger_kernel
        ), 
        data_collator=data_collator,
        tokenizer=tokenizer,
        preprocess_logits_for_metrics=functools.partial(preprocess_logits_for_metrics, strict_option_ids=strict_option_ids),
    )
    letter_idxs_predictions = trainer.predict(dataset, ignore_keys=['past_key_values', 'hidden_states', 'attentions', 'rope_deltas']).predictions
    return letter_idxs_predictions, dataset.datums, trainer.args.process_index

def evaluate_ovobench_results(results: list):
    task_to_counts = {}
    for result in results:
        task = result['task']
        if task not in task_to_counts:
            task_to_counts[task] = {'correct': 0, 'total': 0}
        task_to_counts[task]['total'] += 1
        if result['response'][:len(result['answer'])] == result['answer']:
            task_to_counts[task]['correct'] += 1
    rt_accs, bt_accs, fr_accs = [], [], []
    for task, counts in task_to_counts.items():
        print(f'{task}: {counts["correct"]}/{counts["total"]}={counts["correct"]/counts["total"]}')
        if task in ['OCR', 'ACR', 'ATR', 'STU', 'FPD', 'OJR']:
            rt_accs.append(counts['correct']/counts['total'])
        elif task in ['EPM', 'ASI', 'HLD']:
            bt_accs.append(counts['correct']/counts['total'])
        else:
            fr_accs.append(counts['correct']/counts['total'])
    if rt_accs:
        print(f'Real-Time Visual Perception avg.: {sum(rt_accs)}/{len(rt_accs)}={sum(rt_accs)/len(rt_accs)}')
    if bt_accs:
        print(f'Backward Tracing avg.: {sum(bt_accs)}/{len(bt_accs)}={sum(bt_accs)/len(bt_accs)}')
    if fr_accs:
        print(f'Forward Tracing avg.: {sum(fr_accs)}/{len(fr_accs)}={sum(fr_accs)/len(fr_accs)}')

if __name__ == '__main__':
    config = EWOConfig()
    # 推荐用 set_args_highres(config) 生成 args
    args = set_args_highres(config)
    model, tokenizer = build_model_and_tokenizer(is_training=False, set_vision_inside=True, **asdict(args))
    
    # OVO-Bench evaluation settings
    options = ['No', 'Yes', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E']
    benchmark_path = '/2022233235/videollm-online/livecc/ovo-bench-formatted.jsonl' 
    
    letter_idxs_predictions, benchmark_datums, process_index = mcq_predict(
        model=model, 
        tokenizer=tokenizer, 
        benchmark_path=benchmark_path, 
        options=options, 
        use_liger_kernel=False,
        answer_prefix='The answer is:\n', 
        abcd_previous_str='\n',
        remote_loader=None,
        frame_fps=args.frame_fps,
        max_num_frames=args.max_num_frames,
        add_random_high_res_ratio=0,
        per_device_eval_batch_size=1,  # Reduce batch size for stability
        dataloader_num_workers=2,
    )
    
    if process_index == 0:
        results = []
        for datum, letter_idx_prediction in zip(benchmark_datums, letter_idxs_predictions):
            results.append({
                'id': datum['id'],
                "task": datum['task'],
                "question": datum['question'],
                "options": datum['options'],
                "answer": datum['answer'],
                "response": options[letter_idx_prediction],
            })
        
        # Save results with EWO model identifier
        model_name = "EWO_beaconlivel_h"
        save_json_path = f'evaluation/ovobench/results/{model_name}_it_ovobench_results.json'
        os.makedirs(os.path.dirname(save_json_path), exist_ok=True)
        json.dump(results, open(save_json_path, 'w'), indent=2)
        
        save_txt_path = save_json_path.replace('.json', '.txt')
        save_function_print(
            evaluate_ovobench_results,
            save_txt_path,
            results
        )
        
        print(f"Results saved to: {save_json_path}")
        print(f"Evaluation summary saved to: {save_txt_path}")

# torchrun --standalone --nproc_per_node=8 distributed_evaluate_ovobench_videollmeyewo.py