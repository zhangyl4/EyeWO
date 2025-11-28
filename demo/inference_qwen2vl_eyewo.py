import functools, torch
from liger_kernel.transformers import apply_liger_kernel_to_qwen2_vl
apply_liger_kernel_to_qwen2_vl()
from transformers import AutoProcessor, LogitsProcessor, logging
from livecc_utils import prepare_multiturn_multimodal_inputs_for_generation, get_smart_resized_clip, get_smart_resized_video_reader, _read_video_decord_plus, _spatial_resize_video
from livecc_utils.video_process_patch import VIDEO_MAX_PIXELS, VIDEO_TOTAL_PIXELS, FRAME_FACTOR, VIDEO_MIN_PIXELS, smart_resize, IMAGE_FACTOR
from qwen_vl_utils.vision_process import floor_by_factor
from torchvision import transforms
from qwen_vl_utils import process_vision_info
from transformers import AutoConfig
from dataclasses import asdict
from peft import PeftModel
from functools import partial
import time, os, json
import decord
decord.bridge.set_bridge('native')

from livecc_eyewo.model.modeling_qwen2_vl import Qwen2VLEyeWOForConditionalGeneration


logger = logging.get_logger(__name__)

def updata_config(config, **kwargs):
    overwrite_config = {}
    
    # model args
    overwrite_config["beacon_window"] = kwargs['beacon_window']
    overwrite_config["beacon_stride"] = kwargs['beacon_stride']
    overwrite_config["beacon_attn"] = kwargs['beacon_attn']
    overwrite_config["beacon_attend_prev"] = kwargs['beacon_attend_prev']
    overwrite_config["beacon_sink_size"] = kwargs['beacon_sink_size']
    overwrite_config["beacon_ratio"] = kwargs['beacon_ratio']
    overwrite_config["beacon_ratio_mix"] = kwargs['beacon_ratio_mix']
    overwrite_config["beacon_param"] = kwargs['beacon_param']
    overwrite_config["beacon_pos"] = kwargs['beacon_pos']
    overwrite_config["beacon_parallel_window"] = 1
    overwrite_config["beacon_embed_init"] = "eos"
    overwrite_config["enable_beacon"]=kwargs['enable_beacon']
    overwrite_config["beacon_accum"]=kwargs['beacon_accum']
    
    overwrite_config['beacon_avg_init'] = kwargs['beacon_avg_init']
    overwrite_config['beacon_avg'] = kwargs['beacon_avg']
    overwrite_config['beacon_self_occurrence'] = kwargs['beacon_self_occurrence']
    
    if kwargs['beacon_cache'] is not None:
        part1, part2 = kwargs['beacon_cache'].split('__')
        part1, part2 = [int(x) for x in part1.split('_')],  [int(x) for x in part2.split('_')]
        beacon_cache = [part1, part2]
    else:
        beacon_cache = None
    overwrite_config['beacon_cache'] = beacon_cache
    
    overwrite_config['return_all_logits'] = kwargs['return_all_logits']
    overwrite_config['skip_first'] = kwargs['skip_first']
    overwrite_config['compress_turn'] = kwargs['compress_turn']
    overwrite_config['is_smoothing'] = kwargs['is_smoothing']
    overwrite_config['infer_ct'] = kwargs['infer_ct']
    
    if overwrite_config:
        for k, v in overwrite_config.items():
            setattr(config, k, v)
            del kwargs[k]

    return config



def _spatial_downsample_video(video: torch.Tensor, nframes: int = None, downsample_ratio=2):
    """
    Spatially downsample video for lower resolution processing.
    Reference from lmm_dataset.py
    """
    if not nframes:
        nframes, _, height, width = video.shape
    else:
        height, width = video.shape[2:]
    max_pixels = max(min(VIDEO_MAX_PIXELS, VIDEO_TOTAL_PIXELS / nframes * FRAME_FACTOR), int(VIDEO_MIN_PIXELS * 1.05))
    resized_height, resized_width = smart_resize(
        height,
        width,
        factor=IMAGE_FACTOR,
        min_pixels=VIDEO_MIN_PIXELS,
        max_pixels=max_pixels,
    )
    resized_height, resized_width = floor_by_factor(height / downsample_ratio, IMAGE_FACTOR), floor_by_factor(width / downsample_ratio, IMAGE_FACTOR)
    video = transforms.functional.resize(
        video,
        [resized_height, resized_width],
        interpolation=transforms.InterpolationMode.BICUBIC,
        antialias=True,
    ).float() # need float?
    return video

class ThresholdLogitsProcessor(LogitsProcessor):
    def __init__(self, token_id: int, base_threshold: float, step: float):
        self.token_id = token_id
        self.base_threshold = base_threshold
        self.step = step
        self.count = 0
    
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        threshold = self.base_threshold + self.step * self.count 
        low_confidence = torch.softmax(scores, dim=-1)[:, self.token_id] <= threshold
        if low_confidence.any():
            scores[low_confidence, self.token_id] = -float("inf")
        self.count += 1
        return scores
    
class LiveCCDemoInfer:
    VIDEO_PLAY_END = object()
    VIDEO_PLAY_CONTINUE = object()
    fps = 2
    initial_fps_frames = 6
    streaming_fps_frames = 2
    initial_time_interval = initial_fps_frames / fps
    streaming_time_interval = streaming_fps_frames / fps
    frame_time_interval = 1 / fps

    def __init__(self, model_path: str = None, device: str = 'cuda', config=None):
        model_args = config
        config = AutoConfig.from_pretrained(model_args.pretrained_model_name_or_path)
        config = updata_config(config, **asdict(model_args))
        self.model = Qwen2VLEyeWOForConditionalGeneration.from_pretrained(
            model_args.pretrained_model_name_or_path, torch_dtype="auto", config=config,
            device_map=device, 
            attn_implementation='flash_attention_2'
        )
        self.model = PeftModel.from_pretrained(self.model, model_path, is_trainable=False)
        
        # CRITICAL: Enable _beacon_forward_ct for compressed turn optimization
        # This ensures all forward calls during generation use the CT optimized version
        self.model.forward = partial(self.model.forward, infer_ct=True)
        
        # Verify CT mode is enabled
        logger.info("🚀 EyeWO Inference initialized with _beacon_forward_ct (Compressed Turn) optimization")
        logger.info(f"   - Compress Turn: {config.compress_turn}")
        logger.info(f"   - Beacon Window: {config.beacon_window}")  
        logger.info(f"   - Beacon Stride: {config.beacon_stride}")
        
        self.processor = AutoProcessor.from_pretrained(model_args.pretrained_model_name_or_path, use_fast=False)
        self.streaming_eos_token_id = self.processor.tokenizer(' ...').input_ids[-1]
        self.model.prepare_inputs_for_generation = functools.partial(prepare_multiturn_multimodal_inputs_for_generation, self.model)
        message = {
            "role": "user",
            "content": [
                {"type": "text", "text": 'livecc'},
            ]
        }
        texts = self.processor.apply_chat_template([message], tokenize=False)
        self.system_prompt_offset = texts.index('<|im_start|>user')
        self.system_message = {"role": "system", "content": "You are a helpful AI assistant that can understand and respond to stream video content. If you cannot answer the question at the current moment, output '...'. If the question is related to what you can see, output 'wait' and then provide answer."}
        
    @staticmethod
    def get_phrase_before_timestamp(text_stream, timestamp, start_from: int = 0):
        phrase = ''
        # FIX BUG: if the last word is finished, the pointer will move forward, lead to last element be used repeatedly
        pointer = 0
        for i, (ws, we, word) in enumerate(text_stream[start_from:]):
            if timestamp >= we:
                pointer += 1
                phrase += ' ' + word.strip()
            else:
                break
        return phrase.strip(), start_from + pointer
    
    @torch.inference_mode()
    def live_cc_once_for_evaluation(
        self,
        queries: list[str],
        query_timestamps: list[float],
        video: str,
        video_start: float = None,
        video_end: float = None,
        remote_loader: callable = None,
        max_new_tokens: int = 32,
        repetition_penalty: float = None,
        streaming_eos_base_threshold: float = None, 
        streaming_eos_threshold_step: float = None,
        downsample_ratio: int = 2,
        debug: bool = False,
    ): 
        if hasattr(self.model, "memory"):
            self.model.memory.reset()
        # NOTE: load video
        # 1. read video clip
        if abs(video_start - video_end) < 2:
            video_end = video_start + 2
        original_clip, _, video_pts = _read_video_decord_plus({'video': video, 'video_start': video_start, 'video_end': video_end, 'remote_loader': remote_loader}, return_pts=True, strict_fps=True)
        n_frames = original_clip.shape[0]

        # 2. organize to interleave frames (both low-res and high-res)
        low_res_clips = []
        high_res_clips = []
        
        ## 2.1 initial_fps_frames
        initial_clip = original_clip[:self.initial_fps_frames]
        low_res_clips.append(_spatial_downsample_video(initial_clip, nframes=n_frames, downsample_ratio=downsample_ratio))
        high_res_clips.append(_spatial_resize_video(initial_clip, nframes=n_frames))
        
        remaining_clip = original_clip[self.initial_fps_frames:]
        
        ## 2.2 streaming_fps_frames
        if len(remaining_clip) > 0:
            for i in range(0, len(remaining_clip), self.streaming_fps_frames):
                segment = remaining_clip[i:i + self.streaming_fps_frames]
                low_res_clips.append(_spatial_downsample_video(segment, downsample_ratio=downsample_ratio))
                high_res_clips.append(_spatial_resize_video(segment, nframes=n_frames))
        
        # NOTE: input query stream
        text_streams = []
        if isinstance(queries, str):
            queries = [queries]
            query_timestamps = [query_timestamps]
        for query, query_timestamp in zip(queries, query_timestamps):
            if video_start is not None:
                text_streams.append((0, query_timestamp-video_start, query)) 
            else:
                 text_streams.append((0, query_timestamp, query)) 
        
        if debug:
            print(f'text_streams: {text_streams}')
        
        # NOTE: prepare logit processor
        if streaming_eos_base_threshold is not None:
            logits_processor = [ThresholdLogitsProcessor(self.streaming_eos_token_id, streaming_eos_base_threshold, streaming_eos_threshold_step)]
        else:
            logits_processor = None
        
        # 3. make conversation and send to model
        past_ids = None
        responses = []
        history_conversation = []
        timecosts = []
        user_next_start_from = 0
        newest_query = None
        
        for i, (low_res_clip, high_res_clip) in enumerate(zip(low_res_clips, high_res_clips)):
            if i == 0:
                start_timestamp, stop_timestamp = 0, self.initial_time_interval
            else:
                start_timestamp, stop_timestamp = stop_timestamp, stop_timestamp + self.streaming_time_interval
            
            # Step 1: Send low-resolution video first
            message = {
                "role": "user",
                "content": [
                    {"type": "text", "text": f'Time={start_timestamp:.1f}-{stop_timestamp:.1f}s'},
                    {"type": "video", "video": low_res_clip}
                ]
            }
            
            # HACK: multi user query
            query, user_next_start_from = LiveCCDemoInfer.get_phrase_before_timestamp(text_streams, stop_timestamp, user_next_start_from)
            if query != '':
                if debug:
                    print(f'query: {query}', f'stop_timestamp: {stop_timestamp}')
                message['content'].append({"type": "text", "text": query})
                newest_query = query
            
            if i == 0:
                texts = self.processor.apply_chat_template([self.system_message, message], tokenize=False, add_generation_prompt=True, return_tensors='pt')
            else:
                texts = self.processor.apply_chat_template([message], tokenize=False, add_generation_prompt=True, return_tensors='pt')
            if past_ids is not None:
                texts = '<|im_end|>\n' + texts[self.system_prompt_offset:]
                
            real_start_time = time.time()
            
            # Forward generate for low-res
            inputs = self.processor(
                text=texts,
                images=None,
                videos=[low_res_clip],
                return_tensors="pt",
            )
            inputs.to(self.model.device)
            
            # breakpoint()
            # Generate using _beacon_forward_ct for compressed turn optimization
            outputs = self.model.generate(
                **inputs,
                return_dict_in_generate=True, 
                max_new_tokens=max_new_tokens, repetition_penalty=repetition_penalty, 
                pad_token_id=self.model.config.eos_token_id,
                logits_processor=logits_processor,
            )
            past_ids = outputs.sequences[:, :-1]
            answer = self.processor.decode(outputs.sequences[0, inputs.input_ids.size(1):], skip_special_tokens=True)
            
            if debug:
                print(f'Low-res response: time={start_timestamp:.1f}-{stop_timestamp:.1f}s, answer={answer}')
            
            # Step 2: If model returns "Wait", send high-resolution video for detailed response
            if answer.strip() == "Wait":
                if debug:
                    print(f"Model returned 'Wait', sending high-resolution video...")
                
                # Send focus message with high-resolution video
                focus_message = {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f'Please focus'},
                        {"type": "video", "video": high_res_clip},
                    ]
                }
                
                focus_texts = self.processor.apply_chat_template([focus_message], tokenize=False, add_generation_prompt=True, return_tensors='pt')
                focus_texts = '<|im_end|>\n' + focus_texts[self.system_prompt_offset:]
                
                # Forward generate for high-res
                focus_inputs = self.processor(
                    text=focus_texts,
                    images=None,
                    videos=[high_res_clip],
                    return_tensors="pt",
                )
                focus_inputs.to(self.model.device)
                
                # Generate high-resolution response using _beacon_forward_ct
                logits_processor[0].count = 0
                focus_outputs = self.model.generate(
                    **focus_inputs,
                    return_dict_in_generate=True, 
                    max_new_tokens=max_new_tokens, repetition_penalty=repetition_penalty, 
                    pad_token_id=self.model.config.eos_token_id,
                    logits_processor=logits_processor,
                )
                
                past_ids = focus_outputs.sequences[:, :-1]
                detailed_answer = self.processor.decode(focus_outputs.sequences[0, focus_inputs.input_ids.size(1):], skip_special_tokens=True)
                
                # Use the detailed answer as the final response
                final_answer = detailed_answer
                if debug:
                    print(f'High-res response: {detailed_answer}')
            else:
                final_answer = answer
            
            real_end_time = time.time()
            timecosts.append(real_end_time - real_start_time)
            fps = (i + 1) / sum(timecosts)
            
            responses.append([
                video_start + start_timestamp, 
                video_start + stop_timestamp, 
                final_answer
            ])
            
            if debug:
                print(f'Final: time={start_timestamp:.1f}-{stop_timestamp:.1f}s, answer={final_answer}, fps={fps:.1f}, kv_cache_size={self.model.memory.get_memory_size()}, mode=CT')
            
            # HACK : add conversation list
            if query != '':
                history_conversation.append({
                    "role": "user", 
                    "content": query, 
                    'time': stop_timestamp + video_start, 
                    'fps': fps, 
                    'kv_cache_size': self.model.memory.get_memory_size()
                })
            
            if final_answer != ' ...':
                history_conversation.append({
                    "role": "assistant", 
                    "content": final_answer.replace(' ...', ''), 
                    'time': stop_timestamp + video_start, 
                    'fps': fps, 
                    'kv_cache_size': self.model.memory.get_memory_size()
                })
        
        return responses, history_conversation
    

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='/2022233235/videollm-online/livecc/outputs/livecc_sft_24k480x100_llava178k_sample_lr1e-5/checkpoint-853')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    
    model = LiveCCDemoInfer(args.model_path, args.device)

    video_path = '/2022233235/videollm-online/full_scale_2fps_max384/972f660f-27ad-49ae-bf00-8da9d6d8d708.mp4'
    video_start = 0
    video_end = 300
    queries = ['Can you tell me what object type of the white item is?']
    query_timestamps = [0]
    responses, history_conversation = model.live_cc_once_for_evaluation(
        queries, query_timestamps, video_path, video_start, video_end,
        downsample_ratio=2
    )
    print(json.dumps(history_conversation, indent=4))
    # print(responses)
    