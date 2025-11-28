import functools, torch, typing
# from liger_kernel.transformers import apply_liger_kernel_to_qwen2_vl
# apply_liger_kernel_to_qwen2_vl()
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, LogitsProcessor, logging, SinkCache
from livecc_utils import prepare_multiturn_multimodal_inputs_for_generation, get_smart_resized_clip, get_smart_resized_video_reader, _read_video_decord_plus, _spatial_resize_video
from qwen_vl_utils import process_vision_info

import time, os, json
import decord
decord.bridge.set_bridge('native')

logger = logging.get_logger(__name__)

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
    
class MRopeSinkCache(SinkCache):
    def __init__(self, mrope_section: list[int], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mrope_section = mrope_section
        self.retain_input_idx = None
        # HACK: save all history kv cache to cpu
        self.  = []
        self.full_value_cache = []

    def __getitem__(self, layer_idx: int) -> typing.List[typing.Tuple[torch.Tensor]]:
        """
        Support for backwards-compatible `past_key_value` indexing, e.g. `past_key_value[0][0].shape[2]` to get the
        sequence length.
        """
        if layer_idx < len(self.key_cache):
            return (self.key_cache[layer_idx], self.value_cache[layer_idx])
        else:
            raise KeyError(f"Cache only has {len(self)} layers, attempted to access layer with index {layer_idx}")

    def _get_rerotation_cos_sin(self, key_states: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        # Upcast to float32 temporarily for better accuracy
        # cos/sin: [3, bsz, seq_len, head_dim]
        if key_states.shape[-2] not in self.cos_sin_rerotation_cache:
            cos = cos.to(torch.float32)
            sin = sin.to(torch.float32)
            mrope_section = self.mrope_section * 2
            # 1. Slice as in SinkCache
            seq_len = key_states.shape[-2]
            num_sink_tokens = self.num_sink_tokens
            original_cos = cos[:, :, num_sink_tokens + seq_len :]
            shifted_cos = cos[:, :, num_sink_tokens : -seq_len]
            original_sin = sin[:, :, num_sink_tokens + seq_len :]
            shifted_sin = sin[:, :, num_sink_tokens : -seq_len]
            # 2. mrope split/cat for each
            def mrope_cat(x):
                return torch.cat([m[i % 3] for i, m in enumerate(x.split(mrope_section, dim=-1))], dim=-1).unsqueeze(1)
            original_cos = mrope_cat(original_cos)
            shifted_cos = mrope_cat(shifted_cos)
            original_sin = mrope_cat(original_sin)
            shifted_sin = mrope_cat(shifted_sin)
            # 3. rerotation formula
            rerotation_cos = original_cos * shifted_cos + original_sin * shifted_sin
            rerotation_sin = -original_sin * shifted_cos + original_cos * shifted_sin

            self.cos_sin_rerotation_cache[key_states.shape[-2]] = (
                rerotation_cos.to(key_states.dtype),
                rerotation_sin.to(key_states.dtype),
            )
        return self.cos_sin_rerotation_cache[key_states.shape[-2]]

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: typing.Optional[typing.Dict[str, typing.Any]] = None,
    ) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.

        Parameters:
            key_states (`torch.Tensor`):
                The new key states to cache.
            value_states (`torch.Tensor`):
                The new value states to cache.
            layer_idx (`int`):
                The index of the layer to cache the states for.
            cache_kwargs (`Dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass. The following arguments can be used in `SinkCache`: `sin`,
                `cos` and `partial_rotation_size`. These arguments are used with models using RoPE, to recompute the
                rotation as the tokens are shifted.

        Return:
            A tuple containing the updated key and value states.
        """
        # Optional kwargs for `SinkCache` -- needed on models using RoPE. `partial_rotation_size` is used on models
        # with partially rotated position embeddings, like Phi or Persimmon.
        sin = cache_kwargs.get("sin") # [3, bsz, seq_len, head_dim]
        cos = cache_kwargs.get("cos") # [3, bsz, seq_len, head_dim]
        partial_rotation_size = cache_kwargs.get("partial_rotation_size")
        using_rope = cos is not None and sin is not None

        # Update the number of seen tokens
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]

        # Update the sin/cos cache, which holds sin/cos values for all possible positions
        if using_rope and layer_idx == 0:
            # BC: some models still pass `sin`/`cos` with 2 dims. In those models, they are the full sin/cos. Remove
            # after all RoPE models have a llama-like cache utilization.
            if cos.dim() == 2:
                self._cos_cache = cos
                self._sin_cache = sin
            else:
                if self._cos_cache is None:
                    self._cos_cache = cos
                    self._sin_cache = sin
                elif self._cos_cache.shape[-2] < self.window_length:
                    self._cos_cache = torch.cat([self._cos_cache, cos], dim=-2)
                    self._sin_cache = torch.cat([self._sin_cache, sin], dim=-2)

        # [bsz, num_heads, seq_len, head_dim]
        if len(self.key_cache) <= layer_idx:
            # Empty cache
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
            # HACK: save all history kv cache to cpu
            self.full_key_cache.append(key_states.detach().cpu())
            self.full_value_cache.append(value_states.detach().cpu())
            # HACK: retain input idx
            self.retain_input_idx = range(self.key_cache[layer_idx].shape[-2])

        elif key_states.shape[-2] + self.get_seq_length(layer_idx) < self.window_length:
            # Growing cache
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)
            # HACK: save all history kv cache to cpu
            self.full_key_cache[layer_idx] = torch.cat([self.full_key_cache[layer_idx], key_states.detach().cpu()], dim=-2)
            self.full_value_cache[layer_idx] = torch.cat([self.full_value_cache[layer_idx], value_states.detach().cpu()], dim=-2)
            # HACK: retain input idx
            self.retain_input_idx = range(self.key_cache[layer_idx].shape[-2])
        else:
            # HACK: retain input idx
            all_length = self.key_cache[layer_idx].shape[-2] + key_states.shape[-2]
            self.retain_input_idx = list(range(self.num_sink_tokens)) + list(range(all_length - self.window_length + self.num_sink_tokens, all_length))

            # Shifting cache
            keys_to_keep = self.key_cache[layer_idx][
                :, :, -self.window_length + self.num_sink_tokens + key_states.shape[-2] :
            ]

            # On RoPE models, we need to recompute the Key rotation as the tokens are shifted
            if using_rope:
                rerotation_cos, rerotation_sin = self._get_rerotation_cos_sin(
                    key_states, self._cos_cache[:, :, : self.window_length], self._sin_cache[:, :, : self.window_length]
                )
                if partial_rotation_size is not None:
                    keys_to_keep, keys_pass = (
                        keys_to_keep[..., :partial_rotation_size],
                        keys_to_keep[..., partial_rotation_size:],
                    )

                keys_to_keep = self._apply_key_rotary_pos_emb(keys_to_keep, rerotation_cos, rerotation_sin)
                if partial_rotation_size is not None:
                    keys_to_keep = torch.cat((keys_to_keep, keys_pass), dim=-1)

            # Concatenate sink tokens, shifted & rotated tokens (if needed), and new tokens
            sink_keys = self.key_cache[layer_idx][:, :, : self.num_sink_tokens]
            self.key_cache[layer_idx] = torch.cat([sink_keys, keys_to_keep, key_states], dim=-2)

            sink_values = self.value_cache[layer_idx][:, :, : self.num_sink_tokens]
            values_to_keep = self.value_cache[layer_idx][
                :, :, -self.window_length + self.num_sink_tokens + value_states.shape[-2] :
            ]
            self.value_cache[layer_idx] = torch.cat([sink_values, values_to_keep, value_states], dim=-2)

            # HACK: save all history kv cache to cpu
            self.full_key_cache[layer_idx] = torch.cat([self.full_key_cache[layer_idx], key_states.detach().cpu()], dim=-2)
            self.full_value_cache[layer_idx] = torch.cat([self.full_value_cache[layer_idx], value_states.detach().cpu()], dim=-2)

        return self.key_cache[layer_idx], self.value_cache[layer_idx]
    

class LiveCCDemoInfer:
    VIDEO_PLAY_END = object()
    VIDEO_PLAY_CONTINUE = object()
    fps = 2
    initial_fps_frames = 6
    streaming_fps_frames = 2
    initial_time_interval = initial_fps_frames / fps
    streaming_time_interval = streaming_fps_frames / fps
    frame_time_interval = 1 / fps

    def __init__(self, model_path: str = None, device: str = 'cuda'):
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path, torch_dtype="auto", 
            device_map=device, 
            attn_implementation='flash_attention_2'
        )
        self.processor = AutoProcessor.from_pretrained(model_path, use_fast=False)
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
        self._cached_video_readers_with_hw = {}


    @torch.inference_mode()
    def video_qa(
        self,
        message: str,
        history: list,
        state: dict,
        do_sample: bool = False,
        repetition_penalty: float = 1.05,
        hf_spaces: bool = False,
        **kwargs,
    ): 
        """
        state: dict, (maybe) with keys:
            video_path: str, video path
            video_timestamp: float, current video timestamp
            last_timestamp: float, last processed video timestamp
            last_video_pts_index: int, last processed video frame index
            video_pts: np.ndarray, video pts
            last_history: list, last processed history
        """
        video_path = state.get('video_path', None)
        conversation = []
        if hf_spaces:
            for past_message in history:
                content = [{"type": "text", "text": past_message['content']}]
                if video_path: # only use once
                    content.insert(0, {"type": "video", "video": video_path})
                    video_path = None
                conversation.append({"role": past_message["role"], "content": content})
        else:
            pass # use past_key_values
        past_ids = state.get('past_ids', None)
        content = [{"type": "text", "text": message}]
        if past_ids is None and video_path: # only use once
            content.insert(0, {"type": "video", "video": video_path})
        conversation.append({"role": "user", "content": content})
        image_inputs, video_inputs = process_vision_info(conversation)
        texts = self.processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True, return_tensors='pt')
        if past_ids is not None:
            texts = '<|im_end|>\n' + texts[self.system_prompt_offset:]
        inputs = self.processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
            return_attention_mask=False
        )
        inputs.to(self.model.device)
        if past_ids is not None:
            inputs['input_ids'] = torch.cat([past_ids, inputs.input_ids], dim=1) 
        outputs = self.model.generate(
            **inputs, past_key_values=state.get('past_key_values', None), 
            return_dict_in_generate=True, do_sample=do_sample, 
            repetition_penalty=repetition_penalty,
            max_new_tokens=512,
            pad_token_id=self.model.config.eos_token_id,
        )
        state['past_key_values'] = outputs.past_key_values if not hf_spaces else None
        state['past_ids'] = outputs.sequences[:, :-1] if not hf_spaces else None
        response = self.processor.decode(outputs.sequences[0, inputs.input_ids.size(1):], skip_special_tokens=True)
        return response, state

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
        repetition_penalty: float = 1.05,
        streaming_eos_base_threshold: float = None, 
        streaming_eos_threshold_step: float = None, 
    ): 
        # NOTE: load video
        # 1. read video clip
        if video_start == video_end:
            video_end += 1
        clip, _, video_pts = _read_video_decord_plus({'video': video, 'video_start': video_start, 'video_end': video_end, 'remote_loader': remote_loader}, return_pts=True, strict_fps=True)
        clip = _spatial_resize_video(clip)

        # 2. organize to interleave frames
        interleave_clips = []
        ## 2.1 initial_fps_frames
        interleave_clips.append(clip[:self.initial_fps_frames])
        clip = clip[self.initial_fps_frames:]
        ## 2.2 streaming_fps_frames
        if len(clip) > 0:
            interleave_clips.extend(list(clip.split(self.streaming_fps_frames)))
        
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
        
        
        # NOTE: prepare logit processor
        if streaming_eos_base_threshold is not None:
            logits_processor = [ThresholdLogitsProcessor(self.streaming_eos_token_id, streaming_eos_base_threshold, streaming_eos_threshold_step)]
        else:
            logits_processor = None
        
        # 3. make conversation and send to model
        past_key_values = MRopeSinkCache(mrope_section=self.model.config.rope_scaling["mrope_section"], window_length=2048, num_sink_tokens=1024)
        responses = []
        history_conversation = []
        timecosts = []
        user_next_start_from = 0
        for i, clip in enumerate(interleave_clips):
            if i == 0:
                start_timestamp, stop_timestamp = 0, self.initial_time_interval
            else:
                start_timestamp, stop_timestamp = stop_timestamp, stop_timestamp + self.streaming_time_interval
            message = {
                "role": "user",
                "content": [
                    {"type": "text", "text": f'Time={start_timestamp:.1f}-{stop_timestamp:.1f}s'},
                    {"type": "video", "video": clip}
                ]
            }
            # HACK: multi user query
            query, user_next_start_from = LiveCCDemoInfer.get_phrase_before_timestamp(text_streams, stop_timestamp, user_next_start_from)
            if query != '':
                message['content'].append({"type": "text", "text": query})
            texts = self.processor.apply_chat_template([message], tokenize=False, add_generation_prompt=True, return_tensors='pt')
            if past_key_values.get_seq_length() > 0:
                texts = '<|im_end|>\n' + texts[self.system_prompt_offset:]
            real_start_time = time.time()
            
            # forward generate
            inputs = self.processor(
                text=texts,
                images=None,
                videos=[clip],
                return_tensors="pt",
            )
            
            inputs.to(self.model.device)
            if past_key_values.get_seq_length() > 0:
                inputs['input_ids'] = torch.cat([past_ids, inputs.input_ids], dim=1)
            outputs = self.model.generate(
                **inputs, past_key_values=past_key_values, 
                return_dict_in_generate=True, 
                max_new_tokens=max_new_tokens, repetition_penalty=repetition_penalty, 
                pad_token_id=self.model.config.eos_token_id,
                logits_processor=logits_processor,
            )
            
            real_end_time = time.time()
            timecosts.append(real_end_time - real_start_time)
            fps = (i + 1) / sum(timecosts)
            
            past_key_values = outputs.past_key_values
            past_ids = outputs.sequences[:, :-1][:, past_key_values.retain_input_idx]
            answer = self.processor.decode(outputs.sequences[0, inputs.input_ids.size(1):], skip_special_tokens=True)
            
            
            responses.append([
                video_start + start_timestamp, 
                video_start + stop_timestamp, 
                answer
            ])
            
            print(f'time={start_timestamp:.1f}-{stop_timestamp:.1f}s, answer={answer}, fps={fps:.1f}, kv_cache_size={past_key_values[0][0].shape[2]}')
            
            # HACK : add conversation list
            if query != '':
                history_conversation.append({"role": "user", "content": "text", "text": query, 'time':stop_timestamp + video_start, 'fps': fps, 'kv_cache_size': past_key_values[0][0].shape[2]})
            if answer != ' ...':
                history_conversation.append({"role": "assistant", "content": "text", "text": answer.replace(' ...', ''), 'time':stop_timestamp + video_start, 'fps': fps, 'kv_cache_size': past_key_values[0][0].shape[2]})
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
    responses, history_conversation = model.live_cc_once_for_evaluation(queries, query_timestamps, video_path, video_start, video_end,
                                                                        streaming_eos_base_threshold = 0.98,
                                                                        streaming_eos_threshold_step = 0)
    print(json.dumps(history_conversation, indent=4))
    # print(responses)
    