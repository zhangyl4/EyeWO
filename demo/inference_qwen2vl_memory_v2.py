# V2 版本更新说明：
# 1. 改用 DynamicCache 架构，替代 SinkCache
# 2. 增强的 update 函数：储存 cos/sin 用于重新计算 rotation
# 3. 智能对话轮次选择：保留第一轮对话和最近N轮对话（包含视觉内容）
# 4. 位置嵌入重计算：基于 Qwen2VL 的 M-RoPE 算法重新计算 key rotation
# 5. 每次 generate 后自动执行 refresh_cache，优化内存使用
# 6. 完整的 inputs 支持：传入 image_grid_thw、video_grid_thw、attention_mask 等信息
# 7. 使用 Qwen2VL 的 get_rope_index 方法正确计算 position embeddings
# 8. 对话轮次级别的缓存管理：保留重要历史和最新上下文
# 9. 全局视觉信息存储：update_global 函数存储所有 video_grid_thw 历史
# 10. 智能视觉信息映射：从全局存储中提取对应轮次的视觉信息用于 position_ids 计算

import functools, torch, typing
# from liger_kernel.transformers import apply_liger_kernel_to_qwen2_vl
# apply_liger_kernel_to_qwen2_vl()
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, LogitsProcessor, logging, DynamicCache
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

class MRopeDynamicCache(DynamicCache):
    def __init__(self, mrope_section: list[int], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mrope_section = mrope_section
        # Store original cos and sin for rerotation
        self.original_cos_cache = []
        self.original_sin_cache = []
        # Store current position information
        self.current_position_ids = None
        # Global storage for all video/image information
        self.global_video_grid_thw = []
        self.global_image_grid_thw = []
        self.global_input_ids_history = []
        self.global_attention_mask_history = []
        
        
    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: typing.Optional[typing.Dict[str, typing.Any]] = None,
    ) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        """
        Enhanced update function that stores cos and sin for future rerotation.
        """
        # Extract cos and sin from cache_kwargs if available
        cos = cache_kwargs.get("cos") if cache_kwargs else None
        sin = cache_kwargs.get("sin") if cache_kwargs else None
        
        # Update the number of seen tokens
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]
            
        # Store original cos and sin for this layer if available
        if cos is not None and sin is not None and layer_idx == 0:
            # Store the original cos and sin for future rerotation
            if len(self.original_cos_cache) == 0:
                self.original_cos_cache = cos.clone()
                self.original_sin_cache = sin.clone()
            else:
                # Concatenate new cos/sin values
                self.original_cos_cache = torch.cat([self.original_cos_cache, cos], dim=-2)
                self.original_sin_cache = torch.cat([self.original_sin_cache, sin], dim=-2)
        
        # Call parent's update method
        return super().update(key_states, value_states, layer_idx, cache_kwargs)
    
    def update_global(self, inputs: dict):
        """
        Update global storage with current inputs information.
        This should be called after each generation to maintain complete history.
        
        Args:
            inputs: Current inputs dict containing input_ids, video_grid_thw, image_grid_thw, etc.
        """
        if not inputs:
            print("Warning: Empty inputs provided to update_global")
            return
            
        # Store the current input_ids
        if 'input_ids' in inputs and inputs['input_ids'] is not None:
            self.global_input_ids_history.append(inputs['input_ids'].clone())
        else:
            self.global_input_ids_history.append(None)
        
        # Store the current video_grid_thw
        if 'video_grid_thw' in inputs and inputs['video_grid_thw'] is not None:
            self.global_video_grid_thw.append(inputs['video_grid_thw'].clone())
            video_info = f"shape={inputs['video_grid_thw'].shape}"
        else:
            self.global_video_grid_thw.append(None)
            video_info = "None"
        
        # Store the current image_grid_thw
        if 'image_grid_thw' in inputs and inputs['image_grid_thw'] is not None:
            self.global_image_grid_thw.append(inputs['image_grid_thw'].clone())
            image_info = f"shape={inputs['image_grid_thw'].shape}"
        else:
            self.global_image_grid_thw.append(None)
            image_info = "None"
        
        # Store the current attention_mask
        if 'attention_mask' in inputs and inputs['attention_mask'] is not None:
            self.global_attention_mask_history.append(inputs['attention_mask'].clone())
        else:
            self.global_attention_mask_history.append(None)
        
        # print(f"Updated global storage: entry #{len(self.global_video_grid_thw)}, "
        #       f"video_grid_thw={video_info}, image_grid_thw={image_info}")
    
    def refresh_cache(self, inputs: dict, model, vision_selection_criteria: dict = None):
        """
        Refresh the cache by applying KV selection and rerotation.
        
        Args:
            inputs: Current inputs dict (mainly for reference)
            model: The Qwen2VL model for computing position embeddings
            vision_selection_criteria: Criteria for selecting which KV pairs to keep based on vision content
        """
        if len(self.key_cache) == 0:
            return inputs
        
        # Step 1: Execute KV selection based on input_ids and vision criteria
        selected_indices, keep_turn_indices = self._select_kv_indices(inputs, vision_selection_criteria)
        
        # Step 2: Apply KV selection
        self._apply_kv_selection(selected_indices)
        
        # Step 3: Recompute rotation for selected keys
        if len(selected_indices) > 0:
            # Create selected inputs dict using global storage
            selected_inputs = self._create_selected_inputs_from_global(selected_indices, keep_turn_indices)
            self._rerotate_selected_keys(selected_inputs, model)
            return selected_inputs
        else:
            return inputs
    
    def _select_kv_indices(self, inputs: torch.Tensor, vision_selection_criteria: dict = None) -> list:
        """
        Select which KV indices to keep based on conversation-level selection.
        
        Strategy: Keep the first conversation turn and the most recent N turns,
        preserving both text and vision content for selected turns.
        
        Args:
            input_ids: Current input token ids
            vision_selection_criteria: Dict with 'keep_recent_turns' parameter
            
        Returns:
            List of indices to keep in the cache
            
        Selection logic:
        1. Parse conversation turns from input_ids
        2. Keep first turn (important initial context)
        3. Keep recent N turns (current interaction context)
        4. Remove middle turns to save memory
        """
        # Step 1: Parse input_ids to identify text and vision segments
        text_indices, keep_turn_indices = self._identify_text_segments(inputs, vision_selection_criteria)
        
        # Step 2: Apply temporal filtering (TODO)
        temporal_filtered_indices, temporal_keep_turn_indices = self._apply_temporal_filtering(inputs, vision_selection_criteria)
        
        # # Step 3: Apply visual similarity filtering (TODO)
        # final_indices = self._apply_visual_similarity_filtering(temporal_filtered_indices, vision_selection_criteria)
        
        # merge all indices
        final_indices = list(set(text_indices + temporal_filtered_indices))
        final_keep_turn_indices = list(set(keep_turn_indices + temporal_keep_turn_indices))
        
        return final_indices, final_keep_turn_indices
    
    def _identify_text_segments(self, inputs: dict, vision_selection_criteria: dict = None) -> list:
        """
        Identify conversation turns and keep only the first turn and recent N turns.
        This preserves both text and vision content for selected turns.
        
        Token IDs mapping:
        - 151644: <|im_start|>
        - 151645: <|im_end|>
        - 151652: <|vision_start|>
        - 151656: <|video_pad|>
        - 151653: <|vision_end|>
        """
        input_ids = inputs['input_ids']
        if input_ids.dim() > 1:
            input_ids = input_ids.squeeze(0)
        
        # Get number of recent turns to keep from criteria
        keep_recent_turns = vision_selection_criteria.get('keep_recent_turns', 10) if vision_selection_criteria else 10
        
        # Find conversation turn boundaries
        conversation_turns = self._find_conversation_turns(input_ids)
        
        if len(conversation_turns) <= keep_recent_turns + 1:
            # If total turns <= first + recent, keep everything
            return list(range(len(input_ids))), list(range(len(conversation_turns)))
        
        # Select indices to keep: first turn + recent N turns
        selected_indices = []
        
        # Keep the first conversation turn (usually system + first user message)
        if len(conversation_turns) > 0:
            first_turn = conversation_turns[0]
            selected_indices.extend(range(first_turn['start'], first_turn['end']))
        
        # Keep the recent N turns
        recent_turns = conversation_turns[-keep_recent_turns:]
        for i, turn in enumerate(recent_turns):
            selected_indices.extend(range(turn['start'], turn['end']))
        
        # Remove duplicates and sort
        selected_indices = sorted(list(set(selected_indices)))
        
        # find keep turn indices
        keep_turn_indices = [0] + list(range(len(conversation_turns) - keep_recent_turns, len(conversation_turns)))
        return selected_indices, keep_turn_indices
    
    def _find_conversation_turns(self, input_ids: torch.Tensor) -> list:
        """
        Find conversation turn boundaries in input_ids.
        Each turn consists of user message + assistant response.
        
        Returns:
            List of dicts with 'start' and 'end' indices for each conversation turn
        
        Input_ids example :
        <|im_start|>system
        You are a helpful assistant.<|im_end|>
        <|im_start|>user
        Time=0.0-3.0s<|vision_start|><|video_pad|><|vision_end|>Can you tell me what object type of the white item is?<|im_end|>
        <|im_start|>assistant 
        ...<|im_end|>
        
        """
        turns = []
        turn_starts = []
        
        # First, find all <|im_start|> positions
        for i, token_id in enumerate(input_ids):
            if token_id.item() == 151644:  # <|im_start|>
                turn_starts.append(i)
        
        if not turn_starts:
            return []
        
        # Group consecutive <|im_start|> tokens into conversation turns
        # Each turn must contain: user message + assistant response
        i = 0
        while i < len(turn_starts):
            turn_start = turn_starts[i]
            
            # Check if we have at least 2 more <|im_start|> tokens (user + assistant)
            if i + 1 < len(turn_starts):
                # Find the end of current user message
                current_end = self._find_im_end(input_ids, turn_start)
                
                if current_end is not None:
                    # Look for the next <|im_start|> (assistant message)
                    next_start = turn_starts[i + 1]
                    
                    # Check if next <|im_start|> is close to current end (likely assistant response)
                    if next_start - current_end <= 10:  # Allow some newlines
                        # Find the end of assistant message
                        assistant_end = self._find_im_end(input_ids, next_start)
                        
                        if assistant_end is not None:
                            # Complete turn: user + assistant
                            if i + 2 < len(turn_starts):
                                turn_end = turn_starts[i + 2]  # End at next user message
                            else:
                                turn_end = len(input_ids)  # End of sequence
                            
                            turns.append({'start': turn_start, 'end': turn_end})
                            i += 2  # Skip both user and assistant messages
                        else:
                            # Incomplete assistant message, skip this turn
                            i += 1
                    else:
                        # Next <|im_start|> is too far, skip this turn
                        i += 1
                else:
                    # Incomplete user message, skip this turn
                    i += 1
            else:
                # Not enough tokens for a complete turn, skip
                i += 1
        
        return turns
    
    def _find_im_end(self, input_ids: torch.Tensor, start_pos: int) -> int:
        """Find the position of <|im_end|> token after start_pos."""
        for i in range(start_pos, len(input_ids)):
            if input_ids[i].item() == 151645:  # <|im_end|>
                return i
        return len(input_ids) - 1
    
    def _apply_temporal_filtering(self, inputs: dict, vision_selection_criteria: dict = None) -> list:
        if inputs.get('video_similarity', None) is None:
            return [], []
        video_similarity = inputs['video_similarity']

        
    
    def _apply_visual_similarity_filtering(self, temporal_indices: list, vision_selection_criteria: dict = None) -> list:
        return [],[]
    
    def _apply_kv_selection(self, selected_indices: list):
        """
        Apply the selection by keeping only the specified indices in KV cache.
        
        Args:
            selected_indices: List of indices to keep
        """
        if not selected_indices:
            return
            
        for layer_idx in range(len(self.key_cache)):
            if len(self.key_cache[layer_idx]) > 0:
                # Select specific indices from the cache
                self.key_cache[layer_idx] = self.key_cache[layer_idx][:, :, selected_indices, :]
                self.value_cache[layer_idx] = self.value_cache[layer_idx][:, :, selected_indices, :]
        
        # Update seen tokens count
        self._seen_tokens = len(selected_indices)
        
        # Update cos/sin cache accordingly
        if len(self.original_cos_cache) > 0:
            self.original_cos_cache = self.original_cos_cache[:, :, selected_indices, :]
            self.original_sin_cache = self.original_sin_cache[:, :, selected_indices, :]
    
    def _create_selected_inputs(self, inputs: dict, selected_indices: list) -> dict:
        """
        Create a new inputs dict with only the selected indices.
        
        Args:
            inputs: Original inputs dict
            selected_indices: List of indices to keep
            
        Returns:
            Selected inputs dict
        """
        selected_inputs = {}
        
        # Select input_ids
        input_ids = inputs['input_ids']
        if input_ids.dim() > 1:
            selected_inputs['input_ids'] = input_ids[:, selected_indices]
        else:
            selected_inputs['input_ids'] = input_ids[selected_indices]
        # selected_inputs['original_input_ids'] = input_ids
        # selected_inputs['selected_indices'] = selected_indices
        
        # Handle attention_mask if present
        # if 'attention_mask' in inputs and inputs['attention_mask'] is not None:
        #     attention_mask = inputs['attention_mask']
        #     if attention_mask.dim() > 1:
        #         selected_inputs['attention_mask'] = attention_mask[:, selected_indices]
        #     else:
        #         selected_inputs['attention_mask'] = attention_mask[selected_indices]
        selected_inputs['attention_mask'] = None
        
        # Copy vision-related fields (these don't need index selection)
        for key in ['image_grid_thw', 'video_grid_thw']: # TODO: add more vision-related fields
            if key in inputs:
                selected_inputs[key] = inputs[key]
        
        return selected_inputs
    
    def _create_selected_inputs_from_global(self, selected_indices: list, keep_turn_indices: list) -> dict:
        """
        Create selected inputs dict using global storage based on selected indices.
        
        Args:
            selected_indices: List of token indices to keep
            keep_turn_indices: List of turn indices to keep
        Returns:
            Selected inputs dict with proper video_grid_thw mapping
        """
        if len(self.global_input_ids_history) == 0:
            return {}
            
        # Get the latest full input_ids
        latest_input_ids = self.global_input_ids_history[-1]
        
        # Create selected input_ids
        if latest_input_ids.dim() > 1:
            selected_input_ids = latest_input_ids[:, selected_indices]
        else:
            selected_input_ids = latest_input_ids[selected_indices]
        
        selected_inputs = {
            'input_ids': selected_input_ids,
            'attention_mask': None
        }
        
        # Map selected tokens to their corresponding video/image information
        # This requires finding which global entries contributed to the selected tokens
        selected_video_grid_thw, selected_image_grid_thw = self._map_tokens_to_vision_info(selected_indices, keep_turn_indices)
        
        selected_inputs['video_grid_thw'] = selected_video_grid_thw
        selected_inputs['image_grid_thw'] = selected_image_grid_thw
        
        return selected_inputs
    
    def _map_tokens_to_vision_info(self, selected_indices: list, keep_turn_indices: list):
        """
        Map selected token indices to their corresponding video/image grid information.
        
        Args:
            selected_indices: List of token indices that were selected
            keep_turn_indices: List of turn indices to keep
        Returns:
            Tuple of (selected_video_grid_thw, selected_image_grid_thw)
        """
        selected_video_grid_thw = []
        selected_image_grid_thw = []
        
        # Find all non-None entries in global storage
        # This ensures we have all the video/image information needed for position calculation
        for turn_idx in keep_turn_indices:
            if self.global_video_grid_thw[turn_idx] is not None:
                selected_video_grid_thw.append(self.global_video_grid_thw[turn_idx])
        
        for turn_idx in keep_turn_indices:
            if self.global_image_grid_thw[turn_idx] is not None:
                selected_image_grid_thw.append(self.global_image_grid_thw[turn_idx])
        
        # Concatenate all video grid information
        if selected_video_grid_thw:
            combined_video_grid_thw = torch.cat(selected_video_grid_thw, dim=0)
        else:
            combined_video_grid_thw = None
            
        # Concatenate all image grid information  
        if selected_image_grid_thw:
            combined_image_grid_thw = torch.cat(selected_image_grid_thw, dim=0)
        else:
            combined_image_grid_thw = None
            
        # print(f"Mapped vision info from {len(self.global_video_grid_thw)} global entries: "
        #       f"video_grid_thw={combined_video_grid_thw.shape if combined_video_grid_thw is not None else None}, "
        #       f"image_grid_thw={combined_image_grid_thw.shape if combined_image_grid_thw is not None else None}")
        
        return combined_video_grid_thw, combined_image_grid_thw
    
    def _rerotate_selected_keys(self, selected_inputs: dict, model):
        """
        Rerotate the selected keys using stored cos and sin values.
        Based on Qwen2VL's rope computation logic.
        
        Args:
            selected_inputs: The inputs dict after selection (input_ids, image_grid_thw, video_grid_thw, etc.)
            model: The Qwen2VL model for computing position embeddings
        """
        if len(self.original_cos_cache) == 0 or len(self.key_cache) == 0:
            return
        
        mrope_section = self.mrope_section * 2
        
        # Step 1: Calculate new position embeddings for the selected sequence
        new_position_ids, shifted_cos, shifted_sin = self._compute_new_position_embeddings(
            selected_inputs, model
        )
        original_cos, original_sin = self.original_cos_cache, self.original_sin_cache
        def mrope_cat(x):
            return torch.cat([m[i % 3] for i, m in enumerate(x.split(mrope_section, dim=-1))], dim=-1).unsqueeze(1)
        original_cos = mrope_cat(original_cos)
        shifted_cos = mrope_cat(shifted_cos)
        original_sin = mrope_cat(original_sin)
        shifted_sin = mrope_cat(shifted_sin)
        
        rerotation_cos = original_cos * shifted_cos + original_sin * shifted_sin
        rerotation_sin = -original_sin * shifted_cos + original_cos * shifted_sin
        # DONE: rerotation logic is implemented below using new position embeddings
        rerotation_cos = rerotation_cos.to(self.key_cache[0].dtype)
        rerotation_sin = rerotation_sin.to(self.key_cache[0].dtype)
        
        # Step 2: Apply mrope rotation to all layers
        for layer_idx in range(len(self.key_cache)):
            if len(self.key_cache[layer_idx]) > 0:
                # Get current key states
                current_keys = self.key_cache[layer_idx]
                
                # Apply rerotation with new position embeddings
                rerotated_keys = self._apply_key_rotary_pos_emb(
                    current_keys, 
                    rerotation_cos,
                    rerotation_sin
                )
                
                # Update the cache with rerotated keys
                self.key_cache[layer_idx] = rerotated_keys
    
    def _compute_new_position_embeddings(self, selected_inputs: dict, model):
        """
        Compute new position embeddings for the selected sequence.
        Based on Qwen2VL's position embedding computation.
        
        Args:
            selected_inputs: The inputs dict after selection (input_ids, image_grid_thw, video_grid_thw, etc.)
            model: The Qwen2VL model
            
        Returns:
            Tuple of (position_ids, cos, sin)
        """
        # Get the rotary embedding from the model
        rotary_emb = model.model.rotary_emb
        
        # Extract components from selected_inputs
        selected_input_ids = selected_inputs['input_ids']
        image_grid_thw = selected_inputs.get('image_grid_thw', None)
        video_grid_thw = selected_inputs.get('video_grid_thw', None)
        attention_mask = selected_inputs.get('attention_mask', None)
        
        # Calculate sequence length
        seq_length = selected_input_ids.shape[-1]
        batch_size = selected_input_ids.shape[0] if selected_input_ids.dim() > 1 else 1
        device = selected_input_ids.device
        
        # Method 1: Try to use Qwen2VL's get_rope_index if available
        # This would handle vision tokens properly
        if hasattr(model, 'get_rope_index'):
            try:
                position_ids, rope_deltas = model.get_rope_index(
                    selected_input_ids, image_grid_thw, video_grid_thw, attention_mask
                )
            except Exception as e:
                breakpoint()
            # Compute cos and sin embeddings using the calculated position_ids
            cos, sin = rotary_emb(self.key_cache[0], position_ids)
            
            return position_ids, cos, sin
    
    
    @staticmethod
    def _rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def _apply_key_rotary_pos_emb(
        self, key_states: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        rotated_key_states = (key_states * cos) + (self._rotate_half(key_states) * sin)
        return rotated_key_states


class LiveCCDemoInferV2:
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
        vision_selection_criteria: dict = None,
    ): 
        # NOTE: load video
        # 1. read video clip
        if abs(video_start - video_end) < 2:
            video_end = video_start + 2
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
        past_key_values = MRopeDynamicCache(mrope_section=self.model.config.rope_scaling["mrope_section"])
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
            query, user_next_start_from = LiveCCDemoInferV2.get_phrase_before_timestamp(text_streams, stop_timestamp, user_next_start_from)
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
            
            # Update cache and input ids
            past_key_values = outputs.past_key_values
            past_ids = outputs.sequences[:, :-1]
            
            # NEW: Update global storage with current generation's information
            if vision_selection_criteria is not None:
                # Create full inputs dict for global storage update
                global_inputs = {
                    'input_ids': past_ids,
                    'image_grid_thw': inputs.get('image_grid_thw', None),
                    'video_grid_thw': inputs.get('video_grid_thw', None),
                    'attention_mask': inputs.get('attention_mask', None)
                }
                past_key_values.update_global(global_inputs)
                
                refreshed_input_ids = past_key_values.refresh_cache(global_inputs, self.model, vision_selection_criteria)
                past_ids = refreshed_input_ids['input_ids']
            
            answer = self.processor.decode(outputs.sequences[0, inputs.input_ids.size(1):], skip_special_tokens=True)
            responses.append([
                video_start + start_timestamp, 
                video_start + stop_timestamp, 
                answer
            ])
            
            # if past_key_values.get_seq_length() % 1500 < 100:
            #     breakpoint()
            
            # print(f'time={start_timestamp:.1f}-{stop_timestamp:.1f}s, answer={answer}, fps={fps:.1f}, kv_cache_size={past_key_values.get_seq_length()}')
            
            # HACK : add conversation list
            if query != '':
                history_conversation.append({"role": "user", "content":  query, 'time':stop_timestamp + video_start, 'fps': fps, 'kv_cache_size': past_key_values.get_seq_length()})
            if answer != ' ...':
                history_conversation.append({"role": "assistant", "content":  answer.replace(' ...', ''), 'time':stop_timestamp + video_start, 'fps': fps, 'kv_cache_size': past_key_values.get_seq_length()})
        return responses, history_conversation
    

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='/2022233235/videollm-online/livecc/outputs/livecc_sft_24k480x100_llava178k_sample_lr1e-5/checkpoint-853')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    
    model = LiveCCDemoInferV2(args.model_path, args.device)

    video_path = '/2022233235/videollm-online/full_scale_2fps_max384/972f660f-27ad-49ae-bf00-8da9d6d8d708.mp4'
    video_start = 0
    video_end = 240
    queries = ['Can you tell me what object type of the white item is?']
    query_timestamps = [0]
    
    # Vision selection criteria for cache refresh
    vision_selection_criteria = {
        'keep_ratio': 0.8,                # Not used in conversation-level selection
        'importance_threshold': 0.5,       # Not used in conversation-level selection  
        'temporal_weight': 0.3,           # Not used in conversation-level selection
        'vision_weight': 0.7,             # Not used in conversation-level selection
        'keep_recent_turns': 5           # Keep first turn + recent 10 turns (including vision content)
        # Example configurations:
        # 'keep_recent_turns': 5    # More aggressive compression
        # 'keep_recent_turns': 20   # Less compression, keep more history
    }
    
    responses, history_conversation = model.live_cc_once_for_evaluation(
        queries, query_timestamps, video_path, video_start, video_end,
        streaming_eos_base_threshold=0.90,
        streaming_eos_threshold_step=0,
        vision_selection_criteria=vision_selection_criteria
    )
    # print(json.dumps(history_conversation, indent=4))
