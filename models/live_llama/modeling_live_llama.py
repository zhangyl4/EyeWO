import torch
from torch import nn
from transformers import LlamaForCausalLM, Cache
from transformers.activations import GELUActivation
from transformers.utils import logging

from .configuration_live_llama import LiveLlamaConfig, LiveLlamaConfigMamba, LiveLlamaConfigMambaFt
from ..modeling_live import build_live, LiveMixin
from ..vision_live import build_live_vision_high, build_live_vision, build_dinov2_vision_high, build_dinov2_vision
from transformers import LlamaModel, LlamaForCausalLM
from .modeling_llama_haplo import LlamaHaploForCausalLM
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
import math
from models.beacon_live_llama.modeling_utils import compute_smoothing_loss, compute_vision_shift_idx

logger = logging.get_logger(__name__)

def rotary_position_embedding(q):
    seq_len, dim = q.shape

    position = torch.arange(seq_len, dtype=torch.float).unsqueeze(-1).to(q.device)

    div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float) * -(math.log(1000000.0) / dim)).to(q.device)
    
    pos_emb = position * div_term
    pos_emb = torch.stack([torch.sin(pos_emb), torch.cos(pos_emb)], dim=-1).flatten(-2, -1)
    
    cos_emb = pos_emb[..., 1::2].repeat_interleave(2, dim=-1)
    sin_emb = pos_emb[..., ::2].repeat_interleave(2, dim=-1)
    
    q_alternate = torch.stack([-q[..., 1::2], q[..., ::2]], dim=-1).reshape(q.size())
    
    q_rotated = q * cos_emb + q_alternate * sin_emb

    return q_rotated

def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, cos, sin):
    q_embed = (q * cos) + (rotate_half(q) * sin)
    return q_embed

def find_question(input_tensor:torch.tensor):
    bs, num_token = input_tensor.shape
    mask1 = torch.zeros_like(input_tensor).to(input_tensor.device)

    idx_1502 = (input_tensor == 1502).nonzero(as_tuple=False)
    for i in range(bs):
        row = input_tensor[i]
        
        idx_1502 = (row == 1502).nonzero(as_tuple=True)[0]
        
        if len(idx_1502) > 0:
            start_idx = idx_1502.item()
            
            idx_627 = (row[start_idx:] == 58).nonzero(as_tuple=True)
            if len(idx_627[0]) > 0:
                end_idx = start_idx + idx_627[0][0].item()

                # 将 1502 之后到第一个 627 之前的元素设为 1
                mask1[i, start_idx:end_idx+1] = 1
    
    return mask1.bool()
  


class LiveLlamaForCausalLM(LlamaForCausalLM, LiveMixin):
    config_class = LiveLlamaConfig
    _keys_to_ignore_on_load_missing = ['vision_encoder', 'connector']

    def __init__(self, config: LiveLlamaConfig):
        super().__init__(config)
        self.connector = torch.nn.Sequential(
            torch.nn.Linear(config.vision_hidden_size, config.hidden_size, bias=True),
            GELUActivation(config.hidden_size),
            torch.nn.Linear(config.hidden_size, config.hidden_size, bias=True),
        )

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        frames: torch.FloatTensor = None,
        attention_mask: torch.Tensor = None,
        position_ids: torch.LongTensor = None,
        past_key_values: list[torch.FloatTensor] = None,
        inputs_embeds: torch.FloatTensor = None,
        labels: torch.LongTensor = None,
        use_cache: bool = None,
        output_attentions: bool = None,
        output_hidden_states: bool = None,
        return_dict: bool = None,
        cache_position: torch.LongTensor = None,
        **kwargs,
    ):
        if self.config._attn_implementation == 'vista':
            v_mask = (input_ids == self.config.v_placeholder_id).unsqueeze(2)  
            question_mask = find_question(input_ids).unsqueeze(1) if not kwargs.get('append_prediction', False) else find_question(kwargs.get('all_input_ids')[:,kwargs.get('past_range')]).unsqueeze(1)
            v_mask = v_mask * question_mask
            
            kwargs.update(dict(v_mask=v_mask))
        
        if inputs_embeds is None:
            inputs_embeds = self.joint_embed(input_ids, frames)
        outputs = super().forward(
            attention_mask = attention_mask,
            position_ids = position_ids,
            past_key_values = past_key_values,
            inputs_embeds = inputs_embeds,
            # labels
            use_cache = use_cache,
            output_attentions = output_attentions,
            output_hidden_states = output_hidden_states,
            return_dict = return_dict,
            cache_position=cache_position,
            **kwargs
        )
        loss = None
        if labels is not None:
            logits = outputs[0]
            v_mask = input_ids.flatten(0, 1) == self.config.v_placeholder_id
            if self.config.is_smoothing:
                # Calculate v_mask where labels are not -100
                valid_v_mask = v_mask & (labels.flatten(0, 1) != -100)
                vision_shift_idx = compute_vision_shift_idx(v_mask,self.config.frame_num_tokens)
                vision_shift_idx = vision_shift_idx - valid_v_mask.sum()
                
                total_loss, batch_loss, valid_token_num = compute_smoothing_loss(logits.flatten(0, 1), labels.flatten(),
                                              v_mask, self.config.stream_loss_weight, kwargs.get('response_clip', None),
                                              vision_shift_idx=vision_shift_idx)
                loss = total_loss
            else:
                weight = v_mask * self.config.stream_loss_weight + ~v_mask
                loss = nn.functional.cross_entropy(logits.flatten(0, 1), labels.flatten(), reduction='none') * weight
                loss = loss.sum() / (labels >= 0).sum()

        if not return_dict:
            return (loss,) + outputs[1:] if loss is not None else outputs
        outputs.loss = loss
        return outputs

    def generate_after_embed(self, input_ids, frames, **kwargs):
        return super().generate(inputs_embeds=self.joint_embed(input_ids, frames), **kwargs)
    
class LiveLlamaForCausalLMthreshold(LiveLlamaForCausalLM):
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        frames: torch.FloatTensor = None,
        attention_mask: torch.Tensor = None,
        position_ids: torch.LongTensor = None,
        past_key_values: list[torch.FloatTensor] = None,
        inputs_embeds: torch.FloatTensor = None,
        labels: torch.LongTensor = None,
        use_cache: bool = None,
        output_attentions: bool = None,
        output_hidden_states: bool = None,
        return_dict: bool = None,
        cache_position: torch.LongTensor = None,
        **kwargs,
    ):
        if self.config._attn_implementation == 'vista':
            v_mask = (input_ids == self.config.v_placeholder_id).unsqueeze(2)  
            question_mask = find_question(input_ids).unsqueeze(1) if not kwargs.get('append_prediction', False) else find_question(kwargs.get('all_input_ids')[:,kwargs.get('past_range')]).unsqueeze(1)
            v_mask = v_mask * question_mask
            
            kwargs.update(dict(v_mask=v_mask))
        if inputs_embeds is None:
            inputs_embeds = self.joint_embed(input_ids, frames)
        outputs = super().forward(
            attention_mask = attention_mask,
            position_ids = position_ids,
            past_key_values = past_key_values,
            inputs_embeds = inputs_embeds,
            # labels
            use_cache = use_cache,
            output_attentions = output_attentions,
            output_hidden_states = output_hidden_states,
            return_dict = return_dict,
            cache_position=cache_position,
            **kwargs
        )
        loss = None
        if labels is not None:
            logits = outputs[0]
            v_mask = input_ids.flatten(0, 1) == self.config.v_placeholder_id
            weight = v_mask * self.config.stream_loss_weight + ~v_mask
            loss = nn.functional.cross_entropy(logits.flatten(0, 1), labels.flatten(), reduction='none') * weight
            loss = loss.sum() / (labels >= 0).sum()

            if loss <= self.config.loss_threshold:
            
                logits_flatten_softmax = logits.flatten(0, 1).softmax(dim=-1)
                labels_flatten = labels.flatten()
                low_score_frame_interval_mask = (logits_flatten_softmax[:, self.config.frame_token_interval_id] <= self.config.mask_threshold) & v_mask & (labels_flatten == self.config.frame_token_interval_id)
                labels_flatten[low_score_frame_interval_mask] = -100
                loss = nn.functional.cross_entropy(logits.flatten(0, 1), labels_flatten, reduction='none') * weight
                loss = loss.sum() / (labels >= 0).sum()
        if not return_dict:
            return (loss,) + outputs[1:] if loss is not None else outputs
    
        outputs.loss = loss
        return outputs


class LiveLlamaForCausalLMnoFrameInterval(LiveLlamaForCausalLM):
    def __init__(self, config: LiveLlamaConfig):
        super().__init__(config)
        self.connector = torch.nn.Sequential(
            torch.nn.Linear(config.vision_hidden_size, config.hidden_size, bias=True),
            GELUActivation(config.hidden_size),
            torch.nn.Linear(config.hidden_size, config.hidden_size, bias=True),
        )
        # self.informative_head = nn.Linear(config.hidden_size, 2, bias=False)
        self.relevance_head = nn.Linear(config.hidden_size, 2, bias=False)
    
    def trim_past_key_values(self, past_key_values, start, stop, **kwargs):
        if kwargs.get('frame_mask', None) is not None:
            stop = stop - kwargs.get('frame_mask').sum()
            start = start - kwargs.get('frame_mask').sum() if start != 0 else 0
        return [[past_keys[:,:,start:stop], past_values[:,:,start:stop]] for past_keys, past_values in past_key_values]
    
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        frames: torch.FloatTensor = None,
        attention_mask: torch.Tensor = None,
        position_ids: torch.LongTensor = None,
        past_key_values: list[torch.FloatTensor] = None,
        inputs_embeds: torch.FloatTensor = None,
        labels: torch.LongTensor = None,
        use_cache: bool = None,
        output_attentions: bool = None,
        output_hidden_states: bool = None,
        return_dict: bool = None,
        cache_position: torch.LongTensor = None,
        **kwargs,
    ):
        frame_interval_mask = (input_ids == self.config.frame_token_interval_id)
        if inputs_embeds is None:
            inputs_embeds = self.joint_embed(input_ids[~frame_interval_mask].unsqueeze(0), frames)
            
        outputs = super().forward(
            attention_mask = attention_mask,
            position_ids = position_ids,
            past_key_values = past_key_values,
            inputs_embeds = inputs_embeds,
            # labels
            use_cache = use_cache,
            output_attentions = output_attentions,
            output_hidden_states = True,
            return_dict = return_dict,
            cache_position=cache_position,
            **kwargs
        )
        
        bs, n_token = input_ids.shape
        relevance_logits = self.relevance_head(outputs.hidden_states[-1]).float()
        new_relevance_logits = torch.full((bs, n_token, 2), 0, dtype=outputs[0].dtype, device=outputs[0].device)
        new_relevance_logits[~frame_interval_mask] = relevance_logits
        relevance_logits = new_relevance_logits
        
        if not return_dict:
            new_logits = torch.full((bs, n_token, outputs[0].shape[-1]), 0, dtype=outputs[0].dtype, device=outputs[0].device)
            new_logits[~frame_interval_mask] = outputs[0]
            outputs = (new_logits, ) + outputs[1:]
        else:
            new_logits = torch.full((bs, n_token, outputs.logits.shape[-1]), 0, dtype=outputs.logits.dtype, device=outputs.logits.device)
            new_logits[~frame_interval_mask] = outputs.logits
            outputs.logits = new_logits

        loss = None
        if labels is not None:
            logits = outputs[0]
            v_mask = input_ids.flatten(0, 1) == self.config.v_placeholder_id
            if self.config.is_smoothing:
                text_loss = compute_smoothing_loss(logits.flatten(0, 1), labels.flatten(),
                                              v_mask, 0, self.config, kwargs.get('response_clip', None))
                vision_label = labels.flatten()
                vision_label[~v_mask] = -100
                vision_label[vision_label == 11] = 0
                vision_label[vision_label == 933] = 1
                vision_loss = compute_smoothing_loss(relevance_logits.flatten(0, 1), vision_label,
                                              v_mask, self.config.stream_loss_weight, self.config, kwargs.get('response_clip', None),
                                              response_class=1,continue_class=0)
                loss = text_loss + vision_loss
            else:
                weight = v_mask * 0.0 + ~v_mask
                text_loss = nn.functional.cross_entropy(logits.flatten(0, 1), labels.flatten(), reduction='none') * weight
                text_loss = text_loss.sum() / (labels >= 0).sum()
                
                vision_label = labels.flatten()
                vision_label[~v_mask] = -100
                vision_label[vision_label == 11] = 0
                vision_label[vision_label == 933] = 1
                vision_loss = nn.functional.cross_entropy(relevance_logits.flatten(0, 1), vision_label, reduction='none') * self.config.stream_loss_weight
                vision_loss = vision_loss.sum() / (vision_label >= 0).sum()
                
                loss = text_loss + vision_loss
        
        # change logits to include relevance logits
        if not return_dict:
            logits = outputs[0]
        else:
            logits = outputs.logits
        v_mask = input_ids == self.config.v_placeholder_id
        logits[v_mask] = torch.finfo(logits.dtype).min
        v_logit = logits[v_mask]
        v_logit[:,11] = relevance_logits[v_mask][:,0]
        v_logit[:,933] = relevance_logits[v_mask][:,1]
        logits[v_mask] = v_logit
        if not return_dict:
            outputs = (logits,) + outputs[1:]
        else:
            outputs.logits = logits
            
        
        
        if not return_dict:
            return (loss,) + outputs[1:] if loss is not None else outputs
        outputs.loss = loss
        return outputs
        

class Connection(nn.Module):
    def __init__(self, config):
        super().__init__()
        from mamba_ssm import Mamba2
        self.mamba = Mamba2(
            d_model=config.vision_hidden_size,
            d_state=128,
            d_conv=4,
            expand=2,
        )
        
        self.mamba.load_state_dict(torch.load("/root/videollm-online/test/model_right.pth"))
        self.out_proj = torch.nn.Sequential(
            torch.nn.Linear(config.vision_hidden_size, config.hidden_size, bias=True),
            GELUActivation(config.hidden_size),
            torch.nn.Linear(config.hidden_size, config.hidden_size, bias=True),
        )
        

    def forward(self, x, q_start=None, past_frames=None):        
        T, HW, C = x.shape
        x = x.view(-1, x.shape[-1])
        x = x.unsqueeze(0)
        
        last_token_index = torch.arange(0, T * HW, HW, device=x.device)
        
        if past_frames is not None:
            if past_frames.shape[1] == HW and past_frames.shape[2] == C: # q_start is frames
                past_frames = past_frames.view(-1, past_frames.shape[-1])
                past_frames = past_frames.unsqueeze(0)
            x = torch.cat([past_frames, x], dim=1)
            last_token_index = last_token_index + past_frames.shape[1]
            
        if q_start is not None:
            x = torch.cat([q_start, x], dim=1)
            last_token_index = last_token_index + q_start.shape[1]
        x = self.mamba(x)
        x = self.out_proj(x[:,last_token_index])
        return x.squeeze() # [T x 1, C]


class Connection_decouple(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + config.frame_token_pooled[0] * config.frame_token_pooled[1],config.vision_hidden_size))
        torch.nn.init.kaiming_normal_(self.pos_embed, a=0)
        # self.temporal_pos_embedding = nn.Parameter(torch.zeros(1, 600, config.vision_hidden_size))
        # torch.nn.init.kaiming_normal_(self.temporal_pos_embedding, a=0)
        
        # num_attention_heads = config.num_attention_heads
        # config.num_attention_heads = 1
        # self.rotary_emb = LlamaRotaryEmbedding(config=config)
        # config.num_attention_heads = num_attention_heads
        from mamba_ssm import Mamba2
        self.mamba = Mamba2(
            d_model=config.vision_hidden_size,
            d_state=128,
            d_conv=4,
            expand=2,
        )
        
        self.mamba_text = Mamba2(
            d_model=config.vision_hidden_size,
            d_state=128,
            d_conv=4,
            expand=2,
        )
        
        self.mamba.load_state_dict(torch.load("/root/videollm-online/test/model_right.pth"))
        self.mamba_text.load_state_dict(torch.load("/root/videollm-online/test/model_right2.pth"))
        
        self.out_proj = torch.nn.Sequential(
            torch.nn.Linear(config.vision_hidden_size, config.hidden_size, bias=True),
            GELUActivation(config.hidden_size),
            torch.nn.Linear(config.hidden_size, config.hidden_size, bias=True),
        )
        
        self.in_proj = torch.nn.Sequential(
            torch.nn.Linear(config.hidden_size, config.hidden_size, bias=True),
            GELUActivation(config.hidden_size),
            torch.nn.Linear(config.hidden_size, config.vision_hidden_size, bias=True),
        )

    def forward(self, x, q_start=None, past_frames=None):        
        T, HW, C = x.shape
        
        # copy resp token to the end
        x = torch.cat((x, x[:, :1, :]), dim=1)
    
        # apply pe to text token to compress
        x[:,:-1,:] = x[:,:-1,:] + self.pos_embed
        # position_id = torch.arange(T, device=x.device).repeat_interleave(HW).unsqueeze(0)
        x = x.view(-1, x.shape[-1])
        # cos, sin = self.rotary_emb(x, position_id)
        # x = apply_rotary_pos_emb(x, cos, sin)
        x = x.unsqueeze(0)
        
        # mamba for resp token
        token_index = torch.arange(HW, T * (HW + 1), (HW + 1), device=x.device)
        if past_frames is not None:
            if past_frames.shape[1] == HW and past_frames.shape[2] == C: # q_start is frames
                past_frames = past_frames.view(-1, past_frames.shape[-1])
                past_frames = past_frames.unsqueeze(0)
            x = torch.cat([past_frames, x], dim=1)
            token_index = token_index + past_frames.shape[1]
            token_index = torch.cat([torch.arange(0, past_frames.shape[1]).to(x.device), token_index],dim=0)
            
        # x[:,token_index] = x[:,token_index] + torch.nn.functional.interpolate(self.temporal_pos_embedding, size=(T,), mode='linear', align_corners=False).repeat_interleave(HW, dim=1)
        x[:,token_index] = self.mamba(x[:,token_index]) + x[:,token_index]
        
        # mamba for text token
        token_index = torch.cat([torch.arange(i * (HW + 1), i * (HW + 1) + HW) for i in range(T)], dim=0).to(x.device)
        
        if past_frames is not None:
            token_index = token_index + past_frames.shape[1]
            token_index = torch.cat([torch.arange(0, past_frames.shape[1]).to(x.device), token_index],dim=0)
            
        if q_start is not None:
            if q_start.shape[-1] != C:
                q_start = self.in_proj(q_start)
            x = torch.cat([q_start, x], dim=1)
            token_index = token_index + q_start.shape[1]
            token_index = torch.cat([torch.arange(0, q_start.shape[1]).to(x.device), token_index],dim=0)
        x[:,token_index,:] = self.mamba_text(x[:,token_index,:]) + x[:,token_index,:]
        
        # mlp for alignment
        token_index = torch.cat([torch.tensor([i * (HW + 1), i * (HW + 1) + HW]) for i in range(T)], dim=0).to(x.device)
        x = self.out_proj(x[:,token_index])
        return x.squeeze() # [T x 1, C]


def get_casual_mask(attention_mask, inputs_embeds, cache_position, past_key_values):
    
    if cache_position is None:
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        cache_position = torch.arange(
            past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
        )
    
    past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0   
    using_static_cache = False
    
    dtype, device = inputs_embeds.dtype, inputs_embeds.device
    min_dtype = torch.finfo(dtype).min
    sequence_length = inputs_embeds.shape[1]
    if using_static_cache:
        target_length = past_key_values.get_max_length()
    else:
        target_length = (
            attention_mask.shape[-1]
            if isinstance(attention_mask, torch.Tensor)
            else past_seen_tokens + sequence_length + 1
        )
    
    causal_mask = torch.full((sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device)
    if sequence_length != 1:
        causal_mask = torch.triu(causal_mask, diagonal=1)
    causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
    causal_mask = causal_mask[None, None, :, :].expand(inputs_embeds.shape[0], 1, -1, -1)
    
    return causal_mask

class LiveLlamaForCausalLM_mamba(LlamaForCausalLM, LiveMixin):
    config_class = LiveLlamaConfig
    _keys_to_ignore_on_load_missing = ['vision_encoder', 'connector']

    def __init__(self, config: LiveLlamaConfig):
        super().__init__(config)

        self.connector = Connection(config=config)
        self.loss_vison_head = torch.nn.Linear(config.hidden_size, 2, bias=True)
    
    def text_embed(self, input_ids: torch.Tensor):
        return self.get_input_embeddings()(input_ids.clamp(max=self.vocab_size-1))
    
    def visual_embed(self, frames: torch.Tensor, q_start: torch.Tensor = None, past_frames: torch.Tensor = None):
        if hasattr(self, 'vision_encode'):
            with torch.cuda.amp.autocast():
                frames = self.vision_encode(self.vision_encoder, frames)
                if past_frames is not None and past_frames.dim() == 4:
                    past_frames = self.vision_encode(self.vision_encoder, past_frames)
            frames = frames.to(self.dtype)
        frames = self.connector(frames, q_start, past_frames)
        return frames.view(-1, frames.shape[-1])

    def joint_embed(
        self,
        input_ids: torch.Tensor = None,
        frames: torch.Tensor = None,
        q_start: torch.Tensor = None,
        past_frames: torch.Tensor = None,
    ):
        if frames is None:
            return self.get_input_embeddings()(input_ids)
        if input_ids is None:
            return self.visual_embed(frames, q_start, past_frames)
        inputs_embeds = self.get_input_embeddings()(input_ids.clamp(max=self.vocab_size-1))
        v_mask = input_ids == self.config.v_placeholder_id
        if v_mask.any():
            inputs_embeds[v_mask] = self.visual_embed(frames, q_start, past_frames)
        return inputs_embeds
    
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        frames: torch.FloatTensor = None,
        attention_mask: torch.Tensor = None,
        position_ids: torch.LongTensor = None,
        past_key_values: list[torch.FloatTensor] = None,
        inputs_embeds: torch.FloatTensor = None,
        labels: torch.LongTensor = None,
        use_cache: bool = None,
        output_attentions: bool = None,
        output_hidden_states: bool = None,
        return_dict: bool = None,
        cache_position: torch.LongTensor = None,
        **kwargs,
    ):
        # with torch.no_grad():
        #     question_mask = find_question(input_ids).squeeze() if not kwargs.get('append_prediction', False) else find_question(kwargs.get('all_input_ids')[:,kwargs.get('past_range')]).squeeze()
        #     outputs = super().forward(
        #         attention_mask = attention_mask,
        #         position_ids = position_ids,
        #         past_key_values = past_key_values,
        #         inputs_embeds = self.text_embed(input_ids)[:,question_mask] if not kwargs.get('append_prediction', False) \
        #             else self.text_embed(kwargs.get('all_input_ids')[:,kwargs.get('past_range')])[:,question_mask],
        #         # labels
        #         use_cache = use_cache,
        #         output_attentions = output_attentions,
        #         output_hidden_states = True,
        #         return_dict = return_dict,
        #         cache_position=cache_position,
        #         **kwargs
        #     )
        
        if inputs_embeds is None:
            # inputs_embeds = self.joint_embed(input_ids, frames, outputs.hidden_states[0], None) if not kwargs.get('append_prediction', False) \
            #     else self.joint_embed(input_ids, frames,outputs.hidden_states[0], past_frames=kwargs.get('past_frames'))
                
            inputs_embeds = self.joint_embed(input_ids, frames, None, None) if not kwargs.get('append_prediction', False) \
                else self.joint_embed(input_ids, frames, past_frames=kwargs.get('past_frames'))
                
        outputs = super().forward(
            attention_mask = attention_mask,
            position_ids = position_ids,
            past_key_values = past_key_values,
            inputs_embeds = inputs_embeds,
            # labels
            use_cache = use_cache,
            output_attentions = output_attentions,
            output_hidden_states = output_hidden_states,
            return_dict = return_dict,
            cache_position=cache_position,
            **kwargs
        )
        loss = None
        if labels is not None:
            logits = outputs[0]
            v_mask = input_ids.flatten(0, 1) == self.config.v_placeholder_id
            weight = v_mask * self.config.stream_loss_weight + ~v_mask
            loss = nn.functional.cross_entropy(logits.flatten(0, 1), labels.flatten(), reduction='none') * weight
            loss = loss.sum() / (labels >= 0).sum()

            if self.config.vison_loss:
                v_labels = labels.flatten()[v_mask]
                v_labels[v_labels == self.config.frame_token_interval_id] = 0
                v_labels[v_labels == 933] = 1
                vision_loss = self.loss_vison_head(inputs_embeds.flatten(0, 1)[v_mask])
                weight = (v_labels == 0) + (v_labels == 1) * 10
                vision_loss = nn.functional.cross_entropy(vision_loss, v_labels, reduction='none') * weight
                vision_loss = vision_loss.sum() / (v_labels >= 0).sum()
                loss += vision_loss
        
        if not return_dict:
            return (loss,) + outputs[1:] if loss is not None else outputs
    
        outputs.loss = loss
        return outputs

    def generate_after_embed(self, input_ids, frames, **kwargs):
        return super().generate(inputs_embeds=self.joint_embed(input_ids, frames), **kwargs)


class Connector_fusion(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.connector0 = torch.nn.Sequential(
            torch.nn.Linear(config.vision_hidden_size + config.add_vision_hidden_size, config.hidden_size, bias=True),
            GELUActivation(config.hidden_size),
            torch.nn.Linear(config.hidden_size, config.hidden_size, bias=True),
        )
        
        # self.connector1 = torch.nn.Sequential(
        #     torch.nn.Linear(config.vision_hidden_size, config.hidden_size, bias=True),
        #     GELUActivation(config.hidden_size),
        #     torch.nn.Linear(config.hidden_size, config.hidden_size, bias=True),
        # )

    def forward(self, frames, add_frames):
        frames = torch.cat([frames, add_frames], dim=-1)
        return self.connector0(frames)
    
class Connector_dual(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.connector0 = torch.nn.Sequential(
            torch.nn.Linear(config.vision_hidden_size, config.hidden_size, bias=True),
            GELUActivation(config.hidden_size),
            torch.nn.Linear(config.hidden_size, config.hidden_size, bias=True),
        )
        if config.add_vision_pretrained is not None:
            self.connector1 = torch.nn.Sequential(
                torch.nn.Linear(config.add_vision_hidden_size, config.hidden_size, bias=True),
                GELUActivation(config.hidden_size),
                torch.nn.Linear(config.hidden_size, config.hidden_size, bias=True),
            )
    
    def forward(self, frames, type=0):
        if type == 0:
            return self.connector0(frames)
        else:
            if hasattr(self, 'connector1'):
                return self.connector1(frames)
            else:
                raise ValueError("Connector1 is not initialized")

class LiveLlamaForCausalLMhigh(LlamaForCausalLM, LiveMixin):
    
    config_class = LiveLlamaConfig
    _keys_to_ignore_on_load_missing = ['vision_encoder', 'connector']

    def __init__(self, config: LiveLlamaConfig):
        super().__init__(config)
        if config.add_type == 'fusion':
            self.connector = Connector_fusion(config)
        elif config.add_type == 'dual':
            self.connector = Connector_dual(config)
        else:
            self.connector = torch.nn.Sequential(
                torch.nn.Linear(config.vision_hidden_size, config.hidden_size, bias=True),
                GELUActivation(config.hidden_size),
                torch.nn.Linear(config.hidden_size, config.hidden_size, bias=True),
            )

        if self.config.max_frame_clip_mode_model == 'query':
            
            self.text_mlp=nn.Sequential(
                nn.Linear(config.hidden_size,config.hidden_size),
                nn.GELU(),
            )
            

    def set_vision_inside(self, low_vision_encoder=False, high_vision_encoder=True):
        logger.warning_once("!!! Set vision encoder in the model, only recommended for on in-the-wild inference. "
            "Please dont call this for efficient training & evaluation. Instead, do visual feature pre-extraction.")
        
        if low_vision_encoder:
            self.vision_encoder, self.vision_encode = build_live_vision(self.config)
            if self.config.add_vision_pretrained is not None:
                self.add_vision_encoder, self.add_vision_encode = build_dinov2_vision(self.config)
        if high_vision_encoder:
            self.vision_encoder, self.high_vision_encode = build_live_vision_high(self.config)
            if self.config.add_vision_pretrained is not None:
                self.add_vision_encoder, self.add_high_vision_encode = build_dinov2_vision_high(self.config)
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        def get_w(weights, keyword):
            return {k.split(keyword + ".")[1]: v for k, v in weights.items() if keyword in k}
        """Override the default from_pretrained to extend vocab size according to beacon_size."""
        model = super().from_pretrained(*args, **kwargs)
        config = kwargs.get("config", None)
        if config.pretrain_mm_mlp_adapter:
            mm_projector_weights = torch.load(config.pretrain_mm_mlp_adapter, map_location="cpu")
            
            if hasattr(model.connector, 'connector0'):
                connector_weights = get_w(mm_projector_weights, "connector")
                
                # Handle first layer which has double input size
                first_layer_weight = connector_weights['0.weight']  # [hidden_size, vision_hidden_size]
                first_layer_bias = connector_weights['0.bias']
                
                # Pad the weight matrix with zeros for the second half
                padded_weight = torch.zeros(first_layer_weight.shape[0], first_layer_weight.shape[1]+config.add_vision_hidden_size)
                padded_weight[:, :first_layer_weight.shape[1]] = first_layer_weight
                connector_weights['0.weight'] = padded_weight
                incompatible_keys = model.connector.connector0.load_state_dict(connector_weights)
            else:
                incompatible_keys = model.connector.load_state_dict(get_w(mm_projector_weights, "connector"))
            
            pretrain_mm_mlp_adapter = config.pretrain_mm_mlp_adapter
            print(f"Loaded mm projector weights from {pretrain_mm_mlp_adapter}. Incompatible keys: {incompatible_keys}")

        
        
        return model 
    
    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        model_inputs = super().prepare_inputs_for_generation(input_ids, past_key_values, attention_mask, inputs_embeds, **kwargs)
        model_inputs.update(
            {
                "frames": kwargs.get("frames", None),
                "high_frames": kwargs.get("high_frames", None),
                "high_frames_all": kwargs.get("high_frames_all", None),
                "response_clip": kwargs.get("response_clip", None),
            }
        )
        return model_inputs
    
    
    def generate(
        self,
        input_ids: torch.LongTensor = None,
        input_embeds: torch.Tensor = None,
        frames: torch.Tensor = None,
        high_frames: torch.Tensor = None,
        **kwargs
    ):
        # print(input_ids)
        # print(frames.shape)
        # print(high_frames.shape)
        kwargs.pop('high_frames_all', None)
        kwargs.pop('response_clip', None)
        return super().generate(
            input_ids=input_ids,
            input_embeds=input_embeds,
            frames=frames,
            high_frames=high_frames,
            **kwargs
        )
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        frames: torch.FloatTensor = None,
        high_frames: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        position_ids: torch.LongTensor = None,
        past_key_values: list[torch.FloatTensor] = None,
        inputs_embeds: torch.FloatTensor = None,
        labels: torch.LongTensor = None,
        use_cache: bool = None,
        output_attentions: bool = None,
        output_hidden_states: bool = None,
        return_dict: bool = None,
        cache_position: torch.LongTensor = None,
        **kwargs,
    ):  
        if inputs_embeds is None:
            inputs_embeds = self.joint_embed(input_ids, frames, high_frames)
        outputs = super().forward(
            attention_mask = attention_mask,
            position_ids = position_ids,
            past_key_values = past_key_values,
            inputs_embeds = inputs_embeds,
            # labels
            use_cache = use_cache,
            output_attentions = output_attentions,
            output_hidden_states = output_hidden_states,
            return_dict = return_dict,
            cache_position=cache_position,
            # **kwargs
        )
        loss = None
        if labels is not None:
            logits = outputs[0]
            v_mask = input_ids.flatten(0, 1) == self.config.v_placeholder_id
            weight = v_mask * self.config.stream_loss_weight + ~v_mask
            loss = nn.functional.cross_entropy(logits.flatten(0, 1), labels.flatten(), reduction='none') * weight
            if torch.isnan(loss).any():
                # print((labels>0).sum())
                # print(loss)
                # print(logits)
                # print(inputs_embeds)
                print(torch.isnan(inputs_embeds).any())
                # print(input_ids)
                # print(self.query.data)
                exit()
            loss = loss.sum() / (labels >= 0).sum() if (labels >= 0).sum() > 0 else loss.sum()
            if loss.item() == 0:
                breakpoint()
        # if outputs.loss is not None and torch.isnan(outputs.loss) or outputs.loss == 0:
        if not return_dict:
            return (loss,) + outputs[1:] if loss is not None else outputs
        
        outputs.loss = loss
        return outputs

    def generate_after_embed(self, input_ids, frames, **kwargs):
        return self.generate(inputs_embeds=self.joint_embed(input_ids, frames, kwargs.get('high_frames', None)), **kwargs)

    def visual_embed(self, frames: torch.Tensor=None, high_frames: torch.Tensor = None):
        if frames is None and high_frames is None:
            raise ValueError("At least one of frames or high_frames should be provided.")

        if frames is not None:
            if hasattr(self, 'vision_encode'):
                with torch.cuda.amp.autocast():
                    frames = self.vision_encode(self.vision_encoder, frames)
                frames = frames.to(self.dtype)
            frames = self.connector(frames)
            
        
        if high_frames is not None:
            if hasattr(self, 'high_vision_encode'):
                with torch.cuda.amp.autocast():
                    high_frames = self.high_vision_encode(self.vision_encoder, high_frames)
                high_frames = high_frames.to(self.dtype)
            high_frames = self.connector(high_frames)
            
        if frames is not None and high_frames is None:
            return frames.view(-1, frames.shape[-1])
        elif frames is None and high_frames is not None:
            return high_frames.view(-1, high_frames.shape[-1])
        else:
            return frames.view(-1, frames.shape[-1]), high_frames.view(-1, high_frames.shape[-1])
    
    
    def visual_embed_fusion(self, frames: torch.Tensor=None, high_frames: torch.Tensor = None):
        if frames is None and high_frames is None:
            raise ValueError("At least one of frames or high_frames should be provided.")
        
        if frames is not None:
            if hasattr(self, 'vision_encode'):
                with torch.cuda.amp.autocast():
                    frames_feat = self.vision_encode(self.vision_encoder, frames)
                frames_feat = frames_feat.to(self.dtype)
            else:
                frames_feat = frames
            if hasattr(self, 'add_vision_encode'):
                with torch.cuda.amp.autocast():
                    add_frames_feat = self.add_vision_encode(self.add_vision_encoder, frames)
                add_frames_feat = add_frames_feat.to(self.dtype)
            else:
                add_frames_feat = frames
            frames_feat = self.connector(frames_feat, add_frames_feat)
            
        
        if high_frames is not None:
            if hasattr(self, 'high_vision_encode'):
                with torch.cuda.amp.autocast():
                    high_frames_feat = self.high_vision_encode(self.vision_encoder, high_frames)
                high_frames_feat = high_frames_feat.to(self.dtype)
            else:
                high_frames_feat = high_frames
            if hasattr(self, 'add_high_vision_encode'):
                with torch.cuda.amp.autocast():
                    add_high_frames_feat = self.add_high_vision_encode(self.add_vision_encoder, high_frames)
                add_high_frames_feat = add_high_frames_feat.to(self.dtype)
            else:
                add_high_frames_feat = high_frames
            high_frames_feat = self.connector(high_frames_feat, add_high_frames_feat)
            
        if frames is not None and high_frames is None:
            return frames_feat.view(-1, frames_feat.shape[-1])
        elif frames is None and high_frames is not None:
            return high_frames_feat.view(-1, high_frames_feat.shape[-1])
        else:
            return frames_feat.view(-1, frames_feat.shape[-1]), high_frames_feat.view(-1, high_frames_feat.shape[-1])
            
    
    def visual_embed_dual(self, frames: torch.Tensor=None, high_frames: torch.Tensor = None):
        if frames is None and high_frames is None:
            raise ValueError("At least one of frames or high_frames should be provided.")

        if frames is not None:
            if hasattr(self, 'vision_encode'):
                with torch.cuda.amp.autocast():
                    frames_feat = self.vision_encode(self.vision_encoder, frames)
                frames_feat = frames_feat.to(self.dtype)
            else:
                frames_feat = frames
            frames_feat = self.connector(frames_feat, type=0)
            if hasattr(self, 'add_vision_encode'):
                with torch.cuda.amp.autocast():
                    add_frames_feat = self.add_vision_encode(self.add_vision_encoder, frames)
                add_frames_feat = add_frames_feat.to(self.dtype)
            else:
                add_frames_feat = frames
            add_frames_feat = self.connector(add_frames_feat, type=1)
            # interleave concat frames and add_frames
            frames_feat = torch.cat([frames_feat, add_frames_feat], dim=0)
        
        if high_frames is not None:
            if hasattr(self, 'high_vision_encode'):
                with torch.cuda.amp.autocast():
                    high_frames_feat = self.high_vision_encode(self.vision_encoder, high_frames)
                high_frames_feat = high_frames_feat.to(self.dtype)
            else:
                high_frames_feat = high_frames
            high_frames_feat = self.connector(high_frames_feat, type=0)
            if hasattr(self, 'add_high_vision_encode'):
                with torch.cuda.amp.autocast():
                    add_high_frames_feat = self.add_high_vision_encode(self.add_vision_encoder, high_frames)
                add_high_frames_feat = add_high_frames_feat.to(self.dtype)
            else:
                add_high_frames_feat = high_frames
            add_high_frames_feat = self.connector(add_high_frames_feat, type=1)
            # interleave concat frames and add_frames
            high_frames_feat = torch.cat([high_frames_feat, add_high_frames_feat], dim=0)
            
        if frames is not None and high_frames is None:
            return frames_feat.view(-1, frames_feat.shape[-1])
        elif frames is None and high_frames is not None:
            return high_frames_feat.view(-1, high_frames_feat.shape[-1])
        else:
            return frames_feat.view(-1, frames_feat.shape[-1]), high_frames_feat.view(-1, high_frames_feat.shape[-1])
        
    def joint_embed(
        self,
        input_ids: torch.Tensor = None,
        frames: torch.Tensor = None,
        high_frames: torch.Tensor = None,
    ):
        if frames is None and high_frames is None:
            return self.get_input_embeddings()(input_ids)
        if input_ids is None:
            if self.config.add_type == 'fusion':
                return self.visual_embed_fusion(frames=frames, high_frames=high_frames)
            elif self.config.add_type == 'dual':
                return self.visual_embed_dual(frames=frames, high_frames=high_frames)
            else:
                return torch.cat(self.visual_embed(frames=frames), self.visual_embed(high_frames=high_frames), dim=1)
            
        inputs_embeds = self.get_input_embeddings()(input_ids.clamp(max=self.vocab_size-1))
        v_mask = input_ids == self.config.v_placeholder_id
        hv_mask = input_ids == self.config.high_v_placeholder_id

        if v_mask.any():
            if self.config.add_type == 'fusion':
                low_embedded = self.visual_embed_fusion(frames=frames)
            elif self.config.add_type == 'dual':
                low_embedded = self.visual_embed_dual(frames=frames)
            else:
                low_embedded = self.visual_embed(frames=frames)
                
            low_embedded = self.sample_frames(v_mask, low_embedded, input_ids)
            inputs_embeds[v_mask] = low_embedded
        if hv_mask.any():
            if self.config.add_type == 'fusion':
                high_embedded = self.visual_embed_fusion(high_frames=high_frames)
            elif self.config.add_type == 'dual':
                high_embedded = self.visual_embed_dual(high_frames=high_frames)
            else:
                high_embedded = self.visual_embed(high_frames=high_frames)
            inputs_embeds[hv_mask] = high_embedded
            
        return inputs_embeds
    
    def sample_frames(self, v_mask, low_embedded, input_ids):
        """
        low_embedded: [num_frames, hidden_size]
        v_mask: [batch_size, num_frames]
        """
        
        num_frames = v_mask.sum() // self.config.frame_num_tokens
        
        if self.config.max_frame_clip_mode_model == 'uniform' and low_embedded.shape[0] > v_mask.sum():
            # Calculate indices for uniform sampling
            current_num_frames = low_embedded.shape[0] // self.config.frame_num_tokens
            # Uniformly sample num_frames from current_num_frames
            frame_indices = torch.linspace(0, current_num_frames-1, num_frames, dtype=torch.long)
            indices = []
            for i in frame_indices:
                start = i * self.config.frame_num_tokens
                end = start + self.config.frame_num_tokens
                indices.extend(range(start, end))
            low_embedded = low_embedded[indices]
        elif self.config.max_frame_clip_mode_model == 'query':
            # fork from video-xl pro
            # 1.1 get question input_ids and input_embeds
            question_mask = find_question(input_ids).squeeze()
            question_input_ids = input_ids[0,question_mask]
            question_input_embeds = self.get_input_embeddings()(question_input_ids)
            output_embeds = self.text_mlp(question_input_embeds)
            
            # 2.1 get select_mat
            cls_token = low_embedded[range(0,low_embedded.shape[0],self.config.frame_num_tokens),:]
            select_mat=torch.matmul(cls_token, output_embeds.transpose(0, 1)).mean(dim=-1)
            min_val = torch.min(select_mat)
            max_val = torch.max(select_mat)
            if min_val == max_val:
                select_mat = (select_mat - min_val)
            else:
                select_mat = (select_mat - min_val) / (max_val - min_val)
            low_embedded[range(0,low_embedded.shape[0],self.config.frame_num_tokens),:] = select_mat.unsqueeze(1) + cls_token
            
            #  topk sample
            if low_embedded.shape[0] > v_mask.sum():
                _, top_indices = torch.topk(select_mat,num_frames)
                top_indices=torch.tensor(sorted(top_indices))
                token_indices = []
                for frame_idx in top_indices:
                    start = frame_idx * self.config.frame_num_tokens
                    end = start + self.config.frame_num_tokens
                    token_indices.extend(range(start, end))
                
                low_embedded = low_embedded[token_indices,:]

        elif low_embedded.shape[0] > v_mask.sum():
            raise ValueError(f"Invalid max_frame_clip_mode_model: {self.config.max_frame_clip_mode_model}")
        
        return low_embedded

class LiveLlamaHaploForCausalLMhigh(LlamaHaploForCausalLM, LiveMixin):
    
    config_class = LiveLlamaConfig
    _keys_to_ignore_on_load_missing = ['vision_encoder', 'connector']

    def __init__(self, config: LiveLlamaConfig):
        super().__init__(config)
        if config.add_type == 'fusion':
            self.connector = Connector_fusion(config)
        elif config.add_type == 'dual':
            self.connector = Connector_dual(config)
        else:
            self.connector = torch.nn.Sequential(
                torch.nn.Linear(config.vision_hidden_size, config.hidden_size, bias=True),
                GELUActivation(config.hidden_size),
                torch.nn.Linear(config.hidden_size, config.hidden_size, bias=True),
            )
            
        if self.config.max_frame_clip_mode_model == 'query':
            self.query = nn.Parameter(torch.randn(1, config.hidden_size), requires_grad=True)
            self.query.data.normal_(mean=0.0, std=0.02)
    
    def set_vision_inside(self, low_vision_encoder=False, high_vision_encoder=True):
        logger.warning_once("!!! Set vision encoder in the model, only recommended for on in-the-wild inference. "
            "Please dont call this for efficient training & evaluation. Instead, do visual feature pre-extraction.")
        
        if low_vision_encoder:
            self.vision_encoder, self.vision_encode = build_live_vision(self.config)
            if self.config.add_vision_pretrained is not None:
                self.add_vision_encoder, self.add_vision_encode = build_dinov2_vision(self.config)
        if high_vision_encoder:
            self.vision_encoder, self.high_vision_encode = build_live_vision_high(self.config)
            if self.config.add_vision_pretrained is not None:
                self.add_vision_encoder, self.add_high_vision_encode = build_dinov2_vision_high(self.config)
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        """Override the default from_pretrained to extend vocab size according to beacon_size."""
        model = super().from_pretrained(*args, **kwargs)
        config = kwargs.get("config", None)
        if config.pretrain_mm_mlp_adapter:
            mm_projector_weights = torch.load(config.pretrain_mm_mlp_adapter, map_location="cpu")
            def get_w(weights, keyword):
                return {k.split(keyword + ".")[1]: v for k, v in weights.items() if keyword in k}
            incompatible_keys = model.connector.load_state_dict(get_w(mm_projector_weights, "connector"))
            pretrain_mm_mlp_adapter = config.pretrain_mm_mlp_adapter
            print(f"Loaded mm projector weights from {pretrain_mm_mlp_adapter}. Incompatible keys: {incompatible_keys}")
            
        return model 
    
    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        model_inputs = super().prepare_inputs_for_generation(input_ids, past_key_values, attention_mask, inputs_embeds, **kwargs)
        model_inputs.update(
            {
                "frames": kwargs.get("frames", None),
                "high_frames": kwargs.get("high_frames", None),
                "high_frames_all": kwargs.get("high_frames_all", None),
                "response_clip": kwargs.get("response_clip", None),
            }
        )
        return model_inputs
    
    
    def generate(
        self,
        input_ids: torch.LongTensor = None,
        input_embeds: torch.Tensor = None,
        frames: torch.Tensor = None,
        high_frames: torch.Tensor = None,
        **kwargs
    ):
        # print(input_ids)
        # print(frames.shape)
        # print(high_frames.shape)
        kwargs.pop('high_frames_all', None)
        kwargs.pop('response_clip', None)
        return super().generate(
            input_ids=input_ids,
            input_embeds=input_embeds,
            frames=frames,
            high_frames=high_frames,
            **kwargs
        )
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        frames: torch.FloatTensor = None,
        high_frames: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        position_ids: torch.LongTensor = None,
        past_key_values: list[torch.FloatTensor] = None,
        inputs_embeds: torch.FloatTensor = None,
        labels: torch.LongTensor = None,
        use_cache: bool = None,
        output_attentions: bool = None,
        output_hidden_states: bool = None,
        return_dict: bool = None,
        cache_position: torch.LongTensor = None,
        **kwargs,
    ):
        assert self.config._attn_implementation == 'eager'
        # HACK : get attention_start_ids
        video_token_id = self.config.v_placeholder_id
        high_video_token_id = self.config.high_v_placeholder_id
        num_frames = (input_ids == video_token_id).sum() // self.config.frame_num_tokens
        num_video_tokens = [
                [self.config.frame_num_tokens] * num_frames
                
            ] * input_ids.shape[0]

        num_video_tokens_high = [
                [self.config.frame_num_tokens_high] * high_frames.shape[0]
                
            ] * input_ids.shape[0]
        
        attention_start_ids = []
        for input_id in input_ids:
            start_ids = torch.arange(
                len(input_id), device=input_ids.device)

            video_mask = input_id == video_token_id
            while video_mask.any():
                for num_frame_tokens in num_video_tokens[0]:
                    
                    video_inds = video_mask.cumsum(0)
                    
                    per_vid_mask = ((video_inds <= num_frame_tokens) &
                                    video_mask)
                    
                    try:
                        assert per_vid_mask.sum() == num_frame_tokens
                    except:
                        print(video_mask.sum(), (input_id == video_token_id).sum())
                        print(per_vid_mask.sum(), num_frame_tokens)
                        raise Exception('error')
                    start_ids[per_vid_mask] = start_ids[per_vid_mask].min()
                    video_mask[per_vid_mask] = False
                num_video_tokens.pop(0)
                
            for num_frame_tokens in num_video_tokens_high[0]:
                video_mask = input_id == high_video_token_id
                while video_mask.any():
                    for num_frame_tokens in num_video_tokens_high[0]:
                        video_inds = video_mask.cumsum(0)
                        per_vid_mask = ((video_inds <= num_frame_tokens) &
                                        video_mask)
                        
                        try:
                            assert per_vid_mask.sum() == num_frame_tokens
                        except:
                            raise Exception('error')
                        start_ids[per_vid_mask] = start_ids[per_vid_mask].min()
                        video_mask[per_vid_mask] = False
                    num_video_tokens_high.pop(0)
            attention_start_ids.append(start_ids)
        
        attention_start_ids = torch.stack(attention_start_ids)
        
        if inputs_embeds is None:
            inputs_embeds = self.joint_embed(input_ids, frames, high_frames)
        outputs = super().forward(
            attention_mask = attention_mask,
            position_ids = position_ids,
            past_key_values = past_key_values,
            inputs_embeds = inputs_embeds,
            # labels
            use_cache = use_cache,
            output_attentions = output_attentions,
            output_hidden_states = output_hidden_states,
            return_dict = return_dict,
            cache_position=cache_position,
            attention_start_ids=attention_start_ids
            # **kwargs
        )
        loss = None
        if labels is not None:
            logits = outputs[0]
            v_mask = input_ids.flatten(0, 1) == self.config.v_placeholder_id
            weight = v_mask * self.config.stream_loss_weight + ~v_mask
            loss = nn.functional.cross_entropy(logits.flatten(0, 1), labels.flatten(), reduction='none') * weight
            loss = loss.sum() / (labels >= 0).sum() if (labels >= 0).sum() > 0 else loss.sum()
            if loss.item() == 0:
                breakpoint()
        if not return_dict:
            return (loss,) + outputs[1:] if loss is not None else outputs
        
        outputs.loss = loss
        return outputs

    def generate_after_embed(self, input_ids, frames, **kwargs):
        return self.generate(inputs_embeds=self.joint_embed(input_ids, frames, kwargs.get('high_frames', None)), **kwargs)

    def visual_embed(self, frames: torch.Tensor=None, high_frames: torch.Tensor = None):
        if frames is None and high_frames is None:
            raise ValueError("At least one of frames or high_frames should be provided.")

        if frames is not None:
            if hasattr(self, 'vision_encode'):
                with torch.cuda.amp.autocast():
                    frames = self.vision_encode(self.vision_encoder, frames)
                frames = frames.to(self.dtype)
            frames = self.connector(frames)
            
        
        if high_frames is not None:
            if hasattr(self, 'high_vision_encode'):
                with torch.cuda.amp.autocast():
                    high_frames = self.high_vision_encode(self.vision_encoder, high_frames)
                high_frames = high_frames.to(self.dtype)
            high_frames = self.connector(high_frames)
            
        if frames is not None and high_frames is None:
            return frames.view(-1, frames.shape[-1])
        elif frames is None and high_frames is not None:
            return high_frames.view(-1, high_frames.shape[-1])
        else:
            return frames.view(-1, frames.shape[-1]), high_frames.view(-1, high_frames.shape[-1])
    
    
    def visual_embed_fusion(self, frames: torch.Tensor=None, high_frames: torch.Tensor = None):
        if frames is None and high_frames is None:
            raise ValueError("At least one of frames or high_frames should be provided.")
        
        if frames is not None:
            if hasattr(self, 'vision_encode'):
                with torch.cuda.amp.autocast():
                    frames_feat = self.vision_encode(self.vision_encoder, frames)
                frames_feat = frames_feat.to(self.dtype)
            else:
                frames_feat = frames
            if hasattr(self, 'add_vision_encode'):
                with torch.cuda.amp.autocast():
                    add_frames_feat = self.add_vision_encode(self.add_vision_encoder, frames)
                add_frames_feat = add_frames_feat.to(self.dtype)
            else:
                add_frames_feat = frames
            frames_feat = self.connector(frames_feat, add_frames_feat)
            
        
        if high_frames is not None:
            if hasattr(self, 'high_vision_encode'):
                with torch.cuda.amp.autocast():
                    high_frames_feat = self.high_vision_encode(self.vision_encoder, high_frames)
                high_frames_feat = high_frames_feat.to(self.dtype)
            else:
                high_frames_feat = high_frames
            if hasattr(self, 'add_high_vision_encode'):
                with torch.cuda.amp.autocast():
                    add_high_frames_feat = self.add_high_vision_encode(self.add_vision_encoder, high_frames)
                add_high_frames_feat = add_high_frames_feat.to(self.dtype)
            else:
                add_high_frames_feat = high_frames
            high_frames_feat = self.connector(high_frames_feat, add_high_frames_feat)
            
        if frames is not None and high_frames is None:
            return frames_feat.view(-1, frames_feat.shape[-1])
        elif frames is None and high_frames is not None:
            return high_frames_feat.view(-1, high_frames_feat.shape[-1])
        else:
            return frames_feat.view(-1, frames_feat.shape[-1]), high_frames_feat.view(-1, high_frames_feat.shape[-1])
            
    
    def visual_embed_dual(self, frames: torch.Tensor=None, high_frames: torch.Tensor = None):
        if frames is None and high_frames is None:
            raise ValueError("At least one of frames or high_frames should be provided.")

        if frames is not None:
            if hasattr(self, 'vision_encode'):
                with torch.cuda.amp.autocast():
                    frames_feat = self.vision_encode(self.vision_encoder, frames)
                frames_feat = frames_feat.to(self.dtype)
            else:
                frames_feat = frames
            frames_feat = self.connector(frames_feat, type=0)
            if hasattr(self, 'add_vision_encode'):
                with torch.cuda.amp.autocast():
                    add_frames_feat = self.add_vision_encode(self.add_vision_encoder, frames)
                add_frames_feat = add_frames_feat.to(self.dtype)
            else:
                add_frames_feat = frames
            add_frames_feat = self.connector(add_frames_feat, type=1)
            # interleave concat frames and add_frames
            frames_feat = torch.cat([frames_feat, add_frames_feat], dim=0)
        
        if high_frames is not None:
            if hasattr(self, 'high_vision_encode'):
                with torch.cuda.amp.autocast():
                    high_frames_feat = self.high_vision_encode(self.vision_encoder, high_frames)
                high_frames_feat = high_frames_feat.to(self.dtype)
            else:
                high_frames_feat = high_frames
            high_frames_feat = self.connector(high_frames_feat, type=0)
            if hasattr(self, 'add_high_vision_encode'):
                with torch.cuda.amp.autocast():
                    add_high_frames_feat = self.add_high_vision_encode(self.add_vision_encoder, high_frames)
                add_high_frames_feat = add_high_frames_feat.to(self.dtype)
            else:
                add_high_frames_feat = high_frames
            add_high_frames_feat = self.connector(add_high_frames_feat, type=1)
            # interleave concat frames and add_frames
            high_frames_feat = torch.cat([high_frames_feat, add_high_frames_feat], dim=0)
            
        if frames is not None and high_frames is None:
            return frames_feat.view(-1, frames_feat.shape[-1])
        elif frames is None and high_frames is not None:
            return high_frames_feat.view(-1, high_frames_feat.shape[-1])
        else:
            return frames_feat.view(-1, frames_feat.shape[-1]), high_frames_feat.view(-1, high_frames_feat.shape[-1])
        
    def joint_embed(
        self,
        input_ids: torch.Tensor = None,
        frames: torch.Tensor = None,
        high_frames: torch.Tensor = None,
    ):
        if frames is None and high_frames is None:
            return self.get_input_embeddings()(input_ids)
        if input_ids is None:
            if self.config.add_type == 'fusion':
                return self.visual_embed_fusion(frames=frames, high_frames=high_frames)
            elif self.config.add_type == 'dual':
                return self.visual_embed_dual(frames=frames, high_frames=high_frames)
            else:
                return torch.cat(self.visual_embed(frames=frames), self.visual_embed(high_frames=high_frames), dim=1)
            
        inputs_embeds = self.get_input_embeddings()(input_ids.clamp(max=self.vocab_size-1))
        v_mask = input_ids == self.config.v_placeholder_id
        hv_mask = input_ids == self.config.high_v_placeholder_id

        if v_mask.any():
            if self.config.add_type == 'fusion':
                low_embedded = self.visual_embed_fusion(frames=frames)
            elif self.config.add_type == 'dual':
                low_embedded = self.visual_embed_dual(frames=frames)
            else:
                low_embedded = self.visual_embed(frames=frames)
                
            low_embedded = self.sample_frames(v_mask, low_embedded)
            inputs_embeds[v_mask] = low_embedded
        if hv_mask.any():
            if self.config.add_type == 'fusion':
                high_embedded = self.visual_embed_fusion(high_frames=high_frames)
            elif self.config.add_type == 'dual':
                high_embedded = self.visual_embed_dual(high_frames=high_frames)
            else:
                high_embedded = self.visual_embed(high_frames=high_frames)
            inputs_embeds[hv_mask] = high_embedded
            
        return inputs_embeds
    
    def sample_frames(self, v_mask, low_embedded):
        """
        low_embedded: [num_frames, hidden_size]
        v_mask: [batch_size, num_frames]
        """
        
        num_frames = v_mask.sum() // self.config.frame_num_tokens
        
        if self.config.max_frame_clip_mode_model == 'uniform' and low_embedded.shape[0] > v_mask.sum():
            # Calculate indices for uniform sampling
            current_num_frames = low_embedded.shape[0] // self.config.frame_num_tokens
            # Uniformly sample num_frames from current_num_frames
            frame_indices = torch.linspace(0, current_num_frames-1, num_frames, dtype=torch.long)
            indices = []
            for i in frame_indices:
                start = i * self.config.frame_num_tokens
                end = start + self.config.frame_num_tokens
                indices.extend(range(start, end))
            low_embedded = low_embedded[indices]
        elif self.config.max_frame_clip_mode_model == 'query':
            # fork from video-xl pro
            cls_token = low_embedded[range(0,low_embedded.shape[0],self.config.frame_num_tokens),:]
            select_mat=torch.matmul(cls_token,self.query.transpose(0, 1)).mean(dim=-1)
            min_val = torch.min(select_mat)
            max_val = torch.max(select_mat)
            if min_val == max_val:
                select_mat = (select_mat - min_val)
            else:
                select_mat = (select_mat - min_val) / (max_val - min_val)
            low_embedded[range(0,low_embedded.shape[0],self.config.frame_num_tokens),:] = select_mat.unsqueeze(1) + low_embedded[range(0,low_embedded.shape[0],self.config.frame_num_tokens),:]
            #  topk sample
            if low_embedded.shape[0] > v_mask.sum():
                _, top_indices = torch.topk(select_mat,num_frames)
                top_indices=torch.tensor(sorted(top_indices))
                token_indices = []
                for frame_idx in top_indices:
                    start = frame_idx * self.config.frame_num_tokens
                    end = start + self.config.frame_num_tokens
                    token_indices.extend(range(start, end))
                
                low_embedded = low_embedded[token_indices,:]

        elif low_embedded.shape[0] > v_mask.sum():
            raise ValueError(f"Invalid max_frame_clip_mode_model: {self.config.max_frame_clip_mode_model}")
        
        return low_embedded

class LiveLlamaForCausalLMhighRe(LiveLlamaForCausalLMhigh):
    
    @torch.no_grad()
    def stream_evaluate(
        self,
        input_ids: torch.LongTensor,
        labels: torch.LongTensor,
        frames: torch.Tensor,
        ignore_token_id: int = -100,
        frame_token_interval_threshold: float = 0.0,
        **kwargs
    ):
        # 0. evaluation only supports batch_size = 1
        assert input_ids.size(0) == labels.size(0) == 1
        input_id, label = input_ids[0], labels[0]
        device = input_id.device
        zero = torch.tensor(0, dtype=torch.int, device=device)
        one = torch.tensor(1, dtype=torch.int, device=device)

        # 1. prepare multi-turn start and stop
        turn_stops = ((input_id == self.config.eos_token_id).nonzero() + 1)[:,0].tolist()
        turn_starts = [0] + turn_stops[:-1]
        num_turns = len(turn_starts)

        # 2. forward the full input_ids and labels, get tokenwise logits and losses
        outputs = self.forward(input_ids=input_ids, frames=frames, return_dict=True, use_cache=True, **kwargs)
        # HACK : append forward twice to get high resolution image
        if getattr(self, 'new_input_embed', False):
            inputs_embeds, input_ids, labels = self.new_input_embed(input_ids, frames, kwargs.get('high_frames_all', None),
                                                                    outputs.logits, labels, is_training=True)
            if inputs_embeds is not None:
                outputs = self.forward(inputs_embeds=inputs_embeds, return_dict=True, use_cache=True, **kwargs)
                
                # 0. evaluation only supports batch_size = 1
                assert input_ids.size(0) == labels.size(0) == 1
                input_id, label = input_ids[0], labels[0]
                device = input_id.device
                zero = torch.tensor(0, dtype=torch.int, device=device)
                one = torch.tensor(1, dtype=torch.int, device=device)

                # 1. prepare multi-turn start and stop
                turn_stops = ((input_id == self.config.eos_token_id).nonzero() + 1)[:,0].tolist()
                turn_starts = [0] + turn_stops[:-1]
                num_turns = len(turn_starts)

        logit, past_key_values = outputs.logits[0], outputs.past_key_values

        # 3. compute metrics for each turn
        v_placeholder_id = self.config.v_placeholder_id
        use_interval = self.config.frame_token_interval_id is not None
        frame_token_interval_id = self.config.frame_token_interval_id if use_interval else self.config.eos_token_id
        frame_num_tokens = self.config.frame_token_cls
        if self.config.frame_token_pooled:
            frame_num_tokens += self.config.frame_token_pooled[0] * self.config.frame_token_pooled[1]
        if frame_num_tokens != self.config.frame_num_tokens:
            frame_num_tokens = self.config.frame_num_tokens
        past_num_frames = 0
        lm_ppls, frame_diffs, fluencies, lm_correctness = [], [], [], []
        lm_ppls_away, lm_correctness_away = [], []
        # HACK : to analyze the frame_diffs
        force_rep = kwargs.get('force_rep', False)
        frame_diffs_early, frame_diffs_late, frame_diffs_away  = [], [], []
        frame_diffs_early_count, frame_diffs_late_count, frame_diffs_correct_count, frame_diffs_not_stop_count = torch.tensor(0).to(device), torch.tensor(0).to(device), torch.tensor(0).to(device), torch.tensor(0).to(device)
        
        for r, (turn_start, turn_stop) in enumerate(zip(turn_starts, turn_stops)):
            ## 3.1. we only have two losses: stream loss on frame tokens, and lm loss. prepare corresponding mask according two losses
            turn_label = label[turn_start:turn_stop]
            turn_learn_mask = turn_label != ignore_token_id
            if not turn_learn_mask.any():
                continue
            turn_logit = logit[turn_start:turn_stop]
            turn_input_id = input_id[turn_start:turn_stop]
            turn_v_mask = turn_input_id == v_placeholder_id
            turn_hv_mask = (turn_input_id == self.config.high_v_placeholder_id ) if kwargs.get('high_inference', False) else torch.tensor([False] * len(turn_input_id), device=device)
            turn_num_frames = turn_v_mask.sum() // frame_num_tokens
            turn_stream_mask = turn_v_mask & turn_learn_mask & (~turn_hv_mask)
            turn_lm_mask = turn_learn_mask & (~turn_stream_mask) & (~turn_hv_mask)
            ## 3.2 ppl, offline metric
            if turn_lm_mask.any():
                turn_lm_masked_logit, turn_lm_masked_label = turn_logit[turn_lm_mask], turn_label[turn_lm_mask]
                lm_ppl = torch.nn.functional.cross_entropy(turn_lm_masked_logit, turn_lm_masked_label).exp()
                lm_ppls.append(lm_ppl)
                lm_ppls_away.append(lm_ppl * (r+1) / (num_turns+1) * 2)
                turn_lm_masked_wrong_mask = turn_lm_masked_logit.argmax(dim=-1) != turn_lm_masked_label
                if turn_lm_masked_wrong_mask.any():
                    num_lm_correct_tokens = turn_lm_masked_wrong_mask.nonzero()[0,0]
                else:
                    num_lm_correct_tokens = (~turn_lm_masked_wrong_mask).sum()
                lm_correctness.append(num_lm_correct_tokens / turn_lm_masked_label.numel())
                lm_correctness_away.append(num_lm_correct_tokens / turn_lm_masked_label.numel() * (r+1) / (num_turns+1) * 2)

            ## 3.3. frame_diff (will be casted to time_diff in compute_metrics)
            if turn_stream_mask.any():
                ## 3.3.1: reply before (at) turn_num_frames
                turn_stream_masked_pred_mask = self.calculate_stop_frame(turn_logit, turn_label, turn_input_id, turn_stream_mask, frame_token_interval_threshold=frame_token_interval_threshold, **kwargs)
                if turn_stream_masked_pred_mask.any():
                    frame_diff = turn_stream_mask.sum() - turn_stream_masked_pred_mask.nonzero()[0,0] - 1
                else:
                    ## 3.3.2: the most complex part,reply after turn_num_frames. we assume the 'assistant: ...' not exists
                    turn_last_stream_idx = turn_stream_mask.nonzero()[-1,0]
                    past_key_values_before_assistant = self.trim_past_key_values(past_key_values, 0, turn_start + turn_last_stream_idx + 1)
                    if r == num_turns - 1: # no future frame. we assume the model should receive a signal when streaming ends (e.g. close button).
                        frame_diff = zero
                    else:
                        next_turn_num_frames = (input_id[turn_starts[r+1]:turn_stops[r+1]] == v_placeholder_id).sum() // frame_num_tokens
                        to_append_num_frames = min(next_turn_num_frames, turn_num_frames - 1) # avoid bias. current as center, two equal left/right side
                        if to_append_num_frames == 0:
                            frame_diff = zero
                        else:
                            to_append_frames = frames[past_num_frames+turn_num_frames:past_num_frames+turn_num_frames+to_append_num_frames]
                            frame_placeholder = [v_placeholder_id] * frame_num_tokens
                            if use_interval:
                                frame_placeholder = [frame_token_interval_id] + frame_placeholder
                            to_append_input_id = torch.tensor(frame_placeholder * to_append_num_frames, dtype=torch.long, device=device)
                            
                            # HACK : append high resolution image (. hv hv hv hv) 
                            to_append_high_frames = kwargs.get('high_frames_all', None)[past_num_frames+turn_num_frames:past_num_frames+turn_num_frames+to_append_num_frames]
                            high_frame_placeholder = [self.config.high_v_placeholder_id] * self.config.frame_num_tokens_high
                            if use_interval:
                                high_frame_placeholder = [self.config.high_frame_token_interval_id] + high_frame_placeholder
                            to_append_high_input_id = torch.tensor(high_frame_placeholder, dtype=torch.long, device=device)
                            to_append_input_id = torch.cat([to_append_input_id, to_append_high_input_id])
                            
                            to_append_logits = self.forward(
                                input_ids=to_append_input_id[None],
                                past_key_values=past_key_values_before_assistant,
                                frames=to_append_frames,
                                return_dict=True, use_cache=True,
                                **kwargs
                            ).logits
                            
                            if getattr(self, 'new_input_embed', False):
                                to_append_labels = torch.full_like(to_append_input_id[None], -100)
                                to_append_labels[to_append_input_id[None]==v_placeholder_id] = frame_token_interval_id
                                to_append_labels[:,-1] = 933
                                idx = (to_append_input_id[None]==self.config.v_placeholder_id).nonzero()[-1]
                                to_append_labels[idx[0], idx[1]] = self.config.high_frame_token_interval_id
                                
                                new_to_append_inputs_embeds, new_to_append_input_ids, new_to_append_labels = self.new_input_embed(to_append_input_id[None], to_append_frames, to_append_high_frames,
                                                                                        to_append_logits, to_append_labels, is_training=True)
                                if new_to_append_inputs_embeds is not None:
                                    to_append_inputs_embeds, to_append_input_id, to_append_labels = new_to_append_inputs_embeds, new_to_append_input_ids[0], new_to_append_labels 
                                    to_append_logits = self.forward(inputs_embeds=to_append_inputs_embeds,
                                                                   past_key_values=past_key_values_before_assistant,
                                                                   return_dict=True, use_cache=True, **kwargs).logits
                            
                            # we only use the last idx of each frame
                            append_turn_v_mask = to_append_input_id == v_placeholder_id
                            append_turn_learn_mask = to_append_labels[0] != ignore_token_id
                            append_turn_hv_mask = to_append_input_id == self.config.high_v_placeholder_id
                            try:
                                append_turn_stream_mask = append_turn_v_mask & append_turn_learn_mask & (~append_turn_hv_mask)
                            except:
                                breakpoint()
                            to_append_score_pred_mask = self.calculate_stop_frame(to_append_logits[0], to_append_labels[0], to_append_input_id, append_turn_stream_mask, **kwargs)
                            if to_append_score_pred_mask.any():
                                frame_diff = -(to_append_score_pred_mask.nonzero()[0,0] + 1)
                            else:
                                frame_diffs_not_stop_count += 1
                                frame_diff = -to_append_num_frames
                                
                frame_diffs.append(frame_diff.abs())
                # HACK : to analyze the frame_diffs
                frame_diffs_away.append(frame_diff.abs() * (r) / (num_turns) * 2)
                if frame_diff > 0:
                    frame_diffs_early.append(frame_diff)
                    frame_diffs_early_count += 1
                elif frame_diff < 0:
                    frame_diffs_late.append(frame_diff)
                    frame_diffs_late_count += 1 
                else:
                    frame_diffs_correct_count += 1

            ## 2.6 fluency
            if turn_lm_mask.any() and turn_stream_mask.any():
                num_learn_v_tokens = turn_stream_mask.sum()
                num_learn_valid_tokens = turn_lm_masked_label.numel() + num_learn_v_tokens
                if frame_diff == 0:
                    fluency = (num_learn_v_tokens + num_lm_correct_tokens) / num_learn_valid_tokens
                elif frame_diff > 0:
                    fluency = (num_learn_v_tokens - frame_diff) / num_learn_valid_tokens
                else:
                    fluency = (num_learn_v_tokens - 1) / num_learn_valid_tokens
                fluencies.append(fluency)
            ## 2.7 next turn
            past_num_frames += turn_num_frames
        lm_ppl = torch.stack(lm_ppls).mean() if lm_ppls else one
        frame_diff = torch.stack(frame_diffs).float().mean() if frame_diffs else zero
        fluency = torch.stack(fluencies).float().mean() if fluencies else one
        lm_correctness = torch.stack(lm_correctness).float().mean() if lm_correctness else one
        
        # HACK : to analyze the frame_diffs
        frame_diff_early = torch.stack(frame_diffs_early).float().mean() if frame_diffs_early else zero
        frame_diff_late = torch.stack(frame_diffs_late).float().mean() if frame_diffs_late else zero
        frame_diffs_away = torch.stack(frame_diffs_away).float().mean() if frame_diffs_away else zero
        
        # HACK : to analyze the lm correctness and ppl
        lm_ppls_away = torch.stack(lm_ppls_away).mean() if lm_ppls_away else one
        lm_correctness_away = torch.stack(lm_correctness_away).mean() if lm_correctness_away else one
        
        return torch.stack([lm_ppl, frame_diff, fluency, lm_correctness, 
                            frame_diff_early, frame_diff_late, frame_diffs_away, lm_ppls_away, lm_correctness_away,
                            frame_diffs_early_count, frame_diffs_late_count, frame_diffs_correct_count, frame_diffs_not_stop_count])
    
    def reward_strategies(self, logits, labels, input_ids, **kwargs):
        if logits.dim() == 2:
            logits = logits.unsqueeze(0)
            labels = labels.unsqueeze(0)
            input_ids = input_ids.unsqueeze(0)
        conti_masks = (labels == self.config.frame_token_interval_id) & (input_ids == self.config.v_placeholder_id) # (1, num_tokens)
        if not conti_masks.any():
            return None
        conti_logits = logits[conti_masks][:,self.config.high_frame_token_interval_id] # (num_frames, 1)
        rep_means = conti_logits.mean()
        rep_stds = conti_logits.std()
        rep_masks = (logits[:, :, self.config.high_frame_token_interval_id] > rep_means + 2 * rep_stds) & conti_masks  if kwargs.get("is_training", True) \
            else (logits.argmax(-1) == self.config.high_frame_token_interval_id) & conti_masks # TODO : try different strategy
        
        if not rep_masks.any():
            return None
        
        return rep_masks
    
    def calculate_stop_frame(self, turn_logit, turn_label, turn_input_id, turn_stream_mask, **kwargs):
        # stop strategy has been used in new_input_embed, so this function only need to find the stop frame that not in the continuous frame token and next is start generate token
        # 1. useing reponse strategy to get the need high frame token
        turn_score = turn_logit.softmax(dim=-1)
        rep_masks = self.reward_strategies(turn_logit, turn_label, turn_input_id, **kwargs)
        if rep_masks is not None:
            turn_score[rep_masks[0], self.config.high_frame_token_interval_id] = 1.0
   
        # 2. find the stop frame in stream id
        stop_v_index = None
        turn_stream_pred_mask = (turn_score.argmax(dim=-1) != self.config.frame_token_interval_id) & turn_stream_mask
        if turn_stream_pred_mask.any():
            # breakpoint()
            # 2.1 find all require high frame id in all input_ids and in stream mask
            turn_stream_masked_pred_mask_id = turn_stream_pred_mask[turn_stream_mask].nonzero()
            turn_stream_pred_mask_id = turn_stream_pred_mask.nonzero()
            
            # 2.2 find next token is high frame token and pred generate token
            offset = len(self.config.frame_token_interval_id) if not isinstance(self.config.frame_token_interval_id, int) is not None else 1
            next_high_frame_mask = turn_input_id[turn_stream_pred_mask_id + offset + self.config.frame_num_tokens_high] == self.config.high_v_placeholder_id
            next_generate_mask = turn_label[turn_stream_pred_mask_id + offset + self.config.frame_num_tokens_high] == 933
            satisfy_mask = next_high_frame_mask & next_generate_mask
            
            turn_stream_masked_pred_mask_id = turn_stream_masked_pred_mask_id[satisfy_mask]
            turn_stream_pred_mask_id = turn_stream_pred_mask_id[satisfy_mask]
            
            # 2.3 find the first token in stop id
            if turn_stream_masked_pred_mask_id.any():
                stop_v_index = turn_stream_masked_pred_mask_id[0]
        
        # 3. return the stop frame mask
        turn_stream_masked_pred_mask = turn_stream_mask.new_zeros(turn_stream_mask.sum(), dtype=torch.bool)
        if stop_v_index is not None:
            turn_stream_masked_pred_mask[stop_v_index] = True
        
        return turn_stream_masked_pred_mask
    
    def new_input_embed(self, input_ids: torch.Tensor, frames: torch.Tensor, high_frames: torch.Tensor,
                        logits: torch.Tensor, labels: torch.Tensor, loss: torch.Tensor=None, **kwargs):
        
        # 1. get new input_ids
        # 1.1 find high logit in the continuous frame token
        rep_masks = self.reward_strategies(logits, labels, input_ids, **kwargs)
        if rep_masks is None:
            return None, None, None
        
        # 1.2 replace the high logit with high_v_placeholder_id
        new_input_ids = []
        for i, (input_id, rep_mask) in enumerate(zip(input_ids, rep_masks)):
            new_input_id = []
            for j, (id, mask) in enumerate(zip(input_id, rep_mask)):
                if mask:
                    try:
                        assert input_id[j+1] == self.config.frame_token_interval_id and input_id[j] == self.config.v_placeholder_id, f"not continuous frame token in {input_id[j]} {input_id[j+1] } "
                    except:
                        breakpoint()
                    new_input_id.append(id)
                    new_input_id.append(self.config.high_frame_token_interval_id)
                    new_input_id.extend([self.config.high_v_placeholder_id] * self.config.frame_num_tokens_high)
                    
                else:
                    new_input_id.append(id)
            new_input_ids.append(new_input_id)
        input_ids = torch.tensor(new_input_ids).to(input_ids.device)
        # 2. get new labels
        new_labels = []
        for i, (label, rep_mask) in enumerate(zip(labels, rep_masks)):
            new_label = []
            for j, (l, m) in enumerate(zip(label, rep_mask)):
                if m:
                    assert label[j+1] == -100 and label[j] == self.config.frame_token_interval_id, f"not continuous frame token in {label[j]} {label[j+1]}"
                    new_label.append(self.config.high_frame_token_interval_id)
                    new_label.extend([-100] * self.config.frame_num_tokens_high)
                    new_label.append(l)
                else:
                    new_label.append(l)
            new_labels.append(new_label)
        labels = torch.tensor(new_labels).to(labels.device)
        # 3. get new input_embeds
        hv_indxs = []
        v_count = 0
        for input_id in input_ids:
            for j, id in enumerate(input_id):
                if id == self.config.v_placeholder_id:
                    if input_id[j+2] == self.config.high_v_placeholder_id:
                        hv_indxs.append(v_count)
                    v_count += 1
        assert v_count == frames.size(0), "v_count != frames.size(0)"
        high_frames = high_frames[hv_indxs]
        inputs_embeds = self.joint_embed(input_ids, frames, high_frames)
        
        return inputs_embeds, input_ids, labels
        
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        frames: torch.FloatTensor = None,
        attention_mask: torch.Tensor = None,
        position_ids: torch.LongTensor = None,
        past_key_values: list[torch.FloatTensor] = None,
        inputs_embeds: torch.FloatTensor = None,
        labels: torch.LongTensor = None,
        use_cache: bool = None,
        output_attentions: bool = None,
        output_hidden_states: bool = None,
        return_dict: bool = None,
        cache_position: torch.LongTensor = None,
        **kwargs,
    ):
            
        if inputs_embeds is None:
            inputs_embeds = self.joint_embed(input_ids, frames, kwargs.get('high_frames', None))
        outputs = super().forward(
            attention_mask = attention_mask,
            position_ids = position_ids,
            past_key_values = past_key_values,
            inputs_embeds = inputs_embeds,
            # labels
            use_cache = use_cache,
            output_attentions = output_attentions,
            output_hidden_states = output_hidden_states,
            return_dict = return_dict,
            cache_position=cache_position,
            **kwargs
        )

        loss = None
        if labels is not None:
            logits = outputs[0]
            v_mask = input_ids.flatten(0, 1) == self.config.v_placeholder_id
            weight = v_mask * self.config.stream_loss_weight + ~v_mask
            loss = nn.functional.cross_entropy(logits.flatten(0, 1), labels.flatten(), reduction='none') * weight
            loss = loss.sum() / (labels >= 0).sum()
            
        if loss is not None and loss < 1.0: # TODO : try different strategy
            inputs_embeds, input_ids, labels = self.new_input_embed(input_ids, frames, kwargs.get('high_frames_all', None),
                                                                    logits, labels)
            if inputs_embeds is not None:
                outputs = super().forward(
                    attention_mask = attention_mask,
                    position_ids = position_ids,
                    past_key_values = past_key_values,
                    inputs_embeds = inputs_embeds,
                    # labels
                    use_cache = use_cache,
                    output_attentions = output_attentions,
                    output_hidden_states = output_hidden_states,
                    return_dict = return_dict,
                    cache_position=cache_position,
                    **kwargs
                )
                re_loss = None
                logits = outputs[0]
                v_mask = input_ids.flatten(0, 1) == self.config.v_placeholder_id
                weight = v_mask * self.config.stream_loss_weight + ~v_mask
                re_loss = nn.functional.cross_entropy(logits.flatten(0, 1), labels.flatten(), reduction='none') * weight
                re_loss = re_loss.sum() / (labels >= 0).sum()
                loss = loss + re_loss
            
        if not return_dict:
            return (loss,) + outputs[1:] if loss is not None else outputs
    
        outputs.loss = loss
        return outputs
    

def build_live_llama(**kwargs):
    if kwargs.get('live_version', None) == 'live1_1+' or kwargs.get('live_version', None) == 'livel_h':
        return build_live(config_class=LiveLlamaConfig, model_class=LiveLlamaForCausalLMhigh, **kwargs)
    elif 'live1_1+' in kwargs.get('live_version', None) or 'livel_h' in kwargs.get('live_version', None):
        return build_live(config_class=LiveLlamaConfig, model_class=LiveLlamaHaploForCausalLMhigh, **kwargs)
    else:
        if 'mamba' in kwargs.get('live_version', ''):
            if 'ft' in kwargs.get('live_version', ''):
                return build_live(config_class=LiveLlamaConfigMambaFt, model_class=LiveLlamaForCausalLM_mamba, **kwargs)
            else:
                return build_live(config_class=LiveLlamaConfigMamba, model_class=LiveLlamaForCausalLM_mamba, **kwargs)
        elif 'no_frame_interval' in kwargs.get('live_version', ''):
            return build_live(config_class=LiveLlamaConfig, model_class=LiveLlamaForCausalLMnoFrameInterval, **kwargs)
        elif 'threshold' in kwargs.get('live_version', ''):
            return build_live(config_class=LiveLlamaConfig, model_class=LiveLlamaForCausalLMthreshold, **kwargs)
        else:
            return build_live(config_class=LiveLlamaConfig, model_class=LiveLlamaForCausalLM, **kwargs)

if __name__ == '__main__':
    from ..arguments_live import LiveOnePlusTrainingArguments
    print(LiveOnePlusTrainingArguments().to_dict())
    model, tokenizer = build_live_llama(is_training=True, **LiveOnePlusTrainingArguments().to_dict())
    print(model.config, tokenizer)
