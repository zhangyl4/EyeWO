import torch, os
from peft import LoraConfig, get_peft_model, PeftModel
from transformers import AutoModelForCausalLM, Cache
from transformers.utils import logging

from .tokenization_live import build_live_tokenizer_and_update_config
from .vision_live import build_live_vision
import copy

logger = logging.get_logger(__name__)

class LiveMixin(AutoModelForCausalLM):
    def set_vision_inside(self):
        logger.warning_once("!!! Set vision encoder in the model, only recommended for on in-the-wild inference. "
            "Please dont call this for efficient training & evaluation. Instead, do visual feature pre-extraction.")
        self.vision_encoder, self.vision_encode = build_live_vision(self.config)

    def unset_vision_inside(self):
        del self.vision_encoder
        del self.vision_encode

    def visual_embed(self, frames: torch.Tensor):
        if hasattr(self, 'vision_encode'):
            with torch.cuda.amp.autocast():
                frames = self.vision_encode(self.vision_encoder, frames)
            frames = frames.to(self.dtype)
        frames = self.connector(frames)
        return frames.view(-1, frames.shape[-1])

    def joint_embed(
        self,
        input_ids: torch.Tensor = None,
        frames: torch.Tensor = None,
    ):
        if frames is None:
            return self.get_input_embeddings()(input_ids)
        if input_ids is None:
            return self.visual_embed(frames)
        inputs_embeds = self.get_input_embeddings()(input_ids.clamp(max=self.vocab_size-1))
        v_mask = input_ids == self.config.v_placeholder_id
        if v_mask.any():
            inputs_embeds[v_mask] = self.visual_embed(frames)
        return inputs_embeds

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
        if turn_stops[-1] != len(input_id):
            turn_stops.append(len(input_id))
            turn_starts.append(turn_stops[-2])
        num_turns = len(turn_starts)

        # 2. forward the full input_ids and labels, get tokenwise logits and losses
        outputs = self.forward(input_ids=input_ids, frames=frames, return_dict=True, use_cache=True, return_logits=True, **kwargs)
        logit, past_key_values = outputs.logits[0], outputs.past_key_values
        if hasattr(self, 'memory'):
            self.old_memory = copy.deepcopy(self.memory)

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
                try:
                    turn_lm_masked_logit, turn_lm_masked_label = turn_logit[turn_lm_mask], turn_label[turn_lm_mask]
                except:
                    breakpoint()
                lm_ppl = torch.nn.functional.cross_entropy(turn_lm_masked_logit, turn_lm_masked_label).exp()
                lm_ppls.append(lm_ppl)
                turn_lm_masked_wrong_mask = turn_lm_masked_logit.argmax(dim=-1) != turn_lm_masked_label
                if turn_lm_masked_wrong_mask.any():
                    num_lm_correct_tokens = turn_lm_masked_wrong_mask.nonzero()[0,0]
                else:
                    num_lm_correct_tokens = (~turn_lm_masked_wrong_mask).sum()
                lm_correctness.append(num_lm_correct_tokens / turn_lm_masked_label.numel())

            ## 3.3. frame_diff (will be casted to time_diff in compute_metrics)
            if turn_stream_mask.any():
                ## 3.3.1: reply before (at) turn_num_frames
                turn_score = turn_logit.softmax(dim=-1)
                turn_stream_masked_score = turn_score[turn_stream_mask]
                if frame_token_interval_threshold > 0:
                    lower_threshold_mask = turn_stream_masked_score[:, frame_token_interval_id] < frame_token_interval_threshold
                    turn_stream_masked_score[lower_threshold_mask] = 0
                turn_stream_masked_pred_mask = turn_stream_masked_score.argmax(dim=-1) != frame_token_interval_id  # == 933 # HACK : 933 is the index of the response token
                if turn_stream_masked_pred_mask.any():
                    frame_diff = turn_stream_mask.sum() - turn_stream_masked_pred_mask.nonzero()[0,0] - 1
                else:
                    ## 3.3.2: the most complex part,reply after turn_num_frames. we assume the 'assistant: ...' not exists
                    frame_diff = self.append_frames_diff_pred(num_turns, r, turn_num_frames, turn_starts, turn_stops,
                                                            frames, frame_token_interval_threshold,
                                                            turn_stream_mask, past_key_values, turn_start, device,
                                                            zero, use_interval, past_num_frames, input_id, v_placeholder_id,
                                                            frame_token_interval_id, **kwargs)
                frame_diffs.append(frame_diff.abs())

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
        return torch.stack([lm_ppl, frame_diff, fluency, lm_correctness])
    
    
    def append_frames_diff_pred(self, num_turns, r, turn_num_frames, turn_starts, turn_stops,
                                frames, frame_token_interval_threshold,
                                turn_stream_mask, past_key_values, turn_start, device,
                                zero, use_interval, past_num_frames, input_id, v_placeholder_id,
                                frame_token_interval_id, **kwargs):
        ## 3.3.2: the most complex part,reply after turn_num_frames. we assume the 'assistant: ...' not exists
        turn_last_stream_idx = turn_stream_mask.nonzero()[-1,0]
        frame_num_tokens = kwargs.get("frame_num_tokens")
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
                to_append_logit = self.forward(
                    input_ids=to_append_input_id[None],
                    past_key_values=past_key_values_before_assistant,
                    frames=to_append_frames,
                    return_dict=True, use_cache=True,
                    **{**kwargs, 'append_prediction': True, 'past_input_ids': input_id[:turn_start + turn_last_stream_idx + 1],
                        'past_range':range(0, turn_start + turn_last_stream_idx + 1 + len(to_append_input_id)),
                        "past_frames": frames[:past_num_frames+turn_num_frames]}
                ).logits[0]
                # we only use the last idx of each frame
                idxs = torch.arange(len(frame_placeholder)-1, len(to_append_input_id), len(frame_placeholder), device=device)
                
                to_append_score = to_append_logit[idxs].softmax(dim=-1)
                if frame_token_interval_threshold > 0:
                    lower_threshold_mask = to_append_score[:, frame_token_interval_id] < frame_token_interval_threshold
                    to_append_score[lower_threshold_mask] = 0
                to_append_score_pred_mask = to_append_score.argmax(dim=-1) != frame_token_interval_id # == 933 # HACK : 933 is the index of the response token
                if to_append_score_pred_mask.any():
                    frame_diff = -(to_append_score_pred_mask.nonzero()[0,0] + 1)
                else:
                    frame_diff = -to_append_num_frames
        return frame_diff
    
    
    @torch.no_grad()
    def stream_evaluate_analysis(
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
        logit, past_key_values = outputs.logits[0], outputs.past_key_values

        # 3. compute metrics for each turn
        v_placeholder_id = self.config.v_placeholder_id
        use_interval = self.config.frame_token_interval_id is not None
        frame_token_interval_id = self.config.frame_token_interval_id if use_interval else self.config.eos_token_id
        frame_num_tokens = self.config.frame_token_cls
        if self.config.frame_token_pooled:
            frame_num_tokens += self.config.frame_token_pooled[0] * self.config.frame_token_pooled[1]
        if self.config.frame_num_tokens != frame_num_tokens:
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
            turn_num_frames = turn_v_mask.sum() // frame_num_tokens
            turn_stream_mask = turn_v_mask & turn_learn_mask
            turn_lm_mask = turn_learn_mask & ~turn_stream_mask

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
                turn_score = turn_logit.softmax(dim=-1)
                turn_stream_masked_score = turn_score[turn_stream_mask]
                if frame_token_interval_threshold > 0:
                    lower_threshold_mask = turn_stream_masked_score[:, frame_token_interval_id] < frame_token_interval_threshold
                    turn_stream_masked_score[lower_threshold_mask] = 0
                    
                # HACK : to analyze the force_rep
                if force_rep:
                    # resp_indx = turn_stream_masked_score[:, 933].argmax(dim=-1)
                    # if turn_stream_masked_score[resp_indx, 933] > 0.4:
                    #     turn_stream_masked_score[resp_indx, 933] = 1
                    
                    turn_stream_masked_score_cumsum_933 = turn_stream_masked_score[:, 933].cumsum(dim=-1)
                    turn_stream_masked_score_cumsum_933_threshold = turn_stream_masked_score_cumsum_933.new_full(turn_stream_masked_score_cumsum_933.size(), kwargs.get("force_rep_para1")).cumsum(dim=-1) + kwargs.get("force_rep_para2")
                    resp_mask = turn_stream_masked_score_cumsum_933 > turn_stream_masked_score_cumsum_933_threshold
                    resp_indx = (resp_mask).nonzero(as_tuple=True)[0][0] if (resp_mask).any() else None
                    if resp_indx is not None:
                        turn_stream_masked_score[resp_indx, 933] = 1
                        
                turn_stream_masked_pred_mask = turn_stream_masked_score.argmax(dim=-1) != frame_token_interval_id # == 933 # HACK : 933 is the index of the response token
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
                        to_append_num_frames = min(next_turn_num_frames, turn_num_frames - 1) if not kwargs.get('eval_time_diff_late', False) else next_turn_num_frames # avoid bias. current as center, two equal left/right side
                        if to_append_num_frames == 0:
                            frame_diff = zero
                        else:
                            to_append_frames = frames[past_num_frames+turn_num_frames:past_num_frames+turn_num_frames+to_append_num_frames]
                            frame_placeholder = [v_placeholder_id] * frame_num_tokens
                            if use_interval:
                                frame_placeholder = [frame_token_interval_id] + frame_placeholder
                            to_append_input_id = torch.tensor(frame_placeholder * to_append_num_frames, dtype=torch.long, device=device)
                            to_append_logit = self.forward(
                                input_ids=to_append_input_id[None],
                                past_key_values=past_key_values_before_assistant,
                                frames=to_append_frames,
                                return_dict=True, use_cache=True,
                                **{**kwargs, 'append_prediction': True, 'past_input_ids': input_id[:turn_start + turn_last_stream_idx + 1],
                                   'past_range':range(0, turn_start + turn_last_stream_idx + 1 + len(to_append_input_id)),
                                   "past_frames": frames[:past_num_frames+turn_num_frames]}
                            ).logits[0]
                            # we only use the last idx of each frame
                            idxs = torch.arange(len(frame_placeholder)-1, len(to_append_input_id), len(frame_placeholder), device=device)
                            to_append_score = to_append_logit[idxs].softmax(dim=-1)
                            if frame_token_interval_threshold > 0:
                                lower_threshold_mask = to_append_score[:, frame_token_interval_id] < frame_token_interval_threshold
                                to_append_score[lower_threshold_mask] = 0
                                
                            # HACK : to analyze the force_rep
                            if force_rep:
                                
                                # resp_indx = to_append_score[:,933].argmax(dim=-1)
                                # if to_append_score[resp_indx, 933] > 0.4:
                                #     to_append_score[resp_indx, 933] = 1
                                
                                to_append_score_cumsum_933 = to_append_score[:,933].cumsum(dim=-1) + turn_stream_masked_score_cumsum_933[-1]
                                to_append_score_cumsum_933_threshold = to_append_score_cumsum_933.new_full(to_append_score_cumsum_933.size(), kwargs.get("force_rep_para1")).cumsum(dim=-1) + kwargs.get("force_rep_para2") + turn_stream_masked_score_cumsum_933_threshold[-1]
                                resp_mask = to_append_score_cumsum_933 > to_append_score_cumsum_933_threshold
                                resp_indx = (resp_mask).nonzero(as_tuple=True)[0][0] if (resp_mask).any() else None
    
                                if resp_indx is not None:
                                    to_append_score[resp_indx, 933] = 1
                                
                            to_append_score_pred_mask = to_append_score.argmax(dim=-1) != frame_token_interval_id # == 933 # HACK : 933 is the index of the response token
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

    @torch.no_grad()
    def lm_evaluate_analysis(
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
        use_interval = self.config.frame_token_interval_id is not None
        frame_token_interval_id = self.config.frame_token_interval_id if use_interval else self.config.eos_token_id
        
        # 2. forward the full input_ids and labels, get tokenwise logits and losses
        if getattr(self, 'new_input_embed', False):
            outputs = self.forward(input_ids=input_ids, frames=frames, return_dict=True, use_cache=True, **kwargs)
            new_inputs_embeds, new_input_ids, new_labels = self.new_input_embed(input_ids, frames, kwargs.get('high_frames_all', None),
                                                                    outputs.logits, labels, is_training=False)
            if new_inputs_embeds is not None:
                outputs = self.forward(inputs_embeds=new_inputs_embeds, return_dict=True, use_cache=True, **kwargs)
                input_id, label = new_input_ids[0], new_labels[0]
        else:
            outputs = self.forward(input_ids=input_ids, frames=frames, return_dict=True, use_cache=True, **kwargs)
        
        try:
            logit = outputs.logits[0]
            pred = logit.argmax(dim=-1)
            stream_logit = logit.softmax(dim=-1)[:, [frame_token_interval_id, 13, 933]]
            pred[label == -100] = -100
            return torch.stack([input_id, pred, label, stream_logit[:,0], stream_logit[:,1], stream_logit[:,2]])
        except:
            breakpoint()
        
    
    def trim_past_key_values(self, past_key_values, start, stop):
        return [[past_keys[:,:,start:stop], past_values[:,:,start:stop]] for past_keys, past_values in past_key_values]

def fast_greedy_generate(*, model: LiveMixin, inputs_embeds: torch.Tensor, past_key_values: Cache, eos_token_id: int, inplace_output_ids: torch.Tensor):
    for i in range(inplace_output_ids.size(1)):
        outputs = model(inputs_embeds=inputs_embeds, past_key_values=past_key_values, use_cache=True)
        past_key_values = outputs.past_key_values
        new_token_id = outputs.logits[:, -1:].argmax(dim=-1)
        inplace_output_ids[:, i] = new_token_id
        if new_token_id == eos_token_id:
            break
        inputs_embeds = model.get_input_embeddings()(new_token_id)
    return inplace_output_ids[:, :i+1], past_key_values

def updata_model_config(model, tokenizer, **kwargs):
    model.config.vocab_size = len(tokenizer)
    return model, tokenizer


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
    overwrite_config['pretrain_mm_mlp_adapter'] = kwargs['pretrain_mm_mlp_adapter']
    
    overwrite_config['return_all_logits'] = kwargs['return_all_logits']
    overwrite_config['skip_first'] = kwargs['skip_first']
    overwrite_config['compress_turn'] = kwargs['compress_turn']
    overwrite_config['is_smoothing'] = kwargs['is_smoothing']
    
    # reponse args
    overwrite_config['is_smoothing'] = kwargs['is_smoothing']
    
    # training args
    overwrite_config['only_modules_to_ft'] = kwargs['only_modules_to_ft']
    overwrite_config['adapter_model'] = kwargs['adapter_model']
    
    # vision args
    overwrite_config['add_vision_pretrained'] = kwargs['add_vision_pretrained']
    overwrite_config['add_type'] = kwargs['add_type']
    if config.vision_pretrained == 'google/siglip-so400m-patch14-384':
        config.vision_hidden_size = 1152
        
    # sample strategy
    overwrite_config['max_frame_clip_mode_model'] = kwargs['max_frame_clip_mode_model']
    overwrite_config['max_frame_clip_mode_data'] = kwargs['max_frame_clip_mode_data']
    
    if overwrite_config:
        for k, v in overwrite_config.items():
            setattr(config, k, v)
            del kwargs[k]

    return config

import torch.distributed as dist
def rank0_print(*args):
    if dist.is_initialized():
        if dist.get_rank() == 0:
            print(f"Rank {dist.get_rank()}: ", *args)
    else:
        print(*args)

def build_live(
    *,
    is_training: bool,
    config_class: type,
    model_class: type,
    llm_pretrained: str = None,
    finetune_modules: list[str] = None,
    lora_modules: str = None,
    lora_r: int = None,
    lora_alpha: int = None,
    set_vision_inside: bool = False,
    resume_from_checkpoint: str = '',
    attn_implementation: str = 'flash_attention_2',
    torch_dtype: str | torch.dtype = 'auto',
    enable_vision_lora: bool = False,
    vision_lora_alpha: int = None,
    vision_lora_r: int = None,
    vision_lora_modules: str = None,
    **kwargs
):
    
    # build model
    config = config_class.from_pretrained(llm_pretrained, **kwargs)
    config = updata_config(config, **kwargs)
    model = model_class.from_pretrained(llm_pretrained, config=config, torch_dtype=torch_dtype, attn_implementation=attn_implementation)
    tokenizer = build_live_tokenizer_and_update_config(llm_pretrained, model.config, **kwargs)
    model, tokenizer = updata_model_config(model, tokenizer, **kwargs)
    
    if set_vision_inside:
        if kwargs.get('low_vision_encoder', False) and kwargs.get('high_vision_encoder'):
            model.set_vision_inside(low_vision_encoder=kwargs.get('low_vision_encoder', False), 
                                high_vision_encoder=kwargs.get('high_vision_encoder'))
        else:
            model.set_vision_inside()
    
    
    if is_training:
        # build training params
                    
        if len(config.only_modules_to_ft) > 0: # only ft
            model.requires_grad_(False)
            model.model.requires_grad_(False)
            model.vision_encoder.requires_grad_(False)
            model.connector.requires_grad_(False)
            
            for n, p in model.named_parameters():
                if any([m_name in n for m_name in config.only_modules_to_ft]):
                    p.requires_grad_(True)
                    
            total_params = sum(p.ds_numel if hasattr(p, "ds_numel") else p.numel() for p in model.parameters())
            trainable_params = sum(p.ds_numel if hasattr(p, "ds_numel") else p.numel() for p in model.parameters() if p.requires_grad)
            
            rank0_print(f"Total parameters: ~{total_params/1e6:.2f} M)")
            rank0_print(f"Trainable parameters: ~{trainable_params/1e6:.2f} M)")
            rank0_print(f"Trainable percent: ~{trainable_params/total_params :.8f} M)")
            
        else: # lora ft
            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=lora_modules,
                lora_dropout=0.05,
                task_type="CAUSAL_LM",
                modules_to_save=finetune_modules,
                inference_mode=False,
            )
            if config.adapter_model:
                model = PeftModel.from_pretrained(model, config.adapter_model, is_trainable=True, config=lora_config) # model = get_peft_model(model, config.adapter_model, lora_config)
            else:
                model = get_peft_model(model, lora_config)

            if set_vision_inside:
                if enable_vision_lora:
                    vision_lora_config = LoraConfig(
                        r=vision_lora_r,
                        lora_alpha=vision_lora_alpha,
                        target_modules=vision_lora_modules,
                        lora_dropout=0.05,
                        task_type="FEATURE_EXTRACTION",
                        inference_mode=False,
                    )
                    # model.vision_encoder = get_peft_model(model.vision_encoder, vision_lora_config)
                    if hasattr(model, 'add_vision_encoder'):
                        model.add_vision_encoder = get_peft_model(model.add_vision_encoder, vision_lora_config)
                    rank0_print("Added LoRA to vision encoder")
            
            model.print_trainable_parameters()
    else:
        if resume_from_checkpoint:
            model = PeftModel.from_pretrained(model, resume_from_checkpoint, is_trainable=False)
        else:
            logger.warning(f'!!! Fail to load checkpoint: {resume_from_checkpoint}. Return a new initialized model.')
            
        model.requires_grad_(False)
    return model, tokenizer
