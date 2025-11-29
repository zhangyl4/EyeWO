import torch
from torch.utils.data import Sampler
from functools import partial
from transformers import PreTrainedTokenizer
from transformers.trainer_pt_utils import LabelSmoother

def data_collator(batch: list[list], *, tokenizer: PreTrainedTokenizer, **kwargs):
    batch = list(zip(*batch))
    if len(batch) == 5:
        [batch_text, batch_frames, batch_learn_ranges, batch_sample_idx, batch_evaluation_kwargs], batch_reponse_clip = batch, None
    else:
        batch_text, batch_frames, batch_learn_ranges, batch_sample_idx, batch_evaluation_kwargs, batch_reponse_clip = batch
    batch = tokenizer(batch_text, return_offsets_mapping=True, add_special_tokens=False, return_tensors="pt", padding=True)
    batch_labels = torch.full_like(batch.input_ids, LabelSmoother.ignore_index, dtype=torch.long)
    for text, labels, input_ids, offset_mapping, learn_range in zip(
        batch_text, batch_labels, batch.input_ids, batch.offset_mapping, batch_learn_ranges
    ):
        for learn_r in learn_range:
            start = torch.nonzero(offset_mapping[:,0] == learn_r.start).item()
            if offset_mapping[:,0][-1] >= learn_r.stop:
                stop = torch.nonzero(offset_mapping[:,0] == learn_r.stop).item()
            else: # the last eos token
                stop = len(input_ids)
            labels[start-1:stop-1] = input_ids[start:stop]
            # NOTE: input_ids may out of boundary of len(tokenizer) - 1. (1 is the added vision placeholder)
            # this is because some frames has v_placeholder_id target. so replace it with eos token.
            labels[labels >= len(tokenizer) - 1] = tokenizer.eos_token_id
    batch['labels'] = batch_labels
    batch.pop('offset_mapping')
    batch['frames'] = torch.cat(batch_frames)
    batch['sample_idxs'] = torch.tensor(batch_sample_idx)
    if batch_evaluation_kwargs[0]:
        batch['evaluation_kwargs'] = batch_evaluation_kwargs[0] # evaluation only supports bs = 1, so its okay
    if batch_reponse_clip:
        batch['response_clip'] = torch.tensor(batch_reponse_clip)
    return batch


def data_collator_high(batch: list[list], *, tokenizer: PreTrainedTokenizer, **kwargs):
    batch = list(zip(*batch))
    if len(batch) == 7:
        [batch_text, batch_frames, batch_high_frames, batch_high_frames_all, batch_learn_ranges, batch_sample_idx, batch_evaluation_kwargs], batch_reponse_clip = batch, None
    else:
        batch_text, batch_frames, batch_high_frames, batch_high_frames_all, batch_learn_ranges, batch_sample_idx, batch_evaluation_kwargs, batch_reponse_clip = batch
    batch = tokenizer(batch_text, return_offsets_mapping=True, add_special_tokens=False, return_tensors="pt", padding=True)
    batch_labels = torch.full_like(batch.input_ids, LabelSmoother.ignore_index, dtype=torch.long)
    for text, labels, input_ids, offset_mapping, learn_range in zip(
        batch_text, batch_labels, batch.input_ids, batch.offset_mapping, batch_learn_ranges
    ):
        for learn_r in learn_range:
            start = torch.nonzero(offset_mapping[:,0] == learn_r.start).item()
            if offset_mapping[:,0][-1] >= learn_r.stop:
                stop = torch.nonzero(offset_mapping[:,0] == learn_r.stop).item()
            else: # the last eos token
                stop = len(input_ids)
            labels[start-1:stop-1] = input_ids[start:stop]
            # NOTE: input_ids may out of boundary of len(tokenizer) - 1. (1 is the added vision placeholder)
            # this is because some frames has v_placeholder_id target. so replace it with eos token.
            labels[labels > len(tokenizer) - 1] = tokenizer.eos_token_id
            labels[labels == tokenizer.convert_tokens_to_ids(kwargs.get('high_v_placeholder'))] = tokenizer.eos_token_id
            labels[labels == tokenizer.convert_tokens_to_ids('<v>')] = tokenizer.eos_token_id
    batch['labels'] = batch_labels

    batch.pop('offset_mapping')
    batch['frames'] = torch.cat(batch_frames)
    batch['high_frames'] = torch.cat(batch_high_frames)
    batch['high_frames_all'] = torch.cat(batch_high_frames_all)
    batch['sample_idxs'] = torch.tensor(batch_sample_idx)
    if batch_evaluation_kwargs[0]:
        batch['evaluation_kwargs'] = batch_evaluation_kwargs[0] # evaluation only supports bs = 1, so its okay
    if batch_reponse_clip:
        batch['response_clip'] = torch.tensor(batch_reponse_clip)

    return batch


def get_data_collator(**kwargs):
    if 'live1_1+' in kwargs.get('live_version', None) or 'livel_h' in kwargs.get('live_version', None):
        return partial(data_collator_high, **kwargs)
    else:
        return partial(data_collator, **kwargs)
