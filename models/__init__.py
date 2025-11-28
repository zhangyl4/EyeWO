from transformers import HfArgumentParser
from dataclasses import asdict
from .arguments_live import LiveTrainingArguments, get_args_class
from .live_llama import build_live_llama
from .beacon_live_llama import build_live_beacon_llama
from .modeling_live import fast_greedy_generate


def build_model_and_tokenizer(is_training=True, **kwargs):
    if 'beacon' in kwargs['live_version']:
        kwargs['live_version'] = kwargs['live_version'].replace('beacon', '')
        return build_live_beacon_llama(is_training=is_training, **kwargs)
    else:
        return build_live_llama(is_training=is_training, **kwargs)

def parse_args() -> LiveTrainingArguments:
    args, = HfArgumentParser(LiveTrainingArguments).parse_args_into_dataclasses()
    args, = HfArgumentParser(get_args_class(args.live_version)).parse_args_into_dataclasses()
    return args

def set_args(config):
    args = get_args_class('live1+')
    if config.resume_from_checkpoint is None:
        args = args(resume_from_checkpoint = "chenjoya/videollm-online-8b-v1plus", **asdict(config))
    else:
        args = args(**asdict(config))
    return args

def set_args_highres(config):
    args = get_args_class('beacon_livel_h')
    args = args(**asdict(config))
    return args 