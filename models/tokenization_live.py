import torch
from transformers import AutoTokenizer
from functools import partial

from transformers import HfArgumentParser
from .arguments_live import LiveTrainingArguments, get_args_class

def parse_args() -> LiveTrainingArguments:
    args, = HfArgumentParser(LiveTrainingArguments).parse_args_into_dataclasses()
    args, = HfArgumentParser(get_args_class(args.live_version)).parse_args_into_dataclasses()
    return args
from dataclasses import asdict

from .configuration_live import LiveConfigMixin

def get_stream_placeholder_len(num_frames: int, model_config: LiveConfigMixin) -> str:
    return num_frames * model_config.frame_num_tokens * len(model_config.v_placeholder) + len(model_config.frame_token_interval) * (num_frames - 1)

def get_stream_placeholder_len_high(num_frames: int, model_config: LiveConfigMixin) -> str:
    return num_frames * model_config.frame_num_tokens_high * len(model_config.high_v_placeholder) + len(model_config.frame_token_interval) * (num_frames - 1)

def get_stream_placeholder_jinja2(model_config: LiveConfigMixin) -> str:
    return f"'{model_config.frame_token_interval}'.join([{model_config.frame_num_tokens} * '{model_config.v_placeholder}'] * message['num_frames'])"

def get_stream_placeholder_high_jinja2(model_config: LiveConfigMixin) -> str:
    return f"'{model_config.frame_token_interval}'.join([{model_config.frame_num_tokens_high} * '{model_config.high_v_placeholder}'] * message['num_frames'])"


def get_stream_learn_ranges(num_frames: int, model_config: LiveConfigMixin) -> torch.Tensor:
    if num_frames <= 0:
        raise ValueError(f"num_frames must be positive, got {num_frames}")
        
    len_frame_placeholder_with_interval = model_config.frame_num_tokens * len(model_config.v_placeholder) + len(model_config.frame_token_interval)
    intermediate_interval_idxs = torch.arange(
        len_frame_placeholder_with_interval,
        len_frame_placeholder_with_interval * num_frames + 1,
        len_frame_placeholder_with_interval
    ) - len(model_config.frame_token_interval)
    len_learn = len(model_config.frame_token_interval) if model_config.frame_token_interval else len(model_config.v_placeholder)
    learn_ranges = torch.stack([
        intermediate_interval_idxs,
        intermediate_interval_idxs + len_learn
    ], dim=1)
    return learn_ranges

def get_stream_learn_ranges_high(num_frames: int, model_config: LiveConfigMixin) -> torch.Tensor:
    len_frame_placeholder_with_interval = model_config.frame_num_tokens_high * len(model_config.high_v_placeholder) + len(model_config.high_frame_token_interval)
    intermediate_interval_idxs = torch.arange(
        len_frame_placeholder_with_interval,
        len_frame_placeholder_with_interval * num_frames + 1,
        len_frame_placeholder_with_interval
    ) - len(model_config.high_frame_token_interval)
    len_learn = len(model_config.high_frame_token_interval) if model_config.high_frame_token_interval else len(model_config.v_placeholder)
    learn_ranges = torch.stack([
        intermediate_interval_idxs,
        intermediate_interval_idxs + len_learn
    ], dim=1)
    return learn_ranges

def chat_template(self, stream_placeholder_jinja2: str):
    """
    system prompt
    [<v>,<v>,<v>]
    User: ...
    Assistant: ...\]
    [<v>,<v>]
    Assistant: ...\]
    User: ...
    Assistant: ...\]
    """
    template = (
        "{% if messages[0]['role'] == 'system' %}"
        "{{ bos_token + messages[0]['content'] + '\n' }}" # system
        "{% set messages = messages[1:] %}"
        "{% endif %}"
        "{% for message in messages %}"
        "{% if message['role'] == 'user' %}"
        "{% if add_stream_query_prompt %}"
        "{{ ']\nUser: ' + message['content'] }}"
        "{% else %}"
        "{{ '\nUser: ' + message['content'] }}"
        "{% endif %}"
        "{% elif message['role'] == 'assistant' %}"
        "{{ '\nAssistant: '  + message['content'] + eos_token }}"
        "{% elif message['role'] == 'stream' and message['num_frames'] > 0: %}"
        "{{ '\n[' + STREAM_PLACEHOLDER + ']' }}"
        "{% endif %}"
        "{% endfor %}"
        "{% if add_generation_prompt %}"
        "{{ '\nAssistant:' }}"
        "{% elif add_stream_prompt %}"
        "{{ '\n[' }}"
        "{% elif add_stream_generation_prompt %}"
        "{{ ']\nAssistant:' }}"
        "{% endif %}"
    )
    template = template.replace('STREAM_PLACEHOLDER', stream_placeholder_jinja2)
    return template

def chat_template_high(self, stream_placeholder_jinja2: str,
                  stream_placeholder_high_jinja2: str,
                  ask_high_token: str):
    """
    system prompt
    [<v>,<v>,<v>]
    User: ...
    Assistant: ...\]
    [<v>,<v>]
    Assistant: ...\]
    User: ...
    Assistant: ...\]
    """
    template = (
        "{% if messages[0]['role'] == 'system' %}"
        "{{ bos_token + messages[0]['content'] + '\n' }}" # system
        "{% set messages = messages[1:] %}"
        "{% endif %}"
        "{% for message in messages %}"
        "{% if message['role'] == 'user' %}"
        "{% if add_stream_query_prompt %}"
        "{{ ']\nUser: ' + message['content'] }}"
        "{% else %}"
        "{{ '\nUser: ' + message['content'] }}"
        "{% endif %}"
        "{% elif message['role'] == 'assistant' %}"
        "{{ '\nAssistant: '  + message['content'] + eos_token }}"
        "{% elif message['role'] == 'stream' and message['num_frames'] > 0 %}"
        "{% if loop.index > 1 and loop.index0 + 1 < messages|length and messages[loop.index-2]['role'] == 'stream_high' and loop.index < messages|length and messages[loop.index]['role'] == 'stream_high' %}"
        "{{ STREAM_PLACEHOLDER + ASK_HIGH }}"
        "{% elif loop.index < messages|length and (messages[loop.index]['role'] == 'assistant' or messages[loop.index]['role'] == 'user') %}"
        "{{ '\n[' + STREAM_PLACEHOLDER + ']' }}"
        "{% elif loop.index < messages|length and messages[loop.index]['role'] == 'stream_high' %}"
        "{{ '\n[' + STREAM_PLACEHOLDER + ASK_HIGH }}"
        "{% else %}"
        "{{ '\n[' + STREAM_PLACEHOLDER + ']' }}"
        "{% endif %}"
        "{% elif message['role'] == 'stream_high' and message['num_frames'] > 0 %}"
        "{% if loop.index < messages|length and messages[loop.index]['role'] == 'stream' %}"
        "{{ STREAM_HIGH + ',' }}"
        "{% else %}"
        "{{ STREAM_HIGH + ']' }}"
        "{% endif %}"
        "{% endif %}"
        "{% endfor %}"
        "{% if add_generation_prompt %}"
        "{{ '\nAssistant:' }}"
        "{% elif add_stream_prompt %}"
        "{{ '\n[' }}"
        "{% elif add_stream_generation_prompt %}"
        "{{ ']\nAssistant:' }}"
        "{% endif %}"
    )
    template = template.replace('STREAM_PLACEHOLDER', stream_placeholder_jinja2)
    template = template.replace('STREAM_HIGH', stream_placeholder_high_jinja2)
    template = template.replace('ASK_HIGH', f"'{ask_high_token}'")
    return template

def chat_template_transition(tokenizer):
    return {
        (None, 'system'): tokenizer.bos_token,
        ('system', 'user'): '\n\nUser: ',
        ('system', 'stream'): '\n\n[',
        ('user', 'assistant'): '\nAssistant: ',
        ('user', 'stream'): '\n[',
        ('user', 'user'): '\nUser: ',
        ('assistant', 'user'): f'{tokenizer.eos_token}\nUser: ',
        ('assistant', 'stream'): f'{tokenizer.eos_token}\n[',
        ('stream', 'user'): ']\nUser: ',
        ('stream', 'assistant'): ']\nAssistant: ',
        'assistant': 'Assistant: ',
        'eos_token': tokenizer.eos_token,
    }

def chat_template_offsets(tokenizer):
    return {k:len(v) for k, v in chat_template_transition(tokenizer).items()}


def chat_template_transition_high(tokenizer):
    return {
        (None, 'system'): tokenizer.bos_token,
        ('system', 'user'): '\n\nUser: ',
        ('system', 'stream'): '\n\n[',
        ('user', 'assistant'): '\nAssistant: ',
        ('user', 'stream'): '\n[',
        ('user', 'user'): '\nUser: ',
        ('assistant', 'user'): f'{tokenizer.eos_token}\nUser: ',
        ('assistant', 'stream'): f'{tokenizer.eos_token}\n[',
        ('stream', 'stream_high'): '.',
        ('stream_high', 'user'): ']\nUser: ',
        ('stream_high', 'assistant'): ']\nAssistant: ',
        ('stream_high', 'stream'): ',',
        ('stream', 'user'): ']\nUser: ',
        ('stream', 'assistant'): ']\nAssistant: ',
        'assistant': 'Assistant: ',
        'eos_token': tokenizer.eos_token,
    }

def chat_template_offsets_high(tokenizer):
    return {k:len(v) for k, v in chat_template_transition_high(tokenizer).items()}

def get_learn_ranges(conversation: list[dict], *, chat_template_offsets: dict[tuple, int], model_config: LiveConfigMixin):
    offset = 0
    learn_ranges = []
    last_role = None
    for message in conversation:
        role = message['role']
        offset += chat_template_offsets[(last_role, role)]
        last_role = role
        if role == 'stream':
            if message.get('learn', False):
                ranges = get_stream_learn_ranges(message['num_frames'], model_config) + offset
                # the last one has ]\n, should also consider \n
                ranges[-1, 1] += 1
                if not isinstance(message['learn'], bool):
                    ranges = ranges[:message['learn']]
                learn_ranges.extend([range(r[0], r[1]) for r in ranges])
            offset += get_stream_placeholder_len(message['num_frames'], model_config)
        else:
            if role == 'assistant':
                if message.get('learn', False):
                    learn_ranges.append(range(offset - chat_template_offsets['assistant'], offset + len(message['content']) + chat_template_offsets['eos_token']))
            offset += len(message['content'])
    return learn_ranges

def get_learn_ranges_high(conversation: list[dict], *, chat_template_offsets: dict[tuple, int], model_config: LiveConfigMixin):
    offset = 0
    learn_ranges = []
    last_role = None
    for i, message in enumerate(conversation):
        role = message['role']
        offset += chat_template_offsets[(last_role, role)]
        last_role = role
        if role == 'stream':
            if message.get('learn', False):
                ranges = get_stream_learn_ranges(message['num_frames'], model_config) + offset
                if not isinstance(message['learn'], bool):
                    ranges = ranges[:message['learn']]
                learn_ranges.extend([range(r[0], r[1]) for r in ranges])
            offset += get_stream_placeholder_len(message['num_frames'], model_config)
        elif role == 'stream_high':
            if message.get('learn', False):
                ranges = get_stream_learn_ranges_high(message['num_frames'], model_config) + offset
                # Add extra offset only if there's a next message and it's assistant or user
                has_next = i + 1 < len(conversation)
                needs_extra_offset = has_next and conversation[i + 1]['role'] in ('assistant', 'user')
                ranges[-1, 1] += 1 if needs_extra_offset else 0
                if not isinstance(message['learn'], bool):
                    ranges = ranges[:message['learn']]
                learn_ranges.extend([range(r[0], r[1]) for r in ranges])
            offset += get_stream_placeholder_len_high(message['num_frames'], model_config)
        else:
            if role == 'assistant':
                if message.get('learn', False):
                    learn_ranges.append(range(offset - chat_template_offsets['assistant'], offset + len(message['content']) + chat_template_offsets['eos_token']))
            offset += len(message['content'])
    return learn_ranges

def build_live_tokenizer_and_update_config(llm_pretrained: str, model_config: LiveConfigMixin, **kwargs) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(llm_pretrained, use_fast=True, padding_side='left')
    tokenizer.add_special_tokens({'additional_special_tokens': [model_config.v_placeholder]})
    v_placeholder_id = len(tokenizer) - 1
    if model_config.frame_token_interval:
        frame_token_interval_id = tokenizer.convert_tokens_to_ids(model_config.frame_token_interval)
    else:
        frame_token_interval_id = None
    tokenizer.pad_token = tokenizer.eos_token
    model_config.update(dict(v_placeholder_id=v_placeholder_id, frame_token_interval_id=frame_token_interval_id, eos_token_id=tokenizer.eos_token_id))
    model_config.update(dict(gen_id=933))
    
    if 'live1_1+' in kwargs.get('live_version', None) or 'livel_h' in kwargs.get('live_version', None):
        # add high level stream token
        model_config.update(dict(frame_token_pooled_high=kwargs.get('frame_token_pooled_high')))
        
        tokenizer.add_special_tokens({'additional_special_tokens': [kwargs.get('high_v_placeholder')]})
        model_config.update(dict(high_v_placeholder_id=len(tokenizer) - 1, high_v_placeholder=kwargs.get('high_v_placeholder')))
        
        model_config.update(dict(frame_num_tokens_high=kwargs.get('frame_num_tokens_high')))
        
        high_frame_token_interval_id = tokenizer.convert_tokens_to_ids(kwargs.get('high_frame_token_interval'))
        model_config.update(dict(high_frame_token_interval_id=high_frame_token_interval_id, high_frame_token_interval=kwargs.get('high_frame_token_interval')))
        
        tokenizer.chat_template = chat_template_high(tokenizer, get_stream_placeholder_jinja2(model_config), get_stream_placeholder_high_jinja2(model_config),kwargs.get('high_frame_token_interval'))
        tokenizer.get_learn_ranges = partial(get_learn_ranges_high, chat_template_offsets=chat_template_offsets_high(tokenizer), model_config=model_config)
        
    else:
        tokenizer.chat_template = chat_template(tokenizer, get_stream_placeholder_jinja2(model_config))
        tokenizer.get_learn_ranges = partial(get_learn_ranges, chat_template_offsets=chat_template_offsets(tokenizer), model_config=model_config)
    return tokenizer

if __name__ == '__main__':
    args = parse_args()
    config = LiveConfigMixin.from_pretrained(args.llm_pretrained, **asdict(args))
    tokenizer = build_live_tokenizer_and_update_config(model_config=config, **asdict(args))
    chat = [
        {'role': 'system', 'content': 'cool.'},
        {'role': 'stream', 'num_frames': 2, 'learn': False},
        {'role': 'stream_high', 'num_frames': 1, 'learn': False},
        {'role': 'user', 'content': 'cool?'},
        {'role': 'assistant', 'content': 'cool.', 'learn': True},
        {'role': 'stream', 'num_frames': 3, 'learn': True},
        {'role': 'stream_high', 'num_frames': 1, 'learn': True},
        {'role': 'assistant', 'content': 'so cool.', 'learn': True},
    ]
    prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=False)
    print(prompt)
    learn_ranges = tokenizer.get_learn_ranges(chat)
    print(learn_ranges)
    batch = tokenizer([prompt], return_offsets_mapping=True, add_special_tokens=False, return_tensors="pt", padding=True)
    batch_labels = torch.full_like(batch.input_ids, -100, dtype=torch.long)
    for text, labels, input_ids, offset_mapping, learn_range in zip(
        [prompt], batch_labels, batch.input_ids, batch.offset_mapping, [learn_ranges]
    ):
        for learn_r in learn_range:
            start = torch.nonzero(offset_mapping[:,0] == learn_r.start).item()
            if offset_mapping[:,0][-1] >= learn_r.stop:
                stop = torch.nonzero(offset_mapping[:,0] == learn_r.stop).item()
            else: # the last eos token
                stop = len(input_ids)
            print(tokenizer.decode(input_ids[start:stop]))
            labels[start-1:stop-1] = input_ids[start:stop]
            # NOTE: input_ids may out of boundary of len(tokenizer) - 1. (1 is the added vision placeholder)
            # this is because some frames has v_placeholder_id target. so replace it with eos token.
            labels[labels > len(tokenizer) - 1] = tokenizer.eos_token_id
            labels[labels == tokenizer.convert_tokens_to_ids(args.high_v_placeholder)] = tokenizer.eos_token_id
            labels[labels == tokenizer.convert_tokens_to_ids('<v>')] = tokenizer.eos_token_id
    print(len(tokenizer)-1)
    print(batch.input_ids)
    print(batch_labels)
    print(tokenizer.decode(batch_labels[batch_labels!=-100]))