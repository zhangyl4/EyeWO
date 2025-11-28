import os
from dataclasses import asdict
import argparse
import torch
from peft import PeftModel

from models import build_model_and_tokenizer, parse_args
from transformers import AutoTokenizer
# python -m models.merge_lora --live_version live1_1+ --resume_from_checkpoint chenjoya/videollm-online-8b-v1plus

def merge_lora_weights(
    output_path: str,
    **kwargs
):
    # 1. Load base model and lora weights
    model, tokenizer = build_model_and_tokenizer(
        is_training=False,
        **kwargs
    )
    tokenizer = AutoTokenizer.from_pretrained(kwargs['llm_pretrained'], use_fast=True, padding_side='left')
    model.config.vocab_size = len(tokenizer)
    
    print('load base model and lora weights done')
    # 2. Merge weights
    model = model.merge_and_unload()
    print('merge weights done')

    # 3. Save merged model
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    
    print('save merged model done')
    print(f"Successfully merged model saved to {output_path}")

if __name__ == "__main__":
    args = parse_args()
    output_path = '/2022233235/.cache/huggingface/hub/models--llamaEyeWO-stage2.5-livebase'
    merge_lora_weights(output_path, **asdict(args))