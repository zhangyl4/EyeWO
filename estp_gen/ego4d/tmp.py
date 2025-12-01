EGO_ROOT = '/home/zhangyl/videollm-online/datasets/ego4d/'
CAPTION_ROOT = '/home/zhangyl/videollm-online/datasets/ego4d_action_caption'
MOVE_CAPTION_ROOT = '/home/zhangyl/videollm-online/datasets/ego4d_move_action_caption'

import json
import os
import tqdm

from video_caption_action_scene import AnnotationLoader, BetaAlphaCalculator

EGO_VERSION_ROOT = os.path.join(EGO_ROOT, 'v2')
json_path = os.path.join(EGO_ROOT, 'ego4d.json')
train_path = f'{EGO_VERSION_ROOT}/annotations/refined_narration_stream_train.json'
val_path = f'{EGO_VERSION_ROOT}/annotations/refined_narration_stream_val.json'
origin_path = f'{EGO_VERSION_ROOT}/annotations/all_narrations_redacted.json'
video_root = f'{EGO_VERSION_ROOT}/full_scale_2fps'
alpha = 4.9
device = 'cuda:3'
caption_dir = '/root/videollm-online/tmp5'

annotation_loader = AnnotationLoader(train_path, val_path, origin_path, json_path)
data = annotation_loader.get_data()
origin_narration = annotation_loader.get_origin_narration()

beta_alpha_calculator = BetaAlphaCalculator(data, alpha)
beta_alpha_calculator.compute_beta()
beta_map = beta_alpha_calculator.get_beta_map()
alpha = beta_alpha_calculator.get_alpha()

train_caption = json.load(open(f'{CAPTION_ROOT}/action_caption_train.json'))
val_caption = json.load(open(f'{CAPTION_ROOT}/action_caption_val.json'))
all_caption = {**train_caption, **val_caption}

move_train_caption = json.load(open(f'{MOVE_CAPTION_ROOT}/action_caption_train.json'))
move_val_caption = json.load(open(f'{MOVE_CAPTION_ROOT}/action_caption_val.json'))
move_all_caption = {**move_train_caption, **move_val_caption}

video2scene = json.load(open('/home/zhangyl/videollm-online/data/estp/ego4d/metafile/video2scene.json'))

def merge_caption_wo_action(captions, video_uid, clip_uid, video2scene, origin_narration, w_pre = False):
    caption = captions[video_uid][clip_uid]
    
    caption_texts = ""
    for action_idx, cap in enumerate(caption):
        
        if 'stamp_time' not in cap and w_pre:
            action_narration = 'Time is {} - {}.\n'.format(cap['start_time'], cap['end_time'])
        elif 'stamp_time' not in cap and not w_pre:
            continue
        else:
            action_narration = 'Time is {}.\n'.format(cap['stamp_time'])
        caption_text = 'Detailed Description: \"' + cap['text'] + '\".\n'
        caption_text = action_narration + caption_text
        caption_texts += caption_text + '\n\n'
        
    return caption_texts


def caption_merger(captions, video2scene, origin_narration, w_pre = False):
    for video_uid in captions.keys():
        for clip_uid in captions[video_uid].keys():
            caption_texts = merge_caption_wo_action(captions, video_uid, clip_uid, video2scene, origin_narration, w_pre = False)
            yield caption_texts, video_uid, clip_uid



caption_merger1 = caption_merger(move_all_caption, video2scene, origin_narration)

import json
from openai import OpenAI
import os

def get_llm_response_json(system_prompt, user_prompt):

    client = OpenAI(
        api_key="",
        base_url="https://api.deepseek.com",
    )
    
    messages = [{"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}]
    
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
    )
    
    return response.choices[0].message.content

system_prompt = open('/home/zhangyl/videollm-online/data/estp/ego4d/prompt/caption2qa_system.txt').read().format(NUMBER=5)
user_prompt_templete = open('/home/zhangyl/videollm-online/data/estp/ego4d/prompt/caption2qa_prompt.txt').read()
output_dir = '/home/zhangyl/videollm-online/dataset/move_action_v18/'
os.makedirs(output_dir, exist_ok=True)

system_prompt_a = open('/home/zhangyl/videollm-online/data/estp/ego4d/prompt/action_caption2a_system.txt').read()
user_prompt_templete_a = open('/home/zhangyl/videollm-online/data/estp/ego4d/prompt/action_caption2a_prompt.txt').read()

import tqdm

for i in tqdm.tqdm(range(0,100)):

    caption_texts, video_uid, clip_uid = next(caption_merger1)

    user_prompt = user_prompt_templete.format(caption_texts)
    answer = get_llm_response_json(system_prompt, user_prompt)

    with open(os.path.join(output_dir, f'{video_uid}_{clip_uid}_caption.txt'), 'w') as f:
        f.write(user_prompt)
    with open(os.path.join(output_dir, f'{video_uid}_{clip_uid}_qa.txt'), 'w') as f:
        f.write(answer)
    
    
        
        