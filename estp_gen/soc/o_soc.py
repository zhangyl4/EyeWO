
import json
import os
import tqdm

EGO_ROOT = '/home/zhangyl/videollm-online/datasets/ego4d/'
CAPTION_ROOT = '/home/zhangyl/videollm-online/datasets/ego4d_action_caption/train_0_merge'
MOVE_CAPTION_ROOT = '/home/zhangyl/videollm-online/datasets/ego4d_move_action_caption/train_0_merge'
output_dir = '/home/zhangyl/videollm-online/dataset/soc_o_train_0_merge/'
video_uid_list = list(set(json.load(open("/home/zhangyl/videollm-online/data/estp/split_train_videoids/train_narration_video_uids_split_0.json"))))


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
narrations = annotation_loader.get_data()
origin_narrations = annotation_loader.get_origin_narration()

beta_alpha_calculator = BetaAlphaCalculator(narrations, alpha)
beta_alpha_calculator.compute_beta()
beta_map = beta_alpha_calculator.get_beta_map()
alpha = beta_alpha_calculator.get_alpha()

import json

main_data = json.load(open("/mnt/extra/dataset/ego4d/v2/annotations/fho_main.json"))
filtered_data = json.load(open("/home/zhangyl/videollm-online/data/estp/ego4d/filtered_data.json"))


def judge_other(src: str):
    # 1. remove #
    O_list = ['#O', '#o', ' O ', ' o ']
    for o in O_list:
        if o in src:
            return True

soc_annos = {}

for i, anno in enumerate(main_data['videos']):
    
    video_id = anno['video_uid']
    
    if video_id not in filtered_data.keys():
        continue
    
    soc_annos[video_id] = {}
    for clip in anno['annotated_intervals']:
        clip_uid = clip['clip_uid']
        
        soc_annos[video_id][clip_uid] = []
        for action_narration in clip['narrated_actions']:
            # if action_narration['is_valid_action']:
            if judge_other(action_narration['narration_text']):
                print(action_narration['narration_text'])
                
import json
from openai import OpenAI
import os

import random

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


def clean_text(src: str):
    # 1. remove #
    dst = src.replace('#C', '').replace('#c', '').replace('@c', '').replace('@C', '')
    dst = dst.replace('#O', '').replace('#o', '').replace('@o', '').replace('@O', '')
    dst = dst.replace('#Unsure', '').replace('#unsure', '')
    dst = dst.replace('#', '')
    # 2. remove start&end extra space and ,.
    dst = dst.strip('.,\n ') + '.'
    # 3. make the first word capitalize and remove extra space within the sentence
    words = dst.split()
    dst = ' '.join(words)
    
    return dst

system_prompt_judege = open('/home/zhangyl/videollm-online/data/estp/soc/fho_system_judge.txt').read().format(NUMBER=1)
system_prompt = open('/home/zhangyl/videollm-online/data/estp/soc/fho_system.txt').read().format(NUMBER=1)
user_prompt_templete = open('/home/zhangyl/videollm-online/data/estp/soc/fho_prompt.txt').read()

os.makedirs(output_dir, exist_ok=True)

soc_annos = {}


for i, anno in tqdm.tqdm(enumerate(main_data['videos'])):
    
    video_id = anno['video_uid']
    
    if video_id not in video_uid_list:
        continue
    
    if video_id not in filtered_data.keys():
        continue
    
    soc_annos[video_id] = {}
    for clip in anno['annotated_intervals']:
        clip_uid = clip['clip_uid']
        
        soc_annos[video_id][clip_uid] = []
        for action_narration in clip['narrated_actions']:
            if judge_other(action_narration['narration_text']):
                user_prompt = user_prompt_templete.format(clean_text(action_narration['narration_text']))
                response = get_llm_response_json(system_prompt_judege, user_prompt)
                
                if 'Yes' in response or 'yes' in response:
                    # print(response)
                    soc_annos[video_id][clip_uid].append(action_narration)
                

with open(os.path.join(output_dir, 'soc_select.json'), 'w') as f:
    json.dump(soc_annos, f, indent=4)

soc_annos = json.load(open(os.path.join(output_dir, 'soc_select.json')))
o_soc_annos = {}

for k, annos in tqdm.tqdm(soc_annos.items()):  
    for clip_id, clip_annos in annos.items():
        if len(clip_annos) == 0:
            continue
            
        for i, action in enumerate(clip_annos):
            os.makedirs(os.path.join(output_dir, k, clip_id), exist_ok=True)
            
            # HACK: add narration clip---------------
            if k not in o_soc_annos.keys():
                o_soc_annos[k] = {}
            if action['narration_annotation_uid'] not in o_soc_annos[k].keys():
                o_soc_annos[k][action['narration_annotation_uid']] = []
            
            summs = origin_narrations[k]['summaries']
            is_match = False
            for summ in summs:
                if summ['_annotation_uid'] == action['narration_annotation_uid']:
                    is_match = True
                    break
            if not is_match:
                continue
            
            if action['narration_annotation_uid'] not in filtered_data[k].keys():
                continue
            
            clip_start_time = summ['start_time']
            clip_end_time = summ['end_time']
            # HACK: add narration clip---------------
            
            user_prompt = user_prompt_templete.format(clean_text(action['narration_text']))
            response = get_llm_response_json(system_prompt, user_prompt)
            
            # HACK: add format json---------------
            responselist = response.split('\n')
            for res in responselist:
                if res.startswith('**Q:**'):
                    question = res.split('**Q:**')[1].strip()
                elif res.startswith('**A:**'):
                    answer = res.split('**A:**')[1].strip()
            
            qa = {
                'clip_start_time': clip_start_time,
                'clip_end_time': clip_end_time,
                'question': question,
                'Task Type': 'Object State Change Recognition',
                'conversation': [
                    {
                        'role': 'assistant',
                        'content': answer,
                        'time': action['narration_timestamp_sec'],
                        'start_time': action['start_sec'],
                        'end_time': action['end_sec'],
                    }
                ]
            }
            o_soc_annos[k][action['narration_annotation_uid']].append(qa) 
            # HACK: add format json---------------
            
            
            with open(os.path.join(output_dir, k, clip_id, f'{i}_q.txt'), 'w') as f:
                f.write(user_prompt)
            
            with open(os.path.join(output_dir, k, clip_id, f'{i}_gen.txt'), 'w') as f:
                f.write(response)

with open(os.path.join(output_dir, 'soc_o_train_0_merge.json'), 'w') as f:
    json.dump(o_soc_annos, f, indent=4)