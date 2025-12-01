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

qas = json.load(open('/home/zhangyl/videollm-online/data/estp/annotations/move_action_function_v8.json'))


def merge_caption_wo_action(captions, video_uid, clip_uid, video2scene, origin_narration, w_pre = False):
    caption = captions[video_uid][clip_uid]
    
    caption_texts = {}
    for action_idx, cap in enumerate(caption):
        
        caption_texts[action_idx] = {
            'caption': cap['caption'],
            'reason': None,
            'is_relational': None,
        }
        
        
    return caption_texts

def merge_caption_with_action(captions, narrations, video_uid, clip_uid, video2scene, origin_narration):
    narration = narrations[video_uid][clip_uid]
    caption = captions[video_uid][clip_uid]
    
    caption_texts = ""
    for action_idx, (nar, cap) in enumerate(zip(narration, caption)):
        action_narration = 'Idx is {}, Time is {}. Action narration is \"'.format(action_idx, nar['time']) + nar['text'] + '\".\n'
        caption_text = 'Detailed Description: \"' + cap['text'] + '\" \n'
        caption_text = action_narration + caption_text
        caption_texts += caption_text + '\n'
    
    return caption_texts

def caption_merger(captions, video2scene, origin_narration):
    for video_uid in captions.keys():
        for clip_uid in captions[video_uid].keys():
            caption_texts = merge_caption_wo_action(captions, video_uid, clip_uid, video2scene, origin_narration)
            yield caption_texts, video_uid, clip_uid

caption_merger1 = caption_merger(move_all_caption, video2scene, origin_narration)

def read_qa(video_uid, clip_uid, qas):
    qa = qas[video_uid][clip_uid]
    return qa

def transformqa(qa):
    qa_text = []
    for q in qa:
        qa_text.append({
            'question': q['Question'],
            'answer': q['conversation'][0]['content'],
        })
    return qa_text


import json
from openai import OpenAI

def get_llm_reponse_json(system_prompt, user_prompt):
    client = OpenAI(
            api_key="",
            base_url="https://api.deepseek.com",
        )

    messages = [{"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}]

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        response_format={
            'type': 'json_object'
        }
    )
    return response.choices[0].message.content

mode = 'action'
output_dir = f'/home/zhangyl/videollm-online/dataset/move_action_function_v8_judge_v1_{mode}/'
os.makedirs(output_dir, exist_ok=True)
system_prompt = open('/home/zhangyl/videollm-online/data/estp/ego4d/prompt/judge_relative_system_prompt_v3.txt').read()
user_prompt = open('/home/zhangyl/videollm-online/data/estp/ego4d/prompt/judge_relative_user_prompt.txt').read()
step = 10
from tqdm import tqdm
for n_sample in tqdm(range(0,50)):
    caption_texts, video_uid, clip_uid = next(caption_merger1)
    if mode == 'action':
        caption_texts = merge_caption_wo_action(all_caption, video_uid, clip_uid, video2scene, origin_narration)
        
    for i,qa in enumerate(transformqa(read_qa(video_uid, clip_uid, qas))):

        
        n_caption = len(caption_texts.keys())
        final_answer = {}
        for j in range(0, n_caption, step):
            sample_caption_text = {k:v for k, v in caption_texts.items() if j <= k < j+step}
            question = user_prompt.format(json.dumps(sample_caption_text,indent=4), json.dumps(qa, indent=4))
            answer = get_llm_reponse_json(system_prompt, question)
            final_answer.update(json.loads(answer))
        
        question = user_prompt.format(json.dumps(caption_texts,indent=4), json.dumps(qa, indent=4))   
        with open(os.path.join(output_dir, f'{video_uid}_{clip_uid}_{i}_caption.txt'), 'w') as f:
            f.write(question)
        with open(os.path.join(output_dir, f'{video_uid}_{clip_uid}_{i}_judege.json'), 'w') as f:
            json.dump(final_answer, f, indent=4)
        