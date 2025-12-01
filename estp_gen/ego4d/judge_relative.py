EGO_ROOT = '/home/zhangyl/videollm-online/datasets/ego4d/'
CAPTION_ROOT = '/home/zhangyl/videollm-online/datasets/ego4d_action_caption/valid_1'
MOVE_CAPTION_ROOT = '/home/zhangyl/videollm-online/datasets/ego4d_move_action_caption/valid_1'
SCENE_CAPTION_ROOT = '/home/zhangyl/videollm-online/datasets/ego4d_scene_caption'

import json
import os
import tqdm
import multiprocessing
from functools import partial

from video_caption_action_scene import AnnotationLoader, BetaAlphaCalculator

EGO_VERSION_ROOT = os.path.join(EGO_ROOT, 'v2')
json_path = os.path.join(EGO_ROOT, 'ego4d.json')
train_path = f'{EGO_VERSION_ROOT}/annotations/refined_narration_stream_train.json'
val_path = f'{EGO_VERSION_ROOT}/annotations/refined_narration_stream_val.json'
origin_path = f'{EGO_VERSION_ROOT}/annotations/all_narrations_redacted.json'
video_root = f'{EGO_VERSION_ROOT}/full_scale_2fps'
alpha = 4.9
device = 'cuda:3'

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

scene_train_caption = json.load(open(f'{SCENE_CAPTION_ROOT}/action_caption_train.json'))
scene_val_caption = json.load(open(f'{SCENE_CAPTION_ROOT}/action_caption_val.json'))
scene_all_caption = {**scene_train_caption, **scene_val_caption}

video2scene = json.load(open('/home/zhangyl/videollm-online/data/estp/ego4d/metafile/video2scene.json'))

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

def merge_narration(data, video_uid, clip_uid, video2scene, origin_narration, w_pre = False):
    narration = data[video_uid][clip_uid]
    
    caption_texts = {}
    for action_idx, nar in enumerate(narration):
        
        caption_texts[action_idx] = {
            'caption': nar['text'],
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

def read_qa(video_uid, clip_uid, qas):
    qa = qas[video_uid][clip_uid]
    return qa

def transformqa(qa):
    qa_text = []
    for q in qa:
        if 'visual_cues' in q:
            qa_text.append({
                'question': q['question'],
                # 'answer': q['conversation'][0]['content'],
                'visual_cues': q['visual_cues'],
            })
        else:
            qa_text.append({
                'question': q['question'],
                'answer': q['conversation'][0]['content'],
            })
    return qa_text

def qa_gentor(qas):
    for k,v in qas.items():
        for kk,vv in qas[k].items():
            yield k,kk
            
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

######### =============== judge_relative ================
anno_file = 'soc_valid_1'
postfix = 'judge_v1'
mode = 'narration'
prompt_version = 3
num_workers = 16
######### =============== judge_relative ================

qas = json.load(open(f'/home/zhangyl/videollm-online/data/estp/annotations/{anno_file}.json'))
_qa_gentor = qa_gentor(qas)
tasks = list(_qa_gentor)  # Convert generator to list for multiprocessing

output_dir = f'/home/zhangyl/videollm-online/dataset/{anno_file}_{postfix}_{mode}/'
os.makedirs(output_dir, exist_ok=True)
system_prompt = open(f'/home/zhangyl/videollm-online/data/estp/ego4d/prompt/judge_relative_system_prompt_v{prompt_version}.txt').read()
user_prompt = open('/home/zhangyl/videollm-online/data/estp/ego4d/prompt/judge_relative_user_prompt.txt').read()
step = 20


data_dict = {
    'all_caption': all_caption,
    'scene_all_caption': scene_all_caption,
    'move_all_caption': move_all_caption,
    'data': data,
    'video2scene': video2scene,
    'origin_narration': origin_narration,
    'qas': qas,
}

global_params = {
    'mode': mode,
    'anno_file': anno_file,
    'postfix': postfix,
    'system_prompt': system_prompt,
    'user_prompt': user_prompt,
    'step': 10,
    'data_dict': data_dict,
}


# for video_uid, clip_uid in tqdm.tqdm(_qa_gentor):
#     # try:
#     if mode == 'action':
#         caption_texts = merge_caption_wo_action(all_caption, video_uid, clip_uid, video2scene, origin_narration)
#     elif mode == 'scene':
#         caption_texts = merge_caption_wo_action(scene_all_caption, video_uid, clip_uid, video2scene, origin_narration)
#     elif mode == 'move_action':
#         caption_texts = merge_caption_wo_action(move_all_caption, video_uid, clip_uid, video2scene, origin_narration)
#     elif mode == 'narration':
#         caption_texts = merge_narration(data, video_uid, clip_uid, video2scene, origin_narration)
#     # except:
#     #     continue
    
#     try:
#         qa_list = transformqa(read_qa(video_uid, clip_uid, qas))
#     except:
#         continue

    
#     for i,qa in enumerate(qa_list):

#         if os.path.join(output_dir, f'{video_uid}_{clip_uid}_{i}_caption.txt') in os.listdir(output_dir):
#             continue
        
#         n_caption = len(caption_texts.keys())
#         final_answer = {}
#         for j in range(0, n_caption, step):
#             sample_caption_text = {k:v for k, v in caption_texts.items() if j <= k < j+step}
#             question = user_prompt.format(json.dumps(sample_caption_text,indent=4), json.dumps(qa, indent=4))
#             answer = get_llm_reponse_json(system_prompt, question)
#             final_answer.update(json.loads(answer))
        
#         question = user_prompt.format(json.dumps(caption_texts,indent=4), json.dumps(qa, indent=4))   
#         with open(os.path.join(output_dir, f'{video_uid}_{clip_uid}_{i}_caption.txt'), 'w') as f:
#             f.write(question)
#         with open(os.path.join(output_dir, f'{video_uid}_{clip_uid}_{i}_judge.json'), 'w') as f:
#             json.dump(final_answer, f, indent=4)

import logging
def setup_process_logger():
    """为每个进程设置独立日志文件"""
    pid = os.getpid()
    logger = logging.getLogger(f"Process-{pid}")
    logger.setLevel(logging.INFO)
    
    # 每个进程单独日志文件
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    handler = logging.FileHandler(f'{log_dir}/process_{pid}.log')
    formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

def process_task(task, global_params):
    logger = setup_process_logger()
    
    video_uid, clip_uid = task
    mode = global_params['mode']
    anno_file = global_params['anno_file']
    postfix = global_params['postfix']
    system_prompt = global_params['system_prompt']
    user_prompt = global_params['user_prompt']
    step = global_params['step']
    data = global_params['data_dict']
    
    output_dir = f'/home/zhangyl/videollm-online/dataset/{anno_file}_{postfix}_{mode}/'
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"Start processing: {video_uid}/{clip_uid}")
    
    try:
        if mode == 'action':
            caption_texts = merge_caption_wo_action(data['all_caption'], video_uid, clip_uid, data['video2scene'], data['origin_narration'])
        elif mode == 'scene':
            caption_texts = merge_caption_wo_action(data['scene_all_caption'], video_uid, clip_uid, data['video2scene'], data['origin_narration'])
        elif mode == 'move_action':
            caption_texts = merge_caption_wo_action(data['move_all_caption'], video_uid, clip_uid, data['video2scene'], data['origin_narration'])
        elif mode == 'narration':
            caption_texts = merge_narration(data['data'], video_uid, clip_uid, data['video2scene'], data['origin_narration'])
    except Exception as e:
        logger.info(f"Skipping {video_uid}/{clip_uid} due to error: {str(e)}")
        return
    
    
    try:
        qa_list = transformqa(read_qa(video_uid, clip_uid, data['qas']))
    except Exception as e:
        logger.info(f"Skipping QA processing for {video_uid}/{clip_uid}: {str(e)}")
        return
    
    logger.info(f"Processing {len(qa_list)} QA pairs")
    
    for i, qa in enumerate(qa_list):
        output_prefix = os.path.join(output_dir, f'{video_uid}_{clip_uid}_{i}')
        if os.path.exists(f'{output_prefix}_caption.txt'):
            continue
        
        n_caption = len(caption_texts)
        final_answer = {}
        for j in range(0, n_caption, step):
            sample = {k: v for k, v in caption_texts.items() if j <= k < j+step}
            question = user_prompt.format(json.dumps(sample, indent=4), json.dumps(qa, indent=4))
            try:
                answer = json.loads(get_llm_reponse_json(system_prompt, question))
                final_answer.update(answer)
            except Exception as e:
                print(f"Error processing captions {j}-{j+step} for {video_uid}/{clip_uid}: {str(e)}")
        
        with open(f'{output_prefix}_caption.txt', 'w') as f:
            f.write(user_prompt.format(json.dumps(caption_texts, indent=4), json.dumps(qa, indent=4)))
        with open(f'{output_prefix}_judge.json', 'w') as f:
            json.dump(final_answer, f, indent=4)
            
        logger.info(f"Saved results for QA {i}")
            
if __name__ == '__main__':
    # 主进程日志配置
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [Main] %(message)s',
        handlers=[logging.StreamHandler()]
    )

    
    try:
        logging.info(f"Starting {num_workers} workers")
        
        with multiprocessing.Pool(processes=num_workers) as pool:
            func = partial(process_task, global_params=global_params)
            
            # 使用tqdm显示进度
            for _ in tqdm.tqdm(
                pool.imap(func, tasks),
                total=len(tasks),
                desc="Total progress"
            ):
                pass
                
    except Exception as e:
        logging.error(f"Main process failed: {str(e)}")
    finally:
        logging.info("All tasks completed")