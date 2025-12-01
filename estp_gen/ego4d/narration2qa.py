

import json
import os
import tqdm

from video_caption_action_scene import AnnotationLoader, BetaAlphaCalculator


"""============================================= config ================================================= """
EGO_ROOT = '/home/zhangyl/videollm-online/datasets/ego4d/'
CAPTION_ROOT = '/home/zhangyl/videollm-online/datasets/ego4d_action_caption/train_0_merge'
MOVE_CAPTION_ROOT = '/home/zhangyl/videollm-online/datasets/ego4d_move_action_caption/train_0_merge'
output_dir = '/home/zhangyl/videollm-online/dataset/special_action_train_0_merge/'
video_uid_list = list(set(json.load(open("/home/zhangyl/videollm-online/data/estp/split_train_videoids/train_narration_video_uids_split_0.json"))))
"""==================================================================================================================== """


EGO_VERSION_ROOT = os.path.join(EGO_ROOT, 'v2')
json_path = os.path.join(EGO_ROOT, 'ego4d.json')
train_path = f'{EGO_VERSION_ROOT}/annotations/refined_narration_stream_train.json'
val_path = f'{EGO_VERSION_ROOT}/annotations/refined_narration_stream_val.json'
origin_path = f'{EGO_VERSION_ROOT}/annotations/all_narrations_redacted.json'
video_root = f'{EGO_VERSION_ROOT}/full_scale_2fps'

alpha = 4.9
device = 'cuda:3'

annotation_loader = AnnotationLoader(train_path, val_path, origin_path, json_path)
narrations = annotation_loader.get_data()
origin_narrations = annotation_loader.get_origin_narration()

beta_alpha_calculator = BetaAlphaCalculator(narrations, alpha)
beta_alpha_calculator.compute_beta()
beta_map = beta_alpha_calculator.get_beta_map()
alpha = beta_alpha_calculator.get_alpha()
filtered_data = json.load(open('/home/zhangyl/videollm-online/data/estp/ego4d/filtered_data.json'))

video2scene = json.load(open('/home/zhangyl/videollm-online/data/estp/ego4d/metafile/video2scene.json'))
def merge_narration_wo_action(narrations, video_uid, clip_uid, video2scene, origin_narration):
    narration = narrations[video_uid][clip_uid]
    
    caption_texts = ""
    for action_idx, nar in enumerate(narration):

        action_narration = 'Time is {}.\n'.format(nar['time'])
        caption_text = 'Action Narration: \"' + nar['text'] + '\"\n'
        caption_text = action_narration + caption_text
        caption_texts += caption_text + '\n'
        
    return caption_texts

def narration_merger(narrations, video2scene, origin_narrations, filtered_data):
    for video_uid in filtered_data.keys():
        for clip_uid in filtered_data[video_uid].keys():
            caption_texts = merge_narration_wo_action(narrations, video_uid, clip_uid, video2scene, origin_narrations)
            yield caption_texts, video_uid, clip_uid

caption_merger1 = narration_merger(narrations, video2scene, origin_narrations, filtered_data)

import json
from openai import OpenAI
import os

# def get_llm_response_json(system_prompt, user_prompt):

#     client = OpenAI(
#         api_key="",
#         base_url="https://api.deepseek.com",
#     )
    
#     messages = [{"role": "system", "content": system_prompt},
#             {"role": "user", "content": user_prompt}]
    
#     response = client.chat.completions.create(
#         model="deepseek-chat",
#         messages=messages,
#     )
    
#     return response.choices[0].message.content

def get_llm_response_json(system_prompt, user_prompt):
    client = OpenAI(
            api_key='',
            base_url="https://ark.cn-beijing.volces.com/api/v3",
        )

    messages = [{"role": "system", "content": system_prompt},
                {"role": "user", "content":  user_prompt}]

    response = client.chat.completions.create(
        model="ep-20250209171240-27r8g",
        messages=messages,
        # response_format={
        #     'type': 'json_object'
        # }
    )
    return response.choices[0].message.content

system_prompt = open('/home/zhangyl/videollm-online/data/estp/ego4d/prompt/narration2saction_system.txt').read().format(NUMBER=1)
user_prompt_templete = open('/home/zhangyl/videollm-online/data/estp/ego4d/prompt/narration2saction_prompt.txt').read()

os.makedirs(output_dir, exist_ok=True)



# for video_uid in tqdm.tqdm(video_uid_list):
#     for clip_uid in narrations[video_uid].keys():
        
#         caption_texts = merge_narration_wo_action(narrations, video_uid, clip_uid, video2scene, origin_narrations)

#         user_prompt = user_prompt_templete.format(caption_texts)
#         answer = get_llm_response_json(system_prompt, user_prompt)

#         with open(os.path.join(output_dir, f'{video_uid}_{clip_uid}_caption.txt'), 'w') as f:
#             f.write(user_prompt)
#         with open(os.path.join(output_dir, f'{video_uid}_{clip_uid}_qa.txt'), 'w') as f:
#             f.write(answer)

def process_video(video_uid):
    from tqdm import tqdm  
    
    
    for clip_uid in narrations[video_uid].keys():
        try:
            caption_texts = merge_narration_wo_action(narrations, video_uid, clip_uid, video2scene, origin_narrations)
            
            user_prompt = user_prompt_templete.format(caption_texts)
            answer = get_llm_response_json(system_prompt, user_prompt)

            with open(os.path.join(output_dir, f'{video_uid}_{clip_uid}_caption.txt'), 'w') as f:
                f.write(user_prompt)
            with open(os.path.join(output_dir, f'{video_uid}_{clip_uid}_qa.txt'), 'w') as f:
                f.write(answer)
                
        except Exception as e:
            print(f"Error processing {video_uid}-{clip_uid}: {str(e)}")
            continue
        
if __name__ == "__main__":
    from multiprocessing import Pool
    import tqdm
    
    # 设置进程数（根据GPU数量调整）
    num_processes = 8  # 与cuda设备数量匹配
    
    # 创建进程池
    with Pool(processes=num_processes) as pool:
        # 使用imap保持顺序，使用tqdm显示进度
        results = list(tqdm.tqdm(
            pool.imap(process_video, video_uid_list),
            total=len(video_uid_list),
            desc="Processing Videos"
        ))