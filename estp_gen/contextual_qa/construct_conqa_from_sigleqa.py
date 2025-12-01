# load have process qa .json

import json, sys, os, re
from openai import OpenAI
import openai
from tqdm import tqdm


# gpt key: 
def assignIdx2QaList(qa_list):
    for i, qa in enumerate(qa_list):
        qa['idx'] = i
    return qa_list

def delete_caption(qa_list):
    for qa in qa_list:
        if "visual_cues" in qa.keys():
            del qa["visual_cues"]
        for conv in qa["conversation"]:
            if "caption" in conv.keys():
                del conv["caption"]
    return qa_list

# deepseek
# def get_llm_reponse_json(system_prompt, user_prompt):
#     client = OpenAI(
#             api_key="",
#             base_url="https://api.deepseek.com",
#         )

#     messages = [{"role": "system", "content": system_prompt},
#                 {"role": "user", "content": user_prompt}]

#     response = client.chat.completions.create(
#         model="deepseek-chat",
#         messages=messages,
#         # response_format={
#         #     'type': 'json_object'
#         # }
#     )
#     return response.choices[0].message.content


# doubao
def get_llm_reponse_json(system_prompt, user_prompt):
    client = OpenAI(
            api_key='',
            base_url="https://ark.cn-beijing.volces.com/api/v3",
        )

    messages = [{"role": "system", "content": system_prompt},
                {"role": "user", "content":  user_prompt}]

    response = client.chat.completions.create(
        model="ep-20250209171240-27r8g",
        messages=messages,
        response_format={
            'type': 'json_object'
        }
    )
    return response.choices[0].message.content


# openai
# proxychains -f /home/gentoo/geph.proxyconfig python /home/zhangyl/videollm-online/data/estp/contextual_qa/construct_conqa_from_sigleqa.py
# def get_llm_reponse_json(system_prompt, user_prompt):
#     client = openai.OpenAI(api_key='')

#     messages = [{"role": "system", "content": system_prompt},
#                 {"role": "user", "content": user_prompt}]

#     response = client.chat.completions.create(
#         model="gpt-4o-mini",
#         messages=messages,
#         # response_format={
#         #     'type': 'json_object'
#         # }
#     )
#     return response.choices[0].message.content

import random

processed_single_qa = json.load(open('/home/zhangyl/videollm-online/data/estp/annotation_train/0/estp.json'))
system_prompt = open('/home/zhangyl/videollm-online/data/estp/contextual_qa/from_sigle_qa_system_prompt.txt', 'r').read()
user_prompt_templete = open('/home/zhangyl/videollm-online/data/estp/contextual_qa/from_sigle_qa_user_prompt.txt', 'r').read()
output_dir = '/home/zhangyl/videollm-online/dataset/contextual_qa_single_train_v0_merge/'
os.makedirs(output_dir, exist_ok=True)

for i, video_uid in tqdm(enumerate(processed_single_qa)):
    print(i)
    for j, clip_uid in enumerate(processed_single_qa[video_uid]):
        print(j)
        output_prefix = os.path.join(output_dir, f'{video_uid}_{clip_uid}_{i}')
        if os.path.exists(f'{output_prefix}_qa.txt') and os.path.exists(f'{output_prefix}_group.json'):
            continue
        # if j > 1:
        #     break
        
        qa_list = processed_single_qa[video_uid][clip_uid]
        qa_list_w_idx = assignIdx2QaList(qa_list)
        qa_list_delete_caption = delete_caption(qa_list_w_idx)
        
        user_prompt = user_prompt_templete.format(json.dumps(qa_list_delete_caption, indent=4))
        try:
            answer = get_llm_reponse_json(system_prompt, user_prompt)
            answer = json.loads(answer.replace('```','').replace('json','').strip())
        except Exception as e:
            print(e)
            try:
                qa_num = len(qa_list_delete_caption)
                delete_ratio = 0.5
                left_num = int(qa_num * (1 - delete_ratio))
                left_num = 10
                left_idx = random.sample(range(qa_num), left_num)
                new_qa_list = []
                for idx, qa in enumerate(qa_list_delete_caption):
                    if idx in left_idx:
                        new_qa_list.append(qa)
                user_prompt = user_prompt_templete.format(json.dumps(new_qa_list, indent=4))
                
                answer = get_llm_reponse_json(system_prompt, user_prompt)
                answer = json.loads(answer.replace('```','').replace('json','').strip())
            except Exception as e:
                print(e)
                continue
                    

        with open(f'{output_prefix}_qa.txt', 'w') as f:
            f.write(user_prompt)
        with open(f'{output_prefix}_group.json', 'w') as f:
            json.dump(answer, f, indent=4)
    
    # if i > 5:
    #     break
        