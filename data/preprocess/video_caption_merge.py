# import json
# import os
# import tqdm
# from dataclasses import dataclass
# from tqdm import tqdm



# class captionMerger:
#     def __init__(self, device='cuda:4', prompt_file='/root/videollm-online/data/preprocess/prompt/caption_merge.txt') -> None:
#         from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser
#         self.tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3-8B-Instruct', use_fast=True)
#         self.model = AutoModelForCausalLM.from_pretrained('meta-llama/Meta-Llama-3-8B-Instruct', torch_dtype='auto', attn_implementation='sdpa')
#         self.model.to(device)
#         self.model.eval()
#         self.prompt = open(prompt_file, 'r').read()
#         self.device = device
        
    
#     def merge_wlast(self, caption, last_caption):
#         conversation = [
#             {'role': 'user', 'content': self.prompt.format(caption, last_caption)},
#         ]
#         print(conversation)

#         input_ids = self.tokenizer.apply_chat_template(conversation, return_tensors='pt', add_generation_prompt=True).to(self.device)
#         output_ids = self.model.generate(input_ids, max_length=8192)[:,input_ids.size(1):]
#         answer = self.tokenizer.decode(output_ids[0])
#         print(answer)
#         return answer
    
#     def merge(self, caption_list):
#         old_caption  = caption_list[0]
#         for last_cap in caption_list[1:]:
#             old_caption = self.merge_wlast(old_caption, last_cap)
#         return old_caption


# caption_merger = captionMerger(device='cuda:5')

# class captionLoader:
#     def __init__(self, caption_dir) -> None:
#         self.caption_dir = caption_dir
    
#     def load(self, vdieo_uid, clip_idx):
#         # read txt
#         caption_list = []
#         for file in sorted(os.listdir(os.path.join(self.caption_dir, vdieo_uid, clip_idx))):
#             if file.endswith('.txt') and 'merge' not in file:
#                 with open(os.path.join(self.caption_dir,vdieo_uid,clip_idx,file), 'r') as f:
#                     caption_list.append(''.join(f.readlines()[15:]))
#         return caption_list
    
# class pipelineMain:
#     def __init__(self, caption_dir, caption_merger, caption_loader) -> None:
#         self.caption_merger = caption_merger
#         self.caption_loader = caption_loader
#         self.caption_dir = caption_dir
    
#     def run(self):
#         for dir in tqdm(sorted(os.listdir(self.caption_dir))):
#             if not os.path.isdir(os.path.join(self.caption_dir, dir)):
#                 continue
#             for file in sorted(os.listdir(os.path.join(self.caption_dir, dir))):
#                 if not os.path.isdir(os.path.join(self.caption_dir, dir, file)):
#                     continue
#                 caption_list = self.caption_loader.load(dir, file)
#                 print(os.path.join(self.caption_dir, dir, file, 'caption_list.json'))
#                 json.dump(caption_list, open(os.path.join(self.caption_dir, dir, file, 'caption_list.json'), 'w'), indent=4)
#                 merged_caption = self.caption_merger.merge(caption_list)
#                 print(merged_caption)
#                 with open(os.path.join(self.caption_dir, dir, file, 'merged.txt'), 'w') as f:
#                     f.write(merged_caption)

# caption_dir = '/root/videollm-online/tmp4'
# caption_loader = captionLoader(caption_dir)
# p = pipelineMain(caption_dir, caption_merger, caption_loader)
# p.caption_loader = captionLoader(caption_dir)
# p.run()


import os
import json
path = '/2022233235/datasets/ego4d_action_caption/'
# path = '/2022233235/datasets/ego4d_scene_caption/'
# path = '/2022233235/datasets/ego4d_move_action_caption/'
version = 'train_0'
dir = os.path.join(path, version)
action_caption = {}


for v in os.listdir(dir):
    if not os.path.isdir(os.path.join(dir, v)):
        continue
    action_caption[v] = {}
    for c in os.listdir(os.path.join(dir, v)):
        if c.endswith('.json'):
            with open(os.path.join(dir, v, c), 'r') as f:
                try:
                    caption = json.load(f)
                except:
                    continue
                if len(caption) <= 1:
                    continue
                for cap in caption:
                    cap['text'] = cap['caption']
                    cap['time'] = cap['end_time']
                action_caption[v][c.split('.')[0]] = caption

train_ratio = 0.7
import random
train_video_uid = random.sample(list(action_caption.keys()), int(len(action_caption) * train_ratio))
val_video_uid = list(set(action_caption.keys()) - set(train_video_uid)) 
train_action_caption = {}
val_action_caption = {}
for v in train_video_uid:
    train_action_caption[v] = action_caption[v]
for v in val_video_uid:
    val_action_caption[v] = action_caption[v]


# os.makedirs(os.path.join(path,version + "_merge"), exist_ok=True)
# with open(os.path.join(path,version + "_merge", f'action_caption_train.json'), 'w') as f:
#     json.dump(train_action_caption, f, indent=4)

# with open(os.path.join(path, version + "_merge", f'action_caption_val.json'), 'w') as f:
#     json.dump(val_action_caption, f, indent=4)
    
c = 0
for k ,v in train_action_caption.items():
    c += len(v)
for k ,v in val_action_caption.items():
    c += len(v)
print(c)