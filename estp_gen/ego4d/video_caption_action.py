
from transformers import AutoModel, AutoTokenizer
import torch
import json
import os
import tqdm
import decord
from decord import VideoReader
decord.bridge.set_bridge("torch")
import numpy as np
import math
from PIL import Image
import matplotlib.pyplot as plt
import math

import spacy
nlp = spacy.load("en_core_web_sm")

def sentene2verb(sentence):
    
    doc = nlp(sentence)
    verbs = []
    for token in doc:
        if token.pos_ == "VERB":
            verb_phrase = token.lemma_
            verbs.append(verb_phrase)
    return verbs

def sentene2n(sentence):

    doc = nlp(sentence)
    verbs = []
    for token in doc:
        if token.pos_ == "NOUN":
            verb_phrase = token.lemma_
            verbs.append(verb_phrase)
    return verbs

def ceil_time_by_fps(time: float, fps: int, min_time: float, max_time: float):
    return min(max(math.ceil(time * fps) / fps, min_time), max_time)

def show_image(load_range, frames, output_path=None):
    frames_per_row = 7

    # 计算行数
    rows = math.ceil(len(load_range) / frames_per_row)

    # 创建子图
    fig, axes = plt.subplots(rows, frames_per_row, figsize=(frames_per_row * 4, rows * 4))

    # 将 frames 绘制到子图中
    for i in range(len(load_range)):
        row = i // frames_per_row
        col = i % frames_per_row
        if rows == 1:
            axes[col].imshow(frames[i])
            axes[col].axis('off')
            axes[col].set_title(f"Frame {i}")
        else:
            axes[row, col].imshow(frames[i])
            axes[row, col].axis('off')
            axes[row, col].set_title(f"Frame {i}")

    # 如果最后一行有空的子图格子，关闭它们
    for i in range(len(load_range), rows * frames_per_row):
        fig.delaxes(axes.flatten()[i])

    if output_path is not None:
        plt.savefig(output_path)
    else:
        plt.tight_layout()
        plt.show()


class AnnotationLoader:
    def __init__(self, train_path, val_path):
        self.train_data = json.load(open(train_path))
        self.val_data = json.load(open(val_path))
        self.data = {**self.train_data, **self.val_data}
    
    def get_data(self):
        return self.data

class BetaAlphaCalculator:
    def __init__(self, data, alpha=4.9):
        self.data = data
        self.beta_map = {}
        self.alpha = alpha
    
    def compute_beta(self):
        for video_uid, annotation_uid_narrations in self.data.items():
            for annotation_uid, narrations in annotation_uid_narrations.items():
                if len(narrations) == 0:
                    continue
                total_time = 0
                for i in range(len(narrations) - 1):
                    total_time += narrations[i+1]['time'] - narrations[i]['time']
                self.beta_map[annotation_uid] = total_time / len(narrations)
    
    def get_beta_map(self):
        return self.beta_map
    
    def get_alpha(self):
        return self.alpha

class VideoProcessor:
    def __init__(self, data, beta_map, alpha, video_root, frame_fps=2):
        self.data = data
        self.beta_map = beta_map
        self.alpha = alpha
        self.video_root = video_root
        self.frame_fps = frame_fps
    
    def action2clip(self, path, clip_idx, action_idx):
        annotation_uids = list(self.data[path].keys())
        clip_id = annotation_uids[clip_idx]
        narration = self.data[path][clip_id][action_idx]
        stamp_time = narration['time']
        beta = self.beta_map.get(clip_id, 0)
        start_time = stamp_time - beta / (2 * self.alpha)
        end_time = stamp_time + beta / (2 * self.alpha)
        return stamp_time, start_time, end_time, clip_id
    

    def load_action_clip(self, path, clip_idx, action_idx):
        stamp_time, start_time, end_time, clip_id = self.action2clip(path, clip_idx, action_idx)
        narration = self.data[path][clip_id][action_idx]['text']
        
        video_path = os.path.join(self.video_root, f"{path}.mp4")
        vr = VideoReader(video_path)
        start_frame = int(ceil_time_by_fps(start_time, self.frame_fps, 0, (vr._num_frame-1) / self.frame_fps) * self.frame_fps)
        end_frame = int(ceil_time_by_fps(end_time, self.frame_fps, 0, (vr._num_frame-1) / self.frame_fps)* self.frame_fps) # no need to add 1
        
        load_range = range(start_frame, end_frame)
        frames = vr.get_batch(load_range)
        frames = [Image.fromarray(frame.numpy().astype('uint8')) for frame in frames]
        
        return narration, frames, stamp_time, start_time, end_time, load_range

            

class CaptionGenerator:
    def __init__(self, model_name, tokenizer_name, device='cuda:4', dtype=torch.bfloat16):
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True,
                                               attn_implementation='sdpa', torch_dtype=dtype)
        self.model.eval()
        self.model.to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
            
    def get_caption(self, frames, question):
        msgs = [
            {'role': 'user', 'content': frames + [question]}, 
        ]
        
        # Set decode params for video
        params={}
        params["use_image_id"] = False
        params["max_slice_nums"] = 2 # use 1 if cuda OOM and video resolution >  448*448

        answer = self.model.chat(
            image=None,
            msgs=msgs,
            tokenizer=self.tokenizer,
            **params
        )
        return question, answer
    

class PromptGenerator:
    def __init__(self, prompt_file):
        self.prompt = open(prompt_file).read()
    
    
class PromptGeneratorExpandAction(PromptGenerator):
    def __init__(self, prompt_file):
        super().__init__(prompt_file)
    
    def get_prompt(self, action_narration):
        actions = ' , '.join(sentene2verb(action_narration))
        o = sentene2n(action_narration)
        if 'man' in o:
            o.remove('man')
        if 'woman' in o:
            o.remove('woman')
        if len(o) > 0:
            objects = ' , '.join(o)
        else:
            objects = 'Describe the surrounding environment, objects, and people.'
        return self.prompt.format(action_narration, actions, objects)



def main():
    # 配置路径和参数
    train_path = '/root/videollm-online/datasets/ego4d/v2/annotations/refined_narration_stream_train.json'
    val_path = '/root/videollm-online/datasets/ego4d/v2/annotations/refined_narration_stream_val.json'
    video_root = '/root/videollm-online/datasets/ego4d/v2/full_scale_2fps'
    output_dir = 'tmp5'
    prompt_file = '/root/videollm-online/data/preprocess/prompt/caption_expand.txt'
    video2scene = json.load(open('/root/videollm-online/data/preprocess/metafile/video2scene.json'))
    video_uid_list = open('/root/videollm-online/data/preprocess/metafile/major2scene_case.txt').read().split('\n')
    alpha = 4.9
    
    # 初始化各个模块
    annotation_loader = AnnotationLoader(train_path, val_path)
    data = annotation_loader.get_data()
    
    beta_alpha_calculator = BetaAlphaCalculator(data, alpha)
    beta_alpha_calculator.compute_beta()
    beta_map = beta_alpha_calculator.get_beta_map()
    alpha = beta_alpha_calculator.get_alpha()
    
    video_processor = VideoProcessor(data, beta_map, alpha, video_root)
    caption_generator = CaptionGenerator('openbmb/MiniCPM-V-2_6', 'openbmb/MiniCPM-V-2_6')
    prompt_generator = PromptGeneratorExpandAction(prompt_file)
    # 处理每个视频
    for path in tqdm.tqdm(video_uid_list):
        if path not in data:
            continue
        annotation_uid_narrations = data[path]
        
        for clip_idx, (annotation_uid, narrations) in enumerate(annotation_uid_narrations.items()):
            for action_idx, narration in enumerate(narrations):
                # 加载剪辑
                action_narration, frames, stamp_time, start_time, end_time, load_range = video_processor.load_action_clip(path, clip_idx, action_idx)
                
                # 保存图像
                os.makedirs(f'{output_dir}/{path}/{annotation_uid}', exist_ok=True)
                show_image(load_range, frames, f'{output_dir}/{path}/{annotation_uid}/{action_idx}.png')
                
                # 生成描述
                question = prompt_generator.get_prompt(action_narration)
                question, answer = caption_generator.get_caption(frames, question)
                
                # 保存描述
                with open(f'{output_dir}/{path}/{annotation_uid}/{action_idx}.txt', 'w') as f:
                    f.write(question + '\n')
                    f.write(answer)
            break  # 如果只需要处理第一个剪辑，可以删除这行

if __name__ == '__main__':
    main()