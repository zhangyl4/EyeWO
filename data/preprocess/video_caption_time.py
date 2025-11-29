
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
from PIL import Image, ImageDraw, ImageFont
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
    def __init__(self, train_path, val_path, origin_path):
        self.train_data = json.load(open(train_path))
        self.val_data = json.load(open(val_path))
        self.data = {**self.train_data, **self.val_data}
        
        self.origin_narration = json.load(open(origin_path))['videos']
        
    def get_data(self):
        return self.data
    
    def get_origin_narration(self):
        return self.origin_narration

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
    def __init__(self, data, origin_narration, beta_map, alpha, video_root, frame_fps=2):
        self.data = data
        self.origin_narration = origin_narration
        self.beta_map = beta_map
        self.alpha = alpha
        self.video_root = video_root
        self.frame_fps = frame_fps
        
        from siglip import visionTextAligner
        self.aliger = visionTextAligner()
    
    
    def load_scene_clipv2(self, path, clip_idx, max_frame=32,):

        annotation_uids = list(self.data[path].keys())
        clip_id = annotation_uids[clip_idx]
        
        # load clip
        summs = self.origin_narration[path]['summaries']
        for summ in summs:
            if summ['_annotation_uid'] == clip_id:
                break
            
        start_time, end_time = summ['start_time'], summ['end_time']
        vr = VideoReader(uri=os.path.join(self.video_root, path) + '.mp4')
        start_frame = int(ceil_time_by_fps(start_time, self.frame_fps, 0, vr._num_frame / self.frame_fps) * self.frame_fps)
        end_frame = int(ceil_time_by_fps(end_time, self.frame_fps, 0, vr._num_frame / self.frame_fps)* self.frame_fps) + 1
        load_range = range(start_frame, end_frame)
        frames = vr.get_batch(load_range)
        
        # vision simi
        simi = self.aliger.vision_simi(frames)
        frames = vr.get_batch(load_range)
        frames = [Image.fromarray(v.astype('uint8')) for v in frames.numpy()]
        if simi > 0.8:
            if len(frames) > max_frame:
                # uniformly sample frames
                step = math.ceil(len(frames) / max_frame)
                frames = frames[::step]
                # save frame info
                
                frames = VideoProcessor.add_frame_info(frames, start_frame, self.frame_fps / step)
                
                load_range = range(0,len(frames))
                yield frames, start_frame, end_frame, load_range, (self.frame_fps / step)
        elif simi < 0.6:
            for i in range(0, len(frames), max_frame):
                r_f = frames[i:i+max_frame]
                
                r_f = VideoProcessor.add_frame_info(r_f, start_time + i / self.frame_fps, self.frame_fps)
                yield r_f, start_frame + i, start_frame + (i + len(r_f)), range(0,len(r_f)), self.frame_fps
        else:
            step = 1
            if len(frames) > max_frame:
                # uniformly sample frames
                frames = frames[::self.frame_fps*2]
                step = (self.frame_fps*2)
            for i in range(0, len(frames), max_frame):
                r_f = frames[i:i+max_frame]
                
                r_f = VideoProcessor.add_frame_info(r_f, start_time + i / self.frame_fps, self.frame_fps / step)
                yield r_f, start_frame + i, start_frame + (i + len(r_f)) / self.frame_fps, range(0,len(r_f)), (self.frame_fps / step)

    @staticmethod
    def add_frame_info(frames, start_frame, frame_fps):
        """
        在每一帧左上角添加时间或帧编号信息。
        
        参数:
            frames (list): 视频帧列表，每一帧是一个 PIL Image 对象。
            start_frame (int): 起始帧编号。
            frame_fps (int): 视频的帧率。
        
        返回:
            list: 添加了信息的帧列表。
        """
        font = ImageFont.truetype("/root/videollm-online/data/preprocess/font/ARIAL.TTF", size=100) 
        annotated_frames = []
        for idx, frame in enumerate(frames):
            current_frame = start_frame + idx
            time_seconds = current_frame / frame_fps
            time_text = f"{time_seconds:.2f}s"  # 格式化为秒的小数形式
            frame_text = f"Frame {current_frame}"  # 显示帧编号
            

            draw = ImageDraw.Draw(frame)
            draw.text((10, 10), time_text, fill="red", font=font)
            draw.text((10, 200), frame_text, fill="red", font=font)
            
            annotated_frames.append(frame)
        
        return annotated_frames
            

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
    
    def get_prompt(self, start_time, end_time, subject, fps, origin_fps=2):
        return self.prompt.format((end_time - start_time)/origin_fps, subject, 1/fps)



def main():
    # 配置路径和参数
    import time
    
    time1 = time.time()
    
    train_path = '/root/videollm-online/datasets/ego4d/v2/annotations/refined_narration_stream_train.json'
    val_path = '/root/videollm-online/datasets/ego4d/v2/annotations/refined_narration_stream_val.json'
    origin_path = '/root/videollm-online/datasets/ego4d/v2/annotations/all_narrations_redacted.json'
    video_root = '/root/videollm-online/datasets/ego4d/v2/full_scale_2fps'
    output_dir = 'tmp3'
    prompt_file = '/root/videollm-online/data/preprocess/prompt/caption_streamingbench.txt'
    video2scene = json.load(open('/root/videollm-online/data/preprocess/metafile/video2scene.json'))
    video_uid_list = open('/root/videollm-online/data/preprocess/metafile/major2scene_case.txt').read().split('\n')
    alpha = 4.9
    device = 'cuda:5'
    
    # 初始化各个模块
    annotation_loader = AnnotationLoader(train_path, val_path, origin_path)
    data = annotation_loader.get_data()
    origin_narration = annotation_loader.get_origin_narration()
    
    beta_alpha_calculator = BetaAlphaCalculator(data, alpha)
    beta_alpha_calculator.compute_beta()
    beta_map = beta_alpha_calculator.get_beta_map()
    alpha = beta_alpha_calculator.get_alpha()
    
    video_processor = VideoProcessor(data, origin_narration, beta_map, alpha, video_root)
    caption_generator = CaptionGenerator('openbmb/MiniCPM-V-2_6', 'openbmb/MiniCPM-V-2_6', device=device)
    prompt_generator = PromptGeneratorExpandAction(prompt_file)

    print(f'Initialization time: {time.time() - time1:.2f}s, start captioning...')
    
    for path in tqdm.tqdm(video_uid_list):
        if path not in data:
            continue
        annotation_uid_narrations = data[path]
        
        for clip_idx, (annotation_uid, narrations) in enumerate(annotation_uid_narrations.items()):
            
            subject = ' / '.join(video2scene[path])
        
            clip_gen = video_processor.load_scene_clipv2(path, clip_idx)
            for action_idx, (frames, start_frame, end_frame, load_range, fps) in enumerate(clip_gen):
                os.makedirs(f'{output_dir}/{path}/{annotation_uid}', exist_ok=True)
                show_image(load_range, frames, f'{output_dir}/{path}/{annotation_uid}/{action_idx}.png')

                question = prompt_generator.get_prompt(start_frame, end_frame, subject, fps, origin_fps=video_processor.frame_fps)
                question, answer = caption_generator.get_caption(frames, question)

                with open(f'{output_dir}/{path}/{annotation_uid}/{action_idx}.txt', 'w') as f:
                    f.write(question + '\n')
                    f.write(answer)
                    
                # break
            break
        # break

if __name__ == '__main__':
    main()