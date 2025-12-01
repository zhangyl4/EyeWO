
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
from typing import Dict, Optional, Sequence, List
# from inference_interVL import CaptionGenerator_interVL

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
        plt.close()
    else:
        plt.tight_layout()
        plt.show()
        plt.close()


class AnnotationLoader:
    def __init__(self, train_path, val_path, origin_path, EGO4D_JSON_PATH):
        self.train_data = json.load(open(train_path))
        self.val_data = json.load(open(val_path))
        self.data = {**self.train_data, **self.val_data}
        
        self.origin_narration = json.load(open(origin_path))['videos']
        
        meta_data = json.load(open(EGO4D_JSON_PATH))['videos']
        self.meta_data = {}
        for meta_d in meta_data:
            self.meta_data[meta_d['video_uid']] = meta_d
        
    def get_data(self):
        return self.data
    
    def get_origin_narration(self):
        return self.origin_narration
    
    def get_meta_data(self):
        return self.meta_data

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
    def __init__(self, data, beta_map, alpha, video_root, frame_fps=2, device='cuda:4'):
        self.data = data
        self.beta_map = beta_map
        self.alpha = alpha
        self.video_root = video_root
        self.frame_fps = frame_fps
        
        from siglip import visionTextAligner
        self.aliger = visionTextAligner(device=device)
    
    def action2clip(self, path, clip_idx, action_idx):
        annotation_uids = list(self.data[path].keys())
        clip_id = annotation_uids[clip_idx]
        narration = self.data[path][clip_id][action_idx]
        stamp_time = narration['time']
        beta = self.beta_map.get(clip_id, 0)
        start_time = stamp_time - beta / (2 * self.alpha)
        end_time = stamp_time + beta / (2 * self.alpha)
        return stamp_time, start_time, end_time, clip_id
    
    def load_video(self, path):
        video_path = os.path.join(self.video_root, f"{path}.mp4")
        vr = VideoReader(video_path)
        return vr

    def load_action_clip(self,vr, path, clip_idx, action_idx,clip_start_time, clip_end_time, is_stereo=False):
        stamp_time, start_time, end_time, clip_id = self.action2clip(path, clip_idx, action_idx)
        narration = self.data[path][clip_id][action_idx]['text']
        
        start_frame = int(ceil_time_by_fps(start_time, self.frame_fps, clip_start_time, clip_end_time) * self.frame_fps)
        end_frame = int(ceil_time_by_fps(end_time, self.frame_fps, clip_start_time, clip_end_time)* self.frame_fps)
        
        try:
            load_range = range(start_frame, end_frame)
            frames = vr.get_batch(load_range)
        except:
            breakpoint()
        
        if is_stereo:
            frames = frames[:, :, :frames.shape[2] // 2, :]   
        
        try:
            simi, simi_m = self.aliger.vision_simi(frames, return_m=True)
        except:
            breakpoint()
        
        frames = [Image.fromarray(frame.numpy().astype('uint8')) for frame in frames]
        
        return narration, frames, stamp_time, start_time, end_time, load_range, simi, simi_m

    def load_clip(self, vr, path, clip_idx, start_time, end_time, is_stereo=False, max_frames=32,is_torch=False):
        start_frame = int(ceil_time_by_fps(start_time, self.frame_fps, 0, (vr._num_frame-1) / self.frame_fps) * self.frame_fps)
        end_frame = int(ceil_time_by_fps(end_time, self.frame_fps, 0, (vr._num_frame-1) / self.frame_fps)* self.frame_fps)
        try:
            load_range = range(start_frame, end_frame)
            frames = vr.get_batch(load_range)
        except:
            breakpoint()
        
        if is_stereo:
            frames = frames[:, :, :frames.shape[2] // 2, :]
        
        if not is_torch:
            frames = [Image.fromarray(frame.numpy().astype('uint8')) for frame in frames]
        if len(frames) > max_frames:
            sample_idx = VideoProcessor.get_seq_frames(len(frames), max_frames, 0)
            frames = [frames[i] for i in sample_idx]
            
        return frames, load_range
    
    
    def load_clip_byframe(self, vr, path, clip_idx, start_frame, end_frame, is_stereo=False, max_frames=32, is_torch=False):
        
        try:
            load_range = range(start_frame, end_frame)
            frames = vr.get_batch(load_range)
        except:
            breakpoint()
        
        if is_stereo:
            frames = frames[:, :, :frames.shape[2] // 2, :]
        
        if not is_torch:   
            frames = [Image.fromarray(frame.numpy().astype('uint8')) for frame in frames]
            
        if len(frames) > max_frames:
            sample_idx = VideoProcessor.get_seq_frames(len(frames), max_frames, 0)
            frames = [frames[i] for i in sample_idx]
            
        return frames, load_range
    
    @staticmethod # uniform sample
    def get_seq_frames(total_num_frames, desired_num_frames, start_frame):
        seg_size = float(total_num_frames - 1) / desired_num_frames
        seq = []
        for i in range(desired_num_frames):
            start = int(np.round(seg_size * i))
            end = int(np.round(seg_size * (i + 1)))
            seq.append((start + end) // 2 + start_frame)
        return seq
            

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

def get_vlm_simi(similarity, this_video_feature, pre_frame_n = 1):

    # List to store the mean similarity for each frame
    mean_similarities = []

    # Iterate through each frame starting from the 10th frame (index 9)
    for i in range(pre_frame_n, this_video_feature.size(0)):
        # Select up to the previous 10 frames
        start_idx = max(0, i - pre_frame_n)
        previous_frames = this_video_feature[start_idx:i]

        # Compute cosine similarity between the current frame and previous frames
        current_frame = this_video_feature[i].unsqueeze(0)  # Add batch dimension
        similarities = torch.nn.functional.cosine_similarity(current_frame, previous_frames, dim=1)

        # Calculate the mean similarity for the current frame
        mean_similarity = similarities.mean().item()
        mean_similarities.append(mean_similarity)

    # Convert to tensor if needed
    mean_similarities = torch.tensor(mean_similarities)
    return mean_similarities


def segment_video(anomaly_frames, total_frames, window_len = 10, min_anomalies = 4):
    # 将异常帧列表去重并排序
    anomaly_frames = sorted(set(anomaly_frames))
    # 创建候选帧列表，包括0,所有异常帧,和total_frames
    candidate_frames = sorted(set([0] + anomaly_frames + [total_frames]))
    segments = []
    i = 0
    n = len(candidate_frames)
    while i < n:
        start = candidate_frames[i]
        end = start
        # 尝试扩展end到下一个候选帧，直到满足条件
        j = i + 1
        while j < n:
            end = candidate_frames[j]
            # 计算分段长度
            length = end - start + 1
            # 计算内部异常帧数量
            anomalies_in_segment = sum(1 for frame in anomaly_frames if start <= frame <= end)
            # 检查是否满足条件
            if length > window_len or anomalies_in_segment >= min_anomalies:
                # 记录这个分段
                segments.append((start, end))
                i = j
                break
            j += 1
        else:
            if start < total_frames:
                segments.append((start, total_frames))
            break
    return segments


def seg_video(simi,features,total_frames, load_range, pre_f_n = 1):
    mean_similarities = get_vlm_simi(simi,features, pre_f_n)

    # perform the 3 sigma rule to detect the abnormal frame
    mean = mean_similarities.mean()
    std = mean_similarities.std()
    threshold = mean - 1 * std
    abnormal_frames = torch.where(mean_similarities < threshold)[0]
    # seg
    segments = segment_video((abnormal_frames+pre_f_n).tolist(), total_frames)
    segments = [(start+load_range[0], end+load_range[0]) for start, end in segments]
    return segments


def has_format_placeholder(s):
    return '{' in s and '}' in s

class PromptGenerator:
    def __init__(self, prompt_file):
        self.prompt = open(prompt_file).read()
        self.is_format = has_format_placeholder(self.prompt)
        
    def get_prompt(self):
        return self.prompt
    
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

def flatten_list(nested_list):
    result = []
    for sublist in nested_list:
        result.extend(sublist)
    return result

violent_action = [['cross'],['drive', 'ride'], ['enter'], ['walk'], ['step']]
violent_action = flatten_list(violent_action)
slight_action = [['kneel', 'lay', 'lays', 'sit'], ['stand'], ['check', 'compare', 'examine', 'inspect', 'look', 'observe', 'see', 'view'],['camera']]
slight_action = flatten_list(slight_action)

def sentence2token(sentence):
    a = sentence.split()
    return a

def is_violent_action(scentence):
    for action in violent_action:
        if action in sentence2token(scentence):
            return True
    return False

def is_slight_action(scentence):
    for action in slight_action:
        if action in sentence2token(scentence):
            return True
    return False

def narration2move_action(narrations):
    move_narrations_idx = []
    for action_idx, narration in enumerate(narrations):
        if is_violent_action(narration['text']) or is_slight_action(narration['text']):
            move_narrations_idx.append(action_idx)

    return move_narrations_idx

def parse_device_list(device_list_str: Optional[str]) -> Optional[List[int]]:
    if device_list_str is None:
        return None
    return [int(x) for x in device_list_str.split(',')]


def get_index(bound, fps, max_frame, first_idx=0, num_segments=32):
    """
    Get the frame indices for video segments based on the given parameters.

    Args:
        bound (tuple): A tuple containing the start and end time of the video segment in seconds.
        fps (float): Frames per second of the video.
        max_frame (int): Total number of frames in the video.
        first_idx (int, optional): The index of the first frame. Defaults to 0.
        num_segments (int, optional): Number of segments to divide the video into. Defaults to 32.

    Returns:
        numpy.ndarray: An array of frame indices representing the segments of the video.
    """
    if bound:
        start, end = bound[0], bound[1]
    else:
        start, end = -100000, 100000
    start_idx = max(first_idx, round(start * fps))
    end_idx = min(round(end * fps), max_frame)
    seg_size = float(end_idx - start_idx) / num_segments
    # frame_indices = np.array([
    #     int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
    #     for idx in range(num_segments)
    # ])
    frame_indices = [(int(start_idx + np.round(seg_size * idx)), int(start_idx + np.round(seg_size * (idx+1)))) for idx in range(num_segments)]
    
    return frame_indices


import argparse
def parse_args():
    parser = argparse.ArgumentParser()
    # device
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda:1')
    parser.add_argument('--prompt_file', type=str, default='/root/videollm-online/data/preprocess/prompt/caption_object.txt')
    parser.add_argument('--alpha', type=float, default=5)
    parser.add_argument('--output_dir', type=str, default='/root/videollm-online/datasets/ego4d_uniform_caption')
    parser.add_argument('--device_list', type=str, default=None, help="Comma-separated list of integers e.g. 0,1,2,3")
    parser.add_argument('--model', type=str, default='openbmb/MiniCPM-V-2_6')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--video_root', type=str, default='/2022233235/videollm-online/datasets/ego4d/v2/full_scale_2fps')
    
    
    # pipeline
    parser.add_argument('--save_image',action='store_true', default=False, help='Save the image')
    
    
    args = parser.parse_args()
    return args


def main():

    import time
    time1 = time.time()
    args = parse_args()
    EGO_ANNO_ROOT = '/2022233235/datasets/ego4d/annotations'
    EGO4D_JSON_PATH = "/2022233235/datasets/ego4d/ego4d.json"
    PROJECT_ROOT = '/2022233235/videollm-online'
    fileter_data_path = f'{EGO_ANNO_ROOT}/filtered_data.json'
    video_root = args.video_root
    train_path = f'{EGO_ANNO_ROOT}/refined_narration_stream_train.json'
    val_path = f'{EGO_ANNO_ROOT}/refined_narration_stream_val.json'
    origin_path = f'{EGO_ANNO_ROOT}/all_narrations_redacted.json'
    output_dir = args.output_dir
    prompt_file = args.prompt_file
    video2scene = json.load(open(f'{PROJECT_ROOT}/data/preprocess/metafile/video2scene.json'))
    video_uid_list = os.listdir(video_root)
    video_uid_list = [file.split('.')[0] for file in video_uid_list]
    
    alpha = args.alpha # long
    device = args.device
    
    annotation_loader = AnnotationLoader(train_path, val_path, origin_path, EGO4D_JSON_PATH)
    data = annotation_loader.get_data()
    origin_data = annotation_loader.get_origin_narration()
    meta_data = annotation_loader.get_meta_data()
    filtered_data = json.load(open(fileter_data_path))
    
    beta_alpha_calculator = BetaAlphaCalculator(data, alpha)
    beta_alpha_calculator.compute_beta()
    beta_map = beta_alpha_calculator.get_beta_map()
    alpha = beta_alpha_calculator.get_alpha()
    
    video_processor = VideoProcessor(data, beta_map, alpha, video_root, device=device)
    
    if args.model == 'openbmb/MiniCPM-V-2_6':
        caption_generator = CaptionGenerator('openbmb/MiniCPM-V-2_6', 'openbmb/MiniCPM-V-2_6', device)
    # elif args.model == 'OpenGVLab/InternVL2_5-8B':
    #     caption_generator = CaptionGenerator_interVL('OpenGVLab/InternVL2_5-8B', 'OpenGVLab/InternVL2_5-8B', device)
    prompt_generator = PromptGenerator(prompt_file)
    if prompt_generator.is_format:
        prompt_generator = PromptGeneratorExpandAction(prompt_file)
        
    print('Initialization time:', time.time() - time1)
    
    # all_simis = []
        
    # add gpu list
    if args.device_list is not None:
        device_idx = device.split(':')[-1]
        device_list = parse_device_list(args.device_list)
        device_index = device_list.index(int(device_idx))
        
        # chunk video_uid_list
        video_uid_list = video_uid_list[device_index::len(device_list)]
    
    # check exist files 
    # if os.path.exists(output_dir):
    #     exist_files = os.listdir(output_dir)
    #     exist_files = [file.split('.')[0] for file in exist_files]
    #     # delete exist files
    #     video_uid_list =  [file for file in video_uid_list if file not in exist_files]    
    
    for path in tqdm.tqdm(video_uid_list):
        # fileter_data
        if path not in filtered_data.keys():
            continue
        
        output_json = {}
        output_json[path] = {}
        annotation_uid_narrations = data[path]
        
        # is_stereo
        is_stereo = meta_data[path]['is_stereo']
        
        # load video
        vr = video_processor.load_video(path)
        fps = 2
        max_frame = len(vr) - 1
        
        for clip_idx, (annotation_uid, narrations) in tqdm.tqdm(enumerate(annotation_uid_narrations.items())):
            print(clip_idx)
            # fileter_data
            if annotation_uid not in filtered_data[path].keys():
                continue
            
            output_json[path][annotation_uid] = []

            summs = origin_data[path]['summaries']
            is_match = False
            for summ in summs:
                if summ['_annotation_uid'] == annotation_uid:
                    is_match = True
                    break
            if not is_match:
                continue
            
            clip_start_time = summ['start_time']
            clip_end_time = summ['end_time']
            
            
            # split start t
            frames, load_range = video_processor.load_clip(vr, None, None, clip_start_time,clip_end_time, max_frames=10000, is_torch=True)
            simi, (simi_,features) = video_processor.aliger.vision_simi(frames, return_m=True)
            frame_indices = seg_video(simi,features,len(frames), load_range, pre_f_n = 1)
            print(len(frame_indices))
            # seg_num = math.ceil((clip_end_time - clip_start_time) // alpha)
            # frame_indices = get_index((clip_start_time, clip_end_time), fps, max_frame, first_idx=0, num_segments=seg_num)
            
            os.makedirs(f'{output_dir}/{path}/{annotation_uid}', exist_ok=True)
            if os.path.exists(f'{output_dir}/{path}/{annotation_uid}.json'):
                output_json[path][annotation_uid] = json.load(open(f'{output_dir}/{path}/{annotation_uid}.json'))
            if len(output_json[path][annotation_uid]) == len(frame_indices):
                continue
            
            for action_idx, frame_indice in enumerate(frame_indices[len(output_json[path][annotation_uid]):]):

                start_time = frame_indice[0] / fps
                end_time = frame_indice[-1] / fps
                stamp_time = (start_time + end_time) / 2
                
                frames,load_range = video_processor.load_clip(vr, path, clip_idx, start_time, end_time, is_stereo=is_stereo)
                
                
                if args.save_image:
                    show_image(range(len(frames)), frames, f'{output_dir}/{path}/{annotation_uid}/{action_idx}.png')

                
                question = prompt_generator.get_prompt()
                    
                question, answer = caption_generator.get_caption(frames, question)
                
                output_json[path][annotation_uid].append({
                    'caption': answer,
                    'stamp_time': stamp_time,
                    'start_time': start_time,
                    'end_time': end_time,
                    'action_idx': action_idx,
                })
                
            with open(f'{output_dir}/{path}/{annotation_uid}.json', 'w') as f:
                json.dump(output_json[path][annotation_uid], f, indent=4)
                      
if __name__ == '__main__':
    main()