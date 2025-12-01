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
        end_frame = int(ceil_time_by_fps(end_time, self.frame_fps, clip_start_time, clip_end_time)* self.frame_fps) + 1
        
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

    def load_clip(self, vr, path, clip_idx, start_time, end_time, is_stereo=False, max_frames=32):
        start_frame = int(ceil_time_by_fps(start_time, self.frame_fps, 0, (vr._num_frame-1) / self.frame_fps) * self.frame_fps)
        end_frame = int(ceil_time_by_fps(end_time, self.frame_fps, 0, (vr._num_frame-1) / self.frame_fps)* self.frame_fps) + 1
        
        try:
            load_range = range(start_frame, end_frame)
            frames = vr.get_batch(load_range)
        except:
            breakpoint()
        
        if is_stereo:
            frames = frames[:, :, :frames.shape[2] // 2, :]
            
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
    def __init__(self, model_name, tokenizer_name, device, dtype=torch.float16):
        try:
            import psutil
            import os, time
            
            # 获取当前进程
            process = psutil.Process(os.getpid())
            
            # 监控初始 CPU 内存使用
            print(f"Initial CPU Memory used: {process.memory_info().rss / 1024 / 1024:.2f} MB")
            
            print("Starting model initialization")
            print(f"Model name: {model_name}")
            
            # 加载模型前的内存使用
            print(f"CPU Memory before model loading: {process.memory_info().rss / 1024 / 1024:.2f} MB")
            
            from transformers import AutoModelForCausalLM
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                attn_implementation="sdpa",
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
                use_safetensors=True,
            )
            print(self.model)
            self.model.eval()
            self.model.to(device)
            
            
            # 加载模型后的内存使用
            print(f"CPU Memory after model loading: {process.memory_info().rss / 1024 / 1024:.2f} MB")
            
            # # 定期监控内存使用（每30秒）
            # def monitor_memory():
            #     while True:
            #         print(f"Current CPU Memory: {process.memory_info().rss / 1024 / 1024:.2f} MB")
            #         time.sleep(30)
            
            # # 在后台线程中运行监控
            # import threading
            # monitor_thread = threading.Thread(target=monitor_memory, daemon=True)
            # monitor_thread.start()
            
            print("Step 5: Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
            print("Step 6: Tokenizer loaded")
            
            print("Step 7: Setting model to eval mode...")
            self.model.eval()
            print("Step 8: Model initialization complete")
            
            # 验证模型状态
            print(f"Model device: {next(self.model.parameters()).device}")
            print("Model is ready for inference")
            
        except Exception as e:
            print(f"Error during initialization: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
            
    def get_caption(self, frames, question):
        # 确保输入数据也在正确的设备上
        if isinstance(frames, torch.Tensor):
            frames = frames.to(self.model.device)
        elif isinstance(frames, list):
            frames = [f.to(self.model.device) if isinstance(f, torch.Tensor) else f for f in frames]
        
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

import argparse
def parse_args():
    parser = argparse.ArgumentParser()
    # device
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--prompt_file', type=str, default='/2022233235/videollm-online/data/preprocess/prompt/caption_expand.txt')
    parser.add_argument('--alpha', type=float, default=4.9)
    parser.add_argument('--output_dir', type=str, default='tmp')
    parser.add_argument('--device_list', type=str, default=None, help="Comma-separated list of integers e.g. 0,1,2,3")
    parser.add_argument('--video_root', type=str, default='/2022233235/videollm-online/datasets/ego4d/v2/full_scale_2fps', help="video root")
    
    # pipeline
    parser.add_argument('--caption_last_half', action='store_true', default=False, help='Caption the only last half of the video')
    parser.add_argument('--is_scene', action='store_true', default=False, help='Caption the only secene change')
    parser.add_argument('--save_image',action='store_true', default=False, help='Save the image')
    
    
    args = parser.parse_args()
    return args


def main():
    try:
        print("Starting main function")
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
        caption_generator = CaptionGenerator('openbmb/MiniCPM-V-2_6', 'openbmb/MiniCPM-V-2_6', device)
        
        if args.is_scene:
            prompt_generator = PromptGenerator(prompt_file)
        else:
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

            print(f'GPU {device_idx} starts.')
        
        # check exist files 
        # if os.path.exists(output_dir):
        #     exist_files = os.listdir(output_dir)
        #     exist_files = [file.split('.')[0] for file in exist_files]
        #     # delete exist files
        #     video_uid_list =  [file for file in video_uid_list if file not in exist_files]    

        for i, path in tqdm.tqdm(enumerate(video_uid_list)):
            # fileter_data
            if path not in filtered_data.keys():
                print(path, path in filtered_data.keys())
                continue
            
            output_json = {}
            output_json[path] = {}
            annotation_uid_narrations = data[path]
            
            # is_stereo
            is_stereo = meta_data[path]['is_stereo']
            
            # load video
            vr = video_processor.load_video(path)
            
            for clip_idx, (annotation_uid, narrations) in enumerate(annotation_uid_narrations.items()):
                
                # fileter_data
                if annotation_uid not in filtered_data[path].keys():
                    continue
                
                output_json[path][annotation_uid] = []
                
                valid_narrations_idx = narration2move_action(narrations)

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
                last_start_time = clip_start_time
                
                os.makedirs(f'{output_dir}/{path}/{annotation_uid}', exist_ok=True)
                if os.path.exists(f'{output_dir}/{path}/{annotation_uid}.json'):
                    output_json[path][annotation_uid] = json.load(open(f'{output_dir}/{path}/{annotation_uid}.json'))
                if len(output_json[path][annotation_uid]) == len(narrations):
                    continue
                
                for action_idx, narration in enumerate(narrations):
                    
                    action_narration, frames, stamp_time, start_time, end_time, load_range, simi, simi_m = video_processor.load_action_clip(vr, path, clip_idx, action_idx, clip_start_time, clip_end_time, is_stereo)
                    
                    # all_simis.append(simi)
                    
                    if (((simi > 0.8) or (action_idx not in valid_narrations_idx)) and simi > 0.7) and args.is_scene:
                        continue 
                    
                    if args.save_image:
                        show_image(load_range, frames, f'{output_dir}/{path}/{annotation_uid}/{action_idx}.png')
                    
                    if args.is_scene:
                        question = prompt_generator.get_prompt()
                    else:
                        question = prompt_generator.get_prompt(action_narration)
                        
                    if args.is_scene:
                        if last_start_time < start_time:
                            pre_frames, pre_load_range = video_processor.load_clip(vr, path, clip_idx, last_start_time, start_time, is_stereo)
                            # try:
                            pre_question, pre_answer = caption_generator.get_caption(pre_frames, question)
                            # except:
                                # breakpoint()
                            output_json[path][annotation_uid].append({
                                'caption': pre_answer,
                                'start_time': last_start_time,
                                'end_time': start_time,
                                'pre_scene': True,
                            })
                            last_start_time = end_time
                    
                    if args.caption_last_half:
                        frames = frames[len(frames) // 2:]
                    question, answer = caption_generator.get_caption(frames, question)
                    
                    output_json[path][annotation_uid].append({
                        'caption': answer,
                        'stamp_time': stamp_time,
                        'start_time': start_time,
                        'end_time': end_time,
                        'simi': simi,
                        'action_narration': action_narration,
                        'action_idx': action_idx,
                    })
                
                if os.path.exists(f'{output_dir}/{path}/{annotation_uid}'):  
                    if last_start_time < clip_end_time and args.is_scene:
                        question = prompt_generator.get_prompt()
                        pre_frames, pre_load_range = video_processor.load_clip(vr, path, clip_idx, last_start_time, clip_end_time, is_stereo)
                        try:
                            pre_question, pre_answer = caption_generator.get_caption(pre_frames, question)
                        except:
                            breakpoint()
                        output_json[path][annotation_uid].append({
                            'caption': pre_answer,
                            'start_time': last_start_time,
                            'end_time': start_time,
                            'pre_scene': True,
                        })
                    
                    
                    with open(f'{output_dir}/{path}/{annotation_uid}.json', 'w') as f:
                        json.dump(output_json[path][annotation_uid], f, indent=4)
                        
    except Exception as e:
        print(f"Main function error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()