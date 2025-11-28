import torch, torchvision, transformers, collections
# torchvision.set_video_backend('video_reader')
from dataclasses import asdict
from torchvision.io import read_video
from transformers import AutoProcessor, LlavaNextForConditionalGeneration
from models import build_model_and_tokenizer, parse_args, fast_greedy_generate
from data.utils import ffmpeg_once
import os
import tqdm
import time
import json

logger = transformers.logging.get_logger('liveinfer')

# python -m demo.cli --resume_from_checkpoint ... 


class LiveInfer_llavaNext:
    def __init__(self, device='cuda', system_prompt=None) -> None:
        args = parse_args()
        self.model = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf", torch_dtype=torch.float16, device_map=device)
        processor = AutoProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
        self.tokenizer = processor.tokenizer
        self.processor = processor
        self.model.to(device)
        self.device = device
        
        # visual
        self.frame_fps = 0.125
        self.frame_resolution = 384
        
        # generation
        self.system_prompt = args.system_prompt if system_prompt is None else system_prompt
        print(self.system_prompt)        
        
        # generation
        self._start_ids = {'role': 'user', 'content': [{"type": "text", "text": self.system_prompt}]}
        
        # app
        self.reset()

    def _call_for_response(self, video_time, query):
        conversation_prompt = self.processor.apply_chat_template(self.past_key_values, add_generation_prompt=True)
        inputs = self.processor(
            text=[conversation_prompt], 
            images=self.frame_value if len(self.frame_value) > 0 else None, 
            padding=True, 
            return_tensors="pt"
        ).to(self.model.device)
        generate_ids = self.model.generate(**inputs, max_new_tokens=1024)
        output = self.processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        
        response = output[-1].split("[/INST]")[-1].strip()
        self.past_key_values.append({
            "role": "assistant",
            "content": [{"type": "text", "text": response}]
        })
        if query:
            query = f'(Video Time = {video_time}s) User: {query}'
        response = f'(Video Time = {video_time}s) Assistant:{response}'
        return query, response
    
    def _call_for_streaming(self, ):
        
        if len(self.past_key_values) == 0:
            self.past_key_values.append(self._start_ids)
            return 0, None
    
        while self.frame_embeds_queue:
            # 1. if query is before next frame, response
            if self.query_queue and self.frame_embeds_queue[0][0] > self.query_queue[0][0]:
                video_time, query = self.query_queue.popleft()
                self.past_key_values.append({
                        "role": "user",
                        "content": [{"type": "text", "text": query}]
                    })
                return video_time, query
            
            video_time, frame_embeds = self.frame_embeds_queue.popleft()
            self.past_key_values.append({
                "role": "user",
                "content": [{"type": "image"}]
            })
            self.frame_value.append(frame_embeds)
            
            # 2. if the same time, response after frame at that time
            if self.query_queue and video_time >= self.query_queue[0][0]:
                video_time, query = self.query_queue.popleft()
                self.past_key_values.append({
                        "role": "user",
                        "content": [{"type": "text", "text": query}]
                    })
                return video_time, query
            
            return video_time, None
    
    def reset(self, ):
        self.query_queue = collections.deque()
        self.frame_embeds_queue = collections.deque()
        self.video_time = 0
        self.last_frame_idx = -1
        self.video_tensor = None
        self.past_key_values = []
        self.frame_value = []

    def input_query_stream(self, query, history=None, video_time=None):
        if video_time is None:
            self.query_queue.append((self.video_time, query))
        else:
            self.query_queue.append((video_time, query))
        if not self.past_key_values:
            return f'(NOTE: No video stream here. Please select or upload a video. Then the assistant will answer "{query} (at {self.video_time}s)" in the video stream)'
        return f'(NOTE: Received "{query}" (at {self.video_time}s). Please wait until previous frames have been processed)'
    
    def input_video_stream(self, video_time):
        frame_idx = int(video_time * self.frame_fps)
        if frame_idx > self.last_frame_idx:
            ranger = range(self.last_frame_idx + 1, frame_idx + 1)
            self.frame_embeds_queue.extend([(r / self.frame_fps, frame_embeds) for r, frame_embeds in zip(ranger, self.video_tensor[ranger])])
        self.last_frame_idx = frame_idx
        self.video_time = video_time
    
    def load_video(self, video_path):
        self.video_tensor = read_video(video_path, pts_unit='sec', output_format='TCHW')[0].to(self.device)
        self.num_video_frames = self.video_tensor.size(0)
        self.video_duration = self.video_tensor.size(0) / self.frame_fps
        logger.warning(f'{video_path} -> {self.video_tensor.shape}, {self.frame_fps} FPS')
        
    def __call__(self, ):
        while not self.frame_embeds_queue:
            continue
        if len(self.past_key_values) == 0:
            video_time, query = self._call_for_streaming()
            response = None
            query, response = self._call_for_response(video_time, query)
        video_time, query = self._call_for_streaming()
        response = None
        query, response = self._call_for_response(video_time, query)
        return query, response
    


if __name__ == '__main__':
    sys_prompt = (
        "You act as the AI assistant on the user's AR glasses. The AR glasses continuously receive streaming frames of the user's view. \
        The first few frames might not show any relevant content, so wait until you can see something clearly. Once you detect visible content, begin describing what you see. Are you ready to receive streaming frames?"
    )

    liveinfer = LiveInfer_llavaNext(device='cuda:4', system_prompt=sys_prompt)
    src_video_path = '/root/videollm-online/datasets/coin/videos/d_snHNY3iZE.mp4'
    question = 'What is the action in the video? Format your answer concisely. No extra text output.'
    question_time = 32
    
    liveinfer.reset()
    
    # process video
    name, ext = os.path.splitext(src_video_path)
    ffmpeg_video_path = os.path.join('demo/assets/cache', name + f'_{liveinfer.frame_fps}fps_{liveinfer.frame_resolution}' + ext)
    save_history_path = src_video_path.replace('.mp4', '.json')
    if not os.path.exists(ffmpeg_video_path):
        os.makedirs(os.path.dirname(ffmpeg_video_path), exist_ok=True)
        ffmpeg_once(src_video_path, ffmpeg_video_path, fps=liveinfer.frame_fps, resolution=liveinfer.frame_resolution)
        logger.warning(f'{src_video_path} -> {ffmpeg_video_path}, {liveinfer.frame_fps} FPS, {liveinfer.frame_resolution} Resolution')
    
    
    liveinfer.load_video(ffmpeg_video_path)
    liveinfer.input_query_stream(question, video_time=question_time)
    
    timecosts = []
    pbar = tqdm.tqdm(total=liveinfer.num_video_frames, bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}{postfix}]")
    history = {'video_path': src_video_path, 'frame_fps': liveinfer.frame_fps, 'conversation': []} 
    for i in range(liveinfer.num_video_frames):
        start_time = time.time()
        liveinfer.input_video_stream(i / liveinfer.frame_fps)
        query, response = liveinfer()
        end_time = time.time()
        timecosts.append(end_time - start_time)
        fps = (i + 1) / sum(timecosts)
        pbar.set_postfix_str(f"Average Processing FPS: {fps:.1f}")
        pbar.update(1)
        if query:
            history['conversation'].append({'role': 'user', 'content': query, 'time': liveinfer.video_time, 'fps': fps, 'cost': timecosts[-1]})
            print(query)
        if response:
            history['conversation'].append({'role': 'assistant', 'content': response, 'time': liveinfer.video_time, 'fps': fps, 'cost': timecosts[-1]})
            print(response)
        if not query and not response:
            history['conversation'].append({'time': liveinfer.video_time, 'fps': fps, 'cost': timecosts[-1]})
    json.dump(history, open(save_history_path, 'w'), indent=4)
    print(f'The conversation history has been saved to {save_history_path}.')