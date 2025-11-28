import av
import torch
import numpy as np
from transformers import LlavaNextVideoForConditionalGeneration, LlavaNextVideoProcessor, LlavaNextForConditionalGeneration, AutoProcessor
from data.utils import ffmpeg_once
import os
import math
import json

from tqdm import tqdm
import Levenshtein
from data import COINTaskProcedure, COINProcedure, COINTask, COINNext, COINStep


# python -m demo.coin_benchmark_llavaNext

def default_dump(obj):
    """Convert numpy classes to JSON serializable objects."""
    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, torch.Tensor):
        return obj.detach().cpu().numpy().tolist()
    else:
        return obj

def fuzzy_match(text, choices):
    return min([(Levenshtein.distance(text, choice), choice) for choice in choices])[1]

def read_video_pyav(container, indices):
    '''
    Decode the video with PyAV decoder.
    Args:
        container (`av.container.input.InputContainer`): PyAV container.
        indices (`List[int]`): List of frame indices to decode.
    Returns:
        result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
    '''
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])

def ceil_time_by_fps(time: float, fps: int, min_time: float, max_time: float): # provent time exceed the video duration, and close to frame number for ceil
    return min(max(math.ceil(time * fps) / fps, min_time), max_time)

class COIN_benchmark2video():
    def __init__(self, frame_fps, frame_resolution, num_frames, video_root='datasets/coin/videos', embed_mark='2fps_max384'):
        self.frame_fps = frame_fps
        self.frame_resolution = frame_resolution
        self.num_frames = num_frames
        self.embed_dir = f"{video_root}"
        
    def get_sample_id(self, video_ids, dataset):
        sample_ids = []
        for i, anno in tqdm(enumerate(dataset.annos), desc=f'get sample id...'):
            path = list(anno['load_ranges'].keys())[0]
            video_id = path.split('/')[-1].split('.')[0]
            if video_id in video_ids:
                sample_ids.append(i)
        return sample_ids

    def load_video(self, sample_id, model_name, dataset):
        anno = dataset.annos[sample_id]
        
        src_video_path = list(anno['load_ranges'].keys())[0].split('.')[0] + '.mp4'
        src_video_path = os.path.join(self.embed_dir, src_video_path.split('/')[-1])
        indices = list(anno['load_ranges'].values())[0]
        if len(indices) > self.num_frames:
            indices = np.linspace(indices[0], indices[-1], self.num_frames).astype(int)
        
        name, ext = os.path.splitext(src_video_path)
        ffmpeg_video_path = os.path.join('demo/assets/cache', name + f'_{self.frame_fps}fps_{self.frame_resolution}' + ext)
        if not os.path.exists(ffmpeg_video_path):
            os.makedirs(os.path.dirname(ffmpeg_video_path), exist_ok=True)
            ffmpeg_once(src_video_path, ffmpeg_video_path, fps=self.frame_fps, resolution=self.frame_resolution)

        container = av.open(ffmpeg_video_path)
        total_frames = container.streams.video[0].frames
        video = read_video_pyav(container, indices)
        if model_name != 'video':
            video =[video[0] for i in range(video.shape[0])] 
            
        return video
    
    def get_label_conversation(self, sample_id, dataset):
        return dataset.annos[sample_id]['conversation'], dataset.labels[sample_id]




device = "cuda:7"

video_idx = ['d_snHNY3iZE']
frame_fps = 2
frame_resolution = 384
num_frames = 32
model_name = 'video'
max_num_frames = 1200
augmentation = False
system_prompt: str = (
        "A multimodal AI assistant is helping users with some activities."
        " Below is their conversation, interleaved with the list of video frames received by the assistant."
    )
task_name = 'step'
output_dir = '/root/videollm-online/data/coin'

# Load the model in half-precision
if model_name == 'video':
    model = LlavaNextVideoForConditionalGeneration.from_pretrained("llava-hf/LLaVA-NeXT-Video-7B-hf", torch_dtype=torch.float16, device_map=device)
    processor = LlavaNextVideoProcessor.from_pretrained("llava-hf/LLaVA-NeXT-Video-7B-hf")

else:
    model = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf", torch_dtype=torch.float16, device_map=device)
    processor = AutoProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")

# dataset
dataset = COINStep(split='test', vision_pretrained='google/siglip-large-patch16-384',
                   embed_mark='2fps_max384_1', frame_fps=frame_fps, is_training=False,
                   max_num_frames=max_num_frames, augmentation=augmentation,
                   frame_resolution=frame_resolution, system_prompt=system_prompt, tokenizer=processor.tokenizer)
dataloader = COIN_benchmark2video(frame_fps, frame_resolution, num_frames, video_root=dataset.video_root, embed_mark='2fps_max384_1')


# bad case
print(len(dataset.annos))
bad_case_pred = json.load(open('/root/videollm-online/outputs/coin_benchmarks/live1+v3_evaluate_analysis/COINStep_analysis_multi_step_results.json'))
sample_ids = [p['sample_index'] for p in bad_case_pred if not p['is_correct']]
# sample_ids = dataloader.get_sample_id(video_idx, dataset)


# inference
results = {}
if os.path.exists(f'{output_dir}/{task_name}.json'):
    results = json.load(open(f'{output_dir}/{task_name}.json'))
    sample_ids = [sample_id for sample_id in sample_ids if str(sample_id) not in results]
    
for sample_id in tqdm(sample_ids):
    video = dataloader.load_video(sample_id, model_name, dataset)
    gt_conv, label = dataloader.get_label_conversation(sample_id, dataset)
    if model_name == 'video':
        conversation = [
            {
                
                "role": "user",
                "content": [
                    {"type": "text", "text": 
                        gt_conv[0]['content']},
                    {"type": "video"},
                    ],
            },
        ]

        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = processor(text=prompt, videos=video, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        out = model.generate(**inputs, max_new_tokens=256)
        output = processor.batch_decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        
        
        # evaluate
        prediction = output[0].split('ASSISTANT: ')[1]
        prediction = prediction.lower().rstrip('.')
        match_pred = fuzzy_match(prediction, dataset.categories)
        is_correct = prediction == label or match_pred == label
        
        results[sample_id] = {
            'pred': prediction,
            'match_pred': match_pred,
            'clip_length': video.shape[0],
            'input_length': gt_conv[1]['num_frames'],
            'label': label,
            'correct': is_correct
        }
        
        with open(f'{output_dir}/{task_name}.json', mode='w', encoding='utf-8') as file:
            json.dump(results, file, ensure_ascii=False, indent=4, default=default_dump)
        
    # else:
    #     conversation = [
    #         {
    #             "role": "user",
    #             "content": [
    #                 {"type": "text", "text": 
    #                     "These images are consecutive frames from a video. Can you tell me what the person in the video is doing based on these frames?"},
    #                 ],
    #         },
    #     ]
    #     for i in range(len(video)):
    #         conversation[0]["content"].append({"type": "image"})
        
    #     conversation_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    #     inputs = processor(
    #         text=[conversation_prompt], 
    #         images=video, 
    #         padding=True, 
    #         return_tensors="pt"
    #     ).to(model.device)
    #     generate_ids = model.generate(**inputs, max_new_tokens=1024)
    #     output = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    #     print(output)


correct = 0
for sample_id, result in results.items():
    if result['correct']:
        correct += 1
print(f'Accuracy: {correct / len(sample_ids)}')