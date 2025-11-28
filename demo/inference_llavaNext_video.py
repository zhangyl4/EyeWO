import av
import torch
import numpy as np
from transformers import LlavaNextVideoForConditionalGeneration, LlavaNextVideoProcessor, LlavaNextForConditionalGeneration, AutoProcessor
from data.utils import ffmpeg_once
import os
import math


def ceil_time_by_fps(time: float, fps: int, min_time: float, max_time: float): # provent time exceed the video duration, and close to frame number for ceil
    return min(max(math.ceil(time * fps) / fps, min_time), max_time)

device = "cuda:7"

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

# Load the video as an np.array, sampling uniformly 8 frames (can sample more for longer videos)
src_video_path = "/root/videollm-online/datasets/coin/videos/d_snHNY3iZE.mp4"
frame_fps = 1
frame_resolution = 384
start_time = 23.0
end_time = 35.0
model = 'image'

name, ext = os.path.splitext(src_video_path)
ffmpeg_video_path = os.path.join('demo/assets/cache', name + f'_{frame_fps}fps_{frame_resolution}' + ext)
if not os.path.exists(ffmpeg_video_path):
    os.makedirs(os.path.dirname(ffmpeg_video_path), exist_ok=True)
    ffmpeg_once(src_video_path, ffmpeg_video_path, fps=frame_fps, resolution=frame_resolution)
    
    
start_time = ceil_time_by_fps(start_time, frame_fps, min_time=0, max_time=93)
end_time = ceil_time_by_fps(end_time, frame_fps, min_time=0, max_time=93)
indices = np.arange(int(start_time * frame_fps), int(end_time * frame_fps) + 1).astype(int)
print(indices)

video_path = src_video_path.replace('.mp4', f'_{frame_fps}fps_{frame_resolution}.mp4')
container = av.open(video_path)
total_frames = container.streams.video[0].frames
video = read_video_pyav(container, indices)
if model != 'video':
    video =[video[0] for i in range(video.shape[0])] 

# Load the model in half-precision
if model == 'video':
    model = LlavaNextVideoForConditionalGeneration.from_pretrained("llava-hf/LLaVA-NeXT-Video-7B-hf", torch_dtype=torch.float16, device_map=device)
    processor = LlavaNextVideoProcessor.from_pretrained("llava-hf/LLaVA-NeXT-Video-7B-hf")

    conversation = [
        {

            "role": "user",
            "content": [
                {"type": "text", "text": 
                    "What is the action in the video?"},
                {"type": "video"},
                ],
        },
    ]

    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(text=prompt, videos=video, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    out = model.generate(**inputs, max_new_tokens=256)
    output = processor.batch_decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    print(output)
else:
    model = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf", torch_dtype=torch.float16, device_map=device)
    processor = AutoProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
    
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": 
                    "These images are consecutive frames from a video. Can you tell me what the person in the video is doing based on these frames?"},
                ],
        },
    ]
    for i in range(len(video)):
        conversation[0]["content"].append({"type": "image"})
    
    conversation_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(
        text=[conversation_prompt], 
        images=video, 
        padding=True, 
        return_tensors="pt"
    ).to(model.device)
    generate_ids = model.generate(**inputs, max_new_tokens=1024)
    output = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    print(output)