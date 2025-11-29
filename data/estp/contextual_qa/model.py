import numpy as np
import torch
import torchvision.transforms as T
from decord import VideoReader, cpu
import decord
decord.bridge.set_bridge('torch')
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

def get_index(bound, fps, max_frame, first_idx=0, num_segments=32):
    if bound:
        start, end = bound[0], bound[1]
    else:
        start, end = -100000, 100000
    start_idx = max(first_idx, round(start * fps))
    end_idx = min(round(end * fps), max_frame)
    seg_size = float(end_idx - start_idx) / num_segments
    frame_indices = np.array([
        int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
        for idx in range(num_segments)
    ])
    return frame_indices

def load_video(video_path, bound=None, input_size=448, max_num=1, num_segments=32):
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    max_frame = len(vr) - 1
    fps = float(vr.get_avg_fps())

    pixel_values_list, num_patches_list = [], []
    transform = build_transform(input_size=input_size)
    frame_indices = get_index(bound, fps, max_frame, first_idx=0, num_segments=num_segments)
    for frame_index in frame_indices:
        img = Image.fromarray(vr[frame_index].asnumpy()).convert('RGB')
        img = dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(tile) for tile in img]
        pixel_values = torch.stack(pixel_values)
        num_patches_list.append(pixel_values.shape[0])
        pixel_values_list.append(pixel_values)
    pixel_values = torch.cat(pixel_values_list)
    return pixel_values, num_patches_list


def load_frames(video_path, bound=None, input_size=448, max_num=1, num_segments=32):
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    max_frame = len(vr) - 1
    fps = float(vr.get_avg_fps())
    frame_indices = get_index(bound, fps, max_frame, first_idx=0, num_segments=num_segments)
    
    # frames = [Image.fromarray(vr[frame_index].asnumpy()).convert('RGB') for frame_index in frame_indices]
    frames = vr.get_batch(frame_indices.tolist())
    frames = [Image.fromarray(frame.numpy().astype('uint8')) for frame in frames]
    
    return frames

def process_video(frames, input_size=448, max_num=1):
    transform = build_transform(input_size=input_size)
    pixel_values_list, num_patches_list = [], []
    for img in frames:
        img = dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(tile) for tile in img]
        pixel_values = torch.stack(pixel_values)
        num_patches_list.append(pixel_values.shape[0])
        pixel_values_list.append(pixel_values)
    pixel_values = torch.cat(pixel_values_list)
    return pixel_values, num_patches_list


class interVL:
    def __init__(self, model_name, tokenizer_name, device='cuda:4', dtype=torch.bfloat16):
        
        path = 'OpenGVLab/InternVL2_5-8B'
        model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            trust_remote_code=True).eval().to(device)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True, use_fast=False)
        
        
        self.model = model
        self.tokenizer = tokenizer
        self.generate_config = dict(max_new_tokens=1024, do_sample=True)
        self.device = device
            
    def get_caption(self, frames, question):
        
        pixel_values, num_patches_list = process_video(frames, max_num=1)
        pixel_values = pixel_values.to(torch.bfloat16).to(self.device)
        video_prefix = ''.join([f'Frame{i+1}: <image>\n' for i in range(len(num_patches_list))])


        question = video_prefix + question
        # Frame1: <image>\nFrame2: <image>\n...\nFrame8: <image>\n{question}
        response, history = self.model.chat(self.tokenizer, pixel_values, question, self.generate_config,
                                    num_patches_list=num_patches_list, history=None, return_history=True)

        return question, response
    
class miniCPM:
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
    

class llavaNext:
     def __init__(self, model_name, tokenizer_name, device='cuda:4', dtype=torch.bfloat16):
        
        path = 'OpenGVLab/InternVL2_5-8B'
        model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            trust_remote_code=True).eval().to(device)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True, use_fast=False)
        
        
        self.model = model
        self.tokenizer = tokenizer
        self.generate_config = dict(max_new_tokens=1024, do_sample=True)
        self.device = device