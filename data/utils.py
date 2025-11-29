import random, torch, tqdm, os, subprocess, torchvision, pathlib, submitit, math
from itertools import takewhile
try:
    torchvision.set_video_backend('video_reader')
except:
    pass
from transformers import AutoModel
from torchvision.transforms.functional import to_pil_image, normalize

class DictWithTo(dict):
    def to(self, *args, **kwargs):
        return self

def inverse_preprocess_to_pil_images(frames: torch.Tensor, mean: list, std: list):
    frames = normalize(frames, mean=tuple(-m / s for m, s in zip(mean, std)), std=tuple(1.0 / s for s in std))
    frames = (frames * 255).to(torch.uint8)
    return list(map(to_pil_image, frames))

def rand_bool():
    return bool(random.getrandbits(1))

def case_connect(prefix: str, suffix: str):
    if not prefix:
        return suffix[0].upper() + suffix[1:]
    if not suffix:
        return prefix
    if prefix[-1] == ',' or prefix[-1] == ':':
        return prefix + ' ' + suffix[0].lower() + suffix[1:]
    return prefix + ' ' + suffix[0].upper() + suffix[1:]

def batch_temporal_iou(sequences1: torch.Tensor, sequences2: torch.Tensor):
    area1 = sequences1[:, 1] - sequences1[:, 0]
    area2 = sequences2[:, 1] - sequences2[:, 0]
    l = torch.maximum(sequences1[:,None,0], sequences2[:,0])
    r = torch.minimum(sequences1[:,None,1], sequences2[:,1])
    inter = (r - l).clamp(min=0)
    union = area1[:, None] + area2 - inter
    iou = inter / union
    return iou

def temporal_iou(region1, region2):
    area1 = region1[1] - region1[0]
    area2 = region2[1] - region2[0]
    l = max(region1[0], region2[0])
    r = min(region1[1], region2[1])
    inter = max(0, (r - l))
    union = area1 + area2 - inter
    iou = inter / union
    return iou

def ffmpeg_once(src_path: str, dst_path: str, *, fps: int = None, resolution: int = None, pad: str = '#000000', mode='bicubic'):
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    command = [
        './ffmpeg/ffmpeg',
        '-y',
        '-sws_flags', mode,
        '-i', src_path,
        '-an',
        '-threads', '10',
    ]
    if fps is not None:
        command += ['-r', str(fps)]
    if resolution is not None:
        command += ['-vf', f"scale='if(gt(iw\\,ih)\\,{resolution}\\,-2)':'if(gt(iw\\,ih)\\,-2\\,{resolution})',pad={resolution}:{resolution}:(ow-iw)/2:(oh-ih)/2:color='{pad}'"]
    command += [dst_path]
    subprocess.run(command, check=True)

def distributed_ffmpeg(*, src_root: str, fps: int = None, resolution: int = None, pad: str = '#000000', mode='bicubic'):
    import submitit
    env = submitit.JobEnvironment()
    src_root = src_root.rstrip('/')
    pather = pathlib.Path(src_root)
    src_paths = [str(path) for path in pather.rglob('*') if path.is_file() and str(path).endswith('.mp4')]
    dst_root = src_root
    if fps is not None:
        dst_root += f'_{fps}fps'
    if resolution is not None:
        assert (pad is not None)
        dst_root += f'_max{resolution}'
    for i, src_path in tqdm.tqdm(enumerate(src_paths), desc=f'{src_root} -> {dst_root}'):
        if i % env.num_tasks != env.global_rank:
            continue
        dst_path = src_path.replace(src_root, dst_root)
        if not os.path.exists(dst_path):
            ffmpeg_once(src_path, dst_path, fps=fps, resolution=resolution, pad=pad, mode=mode)
        
def distributed_ffmpeg_image(*, src_root: str, fps: int = None, resolution: int = None, pad: str = '#000000', mode='bicubic'):
    import submitit
    env = submitit.JobEnvironment()
    src_root = src_root.rstrip('/')
    pather = pathlib.Path(src_root)
    src_paths = [str(path) for path in pather.rglob('*') if path.is_file() and str(path).endswith('.jpg')]
    dst_root = src_root
    if fps is not None:
        dst_root += f'_{fps}fps'
    if resolution is not None:
        assert (pad is not None)
        dst_root += f'_max{resolution}'
    for i, src_path in tqdm.tqdm(enumerate(src_paths), desc=f'{src_root} -> {dst_root}'):
        if i % env.num_tasks != env.global_rank:
            continue
        dst_path = src_path.replace(src_root, dst_root)
        ffmpeg_once(src_path, dst_path, fps=fps, resolution=resolution, pad=pad, mode=mode)

def distributed_encode(*, src_root: str, vision_pretrained: str, vision_encode: callable, batch_size: int, embed_mark: str, save_bf16: bool = False, **kwargs):
    env = submitit.JobEnvironment()
    src_root = src_root.rstrip('/')
    model = AutoModel.from_pretrained(vision_pretrained, device_map=f'cuda:{env.local_rank}').vision_model
    model.eval()
    dst_root = f"{src_root}_{embed_mark.split('_')[-1]}_{vision_pretrained.replace('/', '--')}"
    os.makedirs(dst_root, exist_ok=True)
    for i, file in tqdm.tqdm(enumerate(os.listdir(src_root)), desc=f'{src_root} -> {dst_root}'):
        if i % env.num_tasks != env.global_rank:
            continue
        frame_path = os.path.join(src_root, file)
        save_path = os.path.splitext(frame_path)[0] + '.pt'
        save_path = save_path.replace(src_root, dst_root)
        if os.path.exists(save_path):
            continue
        frames = torchvision.io.read_video(frame_path, pts_unit='sec', output_format='TCHW')[0]
        with torch.no_grad():
            frames = torch.cat([vision_encode(model, batch.to(f'cuda:{env.local_rank}')).cpu() for batch in frames.split(batch_size)])
        if save_bf16:
            frames = frames.to(torch.bfloat16)
        torch.save(frames, save_path)

from PIL import Image
import torchvision.transforms as transforms
def distributed_encode_image(*, src_root: str, vision_pretrained: str, vision_encode: callable, batch_size: int, embed_mark: str, save_bf16: bool = False, **kwargs):
    env = submitit.JobEnvironment()
    src_root = src_root.rstrip('/')
    model = AutoModel.from_pretrained(vision_pretrained, device_map=f'cuda:{env.local_rank}').vision_model
    model.eval()
    dst_root = f"{src_root}_{embed_mark.split('_')[-1]}_{vision_pretrained.replace('/', '--')}"
    os.makedirs(dst_root, exist_ok=True)
    transform = transforms.ToTensor()
    
    b_count = 0
    b_read = []
    b_write_list = []
    
    for i, file in tqdm.tqdm(enumerate(os.listdir(src_root)), desc=f'{src_root} -> {dst_root}'):
        if i % env.num_tasks != env.global_rank:
            continue
        frame_path = os.path.join(src_root, file)
        save_path = os.path.splitext(frame_path)[0] + '.pt'
        save_path = save_path.replace(src_root, dst_root)
        frames = Image.open(frame_path).convert('RGB')
        frames_tensor = transform(frames)
        
        b_count += 1
        b_read.append(frames_tensor)
        b_write_list.append(save_path)
        
        if b_count == batch_size:
            image_batch = torch.stack(b_read)
            with torch.no_grad():
                image_batch = vision_encode(model, image_batch.to(f'cuda:{env.local_rank}')).cpu()
            if save_bf16:
                image_batch = image_batch.to(torch.bfloat16)
            for b, save_path in zip(image_batch, b_write_list):
                torch.save(b, save_path)
            
            b_count = 0
            b_read = []
            b_write_list = []

def load_frames(path: str, start: float, end: float, num_threads=10) -> torch.Tensor:
    """
    Return
    torch.Tensor: T x C x H x W
    """
    reader = torchvision.io.VideoReader(path, "video", num_threads=num_threads)
    frames = torch.stack([f['data'] for f in takewhile(lambda x: x['pts'] <= end, reader.seek(start))])
    return frames # T x C x H x W

def round_time_by_fps(time: float, fps: int, min_time: float, max_time: float):
    return min(max(round(time * fps) / fps, min_time), max_time)

def ceil_time_by_fps(time: float, fps: int, min_time: float, max_time: float):
    return min(max(math.ceil(time * fps) / fps, min_time), max_time)

def floor_time_by_fps(time: float, fps: int, min_time: float, max_time: float):
    return min(max(math.floor(time * fps) / fps, min_time), max_time)


from torchvision.io import read_video
import subprocess
import os
import decord
from decord import VideoReader
import numpy as np
decord.bridge.set_bridge("torch")

def split_video(input_file, output_dir, segment_duration):    
    input_filename = os.path.splitext(os.path.basename(input_file))[0]
    output_template = os.path.join(output_dir, f'{input_filename}_part%d.mp4')
    output_path_pattern = os.path.join(output_dir, f'{input_filename}_part')
    
    command = [
        'ffmpeg', '-i', input_file, '-c', 'copy',
        '-map', '0', '-segment_time', str(segment_duration),
        '-f', 'segment', '-reset_timestamps', '1', output_template
    ]
    subprocess.run(command, check=True)
    output_files = []
    i = 0
    while True:
        output_path = f"{output_path_pattern}{i}.mp4"
        if os.path.exists(output_path):
            output_files.append(output_path)
            i += 1
        else:
            break
    return output_files

def split_tensor(tensor, max_duration, frame_fps, new_dir_path, video_id):
    
    chunks = torch.split(tensor, int(max_duration * frame_fps))

    chunk_paths = []
    for i, chunk in enumerate(chunks):
        chunk_filename = f"{video_id}_part{i}.pt"
        chunk_path = os.path.join(new_dir_path, chunk_filename)
        chunk_paths.append(chunk_path)
        if not os.path.exists(chunk_path):
            torch.save(chunk, chunk_path)
    return chunk_paths

# v2 not split video
def get_video_metadata_clip_video(path, frame_fps, max_duration=5000):
    if path.endswith('pt'):
        tensor = torch.load(path, weights_only=True)
        duration = (len(tensor) - 1) / frame_fps
    elif path.endswith('mp4'):
        vr = VideoReader(path)
        duration = (len(vr) - 1) / frame_fps
    else:
        print('error')
    
    if duration <= max_duration or path.endswith('mp4'):
        return duration, path
    else:
        video_id = os.path.splitext(os.path.basename(path))[0]
        parent_dir = os.path.dirname(path)
        parent_dir_name = os.path.basename(parent_dir)
        new_dir_name = f"{parent_dir_name}_long_video"
        new_dir_path = os.path.join(os.path.dirname(parent_dir), new_dir_name)
        os.makedirs(new_dir_path, exist_ok=True)
        chunk_paths = split_tensor(tensor, max_duration, frame_fps, new_dir_path, video_id)
        
        return duration, chunk_paths


# def get_video_metadata_clip_video(path, frame_fps, max_duration=5000):
#     if path.endswith('pt'):
#         tensor = torch.load(path, weights_only=True)
#     elif path.endswith('mp4'):
#         tensor = read_video(path, pts_unit='sec', output_format='TCHW')[0]
#     else:
#         print('error')
#     duration = (len(tensor) - 1) / frame_fps
    
#     if duration <= max_duration:
#         return duration, path
#     else:
#         video_id = os.path.splitext(os.path.basename(path))[0]
#         parent_dir = os.path.dirname(path)
#         parent_dir_name = os.path.basename(parent_dir)
#         new_dir_name = f"{parent_dir_name}_long_video"
#         new_dir_path = os.path.join(os.path.dirname(parent_dir), new_dir_name)
#         os.makedirs(new_dir_path, exist_ok=True)
        
#         if path.endswith('pt'):
#             chunk_paths = split_tensor(tensor, max_duration, frame_fps, new_dir_path, video_id)
#         elif path.endswith('mp4'):
#             chunk_paths = split_video(path, new_dir_path, max_duration)
#         return duration, chunk_paths

def load_frames_pt(path, load_range):
    if isinstance(path, tuple):
        frames = torch.cat([torch.load(chunk_path, weights_only=True) for chunk_path in path])
    else:
        frames = torch.load(path, weights_only=True)
    
    return frames[load_range]


def split_indices_by_video(load_range, frame_lengths):
    ranges = []
    cumulative_frame_count = 0
    
    for i, frame_len in enumerate(frame_lengths):
        video_start = cumulative_frame_count
        video_end = cumulative_frame_count + frame_len
        video_indices = [idx for idx in load_range if video_start <= idx < video_end]
        if video_indices:
            local_indices = [idx - video_start for idx in video_indices]
            ranges.append((i, local_indices))
        cumulative_frame_count += frame_len
    return ranges

def load_frames_mp4(path, load_range):
    if isinstance(path, tuple):
        vrs = [VideoReader(uri=chunk_path) for chunk_path in path]
        frame_lengths = [vr._num_frame for vr in vrs]
        ranges = split_indices_by_video(load_range, frame_lengths)
        frames = []
        for i, local_indices in ranges:
            frames.append(vrs[i].get_batch(local_indices).permute(0, 3, 1, 2))
        frames = torch.cat(frames)
    else:
        vr = VideoReader(uri=path)
        frames = vr.get_batch(load_range).permute(0, 3, 1, 2)
        
    return frames

def load_frames_f(load_ranges: dict[str, range]):
    frames = []
    for path, ranger in load_ranges.items():
        if  (isinstance(path, tuple) and path[0].endswith('.pt')) or (not isinstance(path, tuple) and path.endswith('.pt')):
            frame = load_frames_pt(path, ranger)
        elif (isinstance(path, tuple) and path[0].endswith('.mp4')) or (not isinstance(path, tuple) and path.endswith('.mp4')):
            frame = load_frames_mp4(path, ranger)
        frame.requires_grad_(False)
        frames.append(frame)
    frames = torch.cat(frames)

    return frames

def load_frames_jpg(load_ranges: dict[str, range]):
    frames = []
    for path, ranger in load_ranges.items():
        if ranger == 0:
            continue
        if path.endswith('jpg'):
            image = Image.open(path).convert('RGB')
            image_tensor = torch.tensor(np.array(image), dtype=torch.float32).permute(2, 0, 1)
            image_tensor = image_tensor.unsqueeze(0).repeat(ranger, 1, 1, 1)
        elif path.endswith('pt'):
            image_tensor = torch.load(path, weights_only=True)
            image_tensor = image_tensor.repeat(ranger, 1, 1)
        image_tensor.requires_grad_(False)
        frames.append(image_tensor)
    frames = torch.cat(frames)
    
    return frames

def get_path_with_key(full_path:str, key:str):
    fps_index = full_path.find(key)
    if fps_index != -1:
        path_with_fps = full_path[:fps_index + len(key)]
        return path_with_fps
    else:
        return None
    
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
    
    


### extract video key frame

from PIL import Image
from transformers import AutoProcessor, AutoModel
import torch

# python -m data.preprocess.siglip
class visionTextAligner:
    def __init__(self, model_pretrian="google/siglip-large-patch16-384", device="cuda:4"):
        self.model = AutoModel.from_pretrained(model_pretrian)
        self.model.to(device).eval()
        self.processor = AutoProcessor.from_pretrained(model_pretrian)
        
    def align(self, image_embeds, texts):
        with torch.no_grad():
            inputs = self.processor(text=texts, padding="max_length", return_tensors="pt")
            text_embeds = self.model.get_text_features(**inputs)
            image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
            text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)

            logits_per_text = (torch.matmul(text_embeds, image_embeds.t().to(text_embeds.device)) * self.model.logit_scale.exp()+ self.model.logit_bias)
            
            logits_per_image = logits_per_text.t()
            probs = torch.sigmoid(logits_per_image)
            
        return probs
    
    def vision_feature(self, frames):
        with torch.no_grad():
            inputs = self.processor(images=frames, padding="max_length", return_tensors="pt")
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            image_embeds = self.model.get_image_features(**inputs)
            image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
            return image_embeds
    
    def vision_simi(self, frames, return_m=False):
        with torch.no_grad():
            inputs = self.processor(images=frames, padding="max_length", return_tensors="pt")
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            image_embeds = self.model.get_image_features(**inputs)
            image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
            simi_m = torch.matmul(image_embeds, image_embeds.t().to(image_embeds.device))
            simi = simi_m.min(dim=0).values.mean().cpu().item()

        if return_m:
            return simi, (simi_m.cpu(),image_embeds.cpu())
        
        return simi
    
    def __call__(self, *args: Image.Any, **kwds: Image.Any) -> Image.Any:
        pass
    

def get_vlm_simi(this_video_feature, pre_frame_n = 1):

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


def get_abnormal_frames(features, pre_f_n = 1, std_factor = 1):
    mean_similarities = get_vlm_simi(features, pre_f_n)
    mean = mean_similarities.mean()
    std = mean_similarities.std()
    threshold = mean - std_factor * std
    abnormal_frames = torch.where(mean_similarities < threshold)[0]
    return abnormal_frames


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


def seg_video(features,total_frames, load_range, pre_f_n = 1):
    mean_similarities = get_vlm_simi(features, pre_f_n)

    # perform the 3 sigma rule to detect the abnormal frame
    mean = mean_similarities.mean()
    std = mean_similarities.std()
    threshold = mean - 1 * std
    abnormal_frames = torch.where(mean_similarities < threshold)[0]
    # seg
    segments = segment_video((abnormal_frames+pre_f_n).tolist(), total_frames)
    segments = [(start+load_range[0], end+load_range[0]) for start, end in segments]
    return segments