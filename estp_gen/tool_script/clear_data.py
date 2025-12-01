from functools import partial
import random, torch, tqdm, os, subprocess, torchvision, pathlib, submitit, math, json
from tqdm import tqdm
import submitit, transformers
from dataclasses import dataclass
import decord
from decord import VideoReader
import numpy as np

from models.arguments_live import LiveOnePlusTrainingArguments

def ceil_time_by_fps(time: float, fps: int, min_time: float, max_time: float):
    return min(max(math.ceil(time * fps) / fps, min_time), max_time)

# python -m data.preprocess.clear_data

def judge_data(src_path, narration_data):
    # breakpoint()
    try:
        vr = VideoReader(uri=src_path)
    except:
        return True
    for n in narration_data.values():
        for message in n:
            frame_id = int(ceil_time_by_fps(message['time'], 2, min_time=0, max_time=vr._num_frame) * 2)
            frame = vr.get_batch([frame_id]).permute(0, 3, 1, 2)
            if frame.float().mean() < 0.1:
                return True    
    # count = 0   
    # for i in range(vr._num_frame):
    #     frame = vr.get_batch([i]).permute(0, 3, 1, 2)
    #     # judge frame is black
    #     if frame.float().mean() < 0.1:
    #         count += 1
    # if count / vr._num_frame > 0.5:
    #     return True
    return False

def judge_data_read(src_path):
    # breakpoint()
    try:
        vr = VideoReader(uri=src_path)
    except:
        return True
    
    sample_idx = np.linspace(0, vr._num_frame, 10, endpoint=False, dtype=int).tolist()

    try:
        frames = vr.get_batch(sample_idx).permute(0, 3, 1, 2)
    except:
        return True
    # frames = vr.get_batch(sample_idx).permute(0, 3, 1, 2)
    return False


def distributed_clear_data(src_root, json_path, output_path):
    import submitit
    env = submitit.JobEnvironment()
    decord.bridge.set_bridge("torch")
    src_root = src_root.rstrip('/')
    pather = pathlib.Path(src_root)
    src_paths = [str(path) for path in pather.rglob('*') if path.is_file() and str(path).endswith('.mp4')]
    valid_video_uid = json.load(open(json_path))
    try:
        video_judge = json.load(open(f'{output_path}/{env.global_rank}.json'))
    except:
        video_judge = {}
    for i, src_path in tqdm(enumerate(src_paths)):
        if i % env.num_tasks != env.global_rank:
            continue
        video_id = src_path.split('/')[-1].split('.')[0]
        
        # try:
        #     narration_info = narration_data[video_id]
        # except:
        #     continue
        
        if video_id not in valid_video_uid:
            continue

        if video_id not in video_judge:
            video_judge[video_id] = judge_data_read(src_path)

        json.dump(video_judge, open(f'{output_path}/{env.global_rank}.json', 'w'))
    return video_judge

# python -m data.preprocess.ffmpeg_highImage --frame_fps 1 --frame_resolution 384 --num_tasks 16 --video_dir /mnt/extra/dataset/ego4d/v2/full_scale/
@dataclass
class LiveOnePlusEncodingArguments(LiveOnePlusTrainingArguments):
    num_nodes: int = 1
    num_tasks: int = 16
    video_dir: str = 'datasets/ego4d/v2/full_scale'
    slurm_partition: str = None
    
if __name__ == "__main__":
    args, = transformers.HfArgumentParser(LiveOnePlusEncodingArguments).parse_args_into_dataclasses()
    executor = submitit.AutoExecutor(folder=f"outputs/preprocess/", cluster='local' if args.num_nodes == 1 else 'slurm')
    task = partial(distributed_clear_data, src_root='/mnt/extra/dataset/ego4d/v2/full_scale_2fps/', 
                    json_path='/home/zhangyl/videollm-online/data/estp/all_video_uids.json', 
                    output_path='/home/zhangyl/videollm-online/data/preprocess')
    executor.update_parameters(
        tasks_per_node=args.num_tasks,
        nodes=args.num_nodes,
        slurm_partition=args.slurm_partition,
        cpus_per_task=10,
        mem_gb=240,
        slurm_time='48:00:00',
        timeout_min=60000,
    )
    job = executor.submit(task)
    job.results() 
    
# if __name__ == "__main__":
#     distributed_clear_data('/mnt/extra/dataset/ego4d/v2/full_scale_2fps_max384/', '/home/zhangyl/videollm-online/datasets/ego4d/v2/annotation/refined_narration_stream_train.json', '/home/zhangyl/videollm-online/data/preprocess')