import os, torchvision, transformers, tqdm, time, json, math
import torch.multiprocessing as mp
# torchvision.set_video_backend('video_reader')

from data.utils import ffmpeg_once

from data import Llava1_5ITNarrationStreamHighRes
from .inference import LiveInfer
from .inference_highres import LiveInfer_highres
from pathlib import Path
logger = transformers.logging.get_logger('liveinfer')

# python -m demo.ego4d_narration_videollmonline --resume_from_checkpoint /root/videollm-online/outputs/ego4d_narration_train/live1 --live_version live1
# --resume_from_checkpoint chenjoya/videollm-online-8b-v1plus
# python -m demo.ego4d_narration_videollmonline --resume_from_checkpoint /root/videollm-online/outputs/ego4d_narration_train/live1_1+ --live_version live1_1+
def ceil_time_by_fps(time: float, fps: int, min_time: float, max_time: float):
    return min(max(math.ceil(time * fps) / fps, min_time), max_time)

def main(liveinfer: LiveInfer, src_video_path=None, question=None, question_time=None,load_ranges=None):
    liveinfer.reset()
    if src_video_path is None:
        src_video_path = input("Enter the video path: ")
    if question is None:
        question = input("Enter the question: ")

    name, ext = os.path.splitext(src_video_path)
    
    if str(liveinfer.frame_resolution) in name and str(liveinfer.frame_fps) in name:
        ffmpeg_video_path = src_video_path
    else:
        ffmpeg_video_path = os.path.join('demo/assets/cache', name + f'_{liveinfer.frame_fps}fps_{liveinfer.frame_resolution}' + ext)
        
        save_history_path = src_video_path.replace('.mp4', '.json')
        if not os.path.exists(ffmpeg_video_path):
            os.makedirs(os.path.dirname(ffmpeg_video_path), exist_ok=True)
            ffmpeg_once(src_video_path, ffmpeg_video_path, fps=liveinfer.frame_fps, resolution=liveinfer.frame_resolution)
            logger.warning(f'{src_video_path} -> {ffmpeg_video_path}, {liveinfer.frame_fps} FPS, {liveinfer.frame_resolution} Resolution')
    
    print(f"range: {load_ranges}")
    liveinfer.load_video(ffmpeg_video_path, load_ranges)
    liveinfer.input_query_stream(question, video_time=0)

    timecosts = []
    pbar = tqdm.tqdm(total=liveinfer.num_video_frames, bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}{postfix}]")
    history = {'video_path': src_video_path, 'frame_fps': liveinfer.frame_fps, 'conversation': []}
    all_responses = []
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
            history['conversation'].append({'role': 'user', 'content': query, 'time': liveinfer.video_time + start_time, 'fps': fps, 'cost': timecosts[-1]})
            print(query)
            all_responses.append(query)
        if response:
            history['conversation'].append({'role': 'assistant', 'content': response, 'time': liveinfer.video_time + start_time, 'fps': fps, 'cost': timecosts[-1]})
            print(response)
            all_responses.append(response)
        if not query and not response:
            history['conversation'].append({'time': liveinfer.video_time, 'fps': fps, 'cost': timecosts[-1]})
    # json.dump(history, open(save_history_path, 'w'), indent=4)
    # print(f'The conversation history has been saved to {save_history_path}.')
    return all_responses # last response is answer

if __name__ == '__main__':

    device = int(input("which device: ").strip())
    
    user_input = input("which model choice: ").strip().lower()
    if user_input == 'live1':
        output_file_name = 'ego4d_narration.json'
        liveinfer = LiveInfer(device=f'cuda:{device}')
    elif user_input == 'live1+':
        output_file_name = 'ego4d_narration+.json'
        liveinfer = LiveInfer(device=f'cuda:{device}')
    elif user_input == 'live1_1+':
        output_file_name = 'ego4d_narration_high_fake.json'
        liveinfer = LiveInfer_highres(device=f'cuda:{device}', set_vision_inside=False)
    
    liveinfer.frame_token_interval_threshold = 0.725
    ego4d_narration_test = Llava1_5ITNarrationStreamHighRes(split='TextCaps_22k', 
                                                   frame_fps=liveinfer.frame_fps, is_training=False, augmentation=False,
                                                   embed_mark='2fps_max384_1', embed_mark_high='2fps_max384_1+3x3', vision_pretrained='google/siglip-large-patch16-384',
                                                   tokenizer=liveinfer.tokenizer, system_prompt=liveinfer.system_prompt, max_num_frames=10000)
    
    
    sample_idx = json.load(open('/root/videollm-online/data/ego4d/random_numbers.json'))
    
    # NOTE: EVALUATION
    results = []
    for idx in tqdm.tqdm(sample_idx):
        anno = ego4d_narration_test.annos[idx]
        conversation=ego4d_narration_test.preprocess_conversation(anno['conversation'])
        load_ranges=anno['load_ranges']
        
        video_name = list(load_ranges.keys())[0].split('/')[-1].split('.')[0]
        question = conversation[1]['content']
        
        answer = main(liveinfer, "/root/videollm-online/datasets/ego4d/v2/full_scale_2fps_max384/" + video_name + '.mp4', question, load_ranges=list(load_ranges.values())[0])
        
        # store results
        results.append({"video_id": idx, "conversation": answer})
        json.dump(results, Path(f"/root/videollm-online/data/ego4d/{output_file_name}").open("w"), indent=2)