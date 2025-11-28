import os, torchvision, transformers, tqdm, time, json, math
import torch.multiprocessing as mp
# torchvision.set_video_backend('video_reader')

from data.utils import ffmpeg_once

from data import coinNarrationStream, COIN, build_coin_narration_stream_val
from .inference import LiveInfer
from .inference_highres import LiveInfer_highres
from pathlib import Path
logger = transformers.logging.get_logger('liveinfer')

# python -m demo.coin_narration_videollmonline --resume_from_checkpoint /root/videollm-online/outputs/coin_narration_stream_train/live1 --live_version live1
# python -m demo.coin_narration_videollmonline --resume_from_checkpoint /root/videollm-online/outputs/coin_narration_stream_train/live1_1+ --live_version live1_1+
def ceil_time_by_fps(time: float, fps: int, min_time: float, max_time: float):
    return min(max(math.ceil(time * fps) / fps, min_time), max_time)

class coinNarrationStream_time(coinNarrationStream):
    def __init__(self, *, split: str, frame_fps: int, is_training: bool, augmentation: bool, **kwargs):
        super().__init__(split=split, frame_fps=frame_fps, augmentation=augmentation, is_training=is_training, **kwargs)
        self.is_training = is_training
        self.frame_fps = frame_fps

        annos, origin_annos = self.get_annos(split)
        self.questions = {}
        for video_uid, narrations in tqdm.tqdm(annos.items(), desc=f'narration_stream_{split}...'):
            duration = self.metadata[video_uid]['duration']
            if not narrations:
                continue
            start_time = ceil_time_by_fps(narrations[0]['time'], frame_fps, min_time=0, max_time=duration)
            task = COIN._clean_task(origin_annos[video_uid]["class"])
            instruction = self.instructions[0].copy()
            instruction['content'] = instruction['content'].format(task)
            self.questions[video_uid] = {
                'question': instruction,
                'task': task,
                'start_time': start_time
            }
            
    def get_question(self, video_uid):
        return self.questions[video_uid]
    

def main(liveinfer: LiveInfer, src_video_path=None, question=None, question_time=None):
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
    
    liveinfer.load_video(ffmpeg_video_path, start_time=int(question_time*liveinfer.frame_fps))
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
            print(f"start time is {question_time}",query)
        if response:
            history['conversation'].append({'role': 'assistant', 'content': response, 'time': liveinfer.video_time + start_time, 'fps': fps, 'cost': timecosts[-1]})
            print(f"start time is {question_time}",response)
            all_responses.append(response)
        if not query and not response:
            history['conversation'].append({'time': liveinfer.video_time, 'fps': fps, 'cost': timecosts[-1]})
    # json.dump(history, open(save_history_path, 'w'), indent=4)
    # print(f'The conversation history has been saved to {save_history_path}.')
    return all_responses # last response is answer

if __name__ == '__main__':

    device = 5
    
    user_input = input("which model choice: ").strip().lower()
    if user_input == 'live1':
        output_file_name = 'coin_narration.json'
        liveinfer = LiveInfer(device=f'cuda:{device}')
    elif user_input == 'live1+':
        output_file_name = 'coin_narration.json'
        liveinfer = LiveInfer(device=f'cuda:{device}')
    elif user_input == 'live1_1+':
        output_file_name = 'coin_narration_high.json'
        liveinfer = LiveInfer_highres(device=f'cuda:{device}')
    
    liveinfer.frame_token_interval_threshold = 0.8
    coin_narration_test = coinNarrationStream_time(split='test', 
                                                   frame_fps=liveinfer.frame_fps, is_training=False, augmentation=False,
                                                   embed_mark='2fps_max384_1+3x3', vision_pretrained='google/siglip-large-patch16-384',
                                                   tokenizer=liveinfer.system_prompt, system_prompt=liveinfer.system_prompt, max_num_frames=1200)
    
    # video_ids = ['WgW4sekgPgI', 'M0I6V62QDIQ', 'wfoqCr6FFxE', 'hLuYnt6MQEU', 'pJEbYeh3sHo', 'GjwZc3evESQ', 
    #              '8gN0Hwmtdq8', 'pxfT4zRaqEk', 'LFaKEHlGkXY', 'OdCkXINr60E', 'ETiowy9GJ3c', 
    #              'TQVVmQJjjVE', 'lpwdoT-FDlY', 'MJUi2-anXgc', 'OUG8bwXjm48', '9GxRjGp900A', 'WChbAjPVxj0', 
    #              'ZGCnERtgUtU', 'b2RFa6i8EjM', 'ac1VUY0wFkg', 'fqvlnjZwTEY', 'z-b-R5WYQYY', 'Itos5-EiiuQ', 
    #              'mYFrTHd8NRg', 'DcpwZ5Tqau0', 'MR_olQaxdoQ', 'lIDwYymaxXc']
    
    video_ids = ['yvZ764fCF84', 'otUJa-5slpk', '1ETyUR7zpZM', 'vbsjMBZxZTs', 'aNPXoynTP5s', 'LN3bfOepaB0', 'Lj8YWdeIw7Y', 'qGWvJnsggF0', '2Ziy4XqdXJw', 'TrTTiGNKess']
    video_ids = ['yvZ764fCF84']
    # 'Ax43SwfvnRU'
    # NOTE: EVALUATION
    results = []
    for item in tqdm.tqdm(video_ids):
        question = coin_narration_test.get_question(item)
        answer = main(liveinfer, "/root/videollm-online/datasets/coin/videos/" + item + '.mp4', question['question']['content'], question['start_time'])

        # store results
        results.append({"video_id": item, "conversation": answer})
        json.dump(results, Path(f"/root/videollm-online/data/coin/{output_file_name}").open("w"), indent=2)