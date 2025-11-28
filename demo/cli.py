import os, torchvision, transformers, tqdm, time, json
import torch.multiprocessing as mp
# torchvision.set_video_backend('video_reader')

from data.utils import ffmpeg_once

from .inference import LiveInfer
from .inference_llaveNext import LiveInfer_llavaNext
logger = transformers.logging.get_logger('liveinfer')

# python -m demo.cli --resume_from_checkpoint ... 

def main(liveinfer: LiveInfer, src_video_path=None, question=None, question_time=0.0):
    liveinfer.reset()
    if src_video_path is None:
        src_video_path = input("Enter the video path: ")
    if question is None:
        question = input("Enter the question: ")

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

if __name__ == '__main__':
    # sys_prompt = (
    #     "A multimodal AI assistant is embedded in a simulated embodied robot, assisting users with physical tasks and interactions by analyzing video data from its vision system and responding to user questions."
    #     " Below is their conversation, interleaved with the list of video frames received by the robot and the user's questions."
    # )
    
    # sys_prompt = (
    #     "A multimodal AI assistant is embedded in a simulated embodied robot, tasked with helping users by analyzing streamingframes data from the robot's vision system and responding to questions based on this data."
    #     " Users will provide a question and five possible answer choices. The AI assistant needs to observe the streamingframes and select the most appropriate answer from the given options."
    #     " Below is the conversation, interleaved with the list of video frames received by the robot and the user's question along with the five possible answer choices."
    # )
    sys_prompt = (
        "You act as the Al assistant on user's AR glass. \
        The AR glass is continuously receiving streamingframes of the user's view, \
        and your task is to simply describe what you have seen. Are you ready toreceive streaming frames?"
    )
    
    # sys_prompt = (
    #     "You are an embodied robot placed in an indoor environment. You continuously receive a stream of images captured from the indoor scene."
    #     "Occasionally, the user will interject with a question. Your task is to observe the image stream and, at the appropriate time, provide an accurate response to the user's question."
    # )
    
    liveinfer = LiveInfer(device='cuda:4', system_prompt=None)
    liveinfer = LiveInfer_llavaNext(device='cuda:7', system_prompt=sys_prompt)
    src_video_path = None  # Store the last used src_video_path
    while True:
        user_input = input("Do you want to set a new video path? (yes/no): ").strip().lower()
        if user_input == "yes":
            src_video_path = input("Enter the video path: ")
        question = input("Enter the question: ")
        question_time = float(input("Enter the question time (default 0.0): ").strip() or "0.0")
        main(liveinfer, src_video_path, question, question_time)
        continue_input = input("Do you want to process another video? (yes/no): ").strip().lower()
        if continue_input == "no":
            break
