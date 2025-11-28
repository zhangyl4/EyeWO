import os, torchvision, transformers, tqdm, time, json, re, traceback
import torch.multiprocessing as mp
# torchvision.set_video_backend('video_reader')

from data.utils import ffmpeg_once

from .inference import LiveInfer
from pathlib import Path
logger = transformers.logging.get_logger('liveinfer')
from torch.utils.data import Dataset, DataLoader

# python -m demo.cli --resume_from_checkpoint ... 

def main(liveinfer: LiveInfer, src_video_path=None, question=None):
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
    liveinfer.input_query_stream(question, video_time=int((liveinfer.num_video_frames - 1) / liveinfer.frame_fps)) # question in last
    
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
            history['conversation'].append({'role': 'user', 'content': query, 'time': liveinfer.video_time, 'fps': fps, 'cost': timecosts[-1]})
            print(query)
        if response:
            history['conversation'].append({'role': 'assistant', 'content': response, 'time': liveinfer.video_time, 'fps': fps, 'cost': timecosts[-1]})
            print(response)
            all_responses.append(response)
        if not query and not response:
            history['conversation'].append({'time': liveinfer.video_time, 'fps': fps, 'cost': timecosts[-1]})
    json.dump(history, open(save_history_path, 'w'), indent=4)
    print(f'The conversation history has been saved to {save_history_path}.')
    return all_responses[-1].split("Assistant:")[-1].strip() # last response is answer


class EgoschemaDataset(Dataset):

    video_formats = ['.mp4', '.avi', '.mov', '.mkv']

    def __init__(self, data_folder, data_list):
        self.data_folder = data_folder
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        line = self.data_list[idx]
        q_uid = line['q_uid']

        for fmt in self.video_formats:  # Added this line
            temp_path = os.path.join(self.data_folder, f"{q_uid}{fmt}")
            if os.path.exists(temp_path):
                video_path = temp_path
                break

        question = line['question']
        a0 = line['option 0']
        a1 = line['option 1']
        a2 = line['option 2']
        a3 = line['option 3']
        a4 = line['option 4']
        axs = [a0, a1, a2, a3, a4]
        ops = ['(A)', '(B)', '(C)', '(D)', '(E)']

        instruct = f'Select the best answer to the following multiple-choice question based on the video.\n{question}\nOptions:\n(A) {a0}\n(B) {a1}\n(C) {a2}\n(D) {a3}\n(E) {a4}\nAnswer with the option\'s letter from the given choices directly and only give the best option. The best answer is: ' 

        return {
            'q_uid': q_uid,
            'video_path': video_path,
            'instruct': instruct,
        }

def egoschema_dump(ans_file, line, outputs):
    for idx, output in enumerate(outputs):
        q_uid = line['q_uid'][idx]
        instruct = line['instruct'][idx]
        letters = ['A', 'B', 'C', 'D', 'E']

        output = output.replace('answer', '')
        output = output.replace('Answer', '')
        pred_answer = re.findall('[\(\ ]*[A-E][\)\ ]*', output)
        try:
            
            assert len(pred_answer) >= 1, 'The video \"{}\" instruct: \n\"{}\"\n output: \n\"{}\"\n is not in the expected format'.format(line['q_uid'], instruct, output)
            pred_answer = pred_answer[0].strip()
            pred_answer = pred_answer.strip('()')
            pred_idx = letters.index(pred_answer)
        except:
            traceback.print_exc()
            pred_idx = 2

        ans_file.write(f'{q_uid}, {pred_idx}\n')


if __name__ == '__main__':
    # NOTE：MODEL
    # sys_prompt = (
    #     "A multimodal AI assistant is embedded in a simulated embodied robot, assisting users with physical tasks and interactions by analyzing video data from its vision system and responding to user questions."
    #     " Below is their conversation, interleaved with the list of video frames received by the robot and the user's questions."
    # )
    
    # sys_prompt = (
    #     "A multimodal AI assistant is embedded in a simulated embodied robot, tasked with helping users by analyzing streamingframes data from the robot's vision system and responding to questions based on this data."
    #     " Users will provide a question and five possible answer choices. The AI assistant needs to observe the streamingframes and select the most appropriate answer from the given options."
    #     " Below is the conversation, interleaved with the list of video frames received by the robot and the user's question along with the five possible answer choices."
    # )
    # sys_prompt = (
    #     "You act as the Al assistant on user's AR glass. \
    #     The AR glass is continuously receiving streamingframes of the user's view, \
    #     and your task is to simply describe what you have seen. Are you ready toreceive streaming frames?"
    # )
    
    sys_prompt = (
        "A multimodal AI assistant is helping users with some activities."
        " Below is their conversation, interleaved with the list of video frames received by the assistant."
        " The user will provide a question and five possible answer choices (A, B, C, D, E)."
        " The assistant should analyze the video frames and, based on its observations and reasoning, select the most appropriate answer from A, B, C, D, or E."
    ) 
    liveinfer = LiveInfer(device='cuda:7', system_prompt=sys_prompt)
    liveinfer.frame_fps = 32 / 180
        
    # NOTE: DATASET
    # load dataset
    EGO_PATH = "/root/dataset/egoschema/good_clips_git"
    QUESION_PATH = "/root/dataset/egoschema/EgoSchema/questions.json"
    SUB_ANS_PATH = "/root/dataset/egoschema/EgoSchema/subset_answers.json"
    questions_f = open(QUESION_PATH)
    questions = json.load(questions_f)

    answers_f = open(SUB_ANS_PATH)
    answers = json.load(answers_f)

    sub_questions = []
    for question in questions:
        if question['q_uid'] in answers.keys():
            sub_questions.append(question)
    
    dataset = EgoschemaDataset(EGO_PATH, sub_questions)
    val_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
    
    # NOTE: EVALUATION
    # Iterate over each sample in the ground truth file
    results = []
    for i, line in enumerate(tqdm.tqdm(val_loader)):
        video_path = line['video_path'][0]
        instruct = line['instruct'][0]

        try:
            answer = main(liveinfer, video_path, instruct)
        except:
            traceback.print_exc()
            answer = 'C'

        # store results
        results.append({"question_id": line['q_uid'], "answer": answer})
        json.dump(results, Path("/root/dataset/egoschema/EgoSchema/benchmarking/videollm/llmonline_egoschema.json").open("w"), indent=2)

    
    