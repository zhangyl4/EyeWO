import json, torch, tqdm, random, os

from ..ego4d import Ego4D
from ..stream import StreamMixIn
from ..utils import ceil_time_by_fps, floor_time_by_fps, rand_bool, DictWithTo, get_video_metadata_clip_video, load_frames_f, get_path_with_key, default_dump
from ..utils import visionTextAligner, get_vlm_simi, get_abnormal_frames
from transformers import PreTrainedTokenizer, EvalPrediction

def satisfy_condition(anno):
    high_frame_number = 0
    answer_number = 0
    frame_number = 0
    is_query = False
    frame_number_before_query = 0
    dis_pre_response = []
    for conv in anno['conversation']:
        if conv['role'].lower() == 'user':
            is_query = True
        if conv['role'].lower() == 'assistant':
            answer_number += 1 if is_query else 0
        elif conv['role'].lower() == 'stream': 
            if is_query:
                frame_number += conv['num_frames']
                dis_pre_response.append(conv['num_frames'])
            else:
                frame_number_before_query += conv['num_frames']
        elif conv['role'].lower() == 'stream_high':
            if is_query:
                high_frame_number += conv['num_frames']
            
    
    this_turn_response_clip_number = 0
    response_clip_length = 0
    for response in anno['reponse_clip']:
        this_turn_response_clip_number += response[1] - response[0]
    response_clip_length = this_turn_response_clip_number / (frame_number + frame_number_before_query)
    
    # frame number control
    if frame_number > 1200:
        return False
    
    if frame_number == 0:
        return False
    
    # response clip length control
    if response_clip_length > 0.90 and answer_number > 5:
        return False
    
    if [x for x in dis_pre_response if x > 1000]:
        return False
    
    if len([x for x in dis_pre_response if x <= 2]) > 1:
        return False
    
    if answer_number / (frame_number + frame_number_before_query) > 0.5:
        return False
    
    if answer_number > 60:
        return False
    
    return True


def satisfy_condition_post(anno):
    high_frame_number = 0
    answer_number = 0
    frame_number = 0
    is_query = False
    frame_number_before_query = 0
    dis_pre_response = []
    for conv in anno['conversation']:
        if conv['role'].lower() == 'user':
            is_query = True
        if conv['role'].lower() == 'assistant':
            answer_number += 1 if is_query else 0
        elif conv['role'].lower() == 'stream': 
            if is_query:
                frame_number += conv['num_frames']
                dis_pre_response.append(conv['num_frames'])
            else:
                frame_number_before_query += conv['num_frames']
        elif conv['role'].lower() == 'stream_high':
            if is_query:
                high_frame_number += conv['num_frames']
            
    
    this_turn_response_clip_number = 0
    response_clip_length = 0
    for response in anno['reponse_clip']:
        this_turn_response_clip_number += response[1] - response[0]
    response_clip_length = this_turn_response_clip_number / (frame_number + frame_number_before_query)
    
    
    # not learn case
    if answer_number == 0:
        return False
    
    # frame number control
    if frame_number > 1200:
        return False
    
    if frame_number == 0:
        return False
    
    # response clip length control
    if response_clip_length > 0.90 and answer_number > 5:
        return False
    
    if [x for x in dis_pre_response if x > 1000]:
        return False
    
    # if len([x for x in dis_pre_response if x <= 2]) > 1:
    #     return False
    
    if answer_number / (frame_number + frame_number_before_query) > 0.5:
        return False
    
    if answer_number > 60:
        return False
    
    return True


def time_region2time_stamp(conversation): # TODO: add different time stamp for different task
    for conv in conversation:
        if conv['role'].lower() == 'assistant':
            conv['time'] = (conv['end_time'] + conv['start_time']) / 2.0
    return conversation

class Ego4DESTPSQA(Ego4D, StreamMixIn):
    evaluation_kwargs = DictWithTo(evaluator='lm_evaluate_analysis')
    
    task2number = {
            "Object State Change Recognition": 0,
            "Ego Object State Change Recognition": 0,
            "Object Localization": 0,
            "Action Recognition": 0,
            "Action Reasoning": 0,
            "Object Recognition": 0,
            "Ego Object Localization": 0,
            "Object Function": 0,
            "Task Understanding": 0,
            "Attribute Perception": 0,
            "Information Function": 0,
            "Text-Rich Understanding": 0,
            "Object Relative Context": 0,
            "Task Relative Context": 0,
             "Irrelevant Context": 0,
        }
    
    def get_metadata(self, ):
        metadata_path = f'{self.embed_dir}_metadata.json'
        if  ('2fps_max384_1_google' not in metadata_path): # ('2fps_max384_1+3x3_google' not in metadata_path) and
            self.embed_dir = get_path_with_key(self.embed_dir, 'max384')
            metadata_path = self.embed_dir + "_metadata.json"
        if os.path.exists(metadata_path):
            print(f'load {metadata_path}...')
            metadata = json.load(open(metadata_path))
        else:
            metadata = {}
            for file in tqdm.tqdm(os.listdir(self.embed_dir), desc=f'prepare {metadata_path}...'):
                path = os.path.join(self.embed_dir, file)
                duration, meta_path = get_video_metadata_clip_video(path, self.frame_fps)
                key = os.path.splitext(os.path.basename(path))[0]
                metadata[key] = {'duration': duration, 'path': meta_path}
            json.dump(metadata, open(metadata_path, 'w'), indent=4)
        return metadata
    
    def get_conversation(self, video_uid, clip_uid, anno):
        # 1. gen question time
        if not anno['conversation']:
            return None
        
        is_query = False
        try:
            for conv in anno['conversation']:
                if conv['role'].lower() == 'user':
                    is_query = True
                    break
        except:
            breakpoint()
        
        if is_query:
            if "start_time" in anno:
                anno['clip_start_time'] = anno['start_time']
                anno['clip_end_time'] = anno['end_time']
            return anno
        
        # HACK : if not question skip
        if not is_query and 'question' not in anno:
            return None
        
        
        anno['conversation'] = time_region2time_stamp(anno['conversation'])
        
        conversation = []
        stamps = [msg["time"] for msg in anno["conversation"] if "time" in msg]
        left_time = [anno['conversation'][0]['start_time'] - anno['clip_start_time']] + [anno['conversation'][i]['start_time'] - anno['conversation'][i-1]['end_time'] for i in range(1, len(anno['conversation']))]
        stamps_left_time_pair = list(zip(stamps, left_time))
        if stamps_left_time_pair :
            stamps_left_time_pair.sort(key=lambda x: x[0])
            
            if len(stamps_left_time_pair) > 1 and random.random() < 0.3:
                idx = random.randint(-1, len(stamps_left_time_pair) - 2)
                if idx == -1:
                    t1 = conversation[-1]['end_time'] if conversation else anno['clip_start_time']
                else:
                    t1 = anno["conversation"][idx]["end_time"]
                t2 = anno["conversation"][idx+1]['start_time']

                while t1 > t2-1/self.frame_fps and idx < len(anno["conversation"]) - 2:
                    idx += 1
                    t1 = anno["conversation"][idx]["end_time"]
                    t2 = anno["conversation"][idx+1]['start_time']
                
                if t1 > t2-1/self.frame_fps:
                    return None
                # question_time = (t1 + t2) / 2.0
                # 或者：
                question_time = random.uniform(t1, t2-1/self.frame_fps)
            else:
                t1 = conversation[-1]['time'] if conversation else anno['clip_start_time']
                t2 = stamps[0]
                question_time = random.uniform(t1, t2-1/self.frame_fps)
        
        # 2. insert question
        conversation.append({
            "role": "user",
            "content": anno["question"],
            "time": question_time,
        })

        
        # 3. 如果is_smoothing为True，则将conversation中的time和end_time进行平滑
        if hasattr(self, 'is_smoothing') and self.is_smoothing:
            for answer in sorted(anno["conversation"], key=lambda x: x['time']):
                answer['time'] = random.uniform(answer['time'], answer['end_time'])
         
        # 4. insert answer
        for answer in sorted(anno["conversation"], key=lambda x: x['time']):
            answer_copy = answer.copy()
            if answer['time'] > question_time:
                conversation.append(answer_copy)
        
        
        # 5. 如果user下一个不是assistant，删除user
        i = 0
        while i < len(conversation) - 1:
            if conversation[i]['role'].lower() == 'user' and conversation[i+1]['role'].lower() != 'assistant':
                conversation.pop(i)
            else:
                i += 1
        
        if len(conversation) < 2:
            return None
        
        new_anno = {}
        new_anno['conversation'] = conversation
        new_anno['video_uid'] = video_uid
        new_anno['clip_uid'] = clip_uid
        new_anno['clip_start_time'] = anno['clip_start_time']
        new_anno['clip_end_time'] = anno['clip_end_time']
        new_anno['Task Type'] = anno['Task Type']
        return new_anno
    
    def preprocess_conversation(self, anno):
        # 2. gen sequence
        video_uid = anno['video_uid']
        duration = self.metadata[video_uid]['duration']


        if not anno['conversation']:
            return None
        role = anno['conversation'][0]['role'].lower()
        time = anno['conversation'][0]['time']
        fps_time = ceil_time_by_fps(time, self.frame_fps, 0, duration)
        content = anno['conversation'][0]['content']
        if not (role == 'user' and time >= 0 and time <= duration and content):
            return None
    
        # 1. add random frames before the user
        start_fps_time = ceil_time_by_fps(anno['clip_start_time'], self.frame_fps, 0, duration)
        # TODO: demo experiment
        start_fps_time = fps_time
        waiting_frames = int(fps_time*self.frame_fps) - int(start_fps_time*self.frame_fps) + 1
        
        
        
        conversation = []
        if waiting_frames:
            conversation.append({'role': 'stream', 'num_frames': waiting_frames, 'learn': waiting_frames - 1})
        conversation.append({'role': 'user', 'content': content, 'time': time, 'fps_time': fps_time})
        
        # HACK : response clip
        reponse_clip = []
        
        # 2. for loop to add message
        for message in anno['conversation'][1:]:
            role, content, time = message['role'].lower(), message['content'], message['time']
            fps_time = ceil_time_by_fps(time, self.frame_fps, 0, duration)
            if fps_time > duration:
                break
            if fps_time < conversation[-1]['fps_time']:
                break
            if fps_time == conversation[-1]['fps_time']:
                if role == 'user' and conversation[-1]['role'].lower() == 'user':
                    break
                elif role == 'user':
                    conversation.pop()
                    conversation.extend([{'role': 'user', 'content': content, 'time': time, 'fps_time': fps_time, 'learn': False}])
                    continue
                else: # assistant
                    continue
            if role == 'user':
                fps_time = floor_time_by_fps(time, self.frame_fps, conversation[-1]['fps_time'], duration)
                if fps_time > duration:
                    break
                if fps_time > conversation[-1]['fps_time']:
                    num_next_frames = int((fps_time - conversation[-1]['fps_time']) * self.frame_fps)
                    if num_next_frames >= 1:
                        conversation.append({'role': 'stream', 'num_frames':num_next_frames, 'learn': True})
                    if int((fps_time - conversation[-2]['fps_time']) * self.frame_fps) < 0:
                        breakpoint()
                conversation.append({'role': 'user', 'content': content, 'time': time, 'fps_time': fps_time})
            else:
                fps_time = ceil_time_by_fps(time, self.frame_fps, conversation[-1]['fps_time'], duration)
                if fps_time > duration:
                    break
                if fps_time > conversation[-1]['fps_time']:
                    num_next_frames = int((fps_time - conversation[-1]['fps_time']) * self.frame_fps)
                    if num_next_frames >= 1:
                        conversation.append({'role': 'stream', 'num_frames': num_next_frames, 'learn': True})
                        conversation.append({'role': 'assistant', 'content': content, 'time': time, 'fps_time': fps_time, 'learn': True})

                    if int((fps_time - conversation[-3]['fps_time']) * self.frame_fps) < 0:
                        breakpoint()
                    
                    try:
                        clip_start_frame_idx = int((ceil_time_by_fps(message['start_time'], self.frame_fps, start_fps_time, duration)-start_fps_time)*self.frame_fps)
                        clip_end_frame_idx = int((ceil_time_by_fps(message['time'], self.frame_fps, 0, duration)-start_fps_time)*self.frame_fps+1)
                        clip_start_frame_idx = max(clip_start_frame_idx, clip_end_frame_idx - 50)
                        reponse_clip.append((clip_start_frame_idx, clip_end_frame_idx))
                    except:
                        breakpoint()
        
        # n_f = 0
        # for conv in conversation:
        #     if 'num_frames' in conv and conv['num_frames'] < 0:
        #         breakpoint()
        #     if 'num_frames' in conv:
        #         n_f += conv['num_frames']
        # if len(reponse_clip) > 0 and reponse_clip[-1][1] != n_f:
        #     breakpoint()
        # if int(conversation[-1]['fps_time']*self.frame_fps) + 1 - int(start_fps_time*self.frame_fps) != n_f:
        #     breakpoint()


        try:
            new_anno = {
                'conversation': conversation,
                'load_ranges': {tuple(self.metadata[video_uid]['path']): range(int(start_fps_time*self.frame_fps), int(conversation[-1]['fps_time']*self.frame_fps)+1)} if isinstance(self.metadata[video_uid]['path'], list) else {self.metadata[video_uid]['path']: range(int(start_fps_time*self.frame_fps), int(conversation[-1]['fps_time']*self.frame_fps)+1)},
                'Task Type': anno['Task Type'],
                'reponse_clip': reponse_clip,
            }
        except:
            new_anno = None
        return new_anno
    
    
    # def compute_metrics(self, eval_predictions: EvalPrediction, tokenizer: PreTrainedTokenizer, **kwargs):
        
    #     batch_pred_tensor, sample_idxs = eval_predictions.predictions, eval_predictions.label_ids
    #     batch_pred_tensor[batch_pred_tensor < 0] = tokenizer.bos_token_id # not use clamp(min=0), since 0 is ! in Llama-3 tokenizer and may affect matching
    #     predictions = tokenizer.batch_decode(batch_pred_tensor, skip_special_tokens=True, clean_up_tokenization_spaces=True)

    #      # HACK: for storing the predictions and labels for analysis
    #     results = []
        
    #     for i, prediction in enumerate(predictions): # should be self.labels[sample_idx] to get the correct order
    #         # HACK: for storing the predictions and labels for analysis
    #         results.append({
    #             'sample_index': sample_idxs[i],
    #             'prediction': prediction,
    #         })
            
    #     with open(f'{self.output_dir}/{self.__class__.__name__}_multi_step_results.json', mode='w', encoding='utf-8') as file:
    #         json.dump(results, file, ensure_ascii=False, indent=4, default=default_dump)
            
    #     return dict(accuracy=0) # * 100
        
        
    def __init__(self, *, anno_path: str, frame_fps: int, is_training: bool, **kwargs):
        super().__init__(frame_fps=frame_fps, is_training=is_training, **kwargs)
        self.is_training = is_training
        self.frame_fps = frame_fps
        self.anno_path = anno_path
        annos = json.load(open(self.anno_path))
        self.original_annos = []
        self.annos = []
        self.data_repeat_num = kwargs.get('data_repeat_num', 1)
        
        print("start loading questions...")
        processed_annos_path = self.anno_path.replace('.json', f'_processed_{kwargs.get("is_smoothing", "")}_{kwargs.get("add_random_high_res_ratio", "")}_{self.data_repeat_num}.pth')
        if os.path.exists(processed_annos_path):
            print(f'load {processed_annos_path}...')
            self.annos = torch.load(processed_annos_path)
            c = 0
            new_anno = []
            print(len(self.annos))
            for anno in self.annos:
                c += 1
                if satisfy_condition_post(anno):
                    new_anno.append(anno)
                    if anno['Task Type'].strip() in self.task2number:
                        self.task2number[anno['Task Type'].strip()] += 1
            self.annos = new_anno
            print(len(self.annos))
        else:   
            print(f'{processed_annos_path} not found, start processing...')
            c = 0
            for video_uid in annos.keys():
                for clip_uid in annos[video_uid].keys():
                    for qa in annos[video_uid][clip_uid]:
                        c += 1
                        
                        for _ in range(self.data_repeat_num):
                            anno = self.get_conversation(video_uid, clip_uid, qa)
                            if anno is not None:
                                # if anno['Task Type'].strip() == 'Action Reasoning':
                                #     print(anno)
                                #     breakpoint()
                                processed_anno = self.preprocess_conversation(anno)
                                if processed_anno is not None and satisfy_condition(processed_anno):
                                    self.original_annos.append(anno)
                                    self.annos.append(processed_anno)
                                    
                                    if anno['Task Type'].strip() in self.task2number:
                                        self.task2number[anno['Task Type'].strip()] += 1
                # if c > 50:
                #     break
            
            torch.save(self.annos, processed_annos_path)
                   
        print(f'total {c} questions loaded')
        print(f'{len(self.annos)} questions loaded')
        print(sum(self.task2number.values()))
        print(json.dumps(self.task2number, indent=4))

    def stream_getitem_rewrite(self, *, conversation: list[dict], load_ranges: dict[str, range] | torch.Tensor = None, add_generation_prompt=False, **kwargs):
        # 1. load visual encoding
        if isinstance(load_ranges, torch.Tensor):
            frames = load_ranges
        elif load_ranges is not None:
            conversation, load_ranges = self.max_frames_clip(conversation, load_ranges, self.max_num_frames)
            frames = load_frames_f(load_ranges)
        else:
            frames = torch.tensor([])
        # 2. prepare texts
        if self.augmentation:
            conversation = self.augment(conversation)
        conversation = [{"role": "system", "content": self.system_prompt}] + conversation
        text = self.tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=add_generation_prompt)
        # 3. learn ranges
        learn_ranges = self.tokenizer.get_learn_ranges(conversation) if not add_generation_prompt else []
        return text, frames, learn_ranges # num_conversation = 3 * num_frames_high + 2 (quesion, system_prompt)
    
    def __getitem__(self, index):
        
        anno = self.annos[index]
        return *self.stream_getitem_rewrite(
            conversation=anno['conversation'],
            load_ranges=anno['load_ranges'],
        ), index, self.evaluation_kwargs, anno['reponse_clip']

def build_ego4d_ESTPSQA(**kwargs):
    return Ego4DESTPSQA(**kwargs)


class Ego4DESTPCQA(Ego4DESTPSQA):
    def get_conversation(self, video_uid, clip_uid, anno):
        new_anno = {}
        new_anno['conversation'] = anno['conversation']
        new_anno['video_uid'] = video_uid
        new_anno['clip_uid'] = clip_uid
        new_anno['clip_start_time'] = anno['clip_start_time']
        new_anno['clip_end_time'] = anno['clip_end_time']
        new_anno['Task Type'] = anno['Task Type']

        return new_anno
        
def build_ego4d_ESTPCQA(**kwargs):
    return Ego4DESTPCQA(**kwargs)
        

class Ego4DESTPSQAHighRes(Ego4DESTPSQA):
    def __init__(self, *, frame_fps: int, is_training: bool, **kwargs):
        if kwargs.get("root", None) is not None:
            self.root = kwargs.get("root", None)
            self.video_root = os.path.join(self.root, 'full_scale')
            self.anno_root = os.path.join(self.root, 'annotations')
        
        self.frame_fps = frame_fps
        self.is_training = is_training
        self.embed_dir_high = f"{self.video_root}_{kwargs['embed_mark_high']}_{kwargs['vision_pretrained'].replace('/', '--')}"
        self.metadata_high = self.get_metadata_high()
        self.add_random_high_res_ratio = kwargs.get('add_random_high_res_ratio', None)
        
        self.is_smoothing = kwargs.get('is_smoothing', False)
        print(f'is_smoothing: {self.is_smoothing}')
        print(f'add_random_high_res_ratio: {self.add_random_high_res_ratio}')
        
        super().__init__(frame_fps=frame_fps, is_training=is_training, **kwargs)
        
    
    def get_metadata(self, ):
        metadata_path = f'{self.embed_dir}_metadata.json'
        if  ('2fps_max384_1_google' not in metadata_path): # ('2fps_max384_1+3x3_google' not in metadata_path) and
            self.embed_dir = get_path_with_key(self.embed_dir, 'max384')
            metadata_path = self.embed_dir + "_metadata.json"
        if os.path.exists(metadata_path):
            print(f'load {metadata_path}...')
            metadata = json.load(open(metadata_path))
        else:
            metadata = {}
            for file in tqdm.tqdm(os.listdir(self.embed_dir), desc=f'prepare {metadata_path}...'):
                path = os.path.join(self.embed_dir, file)
                duration, meta_path = get_video_metadata_clip_video(path, self.frame_fps)
                key = os.path.splitext(os.path.basename(path))[0]
                metadata[key] = {'duration': duration, 'path': meta_path}
            json.dump(metadata, open(metadata_path, 'w'), indent=4)
        return metadata
    
    def get_metadata_high(self, ):
        metadata_path = f'{self.embed_dir_high}_metadata.json'
        if ('2fps_max384_1+3x3' not in metadata_path):
            self.embed_dir_high = get_path_with_key(self.embed_dir_high, 'max384')
            metadata_path = self.embed_dir_high + "_metadata.json"
        if os.path.exists(metadata_path):
            print(f'load {metadata_path}...')
            metadata = json.load(open(metadata_path))
        else:
            metadata = {}
            for file in tqdm.tqdm(os.listdir(self.embed_dir_high), desc=f'prepare {metadata_path}...'):
                path = os.path.join(self.embed_dir_high, file)
                duration, meta_path = get_video_metadata_clip_video(path, self.frame_fps)
                key = os.path.splitext(os.path.basename(path))[0]
                metadata[key] = {'duration': duration, 'path': meta_path}
            json.dump(metadata, open(metadata_path, 'w'), indent=4)
        return metadata
    
    
    def _add_stream_and_high_res(self, conversation, high_res_times, num_next_frames, fps_time, add_random_high_res_ratio=None, is_learn=True):
        if add_random_high_res_ratio is None:
            conversation.extend([
                {'role': 'stream', 'num_frames': num_next_frames, 'learn': is_learn},
                {"role": "stream_high", 'num_frames': 1, 'learn': is_learn},
                ])
            high_res_times.append(int(fps_time*self.frame_fps))
        else:
            try:
                add_random_high_res_ratio = float(add_random_high_res_ratio)
            except:
                raise ValueError(f'add_random_high_res_ratio must be a float, but got {add_random_high_res_ratio}')
            if 0 < add_random_high_res_ratio <= 1:
                # Calculate number of high res frames to insert
                num_high_res = max(1, int(num_next_frames * add_random_high_res_ratio))
                # Generate random positions to insert high res frames
                high_res_positions = sorted(random.sample(range(num_next_frames), num_high_res))
                
                current_pos = 0
                for pos in high_res_positions:
                    if pos == 0:
                        continue
                    if pos > current_pos:
                        # Add regular frames up to this position
                        conversation.append({
                            'role': 'stream', 
                            'num_frames': pos - current_pos,
                            'learn': True
                        })
                    # Add high res frame
                    conversation.extend([
                        {"role": "stream_high", 'num_frames': 1, 'learn': is_learn}
                    ])
                    high_res_times.append(int(fps_time*self.frame_fps) - num_next_frames + pos)
                    current_pos = pos
                
                # Add remaining regular frames if any
                if current_pos < num_next_frames:
                    conversation.append({
                        'role': 'stream',
                        'num_frames': num_next_frames - current_pos,
                        'learn': True
                    })
                
                    # add last stream_high
                    conversation.extend([
                        {"role": "stream_high", 'num_frames': 1, 'learn': is_learn}
                    ])  
                    high_res_times.append(int(fps_time*self.frame_fps))
            else:
                # Fallback to default behavior if ratio is invalid
                conversation.extend([
                    {'role': 'stream', 'num_frames': num_next_frames, 'learn': is_learn},
                    {"role": "stream_high", 'num_frames': 1, 'learn': is_learn},
                ])
                high_res_times.append(int(fps_time*self.frame_fps))
    

    def preprocess_conversation(self, anno):
        # 2. gen sequence
        video_uid = anno['video_uid']

        
        duration = self.metadata[video_uid]['duration']

        if not anno['conversation']:
            return None
        role = anno['conversation'][0]['role'].lower()
        time = anno['conversation'][0]['time']
        fps_time = ceil_time_by_fps(time, self.frame_fps, 0, duration)
        content = anno['conversation'][0]['content']
        if not (role == 'user' and time >= 0 and time <= duration and content):
            return None
    
        # 1. add random frames before the user
        start_fps_time = ceil_time_by_fps(anno['clip_start_time'], self.frame_fps, 0, duration)
        # TODO: demo experiment
        start_fps_time = fps_time
        waiting_frames = int(fps_time*self.frame_fps) - int(start_fps_time*self.frame_fps) + 1
        
        conversation = []
        
        # HACK : high resolution image time
        high_res_times = []
        # HACK : response clip
        reponse_clip = []
                
        if waiting_frames:
            conversation.extend([{'role': 'stream', 'num_frames': waiting_frames, 'learn': waiting_frames-1},
                                {"role": "stream_high", 'num_frames': 1, 'learn': False},
                                ])
            high_res_times.append(int(fps_time*self.frame_fps))
        conversation.extend([{'role': 'user', 'content': content, 'time': time, 'fps_time': fps_time}])
        
        # 2. for loop to add message
        for message in anno['conversation'][1:]:
            role, content, time = message['role'].lower(), message['content'], message['time']
            fps_time = ceil_time_by_fps(time, self.frame_fps, 0, duration)
            if fps_time > duration:
                break
            if fps_time < conversation[-1]['fps_time']:
                break
            if fps_time == conversation[-1]['fps_time']:
                if role == 'user' and conversation[-1]['role'].lower() == 'user':
                    break
                elif role == 'user':
                    conversation.pop()
                    conversation.extend([{'role': 'user', 'content': content, 'time': time, 'fps_time': fps_time, 'learn': False}])
                    continue
                else: # assistant
                    continue
            if role == 'user':
                if fps_time > duration:
                    break
                if fps_time > conversation[-1]['fps_time']:
                    num_next_frames = int((fps_time - conversation[-1]['fps_time']) * self.frame_fps)
                    if num_next_frames >= 1:
                        self._add_stream_and_high_res(conversation, high_res_times, num_next_frames, fps_time, self.add_random_high_res_ratio, is_learn=False)
                conversation.extend([{'role': 'user', 'content': content, 'time': time, 'fps_time': fps_time}])
                
                # HACK : not add response clip
                # clip_start_frame_idx = int((ceil_time_by_fps(message['start_time'], self.frame_fps, 0, duration)-start_fps_time)*self.frame_fps)
                # clip_end_frame_idx = int((ceil_time_by_fps(message['end_time'], self.frame_fps, 0, duration)-start_fps_time)*self.frame_fps) # satisfy range
                # reponse_clip.append((clip_start_frame_idx, clip_end_frame_idx))
                
            else:
                if fps_time > duration:
                    break
                if fps_time > conversation[-1]['fps_time']:
                    num_next_frames = int((fps_time - conversation[-1]['fps_time']) * self.frame_fps)
                    if num_next_frames >= 1:
                        self._add_stream_and_high_res(conversation, high_res_times, num_next_frames, fps_time, self.add_random_high_res_ratio)
                        
                    conversation.extend([{'role': 'assistant', 'content': content, 'time': time, 'fps_time': fps_time, 'learn': True}])
                    
                    # HACK : add response clip
                    try:
                        clip_start_frame_idx = int((ceil_time_by_fps(message['start_time'], self.frame_fps, start_fps_time, duration)-start_fps_time)*self.frame_fps)
                        clip_end_frame_idx = int((ceil_time_by_fps(message['time'], self.frame_fps, 0, duration)-start_fps_time)*self.frame_fps+1)
                        clip_start_frame_idx = max(clip_start_frame_idx, clip_end_frame_idx - 50)
                        reponse_clip.append((clip_start_frame_idx, clip_end_frame_idx))
                    except:
                        breakpoint()
        
        
        # total_num_frames = 0
        # for message in conversation:
        #     if message['role'] == 'stream':
        #         total_num_frames += message['num_frames']
        # if total_num_frames != len(range(int(start_fps_time*self.frame_fps), int(fps_time*self.frame_fps)+1)):
        #     breakpoint()
            
        # for i, message in enumerate(conversation):
        #     if message['role'] == 'stream':
        #         if message['num_frames'] <= 0:
        #             breakpoint()
        try:
            new_anno = {
                'conversation': conversation,
                'load_ranges': {tuple(self.metadata[video_uid]['path']): range(int(start_fps_time*self.frame_fps), int(conversation[-1]['fps_time']*self.frame_fps)+1)} if isinstance(self.metadata[video_uid]['path'], list) else {self.metadata[video_uid]['path']: range(int(start_fps_time*self.frame_fps), int(conversation[-1]['fps_time']*self.frame_fps)+1)},
                'load_frame_high': {self.metadata_high[video_uid]['path']: high_res_times},
                'reponse_clip': reponse_clip,
                'Task Type': anno['Task Type'],
            }
        except:
            breakpoint()
            new_anno = None
        return new_anno
    
    

    def max_frames_clip(self, conversation: list[dict], load_ranges: dict[str, range], max_num_frames: int, load_frame_high: dict[str, list[int]]):
        """
        conversation template:
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "stream", "num_frames": 100},
            {"role": "stream_high", "num_frames": 1},
            {"role": "user", "content": "What is the capital of France?"},
            {"role": "stream", "num_frames": 100},
            {"role": "stream_high", "num_frames": 1},
            {"role": "stream", "num_frames": 100},
            {"role": "stream_high", "num_frames": 1},
            {"role": "assistant", "content": "The capital of France is Paris."},
            {"role": "stream", "num_frames": 100},
            {"role": "stream_high", "num_frames": 1},
            {"role": "assistant", "content": "The capital of France is Paris."},
        ]
        """
        
        cum_num_frames = 0
        cum_num_frames_high = 0
        for i, message in enumerate(conversation):
            if message['role'].lower() == 'stream':
                if cum_num_frames + message['num_frames'] > max_num_frames and conversation[i+1]['role'].lower() == 'assistant':
                    conversation = conversation[:i]
                    load_ranges = {path: range(ranger.start, ranger.start + cum_num_frames) for path, ranger in load_ranges.items()}
                    load_frame_high = {path: high_res_times[:cum_num_frames_high] for path, high_res_times in load_frame_high.items()}
                    break
                cum_num_frames += message['num_frames']
                cum_num_frames_high += 1
        return conversation, load_ranges, load_frame_high
    
    def stream_getitem_rewrite(self, *, conversation: list[dict], load_ranges: dict[str, range] | torch.Tensor = None, add_generation_prompt=False, **kwargs):
        # 1. load visual encoding
        if isinstance(load_ranges, torch.Tensor):
            frames = load_ranges
            high_frames = kwargs.get('load_frame_high', None)
        elif load_ranges is not None:
            conversation, load_ranges, load_frame_high = self.max_frames_clip(conversation, load_ranges, self.max_num_frames, kwargs.get('load_frame_high', None))
            frames = load_frames_f(load_ranges)
            high_frames = load_frames_f(load_frame_high)
            load_frame_high_all = {k:v for k,v in zip(load_frame_high.keys(), load_ranges.values())}
            all_high_frames = load_frames_f(load_frame_high_all)
        else:
            frames = torch.tensor([])
            high_frames = torch.tensor([])
        # 2. prepare texts
        if self.augmentation:
            conversation = self.augment(conversation)
        conversation = [{"role": "system", "content": self.system_prompt}] + conversation
        text = self.tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=add_generation_prompt)
        # 3. learn ranges
        learn_ranges = self.tokenizer.get_learn_ranges(conversation) if not add_generation_prompt else []
        return text, frames, high_frames, all_high_frames, learn_ranges # num_conversation = 3 * num_frames_high + 2 (quesion, system_prompt)
    
    def __getitem__(self, index):
        # delete_data = [7053, 2892, 3443, 7094, 3811, 6514, 7360, 6762, 1149, 1834, 1164, 203, 3604, 7042, 5645, 5607, 498, 6618, 1395, 4792, 2858, 5627, 2756, 2874, 5575, 2323, 4622, 6919, 525, 5970, 2739, 3724, 744, 6305, 7262, 5709, 4581, 6926, 4330, 5733, 795, 3309, 6167, 379, 2483, 4073, 6566, 2468, 2284, 7454, 647, 535, 1484, 7138, 1113, 937, 4378, 950, 1840, 4476, 1308, 3867, 3249, 2308]
        # if index in delete_data:
        #     # If index is in delete_data, randomly sample from remaining indices
        #     valid_indices = [i for i in range(len(self.annos)) if i not in delete_data]
        #     if len(valid_indices) == 0:
        #         raise ValueError("No valid indices remaining after deletion")
        #     index = np.random.choice(valid_indices)
        
        # Calculate total number of frames
        # total_frames = 0
        # total_high_frames = 0
        # for message in self.annos[index]['conversation']:
        #     if message['role'] == 'stream':
        #         total_frames += message['num_frames']
        #     elif message['role'] == 'stream_high':
        #         total_high_frames += message['num_frames']
        
        # print(f'total_frames: {total_frames}, total_high_frames: {total_high_frames}')
        # print(f'index: {index}')
        # print(self.annos[index]['conversation'])
        
        anno = self.annos[index]
        return *self.stream_getitem_rewrite(
            conversation=anno['conversation'],
            load_ranges=anno['load_ranges'],
            load_frame_high=anno['load_frame_high'],
        ), index, self.evaluation_kwargs, anno['reponse_clip']

def build_ego4d_ESTPSQAHighRes(**kwargs):
    return Ego4DESTPSQAHighRes(**kwargs)

class Ego4DESTPCQAHighRes(Ego4DESTPSQAHighRes):
    def get_conversation(self, video_uid, clip_uid, anno):
        new_anno = {}
        new_anno['conversation'] = anno['conversation']
        new_anno['video_uid'] = video_uid
        new_anno['clip_uid'] = clip_uid
        new_anno['clip_start_time'] = anno['clip_start_time'] if 'clip_start_time' in anno else anno['start_time']
        new_anno['clip_end_time'] = anno['clip_end_time'] if 'clip_end_time' in anno else anno['end_time']
        new_anno['Task Type'] = anno['Task Type']
        return new_anno
    
    
def build_ego4d_ESTPCQAHighRes(**kwargs):
    return Ego4DESTPCQAHighRes(**kwargs)


from transformers import EvalPrediction
import numpy as np

class Ego4DESTPSQAHighResGen(Ego4DESTPSQAHighRes):
    def __init__(self, *, frame_fps: int, is_training: bool, **kwargs):
        super().__init__(frame_fps=frame_fps, is_training=is_training, **kwargs)
        self.evaluation_kwargs = DictWithTo(evaluator='lm_evaluate_analysis')
        self.output_dir = kwargs.get('output_dir', None)
        
    def compute_metrics(self, eval_predictions: EvalPrediction, *args, **kwargs):
        np.save(f'{self.output_dir}/{self.__class__.__name__}.npy', eval_predictions.predictions)
        return {
            f'not_metric': 0,
        }
    
def build_ego4d_ESTPSQAHighResGen(**kwargs):
    return Ego4DESTPSQAHighResGen(**kwargs)


class Ego4DESTPCQAHighResGen(Ego4DESTPCQAHighRes):
    def __init__(self, *, frame_fps: int, is_training: bool, **kwargs):
        super().__init__(frame_fps=frame_fps, is_training=is_training, **kwargs)
        self.evaluation_kwargs = DictWithTo(evaluator='lm_evaluate_analysis')
        self.output_dir = kwargs.get('output_dir', None)
        
    def compute_metrics(self, eval_predictions: EvalPrediction, *args, **kwargs):
        np.save(f'{self.output_dir}/{self.__class__.__name__}.npy', eval_predictions.predictions)
        return {
            f'not_metric': 0,
        }

def build_ego4d_ESTPCQAHighResGen(**kwargs):
    return Ego4DESTPSQAHighResGen(**kwargs)


class HighResInsertor:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.mode = kwargs.get('mode', 'random')
        if kwargs.get('feature_dir', None) is not None:
            self.aliger = kwargs.get('feature_dir', None)
        else:
            self.aliger = visionTextAligner(device='cuda:0')
        
        
    def InsertHighRes(self, anno_path, output_dir, **kwargs):
        def count_last_continuous_neg100(arr):
            count = 0
            # 逆向遍历数组，从最后一个元素开始
            for num in reversed(arr):
                if num == -100:
                    count += 1
                else:
                    break  # 遇到第一个非 -100 的数，停止计数
            return count
        
        def _add_high_res_policy(original_conv, error_index, error_logits, reponse_start, reponse_end, policy_mode='first'):
            new_conv = []
            insert_position = []
            
            early_const = 1
            # Insert stream_high at first error position if not in response range
            if policy_mode == 'high' and not (reponse_start-early_const <= error_index[0] < reponse_end):
                # Find position with highest error logit
                max_logit = error_logits[0]
                max_pos = error_index[0]
                for j in range(1, len(error_index)):
                    if error_index[j] > 0 and not (reponse_start-early_const <= error_index[j] < reponse_end):
                        if error_logits[j] > max_logit:
                            max_logit = error_logits[j]
                            max_pos = error_index[j]
                insert_position = [max_pos]
            else:
                for i, logit_idx in enumerate(error_index):
                    if logit_idx > 0 and not (reponse_start-early_const <= logit_idx < reponse_end):
                        insert_position.append(logit_idx)
                        if policy_mode == 'first':
                            break
                        elif policy_mode == 'all':
                            continue
                        else:
                            raise ValueError(f'Invalid policy mode: {policy_mode}')
                        
            current_num_frames = 0
            for i, logit_idx in enumerate(insert_position):
                new_conv.append({
                    'role': 'stream', 
                    'num_frames': logit_idx+1-current_num_frames,
                    'learn': True
                })
                new_conv.append({
                    'role': 'stream_high', 
                    'num_frames': 1,
                    'learn': True
                })
                current_num_frames = logit_idx+1
                
            new_conv.append({
                'role': 'stream',
                'num_frames': original_conv['num_frames'] - current_num_frames if insert_position else  original_conv['num_frames'],
                'learn': original_conv['learn'] if isinstance(original_conv['learn'], bool) else original_conv['num_frames'] - logit_idx - 2,
            })
            
            
            return new_conv, insert_position
        
        def _add_response_clip(insert_position, abnormal_frame_idx, frame_num):
            insert_response_clip = []
            satisfied_idx = []
            max_position = max(insert_position)
            for abnormal_frame in abnormal_frame_idx:
                abnormal_frame = abnormal_frame - frame_num
                if 0 <= abnormal_frame <= max_position:
                    satisfied_idx.append(abnormal_frame.item())

            if len(satisfied_idx) > 0:
                for pos in insert_position:
                    # Find closest satisfied_idx before pos
                    closest_start = None
                    closest_end = pos
                    for idx in satisfied_idx:
                        if idx <= pos:
                            closest_start = idx
                        else:
                            break
                    if closest_start is not None:
                        insert_response_clip.append((closest_start+frame_num, closest_end+frame_num+1))
                        
            return insert_response_clip

                    
            
        # 0 init
        # 0.1 init config
        token_args = json.load(open(os.path.join(output_dir, 'tokenizer_config.json')))
        for k,v in token_args['added_tokens_decoder'].items():
            if v['content'] == '<hv>':
                frame_v_placeholder_high_id = int(k)
            if v['content'] == '<v>':
                frame_v_placeholder_id = int(k)

        # low resolution
        frame_num_tokens = kwargs.get('frame_num_tokens')
        frame_token_interval_id = kwargs.get('frame_token_interval_id')

        # high resolution
        frame_num_tokens_high = kwargs.get('frame_num_tokens_high')
        high_frame_token_interval_id = kwargs.get('high_frame_token_interval_id')
        
        # stream config
        stream_start_id = 58
        stream_end_id = 933
        
        # 0.2 load processed annos
        processed_annos_path = anno_path.replace('.json', f'_processed_{kwargs.get("is_smoothing", "")}_{kwargs.get("add_random_high_res_ratio", "")}_{kwargs.get("data_repeat_num", "")}.pth')
        if os.path.exists(processed_annos_path):
            
            processed_annos = torch.load(processed_annos_path)
        else:
            raise ValueError(f'Processed annotations not found at {processed_annos_path}')
        
        new_annos = []
        for anno in processed_annos:
            if satisfy_condition_post(anno):
                new_annos.append(anno)
        processed_annos = new_annos
        
        
        output_logits_path = os.path.join(output_dir, 'Ego4DESTPSQAHighResGen.npy')
        if os.path.exists(output_logits_path):
            output_logits = np.load(output_logits_path)
        else:
            raise ValueError(f'Output logits not found at {output_logits_path}')
        
        assert len(processed_annos) == output_logits.shape[0], f'Length of processed annotations and output logits must be the same, but got {len(processed_annos)} and {output_logits.shape[0]}'
        
        final_processed_annos = []
        for anno, output_id in tqdm.tqdm(zip(processed_annos, output_logits), total=len(processed_annos)):
            
            # 1. parser output
            # 1.1 split output to 6 parts
            num_conver = (len(output_id) - count_last_continuous_neg100(output_id)) // 6
            output = output_id
            input_id = output[:num_conver]
            pred = output[num_conver:2*num_conver]
            label = output[2*num_conver:3*num_conver]
            logit1 = output[3*num_conver:4*num_conver]
            logit2 = output[4*num_conver:5*num_conver]
            logit3 = output[5*num_conver:6*num_conver]
            
            # check stream numbers
            v_token_num = (input_id == frame_v_placeholder_id).sum() / frame_num_tokens
            hv_token_num = (input_id == frame_v_placeholder_high_id).sum() / frame_num_tokens_high
            
            stream_num = 0
            stream_high_num = 0
            for conv in anno['conversation']:
                if conv['role'].lower() == 'stream':
                    stream_num += conv['num_frames']
                elif conv['role'].lower() == 'stream_high':
                    stream_high_num += 1
                    
            assert v_token_num == stream_num, f'Number of v tokens {v_token_num} must be equal to stream number {stream_num}'
            assert hv_token_num == stream_high_num, f'Number of hv tokens {hv_token_num} must be equal to stream high number {stream_high_num}'
            
            # 1.2 split pred to each turn
            stream_turn_start = ((input_id == stream_start_id).nonzero())[0].tolist()
            stream_turn_end = ((input_id == stream_end_id).nonzero())[0].tolist()

            # 1.3 extract video features for add response clip
            if self.mode == 'simi':
                if isinstance(self.aliger, str):
                    load_ranges = {os.path.join(self.aliger, os.path.basename(k).replace('.mp4', '.pt')):v for k,v in anno['load_ranges'].items()}
                    video_features = load_frames_f(load_ranges)
                else:
                    video_features = load_frames_f(anno['load_ranges'])
                    video_features = self.aliger.vision_feature(video_features)
                abnormal_frame_idx = get_abnormal_frames(video_features)
            
            # 2.1 travel conversation
            new_conversation = []
            turn_idx = 0
            learn_turn_idx = 0
            frame_num = 0
            error_data = False
            for conv_id, conv in enumerate(anno['conversation']):
                if conv['role'].lower() == 'user':
                    new_conversation.append(conv)
                elif conv['role'].lower() == 'assistant':
                    new_conversation.append(conv)
                elif conv['role'].lower() == 'stream':
                    # 2.2 for each stream, find error logits to insert high resolution
                    stream_start = stream_turn_start[turn_idx]
                    stream_end = stream_turn_end[turn_idx]
                    stream_input_id = input_id[stream_start:stream_end]
                    stream_logits1 = logit1[stream_start:stream_end]
                    stream_logits2 = logit2[stream_start:stream_end]
                    stream_logits3 = logit3[stream_start:stream_end]
                    
                    stream_label = label[stream_start:stream_end]
                    stream_pred_mask = (stream_label != -100) & (stream_label != 933)
                    stream_label = stream_label[stream_pred_mask]
                    stream_pred = pred[stream_start:stream_end]
                    stream_pred = stream_pred[stream_pred_mask]
                    
                    # 2.3 insert high resolution in error posistion
                    error_index = (stream_pred != stream_label).nonzero()[0].tolist() # index
                    error_logits2 = [stream_logits2[i] for i in error_index]
                    if len(error_index) == 0:
                        new_conversation.append(conv)
                        new_conversation.append(anno['conversation'][conv_id+1])
                        turn_idx += 1
                        if conv['learn']:
                            learn_turn_idx += 1
                        frame_num += conv['num_frames']
                        continue
                    try:
                        reponse_start, reponse_end = anno['reponse_clip'][learn_turn_idx] 
                    except:
                        error_data = True
                        break
                    reponse_start, reponse_end = reponse_start - frame_num, reponse_end - frame_num
                    new_conv, insert_position = _add_high_res_policy(conv, error_index, error_logits2, reponse_start, reponse_end, kwargs.get('high_res_policy_mode', 'first'))
                    
                    
                    # 2.4 insert response clip
                    if len(insert_position) > 0 and self.mode == 'simi':
                        insert_response_clip = _add_response_clip(insert_position, abnormal_frame_idx, frame_num)
                        anno['reponse_clip'].extend(insert_response_clip)
                        
                    # 2.5 update conversation
                    new_conversation.extend(new_conv)
                    new_conversation.append(anno['conversation'][conv_id+1])
                    turn_idx += 1
                    if conv['learn']:
                        learn_turn_idx += 1
                    frame_num += conv['num_frames']    
                    
                else:
                    continue
            
            
            # 3. check error data
            if error_data:
                # Skip this annotation if there's an error in the data
                continue
            
            anno['conversation'] = new_conversation
            
            stream_num = 0
            stream_high_num = 0
            for conv in anno['conversation']:
                if conv['role'].lower() == 'stream':
                    stream_num += conv['num_frames']
                elif conv['role'].lower() == 'stream_high':
                    stream_high_num += 1
            try:
                assert stream_num == v_token_num, f'Number of stream tokens {stream_num} must be equal to v tokens {v_token_num}'
            except:
                breakpoint()
            # 3. store high frame position
            # Find all high resolution frame indices
            total_num_frames = 0
            high_frame_indices = []
            start_frame = list(anno['load_ranges'].values())[0][0]
            for idx, conv in enumerate(new_conversation):
                if conv['role'].lower() == 'stream':
                    total_num_frames += conv['num_frames']
                elif conv['role'].lower() == 'stream_high':
                    high_frame_indices.append(start_frame + total_num_frames - 1)
            for k,v in anno['load_frame_high'].items():
                anno['load_frame_high'][k] = high_frame_indices
                
            
            final_processed_annos.append(anno)
        # 4. save new anno
        processed_annos_path_calibration = anno_path.replace('.json', f'_processed_{kwargs.get("is_smoothing", "")}_{kwargs.get("high_res_policy_mode", "first")}_{kwargs.get("data_repeat_num", "")}.pth')
        torch.save(final_processed_annos, processed_annos_path_calibration)




# python -m data.estp.livechat
if __name__ == '__main__':
    # data = build_ego4d_ESTPSQA_train(
    #     anno_path='/root/videollm-online/estp.json',
    #     is_training=True, augmentation=False, embed_mark='2fps_max384_1', system_prompt='', tokenizer=None,
    #     frame_fps=2, vision_pretrained='google/siglip-large-patch16-384',
    #     max_num_frames=600,
    # )
    # data_high = build_ego4d_ESTPSQAHighRes(
    #     anno_path='/root/videollm-online/estp_bench_sq.json',
    #     is_training=True, augmentation=False, embed_mark='2fps_max384_1',embed_mark_high='2fps_max384_1+7x7', system_prompt='', tokenizer=None,
    #     frame_fps=2, vision_pretrained='google/siglip-large-patch16-384',
    #     max_num_frames=600,
    #     add_random_high_res_ratio='0.00',
    # )

    # data_sqa_train = build_ego4d_ESTPSQA(
    #     anno_path='/2022233235/datasets/ESTP_IT/estp.json',
    #     is_training=True, augmentation=False, embed_mark='2fps_max384_1+3x3',embed_mark_high='2fps_max384_1+7x7', system_prompt='', tokenizer=None,
    #     frame_fps=2, vision_pretrained='google/siglip-large-patch16-384',
    #     is_smoothing=True,
    #     max_num_frames=12000,
    #     root="/2022233235/datasets/ESTP_IT/"
    # )
    
    # data_cqa_train = build_ego4d_ESTPCQA(
    #     anno_path='/2022233235/datasets/ESTP_IT/estp_cqa_with_time.json',
    #     is_training=True, augmentation=False, embed_mark='2fps_max384_1+3x3',embed_mark_high='2fps_max384_1+7x7', system_prompt='', tokenizer=None,
    #     frame_fps=2, vision_pretrained='google/siglip-large-patch16-384',
    #     is_smoothing=True,
    #     max_num_frames=12000,
    #     root="/2022233235/datasets/ESTP_IT/"
    # )
    
    
    data_sqa_train = build_ego4d_ESTPSQAHighRes(
        anno_path='/2022233235/datasets/ESTP_IT/estp.json',
        is_training=True, augmentation=False, embed_mark='2fps_max384_1+3x3',embed_mark_high='2fps_max384_1+7x7', system_prompt='', tokenizer=None,
        frame_fps=2, vision_pretrained='google/siglip-large-patch16-384',
        is_smoothing=True,
        max_num_frames=12000,
        add_random_high_res_ratio='0.00',
        root="/2022233235/datasets/ESTP_IT/"
    )
    
    # data_cqa_train = build_ego4d_ESTPCQAHighRes(
    #     anno_path='/2022233235/datasets/ESTP_IT/estp_cqa_with_time.json',
    #     is_training=True, augmentation=False, embed_mark='2fps_max384_1+3x3',embed_mark_high='2fps_max384_1+7x7', system_prompt='', tokenizer=None,
    #     frame_fps=2, vision_pretrained='google/siglip-large-patch16-384',
    #     is_smoothing=True,
    #     max_num_frames=12000,
    #     add_random_high_res_ratio='high',
    #     root="/2022233235/datasets/ESTP_IT/"
    # )
    
    # breakpoint()
    # inserter = HighResInsertor(feature_dir="/2022233235/videollm-online/datasets/ESTP_IT/ESTP_IT/",
    #                            mode='simi')
    # inserter.InsertHighRes(
    #     anno_path='/2022233235/videollm-online/datasets/ESTP_IT/estp_cqa_with_time.json',
    #     output_dir='./outputs/ego4d_ESTPSQA/beaconlivel_h_stage2_5_livebase_cqa/',
    #     add_random_high_res_ratio='0.00',
    #     is_smoothing=True,
    #     frame_num_tokens=10,
    #     frame_token_interval_id=11,
    #     frame_num_tokens_high=50,
    #     high_frame_token_interval_id=13,
    #     data_repeat_num=1,
    #     high_res_policy_mode='high',
    # )

    
    
