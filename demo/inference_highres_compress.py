import torch, torchvision, transformers, collections
# torchvision.set_video_backend('video_reader')
from dataclasses import asdict
from torchvision.io import read_video

from models import build_model_and_tokenizer, parse_args, get_args_class, set_args_highres

logger = transformers.logging.get_logger('liveinfer')

# python -m demo.cli --resume_from_checkpoint ... 

def fast_greedy_generate(*, model, input_ids: torch.Tensor, past_key_values, eos_token_id: int, inplace_output_ids: torch.Tensor, beacon_skip_first=None, infer_ct=False):
    for i in range(inplace_output_ids.size(1)):
        outputs = model(input_ids=input_ids, past_key_values=None, use_cache=True, beacon_skip_first=beacon_skip_first, infer_ct=infer_ct)
        past_key_values = outputs.past_key_values
        # breakpoint()
        new_token_id = outputs.logits[:, -1:].argmax(dim=-1)
        inplace_output_ids[:, i] = new_token_id
        if new_token_id == eos_token_id:
            break
        input_ids = new_token_id
    return inplace_output_ids[:, :i+1], past_key_values

class LiveInfer_highres:
    def __init__(self, device='cuda', system_prompt=None, set_vision_inside=True, config=None) -> None:
        if config is None:
            args = parse_args()
        else:
            args = set_args_highres(config)
        
        self.model, self.tokenizer = build_model_and_tokenizer(is_training=False, set_vision_inside=set_vision_inside, **asdict(args))
        self.model.to(device)
        self.device = device
        
        # visual
        self.hidden_size = self.model.config.hidden_size
        self.frame_fps = args.frame_fps
        print(self.frame_fps)
        self.frame_interval = 1 / self.frame_fps
        self.frame_resolution = self.model.config.frame_resolution
        self.frame_num_tokens = self.model.config.frame_num_tokens
        
        self.frame_v_placeholder = self.model.config.v_placeholder * self.frame_num_tokens
        self.frame_token_interval_id = self.model.config.frame_token_interval_id
        self.frame_placeholder_ids = torch.tensor(self.model.config.v_placeholder_id).repeat(self.model.config.frame_num_tokens).reshape(1,-1).to(self.device)
        
        # high resolution
        self.frame_num_tokens_high = self.model.config.frame_num_tokens_high
        self.frame_v_placeholder_high = self.model.config.high_v_placeholder * self.frame_num_tokens_high
        self.high_frame_token_interval = self.model.config.high_frame_token_interval
        self.high_frame_token_interval_id = self.model.config.high_frame_token_interval_id
        self.high_frame_placeholder_ids = torch.tensor(self.model.config.high_v_placeholder_id).repeat(self.model.config.frame_num_tokens_high).reshape(1,-1).to(self.device)
        
        
        # generation
        self.system_prompt = args.system_prompt if system_prompt is None else system_prompt
        self.inplace_output_ids = torch.zeros(1, 100, device=self.device, dtype=torch.long)
        self.frame_token_interval_threshold = config.frame_token_interval_threshold
        self.frame_token_interval_threshold_high = config.frame_token_interval_threshold_high
        self.eos_token_id = self.model.config.eos_token_id
        self._start_ids = self.tokenizer.apply_chat_template([{'role': 'system', 'content': self.system_prompt}], add_stream_prompt=True, return_tensors='pt').to(self.device)
        self._added_stream_prompt_ids = self.tokenizer.apply_chat_template([{}], add_stream_prompt=True, return_tensors='pt').to(self.device)
        self._added_stream_generation_ids = self.tokenizer.apply_chat_template([{}], add_stream_generation_prompt=True, return_tensors='pt').to(self.device)
        self.generation_ids = torch.tensor([933], device=self.device, dtype=torch.long)
        
        # compress
        self.beacon_skip_first = self._start_ids.shape[-1] - 1 # minus 1 for the stream prompt
        self.compress_turn = self.model.config.compress_turn
        self.infer_ct = True if self.compress_turn is not None else False
        
        # app
        self.reset()

    
    def _call_for_query(self, video_time, query):
        self.last_ids = self.tokenizer.apply_chat_template([{'role': 'user', 'content': query}], add_stream_query_prompt=True, return_tensors='pt').to(self.device)
        
        outputs = self.model(input_ids=self.last_ids, use_cache=True, past_key_values=None, beacon_skip_first=self.beacon_skip_first, infer_ct=self.infer_ct)
        self.last_ids = self._added_stream_prompt_ids # outputs.logits[:, -1:].argmax(dim=-1) 
        query = f'(Video Time = {video_time}s) User: {query}'
        return query
    
    def _call_for_response(self, video_time, query, need_response):
        if query is not None:
            self.last_ids = self.tokenizer.apply_chat_template([{'role': 'user', 'content': query}], add_stream_query_prompt=True, add_generation_prompt=True, return_tensors='pt').to(self.device)
        else:
            assert self.last_ids == 933, f'{self.last_ids} != 933' # HACK, 933 = ]\n
            self.last_ids = self._added_stream_generation_ids
        output_ids, self.past_key_values = fast_greedy_generate(model=self.model, input_ids=self.last_ids, past_key_values=None, eos_token_id=self.eos_token_id, inplace_output_ids=self.inplace_output_ids, 
                                                                beacon_skip_first=self.beacon_skip_first, infer_ct=self.infer_ct)

        self.last_ids = output_ids[:, -1:]
        if self.last_ids != self.eos_token_id:
            self.last_ids = torch.tensor([[self.eos_token_id]], device=self.device, dtype=torch.long)
            
        if query:
            query = f'(Video Time = {video_time}s) User: {query}'
        response = f'(Video Time = {video_time}s) Assistant:{self.tokenizer.decode(output_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)}'
        return query, response
    
    def _call_for_streaming(self, ):
        while self.frame_queue:
            # 1. if query is before next frame, response
            if self.query_queue and self.frame_queue[0][0] > self.query_queue[0][0]:
                video_time, query = self.query_queue.popleft()
                return video_time, query, False
            video_time, frame = self.frame_queue.popleft()
            if not self.past_key_values:
                self.last_ids = self._start_ids
            elif self.last_ids.shape[1] == 1 and self.last_ids == self.eos_token_id:
                self.last_ids = torch.cat([self.last_ids, self._added_stream_prompt_ids], dim=1)
                
            # HACK: replace frame_placeholder_ids with last_ids
            input_ids = torch.cat([self.last_ids, self.frame_placeholder_ids], dim=1)
            # inputs_embeds = torch.cat([
            #     self.model.get_input_embeddings()(self.last_ids).view(1, -1, self.hidden_size),
            #     frame_embeds.view(1, -1, self.hidden_size),
            # ], dim=1)
            
            outputs = self.model(input_ids=input_ids, use_cache=True, past_key_values=None, frames = frame, beacon_skip_first=self.beacon_skip_first, infer_ct=self.infer_ct)
            self.past_key_values = outputs.past_key_values
            # 2. if the same time, response after frame at that time
            if self.query_queue and video_time >= self.query_queue[0][0]:
                
                # HACK: for high resolution
                input_ids = torch.cat([torch.tensor(self.high_frame_token_interval_id).reshape(1,-1).to(self.device),
                                       self.high_frame_placeholder_ids], dim=1)
                
                # inputs_embeds_high = torch.cat([
                #     self.model.get_input_embeddings()(torch.tensor(self.high_frame_token_interval_id).to(self.device)).view(1, -1, self.hidden_size),
                #     frame_embeds_high.view(1, -1, self.hidden_size),
                # ], dim=1)
                
                outputs_high = self.model(input_ids=input_ids, use_cache=True, past_key_values=None, high_frames=frame, beacon_skip_first=self.beacon_skip_first, infer_ct=self.infer_ct)
                self.past_key_values = outputs_high.past_key_values
                
                video_time, query = self.query_queue.popleft()
                return video_time, query, False
            # 3. if the next is frame but next is not interval, then response
            next_score = outputs.logits[:,-1:].softmax(dim=-1)
            if self.frame_token_interval_threshold is not None:
                if next_score[:,:,self.frame_token_interval_id] < self.frame_token_interval_threshold:
                    next_score[:,:,self.frame_token_interval_id].zero_()
            self.last_ids = next_score.argmax(dim=-1)
            if self.last_ids != self.frame_token_interval_id: 
                # HACK: for high resolution
                input_ids = torch.cat([torch.tensor(self.high_frame_token_interval_id).reshape(1,-1).to(self.device),
                                       self.high_frame_placeholder_ids], dim=1)
                # inputs_embeds_high = torch.cat([
                #     self.model.get_input_embeddings()(torch.tensor(self.high_frame_token_interval_id).to(self.device)).view(1, -1, self.hidden_size),
                #     frame_embeds_high.view(1, -1, self.hidden_size),
                # ], dim=1)
                
                outputs_high = self.model(input_ids=input_ids, use_cache=True, past_key_values=None, high_frames=frame, beacon_skip_first=self.beacon_skip_first, infer_ct=self.infer_ct)
                self.past_key_values = outputs_high.past_key_values
                
                next_score_high = outputs_high.logits[:,-1:].softmax(dim=-1)
                if self.frame_token_interval_threshold_high is not None: 
                    if next_score_high[:,:,self.frame_token_interval_id] < self.frame_token_interval_threshold_high:
                        next_score_high[:,:,self.frame_token_interval_id].zero_()
                self.last_ids = next_score_high.argmax(dim=-1)
                if self.last_ids == self.generation_ids:
                    return video_time, None, True
                
        return None, None, False
    
    def reset(self, ):
        self.query_queue = collections.deque()
        self.frame_embeds_queue = collections.deque()
        self.frame_embeds_queue_high = collections.deque()
        self.frame_queue = collections.deque()
        self.video_time = 0
        self.last_frame_idx = -1
        self.video_tensor = None
        self.last_ids = torch.tensor([[]], device=self.device, dtype=torch.long)
        self.past_key_values = None
        self.model.memory.reset()

    def input_query_stream(self, query, history=None, video_time=None):
        if video_time is None:
            self.query_queue.append((self.video_time, query))
        else:
            self.query_queue.append((video_time, query))
        if not self.past_key_values:
            return f'(NOTE: No video stream here. Please select or upload a video. Then the assistant will answer "{query} (at {self.video_time}s)" in the video stream)'
        return f'(NOTE: Received "{query}" (at {self.video_time}s). Please wait until previous frames have been processed)'
    
    # def input_video_stream(self, video_time):
    #     frame_idx = int(video_time * self.frame_fps)
    #     if frame_idx > self.last_frame_idx:
    #         ranger = range(self.last_frame_idx + 1, frame_idx + 1)
    #         frames_embeds, high_frame_embeds = self.model.visual_embed(self.video_tensor[ranger], self.video_tensor[ranger])
    #         frames_embeds = frames_embeds.split(self.frame_num_tokens)
    #         high_frame_embeds = high_frame_embeds.split(self.frame_num_tokens_high)
    #         self.frame_embeds_queue.extend([(r / self.frame_fps, frame_embeds) for r, frame_embeds in zip(ranger, frames_embeds)])
    #         self.frame_embeds_queue_high.extend([(r / self.frame_fps, high_frame_embeds) for r, high_frame_embeds in zip(ranger, high_frame_embeds)])
    #     self.last_frame_idx = frame_idx
    #     self.video_time = video_time

    def input_video_stream(self, video_time):
        frame_idx = int(video_time * self.frame_fps)
        if frame_idx > self.last_frame_idx:
            ranger = range(self.last_frame_idx + 1, frame_idx + 1)
            self.frame_queue.extend([(r / self.frame_fps, self.video_tensor[r].unsqueeze(0)) for r in ranger])
        self.last_frame_idx = frame_idx
        self.video_time = video_time
        
    def load_video(self, video_path, load_ranges=None):
        self.video_tensor = read_video(video_path, pts_unit='sec', output_format='TCHW')[0].to(self.device)
        if load_ranges:
            self.video_tensor = self.video_tensor[load_ranges]
        self.num_video_frames = self.video_tensor.size(0)
        self.video_duration = self.video_tensor.size(0) / self.frame_fps
        logger.warning(f'{video_path} -> {self.video_tensor.shape}, {self.frame_fps} FPS')

    def __call__(self, ):
        while not self.frame_queue:
            continue
        video_time, query, need_response = self._call_for_streaming()
        if query is not None:
            query = self._call_for_query(video_time, query)
        response = None
        if need_response and video_time is not None:
            query, response = self._call_for_response(video_time, None, need_response)

        return query, response
    
    # orgin call for response right now after query
    # def __call__(self, ):
    #     while not self.frame_queue:
    #         continue
    #     video_time, query, need_response = self._call_for_streaming()
        
    #     response = None
    #     if  video_time is not None:
    #         query, response = self._call_for_response(video_time, query, need_response)
    #     return query, response