import json, torch, tqdm, os, submitit, random
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser
from dataclasses import dataclass, asdict

from models.arguments_live import LiveOnePlusTrainingArguments
from .templates import Templates
from ..utils import ceil_time_by_fps
from ..ego4d import Ego4D, Ego4DRefinedNarrationStream
import sys
sys.path.insert(0, '/root/dataset/open-eqa/openeqa/utils/')
from openai_utils import call_openai_api, prepare_openai_messages

# python -m data.livechat.ego4d_qa_generation

@dataclass
class LiveOnePlusQAGenerationArguments(LiveOnePlusTrainingArguments):
    split = 'train'
    is_training = False
    augmentation = False
    num_nodes: int = 1
    num_gpus: int = 8
    num_queries_each_conversation: int = 3
    num_conversations_each_video: int = 10
    slurm_partition: str = None
    


narration_templates = "start time: {}s - action narration: {}"
scene_description_templates = "start time: {}s - scene description: {}"
system_prompt = ""

class Ego4DQAGeneration(Ego4DRefinedNarrationStream):

    def __init__(self,num_conversations_each_video, **kwargs):
        split = kwargs['split']
        annos = self.get_annos(split=split)
        self.annos = []
        self.num_conversations_each_video = num_conversations_each_video
        
        for video_uid, _annotation_uid_narrations in tqdm.tqdm(annos.items(), desc=f'narration_stream_{split}...'):
            for narrations in _annotation_uid_narrations.values():
                if not narrations:
                    continue
                prompt_narration = []
                for narration in narrations:
                    prompt_narration.append(narration_templates.format(narration['time'], narration['text']))
                
                self.annos.append({'video_uid': video_uid, 'action_narration': prompt_narration})
                    
                    
        self.tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3-8B-Instruct', use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained('meta-llama/Meta-Llama-3-8B-Instruct', torch_dtype='auto', attn_implementation='sdpa')
        self.model.to('cuda:5')
        self.model.eval()

    @torch.no_grad()
    def __call__(self, index):
        anno = self.annos[index]
        video_uid, action_narration = anno['video_uid'], anno['action_narration']
        print(action_narration)
        # for nt in range(self.num_conversations_each_video):
            # example = ''
            # for ui in range(len(user_queries)):
            #     example += f"\n{user_times[ui]}s User: {user_queries[ui]}\n{user_times[ui]}s Assistant: ..."
            #     for i, t in enumerate(timestamps):
            #         if t < user_times[ui]:
            #             continue
            #         if ui+1 < len(user_times) and t >= user_times[ui+1]: 
            #             break
            #         example += f"\n{t}s Assistant: ..."
            # input_ids = self.tokenizer.apply_chat_template([
            #     {'role': 'user', 'content': prompt + '\n' + example},
            # ], return_tensors='pt', add_generation_prompt=True).cuda()
            # output_ids = self.model.generate(input_ids, max_length=8192)[:,input_ids.size(1):]
            # text = self.tokenizer.decode(output_ids[0])
            # lines = [t.replace('<|eot_id|>', '') for t in text.split('\n') if t and ('User:' in t or 'Assistant:' in t)]
            # try:
            #     anno = {'video_uid': video_uid, 'conversation': []}
            #     for line in lines:
            #         role = 'User' if 'User:' in line else 'Assistant'
            #         role_index = line.index(role)
            #         time = float(line[:role_index].rstrip(' s'))
            #         content = line[role_index+len(role)+2:]
            #         anno['conversation'].append({'role': role.lower(), 'content': content, 'time': time})
            #     os.makedirs(f'{Ego4D.anno_root}/livechat/', exist_ok=True)
            #     json.dump(anno, open(f'{Ego4D.anno_root}/livechat/{video_uid}_{index}_{nt}.json', 'w'), indent=4)
            # except:
            #     print('\n---\n' + text + '\n---\n')
    

# def distributed_livechat_generation(args):
#     env = submitit.JobEnvironment()
#     torch.cuda.set_device(env.local_rank)
#     generator = Ego4DQAGeneration(**asdict(args))
#     for i in tqdm.tqdm(range(len(generator))):
#         if i % env.num_tasks != env.global_rank:
#             continue
#         generator(i)
    
# if __name__ == "__main__":
#     args, = HfArgumentParser(LiveOnePlusLiveChatGenerationArguments).parse_args_into_dataclasses()
#     executor = submitit.AutoExecutor(folder=f"outputs/preprocess/", cluster='local' if args.num_nodes == 1 else 'slurm')
#     executor.update_parameters(
#         tasks_per_node=args.num_gpus,
#         nodes=args.num_nodes,
#         gpus_per_node=args.num_gpus,
#         cpus_per_task=10,
#         slurm_partition=args.slurm_partition,
#         mem_gb=240,
#         slurm_time='24:00:00',
#         timeout_min=600,
#     )
#     job = executor.submit(distributed_livechat_generation, args)

if __name__ == "__main__":
    args, = HfArgumentParser(LiveOnePlusQAGenerationArguments).parse_args_into_dataclasses()
    generator = Ego4DQAGeneration(**{**asdict(args), 'split': 'train', 'is_training': False})
    for i in tqdm.tqdm(range(len(generator))):
        generator(i)
        breakpoint()