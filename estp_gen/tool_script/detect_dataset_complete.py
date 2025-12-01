import json
import os
from argparse import ArgumentParser
import tqdm


if __name__ == "__main__": 
    parser = ArgumentParser()
    parser.add_argument("--json_dir", type=str, default='/home/zhangyl/videollm-online/datasets/ego4d/v2/annotations/')
    parser.add_argument("--subset", type=str, default='narration')
    parser.add_argument("--video_dir", type=str, default='/mnt/extra/dataset/ego4d/v2/full_scale_2fps/')
    args = parser.parse_args()

    
    if args.subset == 'narration':
        json_path = os.path.join(args.json_dir, f'refined_narration_stream_train.json')
        annos = json.load(open(json_path))
        all_video_id = list(annos.keys())
        
        json_path = os.path.join(args.json_dir, f'refined_narration_stream_val.json')
        annos = json.load(open(json_path))
        all_video_id += list(annos.keys())
        all_video_id = set(all_video_id)
        print(f'All video id: {len(all_video_id)}')
        
    elif args.subset == 'goalstep':
        EGO4D_ANNO_ROOT = '/mnt/extra/dataset/ego4d/v2/annotations/'
        sources = json.load(open(os.path.join(EGO4D_ANNO_ROOT,'goalstep_train.json')))['videos']
        sources += json.load(open(os.path.join(EGO4D_ANNO_ROOT,'goalstep_val.json')))['videos']
        all_video_id = set([source['video_uid'] for source in sources])
        print(f'All video id: {len(all_video_id)}')
    else:
        file_list_path = args.subset
        if not os.path.exists(file_list_path):
            raise ValueError(f'File list path {file_list_path} not exist')
        else:
            with open(file_list_path, 'r') as f:
                all_video_id = json.load(f)
                print(f'All video id: {len(all_video_id)}')
        
    
    exist_video_id = []
    for file in os.listdir(args.video_dir):
        if file.endswith('.mp4'):
            exist_video_id.append(file.split('.')[0])
            
    not_exist_video_id = list(set(all_video_id) - set(exist_video_id))
    with open(f'/home/zhangyl/videollm-online/data/preprocess/not_exist.txt', 'w') as f:
        f.write('\n'.join(not_exist_video_id))
    print(f'{len(not_exist_video_id)} videos not exist')