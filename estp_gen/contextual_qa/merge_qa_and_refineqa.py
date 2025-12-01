import os
from tqdm import tqdm
output_dir = '/home/zhangyl/videollm-online/dataset/contextual_qa_single_v2/'

def timeSatisfy(qa1,qa2):
    last_time = qa1['conversation'][0]['end_time']
    is_satisfy = False
    for conv in qa2['conversation']:
        if conv['end_time'] > last_time:
            is_satisfy = True
            break
    
    return is_satisfy


group_qas = {}
for i, video_uid in tqdm(enumerate(processed_single_qa)):
    for clip_uid in processed_single_qa[video_uid]:
        output_prefix = os.path.join(output_dir, f'{video_uid}_{clip_uid}_{i}')
        try:
            group_id = json.load(open(f'{output_prefix}_group.json', 'r'))['groups']
        except:
            continue
        
        # group qa, delete time repeat and number of qa less than 3
        for group in group_id:
            print(group)
            a_group_qas = []
            for idx in group['question_ids']:
                if not a_group_qas:
                    a_group_qas.append(processed_single_qa[video_uid][clip_uid][idx])
                    continue
                if timeSatisfy(a_group_qas[-1], processed_single_qa[video_uid][clip_uid][idx]):
                    a_group_qas.append(processed_single_qa[video_uid][clip_uid][idx])
            
            # print(json.dumps(a_group_qas, indent=4))
            if len(a_group_qas) <= 3:
                continue
        
            if a_group_qas:
                if video_uid not in group_qas.keys():
                    group_qas[video_uid] = {}
                if clip_uid not in group_qas[video_uid].keys():
                    group_qas[video_uid][clip_uid] = []
                a_group_qas = {
                    'reason': group['reason'],
                    'qas': a_group_qas
                }
                group_qas[video_uid][clip_uid].append(a_group_qas)
    #     break
    # break


def sumDict(a):
    c = 0
    for key in a:
        c += len(a[key])
    return c

print(json.dumps(group_qas, indent=4))
print(sumDict(group_qas))
        