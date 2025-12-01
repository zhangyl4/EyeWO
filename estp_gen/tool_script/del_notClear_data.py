import json
import os

def delete_files_from_txt(folder_path, files_to_delete, postfix='.pt'):
    # 遍历需要删除的文件
    for file_name in files_to_delete:
        # 生成完整的文件路径
        file_path = os.path.join(folder_path, file_name + postfix)

        # 检查文件是否存在，若存在则删除
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                print(f"已删除文件: {file_path}")
            except Exception as e:
                print(f"删除文件 {file_path} 时出错: {e}")
        else:
            print(f"文件未找到: {file_path}")


json_dir = '/home/zhangyl/videollm-online/data/preprocess'
video_dir = '/mnt/extra/dataset/ego4d/v2/full_scale_2fps'
num_task = 16
all_dict = {}
bad_data_count = 0
for i in range(num_task):
    with open(f'{json_dir}/{i}.json', 'r') as f:
        all_dict.update(json.load(f))

print(len(all_dict))
for k, v in all_dict.items():
    if v:
        bad_data_count += 1
print(bad_data_count)

file_delete_list = [k for k, v in all_dict.items() if v]
print(len(file_delete_list))
print(len(file_delete_list))
delete_files_from_txt(video_dir, file_delete_list, postfix='.mp4')

