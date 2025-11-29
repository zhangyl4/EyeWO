import torch
import torch.multiprocessing as mp
import subprocess
import os

def intlist2str(sep, intlist):
    a = ''
    for i in intlist:
        a+= str(i) + sep if i != intlist[-1] else str(i)
    return a

def run_on_gpu(device_id, prompt_file, output_dir,device_list, video_root):
    # 设置当前进程使用的 GPU
    torch.cuda.set_device(device_id)
    # 构造命令
    command = [
        "python", "/2022233235/videollm-online/data/preprocess/video_caption_action_scene.py",
        "--prompt_file", prompt_file,
        "--output_dir", output_dir,
        "--device", f"cuda:{device_id}",
        "--device_list", intlist2str(',', device_list),
        "--alpha", "4.9", # "--is_scene",
        "--video_root",video_root
    ]
    
    # 日志文件路径
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir,f"log_gpu_{device_id}.txt")
    
    # 运行命令并将输出重定向到日志文件
    with open(log_file, "w") as f:
        subprocess.run(command, stdout=f, stderr=subprocess.STDOUT)

if __name__ == "__main__":
    # 输入参数
    prompt_file = "/2022233235/videollm-online/data/preprocess/prompt/caption_expand.txt"
    output_dir = "/2022233235/videollm-online/datasets/ego4d_action_caption/train_0"
    video_root = "/2022233235/datasets/full_scale_2fps_train_0/"
    
    # GPU 列表
    device_list = [0,1,2,3,4,5,6,7]
    print(intlist2str(',', device_list))
    
    
    # 使用多进程并行运行
    processes = []
    for device_id in device_list:
        p = mp.Process(target=run_on_gpu, args=(device_id, prompt_file, output_dir, device_list,video_root))
        p.start()
        processes.append(p)
        print(f'{device_id} starts.')
    
    # 等待所有进程完成
    for p in processes:
        p.join()