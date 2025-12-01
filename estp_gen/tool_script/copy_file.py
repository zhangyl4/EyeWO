import os
import shutil
import json
import tqdm

def copy_videos(source_folder, target_folder, video_list_file):
    """
    Copy videos from source folder to target folder based on video list
    
    Args:
        source_folder (str): Path to source folder containing videos
        target_folder (str): Path to target folder to copy videos to
        video_list_file (str): Path to text file containing list of video filenames to copy
    """
    # Create target folder if it doesn't exist
    os.makedirs(target_folder, exist_ok=True)
    
    # Read video list
    with open(video_list_file, 'r') as f:
        videos_to_copy = json.load(f)

    # Copy each video
    for video_name in tqdm.tqdm(videos_to_copy):
        video_name = video_name + ''
        source_path = os.path.join(source_folder, video_name)
        target_path = os.path.join(target_folder, video_name)
        
        if os.path.exists(source_path):
            shutil.copy2(source_path, target_path)
        else:
            print(f"Warning: Source video not found: {source_path}")

if __name__ == "__main__":
    # Example usage
    source_folder = "/mnt/extra/dataset/ego4d/v2/full_scale_2fps/"
    target_folder = "/mnt/extra/dataset/ego4d/v2/ESTP_IT_full_scale_2fps/" 
    video_list_file ='/home/zhangyl/videollm-online/data/preprocess/it.json'
    
    copy_videos(source_folder, target_folder, video_list_file)
