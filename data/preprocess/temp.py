import os

def count_files_in_directory(directory):
    """Count number of files in specified directory and subdirectories"""
    total_files = 0
    
    # Walk through directory tree
    total_files = 0
    for file in os.listdir(directory):
        if not os.path.isdir(os.path.join(directory, file)):
            continue
        for f in os.listdir(os.path.join(directory, file)):
            if f.endswith('.json'):
                total_files += 1
    # for root, dirs, files in os.walk(directory):
    #     total_files += len(files)
        
    return total_files

if __name__ == '__main__':
    # Example usage
    directory = "/2022233235/videollm-online/datasets/ego4d_move_action_caption/train_0"
    num_files = count_files_in_directory(directory)
    print(f"Total number of files: {num_files}")
