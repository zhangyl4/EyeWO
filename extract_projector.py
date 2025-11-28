import torch
from safetensors import safe_open
import os

def extract_projector_weights(model_path: str, output_path: str):
    """
    从safetensors格式的模型文件中提取connector权重并保存为bin格式
    
    Args:
        model_path: safetensors模型文件路径
        output_path: 输出的mm_projector.bin文件路径
    """
    # 初始化一个字典来存储提取的权重
    projector_weights = {}
    
    # 打开safetensors文件
    with safe_open(model_path, framework="pt", device="cpu") as f:
        # 遍历所有张量
        for key in f.keys():
            # 只提取包含"connector"的权重
            if "connector" in key:
                projector_weights[key] = f.get_tensor(key)
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 保存提取的权重
    if projector_weights:
        torch.save(projector_weights, output_path)
        print(f"成功提取connector权重并保存到: {output_path}")
    else:
        print("未找到connector相关权重")

def main():
    # 设置路径
    model_path = "/2022233235/.cache/huggingface/hub/models--videollm-online-8b-v2plus-coin/model-00004-of-00004.safetensors"
    output_path = "/2022233235/.cache/huggingface/hub/models--videollm-online-8b-v2plus-coin/mm_projector.bin"
    
    # 提取并保存权重
    extract_projector_weights(model_path, output_path)

if __name__ == "__main__":
    main() 