import torch
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import requests
import math
from torch import Tensor
from functools import partial
import numpy as np

def dinov2_v1():
    url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    image = Image.open(requests.get(url, stream=True).raw)

    processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
    model = AutoModel.from_pretrained('facebook/dinov2-base')

    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    last_hidden_states = outputs[0]

    # We have to force return_dict=False for tracing
    model.config.return_dict = False

    with torch.no_grad():
        traced_model = torch.jit.trace(model, [inputs.pixel_values])
        traced_outputs = traced_model(inputs.pixel_values)

    print((last_hidden_states - traced_outputs[0]).abs().max())

    return last_hidden_states

def _dinov2_vision_encode(vision_model: AutoModel, frames: Tensor, processor: AutoImageProcessor, frame_token_cls: bool, frame_token_pooled: tuple):
    with torch.amp.autocast('cuda'):
        frames = frames.permute(0, 2, 3, 1)
        
        frames_np = frames.cpu().numpy()
        
        inputs = processor(images=frames_np, return_tensors="pt")
        inputs = {k: v.to(frames.device) for k, v in inputs.items()}
        
        outputs = vision_model(**inputs)
        last_hidden_states = outputs[0]
        
        if frame_token_pooled:
            s = int(math.sqrt(last_hidden_states.shape[1]))
            spatial_tokens = torch.nn.functional.adaptive_avg_pool2d(
                last_hidden_states[:,1:].reshape(
                    last_hidden_states.shape[0], s, s, last_hidden_states.shape[-1]
                ).permute(0, 3, 1, 2),
                frame_token_pooled
            ).flatten(2, 3).permute(0, 2, 1)
            if not frame_token_cls:
                return spatial_tokens
        if frame_token_cls:
            cls_token = last_hidden_states[:,0].unsqueeze(1)
            if not frame_token_pooled:
                return cls_token
    return torch.cat([cls_token, spatial_tokens], dim=1)

def dinov2_v2():
    frame_token_cls = True
    frame_token_pooled_high = (24, 24)
    model = AutoModel.from_pretrained('facebook/dinov2-large')
    url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    image = Image.open(requests.get(url, stream=True).raw)
    frames = torch.tensor(np.array(image)).permute(2,0,1).unsqueeze(0).float() # [1, 3, H, W]
    processor = AutoImageProcessor.from_pretrained('facebook/dinov2-large')
    
    
    
    encode_fn = partial(_dinov2_vision_encode, processor=processor, frame_token_cls=frame_token_cls, frame_token_pooled=frame_token_pooled_high)
    
    vision_tokens = encode_fn(model, frames=frames)
    
    return vision_tokens


a = dinov2_v2()
b = dinov2_v1()

print((a - b).abs().max())
