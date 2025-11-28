import math, torch
from functools import partial
from torch import nn, Tensor
from torchvision.transforms.functional import normalize
from transformers import AutoModel, AutoImageProcessor
from transformers.utils.constants import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD

from .configuration_live import LiveConfigMixin

def _siglip_vision_encode(vision_model: nn.Module, frames: Tensor, frame_token_cls: bool, frame_token_pooled: tuple,
    mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5], rescale_factor=0.00392156862745098, **kwargs):
    frames = normalize(frames * rescale_factor, mean=mean, std=std)
    
    # Split frames if more than 3000
    batch_size = 1000
    if frames.shape[0] > batch_size:
        
        outputs = []
        for i in range(0, frames.shape[0], batch_size):
            batch_frames = frames[i:i+batch_size]
            with torch.cuda.amp.autocast():
                vision_outputs = vision_model(batch_frames)
                last_hidden_state = vision_outputs.last_hidden_state
                if frame_token_pooled:
                    s = int(math.sqrt(last_hidden_state.shape[1]))
                    spatial_tokens = torch.nn.functional.adaptive_avg_pool2d(
                        last_hidden_state.reshape(
                            last_hidden_state.shape[0], s, s, last_hidden_state.shape[-1]
                        ).permute(0, 3, 1, 2),
                        frame_token_pooled
                    ).flatten(2, 3).permute(0, 2, 1)
                    if not frame_token_cls:
                        outputs.append(spatial_tokens)
                        continue
                if frame_token_cls:
                    cls_token = vision_outputs.pooler_output[:, None]
                    if not frame_token_pooled:
                        outputs.append(cls_token)
                        continue
                outputs.append(torch.cat([cls_token, spatial_tokens], dim=1))
        return torch.cat(outputs, dim=0)
    
    # Original processing for smaller batches
    with torch.cuda.amp.autocast():
        vision_outputs = vision_model(frames)
        last_hidden_state = vision_outputs.last_hidden_state
        if frame_token_pooled:
            s = int(math.sqrt(last_hidden_state.shape[1]))
            spatial_tokens = torch.nn.functional.adaptive_avg_pool2d(
                last_hidden_state.reshape(
                    last_hidden_state.shape[0], s, s, last_hidden_state.shape[-1]
                ).permute(0, 3, 1, 2),
                frame_token_pooled
            ).flatten(2, 3).permute(0, 2, 1)
            if not frame_token_cls:
                return spatial_tokens
        if frame_token_cls:
            cls_token = vision_outputs.pooler_output[:, None]
            if not frame_token_pooled:
                return cls_token
    return torch.cat([cls_token, spatial_tokens], dim=1)

def _clip_vision_encode(vision_model: nn.Module, frames: Tensor, frame_token_cls: bool, frame_token_pooled: tuple,
    mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD, rescale_factor=0.00392156862745098, **kwargs):
    frames = normalize(frames * rescale_factor, mean=mean, std=std)
    with torch.cuda.amp.autocast():
        vision_outputs = vision_model(frames)
        last_hidden_state = vision_outputs.last_hidden_state
        if frame_token_pooled:
            s = int(math.sqrt(last_hidden_state.shape[1]))
            spatial_tokens = torch.nn.functional.adaptive_avg_pool2d(
                last_hidden_state[:,1:].reshape(
                    last_hidden_state.shape[0], s, s, last_hidden_state.shape[-1]
                ).permute(0, 3, 1, 2),
                frame_token_pooled
            ).flatten(2, 3).permute(0, 2, 1)
            if not frame_token_cls:
                return spatial_tokens
        if frame_token_cls:
            cls_token = last_hidden_state[:,0]
            if not frame_token_pooled:
                return cls_token
    return torch.cat([cls_token, spatial_tokens], dim=1)

def _dinov2_vision_encode(vision_model: AutoModel, frames: Tensor, processor: AutoImageProcessor, frame_token_cls: bool, frame_token_pooled: tuple):
    with torch.amp.autocast('cuda'):
        frames = frames.permute(0, 2, 3, 1)
        frames_np = frames.cpu().numpy()
        
        # Split into chunks if number of frames is too large
        chunk_size = 1500
        if len(frames) > chunk_size:
            
            outputs_list = []
            for i in range(0, len(frames), chunk_size):
                chunk_frames = frames_np[i:i+chunk_size]
                inputs = processor(images=chunk_frames, return_tensors="pt")
                inputs = {k: v.to(frames.device) for k, v in inputs.items()}
                chunk_outputs = vision_model(**inputs)
                outputs_list.append(chunk_outputs[0])
            last_hidden_states = torch.cat(outputs_list, dim=0)
        else:
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

def build_live_vision(config: LiveConfigMixin):
    model = AutoModel.from_pretrained(config.vision_pretrained).vision_model
    if 'google/siglip-so400m-patch14-384' == config.vision_pretrained or 'google/siglip-large-patch16-384' == config.vision_pretrained: # google/siglip-so400m-patch14-384 # google/siglip-large-patch16-384
        return model, partial(_siglip_vision_encode, frame_token_cls=config.frame_token_cls, frame_token_pooled=config.frame_token_pooled)
    elif 'laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90k' == config.vision_pretrained or 'openai/clip-vit-large-patch14-336' == config.vision_pretrained:
        return model, partial(_clip_vision_encode, config)
    else:
        raise ValueError(f'Unverified vision_pretrained: {config.vision_pretrained}')

def build_live_vision_high(config: LiveConfigMixin):
    model = AutoModel.from_pretrained(config.vision_pretrained).vision_model
    if 'google/siglip-so400m-patch14-384' == config.vision_pretrained or 'google/siglip-large-patch16-384' == config.vision_pretrained:
        return model, partial(_siglip_vision_encode, frame_token_cls=config.frame_token_cls, frame_token_pooled=config.frame_token_pooled_high)
    elif 'laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90k' == config.vision_pretrained or 'openai/clip-vit-large-patch14-336' == config.vision_pretrained:
        return model, partial(_clip_vision_encode, config)
    else:
        raise ValueError(f'Unverified vision_pretrained: {config.vision_pretrained}')

def build_dinov2_vision(config: LiveConfigMixin):
    model = AutoModel.from_pretrained(config.add_vision_pretrained)
    processor = AutoImageProcessor.from_pretrained(config.add_vision_pretrained)
    return model, partial(_dinov2_vision_encode, processor=processor, frame_token_cls=config.frame_token_cls, frame_token_pooled=config.frame_token_pooled)

def build_dinov2_vision_high(config: LiveConfigMixin):
    model = AutoModel.from_pretrained(config.add_vision_pretrained)
    processor = AutoImageProcessor.from_pretrained(config.add_vision_pretrained)
    return model, partial(_dinov2_vision_encode, processor=processor, frame_token_cls=config.frame_token_cls, frame_token_pooled=config.frame_token_pooled_high)


if __name__ == "__main__":
    from configuration_live import LiveConfigMixin
    config = LiveConfigMixin()
    model, encode_fn = build_live_vision(config)
    print(model)
    print(encode_fn)
