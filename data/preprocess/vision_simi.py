from PIL import Image
import requests
from transformers import AutoProcessor, AutoModel
import torch
from ..ego4d.narration import build_ego4d_refined_narration_stream_val
from ..utils import load_frames_f
from tqdm import tqdm

# python -m data.preprocess.vision_simi



dataset2 = build_ego4d_refined_narration_stream_val(
    frame_fps=2, is_training=False, augmentation=False,
    system_prompt='', tokenizer=None,
    vision_pretrained='google/siglip-large-patch16-384',
    embed_mark='2fps_max384_1',
    max_num_frames = 10000,
)


simis = []
for i, anno in tqdm(enumerate(dataset2.annos)):
    load_ranges = anno['load_ranges']
    image_embeds = load_frames_f(load_ranges)[:,0,:].to(torch.float32)
    image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
    simi = torch.matmul(image_embeds, image_embeds.t().to(image_embeds.device))
    simis.append(simi.min(dim=0).values.mean())

torch.save(simis, '/root/videollm-online/data/preprocess/simis.pt')


