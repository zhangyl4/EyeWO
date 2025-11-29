from PIL import Image
import requests
from transformers import AutoProcessor, AutoModel
import torch
from ..ego4d.narration import build_ego4d_refined_narration_stream_val
from ..utils import load_frames_f

# python -m data.preprocess.siglip

class visionTextAligner:
    def __init__(self, model_pretrian="google/siglip-large-patch16-384", device="cuda:4"):
        self.model = AutoModel.from_pretrained(model_pretrian)
        self.processor = AutoProcessor.from_pretrained(model_pretrian)
        
    def align(self, image_embeds, texts):
        with torch.no_grad():
            inputs = self.processor(text=texts, padding="max_length", return_tensors="pt")
            text_embeds = self.model.get_text_features(**inputs)
            image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
            text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)

            logits_per_text = (torch.matmul(text_embeds, image_embeds.t().to(text_embeds.device)) * self.model.logit_scale.exp()+ self.model.logit_bias)
            
            logits_per_image = logits_per_text.t()
            probs = torch.sigmoid(logits_per_image)
            
        return probs
    
    def vision_simi(self, frames):
        with torch.no_grad():
            inputs = self.processor(images=frames, padding="max_length", return_tensors="pt")
            image_embeds = self.model.get_image_features(**inputs)
            image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
            simi = torch.matmul(image_embeds, image_embeds.t().to(image_embeds.device))
            simi = simi.min(dim=0).values.mean()

        return simi
    
    def __call__(self, *args: Image.Any, **kwds: Image.Any) -> Image.Any:
        pass


dataset2 = build_ego4d_refined_narration_stream_val(
    frame_fps=2, is_training=False, augmentation=False,
    system_prompt='', tokenizer=None,
    vision_pretrained='google/siglip-large-patch16-384',
    embed_mark='2fps_max384_1',
    max_num_frames = 10000,
)

aliger = visionTextAligner()

for i, anno in enumerate(dataset2.annos):
    load_ranges = anno['load_ranges']
    frames = load_frames_f(load_ranges)[:,0,:].to(torch.float32)
    texts = []
    for j, sentence in enumerate(anno['conversation']):
        if sentence['role'] == 'assistant':
            texts.append(sentence['content'])
    probs = aliger.align(frames, texts)
    breakpoint()


