from PIL import Image
from transformers import AutoProcessor, AutoModel
import torch


# python -m data.preprocess.siglip
class visionTextAligner:
    def __init__(self, model_pretrian="google/siglip-large-patch16-384", device="cuda:4"):
        self.model = AutoModel.from_pretrained(model_pretrian)
        self.model.to(device).eval()
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
    
    def vision_feature(self, frames):
        with torch.no_grad():
            inputs = self.processor(images=frames, padding="max_length", return_tensors="pt")
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            image_embeds = self.model.get_image_features(**inputs)
            image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
            return image_embeds
    
    def vision_simi(self, frames, return_m=False):
        with torch.no_grad():
            inputs = self.processor(images=frames, padding="max_length", return_tensors="pt")
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            image_embeds = self.model.get_image_features(**inputs)
            image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
            simi_m = torch.matmul(image_embeds, image_embeds.t().to(image_embeds.device))
            simi = simi_m.min(dim=0).values.mean().cpu().item()

        if return_m:
            return simi, (simi_m.cpu(),image_embeds.cpu())
        
        return simi
    
    def __call__(self, *args: Image.Any, **kwds: Image.Any) -> Image.Any:
        pass





