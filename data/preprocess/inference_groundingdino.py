import requests

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection 

model_id = "IDEA-Research/grounding-dino-base"
device = "cuda" if torch.cuda.is_available() else "cpu"

processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

image_path = '/root/videollm-online/datasets/ego4d_it_train/0/2.jpg'
image = Image.open(image_path)
# Check for cats and remote controls
# VERY important: text queries need to be lowercased + end with a dot
text = "a traffic light. a person."

inputs = processor(images=image, text=text, return_tensors="pt").to(device)

with torch.no_grad():
    outputs = model(**inputs)


results = processor.post_process_grounded_object_detection(
    outputs,
    inputs.input_ids,
    box_threshold=0.4,
    text_threshold=0.3,
    target_sizes=[image.size[::-1]]
)

print(results)