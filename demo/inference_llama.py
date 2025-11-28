import torch
from transformers import LlavaNextForConditionalGeneration, AutoProcessor, AutoModelForCausalLM
from PIL import Image
import requests
import time

# Load the model in half-precision

model = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf", torch_dtype=torch.float16, device_map="cuda:1")
processor = AutoProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")

# processor = AutoProcessor.from_pretrained("lmms-lab/llama3-llava-next-8b")
# model = AutoModelForCausalLM.from_pretrained("lmms-lab/llama3-llava-next-8b")

past_key_values = []

# Function to load images dynamically from URLs

def load_images(image_urls):
    images = []
    for url in image_urls:
        if url.startswith('http'):  # Handle URLs
            image = Image.open(requests.get(url, stream=True).raw)
        else:  # Handle local file paths
            image = Image.open(url)
        images.append(image)
    return images

# Example list of image URLs (can be modified for n images)

image_urls = [
    "https://www.ilankelman.org/stopsigns/australia.jpg", 
    "http://images.cocodataset.org/val2017/000000039769.jpg", 
    "/root/videollm-online/demo/llava_next_demo_image/pngtree-cute-christmas-snowman-clip-art-png-image_6467805.png"
]

# Load images dynamically
images = load_images(image_urls)


# Define system prompt
system_prompt = {
    "role": "user",
    "content": [{"type": "text", "text": "You act as the Al assistant on user's AR glass. \
                The AR glass is continuously receiving streamingframes of the user's view, \
                and your task is to simply describe what you have seen. Are you ready toreceive streaming frames?"}]
}

# Apply the chat template for the conversation
conversation = [system_prompt
                ] # llm can not see other conversation history in this conversation


def ask_question(conversation, images, processor, model):
    conversation_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(
        text=[conversation_prompt], 
        images=images, 
        padding=True, 
        return_tensors="pt"
    ).to(model.device)
    generate_ids = model.generate(**inputs, max_new_tokens=1024)
    output = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    
    response = output[-1].split("[/INST]")[-1].strip()
    conversation.append({
        "role": "assistant",
        "content": [{"type": "text", "text": response}]
    })
    
    return conversation, response
    
    
for i in range(4):
    conversation, response = ask_question(conversation, 
                                          images[:i] if i > 0 else None,
                                          processor, model)
    conversation.append({
        "role": "user",
        "content": [{"type": "image"}]
    })
    print(response)

print(conversation)



