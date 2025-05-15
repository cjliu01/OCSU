import torch
import argparse
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
import json
import os

model_path = "ckpts/mol-vl-7b"  # local_model_path

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto",          
)
processor = AutoProcessor.from_pretrained(model_path, size={"shortest_edge": 56 * 56, "longest_edge": 28 * 28 * 1280})
processor.chat_template = "{% set image_count = namespace(value=0) %}{% set video_count = namespace(value=0) %}{% for message in messages %}{% if loop.first and message['role'] != 'system' %}<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n{% endif %}<|im_start|>{{ message['role'] }}\n{% if message['content'] is string %}{{ message['content'] }}<|im_end|>\n{% else %}{% for content in message['content'] %}{% if content['type'] == 'image' or 'image' in content or 'image_url' in content %}{% set image_count.value = image_count.value + 1 %}{% if add_vision_id %}Picture {{ image_count.value }}: {% endif %}<|vision_start|><|image_pad|><|vision_end|>{% elif content['type'] == 'video' or 'video' in content %}{% set video_count.value = video_count.value + 1 %}{% if add_vision_id %}Video {{ video_count.value }}: {% endif %}<|vision_start|><|video_pad|><|vision_end|>{% elif 'text' in content %}{{ content['text'] }}{% endif %}{% endfor %}<|im_end|>\n{% endif %}{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"

model.eval()

def generate_response(image_path, prompt):
    """
        Generate the model's response
        
        Parameters:
            image_path: The path of the image
            question: The user's question (optional, used for visual question answering tasks)
            max_length: The maximum length of the generated text
    """
    image = Image.open(image_path)
    conversation = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role":"user",
            "content":[
                {
                    "type": "image",
                },
                {
                    "type":"text",
                    "text": f"{prompt}"
                }
            ]
        }
    ]


    # Preprocess the inputs
    text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(text=[text_prompt], images=[image], padding=True, return_tensors="pt")
    inputs = inputs.to('cuda')

    # Inference: Generation of the output
    output_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
    output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
    
    return output_text


def process_ocsu(args):
    samples_to_process = []
    
    with open(args.jsonl_path, 'r') as f:
        for i, line in enumerate(f):
            if line.strip():
                try:
                    sample = json.loads(line.strip())
                    samples_to_process.append((i, sample))
                except json.JSONDecodeError:
                    print(f"Error parsing JSON at line {i}")
                    continue
    for index, sample in samples_to_process:
        print(f"\nProcessing sample {index}...")
        
        # Get the image path for the sample
        image_path = sample["images"][0]
        predicted_smiles = None
        try:
            response = generate_response(image_path, prompt='What is the SMILES of the molecule?')
            predicted_smiles = response[:-1].replace("The SMILES is ", "")
        except Exception as e:
            print(f"Error extracting SMILES: {e}")
            predicted_smiles = f"ERROR: {str(e)}"
            
        sample["ocsu"] = predicted_smiles if predicted_smiles else "ERROR: Processing failed"

        with open(args.output, 'a') as f:
            f.write(json.dumps(sample) + "\n")
            print(f"Appended result for sample {index} to {args.output}")
            

def main():
    parser = argparse.ArgumentParser(description="Query Qwen-vl models for molecular recognition")
    parser.add_argument("--jsonl_path", type=str, default='test.jsonl', 
                       help="Path to the JSONL file with molecule data (mandatory)")
    parser.add_argument("--output", type=str, default='test_ocsu.jsonl', 
                       help="Output file to save the full response (only used with sample_index)")
    
    args = parser.parse_args()
    
    print(f"args: {args}")
    process_ocsu(args)

if __name__ == '__main__':
    main()
    
    

