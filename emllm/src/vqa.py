"""
Main entrypoint for Visual-Question Answering with a pretrained model
Example run: python src/vqa.py --model_path="./model_checkpoints/04-23_18-53-28/checkpoint-1000" --image_path="./data/images/000000581899.jpg" --user_question="What is in the image?"
"""

import argparse
import sys
import logging

import torch
from PIL import Image

from transformers import AutoImageProcessor, AutoTokenizer, BitsAndBytesConfig

from utils.constants import IMAGE_TOKEN, PAD_TOKEN, LORA_CONFIG, IGNORE_INDEX

from model.modeling_llama import MultimodalLlamaForConditionalGeneration
from processing_llama import MultiModalLlamaProcessor
from utils.utils import get_available_device

from dataset.data_utils import preprocess_multimodal,get_preprocess_func
import copy


logging.basicConfig(level=logging.INFO,
                    format=f'[{__name__}:%(levelname)s] %(message)s')



def preprocess(image_file,text_input,tokenizer,config,processor):
    if image_file!='':
        image = Image.open(image_file).convert('RGB')
        def expand2square(pil_img, background_color):
            width, height = pil_img.size
            if width == height:
                return pil_img
            elif width > height:
                result = Image.new(pil_img.mode, (width, width), background_color)
                result.paste(pil_img, (0, (width - height) // 2))
                return result
            else:
                result = Image.new(pil_img.mode, (height, height), background_color)
                result.paste(pil_img, ((height - width) // 2, 0))
                return result
        image = expand2square(image, tuple(int(x*255) for x in processor.image_mean))
        image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
    
    sources= [[
            {
                "from": "human",
                "value": "<image>\n"+text_input  if image_file !='' else text_input
            },
            {
                "from": "gpt",
                "value": ""
            },
        ]]
    
    
    preprocess_func = get_preprocess_func(config.text_model_id)
    data_dict = preprocess_func(
            sources,
            tokenizer,
            has_image=(image_file !=''))

    data_dict = dict(input_ids=data_dict["input_ids"][0])
    if image_file !='':
        data_dict['pixel_values'] = image
    
    
    input_ids = data_dict["input_ids"].unsqueeze(0)
    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids,
        batch_first=True,
        padding_value=tokenizer.pad_token_id)
    input_ids = input_ids[:, :tokenizer.model_max_length]
    
    batch = dict(
        input_ids=input_ids.to(device),
        attention_mask=input_ids.ne(tokenizer.pad_token_id).to(device),
    )

    if 'pixel_values' in data_dict:
        images = [data_dict['pixel_values']]
        if all(x is not None and x.shape == images[0].shape for x in images):
            batch['pixel_values'] = torch.stack(images).to(device)
        else:
            batch['pixel_values'] = images.to(device)
            
    return batch

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path',
                        required=True,
                        help='Path to the pretrained model weights')

    parser.add_argument('--image_path',
                        required=True,
                        help='Path to the prompt image file')

    parser.add_argument('--user_question',
                        required=True,
                        help='The question to ask about the image to the model. Example: "What is in the image?"')

    args = parser.parse_args(sys.argv[1:])
    logging.info(f"Parameters received: {args}")

    logging.info("Loading pretrained model...")
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False)
    IMAGE_TOKEN_INDEX=tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)
    
    
    multimodal_llama_model = MultimodalLlamaForConditionalGeneration.from_pretrained(args.model_path,
                                                                                     device_map="cuda",
                                                                                     torch_dtype=torch.bfloat16,
                                                                                     ).eval()
    
    multimodal_llama_model.config.pad_token_index=tokenizer.pad_token_id

    if tokenizer.pad_token is None or tokenizer.pad_token != PAD_TOKEN:
        tokenizer.add_special_tokens({'pad_token': PAD_TOKEN})
        tokenizer.pad_token = PAD_TOKEN
    
    image_processor = AutoImageProcessor.from_pretrained(multimodal_llama_model.config.vision_model_id)
    processor = MultiModalLlamaProcessor(image_processor=image_processor, tokenizer=tokenizer)
    
    multimodal_llama_model.config.image_token_index=IMAGE_TOKEN_INDEX

    logging.info("Running model for VQA...")

    prompt = (f"<|start_header_id|>user<|end_header_id|> <image>\n{args.user_question} <|eot_id|>\n"
              f"<|start_header_id|>assistant<|end_header_id|>:")
    inputs=preprocess(args.image_path,args.user_question,tokenizer,multimodal_llama_model.config,processor.image_processor)

    output = multimodal_llama_model.generate(**inputs,
                do_sample=True,
                temperature=0.2,
                top_k=40,
                top_p=0.9,
                num_beams=1,
                repetition_penalty=1.1,
                # no_repeat_ngram_size=3,
                max_new_tokens=100 )
    logging.info(f"Model answer: {processor.decode(output[0][2:], skip_special_tokens=True)}")
