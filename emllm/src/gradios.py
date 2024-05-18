import argparse
import os
import gradio as gr
import mdtex2html
import torch
from PIL import Image
import model_utils
from transformers import GenerationConfig
from transformers import AutoImageProcessor, AutoTokenizer
from utils.constants import IMAGE_TOKEN, PAD_TOKEN
from model.modeling_llama import MultimodalLlamaForConditionalGeneration
from processing_llama import MultiModalLlamaProcessor


DEFAULT_GENERATION_CONFIG = GenerationConfig(
    max_new_tokens=512,
    do_sample=True,
    top_p=0.9,
    top_k=40,
    num_beams=1,
    temperature=0.3,
    num_return_sequences=1,
    # no_repeat_ngram_size=15,
    repetition_penalty=1.1,
)

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', default=None, type=str, required=True,
                    help="Path to the merged VisualCLA model")
parser.add_argument('--gpus', default="0", type=str,
                    help="GPU(s) to use for inference")
parser.add_argument('--share', default=False, action='store_true',
                    help='share gradio domain name')
parser.add_argument('--only_cpu',action='store_true',
                    help='Only use CPU for inference')
args = parser.parse_args()
share = args.share
if args.only_cpu is True:
    args.gpus = ""
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
model = None
tokenizer = None
processor = None

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def get_model_and_tokenizer_and_processor():
    
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
    
    return multimodal_llama_model,tokenizer,processor

# Borrowed from VisualGLM (https://github.com/THUDM/VisualGLM-6B)
def add_message(history, message):
    for x in message["files"]:
        history.append(((x,), None))
    if message["text"] is not None:
        history.append((message["text"], None))
    return history, gr.MultimodalTextbox(value=None, interactive=False)


def predict( chatbot, max_new_tokens, top_p, top_k, temperature, history):
    print(chatbot,flush=True)
    image_path=Image.open(chatbot[0][0][0]).convert('RGB')
    input_text=chatbot[-1][0]
    global tokenizer
    DEFAULT_GENERATION_CONFIG.top_p = top_p
    DEFAULT_GENERATION_CONFIG.top_k = top_k
    DEFAULT_GENERATION_CONFIG.max_new_tokens = max_new_tokens
    DEFAULT_GENERATION_CONFIG.temperature = temperature
    if image_path is None:
        return [(input_text, "图片不能为空。请重新上传图片。")], []
    with torch.no_grad():
        strean_generator = model_utils.chat_in_stream(model, image=image_path, text=input_text, history=history, generation_config=DEFAULT_GENERATION_CONFIG,tokenizer=tokenizer,processor=processor)
        for response, history in strean_generator:
            chatbot[-1] = [input_text,response]
            yield chatbot, history

def main():
    global model,tokenizer,processor,device
    print("Loading the model...")
    model, tokenizer,processor = get_model_and_tokenizer_and_processor()

    if device == torch.device('cpu'):
        model.float()
    model.eval()

    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column(scale=3.5):
                chatbot = gr.Chatbot(
        [],
        elem_id="chatbot",
        bubble_full_width=False
    )
                user_input = gr.MultimodalTextbox(interactive=True, file_types=["image"], placeholder="Enter message or upload file...", show_label=False)
                with gr.Row():
                    emptyBtn = gr.Button("Clear")
            with gr.Column(scale=2.5):
                    max_new_tokens = gr.Slider(0, 1024, value=512, step=1.0, label="Max new tokens", interactive=True)
                    top_p = gr.Slider(0, 1, value=0.9, step=0.01, label="Top P", interactive=True)
                    top_k = gr.Slider(0, 100, value=40, step=1, label="Top K", interactive=True)
                    temperature = gr.Slider(0, 1, value=0.3, step=0.01, label="Temperature", interactive=True)

        history = gr.State([])
        
        x1=user_input.submit(add_message, [chatbot,user_input], [chatbot,user_input])
        x2=x1.then(predict, [chatbot, max_new_tokens, top_p, top_k, temperature, history], [chatbot, history],api_name="bot_response")
        x3=x2.then(lambda: gr.MultimodalTextbox(interactive=True), None, [user_input])
        
        emptyBtn.click(lambda: ( [], []), outputs=[chatbot, history], show_progress=True)

        print(gr.__version__)

        demo.queue()
        demo.launch(inbrowser=True, server_name='0.0.0.0', server_port=8090)

if __name__ == '__main__':
    main()