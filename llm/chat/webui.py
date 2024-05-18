import torch
from transformers import AutoModelForCausalLM,LlamaTokenizer,StoppingCriteria,BitsAndBytesConfig
import gradio as gr
import argparse
import os
from queue import Queue
from threading import Thread
import traceback
import gc
import json
import requests
from typing import Iterable, List

DEFAULT_SYSTEM_PROMPT = """You are a kind and helpful assistant"""

TEMPLATE_WITH_SYSTEM_PROMPT = (
    "[INST] <<SYS>>\n"
    "{system_prompt}\n"
    "<</SYS>>\n\n"
    "{instruction} [/INST]"
)

TEMPLATE_WITHOUT_SYSTEM_PROMPT = "[INST] {instruction} [/INST]"

parser = argparse.ArgumentParser()
parser.add_argument(
    '--base_model',
    default=None,
    type=str,
    required=True,
    help='Base model path')
parser.add_argument(
    '--gpus',
    default="0",
    type=str,
    help='If None, cuda:0 will be used. Inference using multi-cards: --gpus=0,1,... ')
parser.add_argument('--share', default=True, help='Share gradio domain name')
parser.add_argument('--port', default=19324, type=int, help='Port of gradio demo')
parser.add_argument(
    '--max_memory',
    default=1024,
    type=int,
    help='Maximum number of input tokens (including system prompt) to keep. If exceeded, earlier history will be discarded.')
parser.add_argument(
    '--load_in_8bit',
    action='store_true',
    help='Use 8 bit quantized model')
parser.add_argument(
    '--load_in_4bit',
    action='store_true',
    help='Use 4 bit quantized model')
parser.add_argument(
    '--alpha',
    type=str,
    default="1.0",
    help="The scaling factor of NTK method, can be a float or 'auto'. ")
parser.add_argument(
    '--defulted_system_prompt',
    type=str,
    default="You are a kind and helpful assistant",
    help="The default system prompt")
args = parser.parse_args()


import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)


from attn_and_long_ctx_patches import apply_attention_patch
apply_attention_patch(use_memory_efficient_attention=True)

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus


def setup():
    global tokenizer, model, device, share, port, max_memory,DEFAULT_SYSTEM_PROMPT
    
    
    DEFAULT_SYSTEM_PROMPT=args.defulted_system_prompt
    
    max_memory = args.max_memory
    port = args.port
    share = args.share == 'True' or args.share is True
    load_type = torch.float16
    device = torch.device(0)
    
    tokenizer = LlamaTokenizer.from_pretrained(args.base_model, legacy=True)
    if args.load_in_4bit or args.load_in_8bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=args.load_in_4bit,
            load_in_8bit=args.load_in_8bit,
            bnb_4bit_compute_dtype=load_type,
        )

    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=load_type,
        low_cpu_mem_usage=True,
        device_map='auto',
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
        quantization_config=quantization_config if (args.load_in_4bit or args.load_in_8bit) else None,
        trust_remote_code=True
    )

    model_vocab_size = base_model.get_input_embeddings().weight.size(0)
    tokenizer_vocab_size = len(tokenizer)
    print(f"Vocab of the base model: {model_vocab_size}")
    print(f"Vocab of the tokenizer: {tokenizer_vocab_size}")
    if model_vocab_size != tokenizer_vocab_size:
        print("Resize model embeddings to fit tokenizer")
        base_model.resize_token_embeddings(tokenizer_vocab_size)

    model = base_model
    model.eval()

def reset_user_input():
    return gr.update(value='')


def reset_state():
    return []


def generate_prompt(instruction, response="", with_system_prompt=True, system_prompt=DEFAULT_SYSTEM_PROMPT):
    if with_system_prompt is True:
        prompt = TEMPLATE_WITH_SYSTEM_PROMPT.format_map({'instruction': instruction,'system_prompt': system_prompt})
    else:
        prompt = TEMPLATE_WITHOUT_SYSTEM_PROMPT.format_map({'instruction': instruction})
    if len(response)>0:
        prompt += " " + response
    return prompt

def user(user_message, history):
    return gr.update(value="", interactive=False), history + \
        [[user_message, None]]


class Stream(StoppingCriteria):
    def __init__(self, callback_func=None):
        self.callback_func = callback_func

    def __call__(self, input_ids, scores) -> bool:
        if self.callback_func is not None:
            self.callback_func(input_ids[0])
        return False


class Iteratorize:
    def __init__(self, func, kwargs=None, callback=None):
        self.mfunc = func
        self.c_callback = callback
        self.q = Queue()
        self.sentinel = object()
        self.kwargs = kwargs or {}
        self.stop_now = False

        def _callback(val):
            if self.stop_now:
                raise ValueError
            self.q.put(val)

        def gentask():
            try:
                ret = self.mfunc(callback=_callback, **self.kwargs)
            except ValueError:
                pass
            except Exception:
                traceback.print_exc()

            clear_torch_cache()
            self.q.put(self.sentinel)
            if self.c_callback:
                self.c_callback(ret)

        self.thread = Thread(target=gentask)
        self.thread.start()

    def __iter__(self):
        return self

    def __next__(self):
        obj = self.q.get(True, None)
        if obj is self.sentinel:
            raise StopIteration
        else:
            return obj

    def __del__(self):
        clear_torch_cache()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_now = True
        clear_torch_cache()


def clear_torch_cache():
    gc.collect()
    if torch.cuda.device_count() > 0:
        torch.cuda.empty_cache()


def post_http_request(prompt: str,
                      api_url: str,
                      n: int = 1,
                      top_p: float = 0.9,
                      top_k: int = 40,
                      temperature: float = 0.2,
                      max_tokens: int = 512,
                      presence_penalty: float = 1.0,
                      use_beam_search: bool = False,
                      stream: bool = False) -> requests.Response:
    headers = {"User-Agent": "Test Client"}
    pload = {
        "prompt": prompt,
        "n": n,
        "top_p": 1 if use_beam_search else top_p,
        "top_k": -1 if use_beam_search else top_k,
        "temperature": 0 if use_beam_search else temperature,
        "max_tokens": max_tokens,
        "use_beam_search": use_beam_search,
        "best_of": 5 if use_beam_search else n,
        "presence_penalty": presence_penalty,
        "stream": stream,
    }
    print(pload)

    response = requests.post(api_url, headers=headers, json=pload, stream=True)
    return response


def get_streaming_response(response: requests.Response) -> Iterable[List[str]]:
    for chunk in response.iter_lines(chunk_size=8192,
                                     decode_unicode=False,
                                     delimiter=b"\0"):
        if chunk:
            data = json.loads(chunk.decode("utf-8"))
            output = data["text"]
            yield output

@torch.no_grad()
def predict(
    history,
    system_prompt,
    max_new_tokens=128,
    top_p=0.9,
    temperature=0.2,
    top_k=40,
    repetition_penalty=1.1
):
    if len(system_prompt) == 0:
        system_prompt = DEFAULT_SYSTEM_PROMPT
    while True:
        print("len(history):", len(history))
        print("history: ", history)
        history[-1][1] = ""
        if len(history) == 1:
            input = history[0][0]
            prompt = generate_prompt(input,response="", with_system_prompt=True, system_prompt=system_prompt)
        else:
            input = history[0][0]
            response = history[0][1]
            prompt = generate_prompt(input, response=response, with_system_prompt=True, system_prompt=system_prompt)+'</s>'
            for hist in history[1:-1]:
                input = hist[0]
                response = hist[1]
                prompt = prompt + '<s>'+generate_prompt(input, response=response, with_system_prompt=False)+'</s>'
            input = history[-1][0]
            prompt = prompt + '<s>'+generate_prompt(input, response="", with_system_prompt=False)

        input_length = len(tokenizer.encode(prompt, add_special_tokens=True))
        print(f"Input length: {input_length}")
        if input_length > max_memory and len(history) > 1:
            print(f"The input length ({input_length}) exceeds the max memory ({max_memory}). The earlier history will be discarded.")
            history = history[1:]
            print("history: ", history)
        else:
            break

    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    generate_params = {
        'input_ids': input_ids,
        'max_new_tokens': max_new_tokens,
        'top_p': top_p,
        'temperature': temperature,
        'top_k': top_k,
        'repetition_penalty': repetition_penalty,
        'eos_token_id': tokenizer.eos_token_id,
    }

    def generate_with_callback(callback=None, **kwargs):
        if 'stopping_criteria' in kwargs:
            kwargs['stopping_criteria'].append(Stream(callback_func=callback))
        else:
            kwargs['stopping_criteria'] = [Stream(callback_func=callback)]
        clear_torch_cache()
        with torch.no_grad():
            model.generate(**kwargs)

    def generate_with_streaming(**kwargs):
        return Iteratorize(generate_with_callback, kwargs, callback=None)

    with generate_with_streaming(**generate_params) as generator:
        for output in generator:
            next_token_ids = output[len(input_ids[0]):]
            if next_token_ids[0] == tokenizer.eos_token_id:
                break
            new_tokens = tokenizer.decode(
                next_token_ids, skip_special_tokens=True)
            if isinstance(tokenizer, LlamaTokenizer) and len(next_token_ids) > 0:
                if tokenizer.convert_ids_to_tokens(int(next_token_ids[0])).startswith('▁'):
                    new_tokens = ' ' + new_tokens

            history[-1][1] = new_tokens
            yield history
            if len(next_token_ids) >= max_new_tokens:
                break

setup()

with gr.Blocks() as demo:
    chatbot = gr.Chatbot()
    with gr.Row():
        with gr.Column(scale=4):
            with gr.Column(scale=3):
                system_prompt_input = gr.Textbox(
                    show_label=True,
                    label="System Prompt(Effective only before the start of the conversation or after clearing history)",
                    placeholder=DEFAULT_SYSTEM_PROMPT,
                    lines=1).style(
                    container=True)
            with gr.Column(scale=12):
                user_input = gr.Textbox(
                    show_label=True,
                    label="User Instruction",
                    placeholder="Shift + Enter To send instruction...",
                    lines=10).style(
                    container=True)
            with gr.Column(min_width=32, scale=1):
                submitBtn = gr.Button("Submit", variant="primary")
        with gr.Column(scale=1):
            emptyBtn = gr.Button("Clear History")
            max_new_token = gr.Slider(
                0,
                4096,
                value=512,
                step=1.0,
                label="Maximum New Token Length",
                interactive=True)
            top_p = gr.Slider(0, 1, value=0.9, step=0.01,
                              label="Top P", interactive=True)
            temperature = gr.Slider(
                0,
                1,
                value=0.2,
                step=0.01,
                label="Temperature",
                interactive=True)
            top_k = gr.Slider(1, 40, value=40, step=1,
                              label="Top K", interactive=True)
            repetition_penalty = gr.Slider(
                1.0,
                3.0,
                value=1.1,
                step=0.1,
                label="Repetition Penalty",
                interactive=True,
                visible=True)

    params = [user_input, chatbot]
    predict_params = [
        chatbot,
        system_prompt_input,
        max_new_token,
        top_p,
        temperature,
        top_k,
        repetition_penalty]

    submitBtn.click(
        user,
        params,
        params,
        queue=False).then(
        predict,
        predict_params,
        chatbot).then(
            lambda: gr.update(
                interactive=True),
        None,
        [user_input],
        queue=False)

    user_input.submit(
        user,
        params,
        params,
        queue=False).then(
        predict,
        predict_params,
        chatbot).then(
            lambda: gr.update(
                interactive=True),
        None,
        [user_input],
        queue=False)

    submitBtn.click(reset_user_input, [], [user_input])

    emptyBtn.click(reset_state, outputs=[chatbot], show_progress=True)

demo.queue().launch(
    share=share,
    inbrowser=True,
    server_name='0.0.0.0',
    server_port=port)