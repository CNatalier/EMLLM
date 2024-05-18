import argparse
import os
import torch
from transformers import AutoModelForCausalLM, LlamaTokenizer,GenerationConfig,BitsAndBytesConfig
import sys

DEFAULT_SYSTEM_PROMPT = """You are a kind and helpful assistant. 你是一个善良且乐于助人的助手。"""

TEMPLATE = (
    "[INST] <<SYS>>\n"
    "{system_prompt}\n"
    "<</SYS>>\n\n"
    "{instruction} [/INST]"
)

HISTORY=""

parser = argparse.ArgumentParser()
parser.add_argument('--base_model', default=None, type=str, required=True)
parser.add_argument('--tokenizer_path', default=None, type=str)
parser.add_argument('--data_file', default=None, type=str, help="A file that contains instructions (one instruction per line)")
parser.add_argument('--gpus', default="0", type=str)
parser.add_argument('--alpha', type=str, default="1.0", help="The scaling factor of NTK method, can be a float or 'auto'. ")
parser.add_argument('--load_in_8bit', action='store_true', help="Load the LLM in the 8bit mode")
parser.add_argument('--load_in_4bit', action='store_true', help="Load the LLM in the 4bit mode")
parser.add_argument('--history', action='store_true', help="Load the LLM with memory in the chats")
parser.add_argument('--system_prompt', type=str, default=DEFAULT_SYSTEM_PROMPT, help="The system prompt of the prompt template.")
args = parser.parse_args()


if args.load_in_8bit and args.load_in_4bit:
    raise ValueError("Only one quantization method can be chosen for inference. Please check your arguments")

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)


from attn_and_long_ctx_patches import apply_attention_patch
apply_attention_patch(use_memory_efficient_attention=True)


generation_config = GenerationConfig(
    temperature=0.2,
    top_k=40,
    top_p=0.9,
    do_sample=True,
    num_beams=1,
    repetition_penalty=1.1,
    max_new_tokens=400
)

def generate_prompt(instruction, system_prompt=DEFAULT_SYSTEM_PROMPT):
    if args.history:
        system_prompt=HISTORY+system_prompt
    prompt=TEMPLATE.format_map({'instruction': instruction,'system_prompt': system_prompt})
    return prompt

if __name__ == '__main__':
    load_type = torch.float16
    device = torch.device(0)
    if args.tokenizer_path is None:
        args.tokenizer_path = args.base_model

    
    tokenizer = LlamaTokenizer.from_pretrained(args.tokenizer_path, legacy=True)
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
        quantization_config=quantization_config if (args.load_in_4bit or args.load_in_8bit) else None,
        trust_remote_code=True
    )

    model_vocab_size = base_model.get_input_embeddings().weight.size(0)
    tokenizer_vocab_size = len(tokenizer)
    print(f"Vocab of the base model: {model_vocab_size}")
    print(f"Vocab of the tokenizer: {tokenizer_vocab_size}")
    if model_vocab_size!=tokenizer_vocab_size:
        print("Resize model embeddings to fit tokenizer")
        base_model.resize_token_embeddings(tokenizer_vocab_size)
    
    model = base_model

    model.eval()

    with torch.no_grad():
        print('='*85)
        sys_output="Start inference with history mode." if args.history else "Start inference with once-chat mode."
        print(sys_output)

        while True:
            raw_input_text = input("Input:")
            if len(raw_input_text.strip())==0:
                break
            
            input_text = generate_prompt(instruction=raw_input_text, system_prompt=args.system_prompt)
            
            inputs = tokenizer(input_text,return_tensors="pt")  #add_special_tokens=False ?

            generation_output = model.generate(
                input_ids = inputs["input_ids"].to(device),
                attention_mask = inputs['attention_mask'].to(device),
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                generation_config = generation_config
            )
            s = generation_output[0]
            output = tokenizer.decode(s,skip_special_tokens=True)
            response = output.split("[/INST]")[-1].strip()
            if args.history:
                HISTORY+=f"{raw_input_text} [/INST] {response} [INST] "

            print("Response: ",response)
            print("\n")