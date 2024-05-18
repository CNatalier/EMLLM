import logging
import math
import gc
import traceback
from queue import Queue
from threading import Thread
from PIL import Image
from copy import deepcopy
import torch
from typing import Union
from dataset.data_utils import preprocess_multimodal,get_preprocess_func

from transformers import  CLIPImageProcessor
from transformers import LlamaTokenizer
from transformers.image_utils import ImageInput
import transformers
from transformers import (
    GenerationConfig, 
    LogitsProcessorList, 
    TemperatureLogitsWarper, 
    LogitsWarper
)
from transformers.generation.logits_process import LogitNormalization

logger = logging.getLogger(__name__)


DEFAULT_GENERATION_CONFIG = GenerationConfig(
    max_new_tokens=512,
    min_length=0,
    do_sample=True,
    top_p=0.9,
    top_k=40,
    num_beams=1,
    temperature=0.5,
    num_return_sequences=1,
    no_repeat_ngram_size=15,
    repetition_penalty=1.1
)

def preprocess(image_file,tokenizer,config,processor,sources):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    if image_file:
        #image = Image.open(image_file).convert('RGB')
        image = image_file
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
    
    sources=[sources]
    # sources= [[
    #         {
    #             "from": "human",
    #             "value": f"<image>\n{text_input}"
    #         },
    #         {
    #             "from": "gpt",
    #             "value": ""
    #         },
    #         # {
    #         #     "from": "human",
    #         #     "value": "这个男人和女孩可能是什么关系？"
    #         # },
    #         # {
    #         #     "from": "gpt",
    #         #     "value": ""
    #         # }
    #     ]]
    
    
    preprocess_func = get_preprocess_func(config.text_model_id)
    data_dict = preprocess_func(
            sources,
            tokenizer,
            has_image=(image_file is not None))

    data_dict = dict(input_ids=data_dict["input_ids"][0])
    if image_file is not None:
        data_dict['pixel_values'] = image
    
    
    input_ids = data_dict["input_ids"].unsqueeze(0)
    # print(input_ids)
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

# @torch.inference_mode()
# def infer(image_path,input,tokenizer,model,processor):
#     inputs=preprocess(image_path,tokenizer,model.config,processor.image_processor,input)

#     output = model.generate(**inputs,
#                 do_sample=True,
#                 temperature=0.3,
#                 top_k=40,
#                 top_p=0.9,
#                 num_beams=1,
#                 repetition_penalty=1.1,
#                 # no_repeat_ngram_size=3,
#                 max_new_tokens=256 )
    
#     result=processor.decode(output[0][2:], skip_special_tokens=True)
#     result = result[result.rfind("ASSISTANT: ")+ len("ASSISTANT: "):]
    
#     return result


@torch.inference_mode()
def chat_in_stream(model, image : Union[str,ImageInput], text : str, history = [], generation_config = None,tokenizer=None,processor=None):

    generation_config.bos_token_id = generation_config.bos_token_id or tokenizer.bos_token_id
    
    if len(history)==0:
        text="<image>\n"+text
    inputs=history+[{"from":"human", "value":text},{"from":"gpt", "value":""}]
    
    test_input=preprocess(image,tokenizer,model.config,processor.image_processor,inputs)
    
    generate_params = generation_config.to_dict()
    generate_params['input_ids'] = test_input['input_ids']
    generate_params['attention_mask'] = test_input['attention_mask']
    generate_params['pixel_values'] = test_input['pixel_values']
    generate_params['eos_token_id']=tokenizer.eos_token_id
    generate_params['pad_token_id']=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id


    
    history.append({"from":"human", "value":text})
        

    origin_size = len(test_input['input_ids'][0])
    eos_token_id = tokenizer.eos_token_id

    response = ''
    old_history = deepcopy(history)
    
    # with torch.no_grad():
    #     y=model.generate(**generate_params)
    #     print(y,flush=True)

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
            next_token_ids = output[2:]
            if next_token_ids[0] == eos_token_id:
                break
            next_tokens = tokenizer.decode(
                next_token_ids, skip_special_tokens=True)
            if type(tokenizer) is LlamaTokenizer and len(next_token_ids) > 0:
                if tokenizer.convert_ids_to_tokens(int(next_token_ids[0])).startswith('▁'):
                    next_tokens = ' ' + next_tokens
            response = next_tokens
            
            response=response[response.rfind("ASSISTANT: ")+ len("ASSISTANT: "):]

            history = deepcopy(old_history)
            history.append({"from":"gpt", "value":response})

            yield response, history
            if len(test_input['input_ids'][0]) > origin_size + generation_config.max_new_tokens:
                break

        print("Response:", response)
        print("History:", history)


class TailFreeLogitsWarper(LogitsWarper):
    def __init__(self, tfs: float, filter_value: float = -float("Inf"), min_tokens_to_keep: int = 1):
        tfs = float(tfs)
        if tfs < 0 or tfs > 1.0:
            raise ValueError(f"`tfs` has to be a float >= 0 and <= 1, but is {tfs}")
        self.tfs = tfs
        self.filter_value = filter_value
        self.min_tokens_to_keep = min_tokens_to_keep

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        sorted_logits, sorted_indices = torch.sort(scores, descending=True)
        probs = sorted_logits.softmax(dim=-1)

        # Compute second derivative normalized CDF
        d2 = probs.diff().diff().abs()
        normalized_d2 = d2 / d2.sum(dim=-1, keepdim=True)
        normalized_d2_cdf = normalized_d2.cumsum(dim=-1)

        # Remove tokens with CDF value above the threshold (token with 0 are kept)
        sorted_indices_to_remove = normalized_d2_cdf > self.tfs

        # Centre the distribution around the cutoff as in the original implementation of the algorithm
        sorted_indices_to_remove = torch.cat(
            (
                torch.zeros(scores.shape[0], 1, dtype=torch.bool, device=scores.device),
                sorted_indices_to_remove,
                torch.ones(scores.shape[0], 1, dtype=torch.bool, device=scores.device),
            ),
            dim=-1,
        )

        if self.min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep
            sorted_indices_to_remove[..., : self.min_tokens_to_keep] = 0

        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        scores = scores.masked_fill(indices_to_remove, self.filter_value)
        return scores


class TopALogitsWarper(LogitsWarper):
    def __init__(self, top_a: float, filter_value: float = -float("Inf"), min_tokens_to_keep: int = 1):
        top_a = float(top_a)
        if top_a < 0 or top_a > 1.0:
            raise ValueError(f"`top_a` has to be a float >= 0 and <= 1, but is {top_a}")
        self.top_a = top_a
        self.filter_value = filter_value
        self.min_tokens_to_keep = min_tokens_to_keep

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        sorted_logits, sorted_indices = torch.sort(scores, descending=True)
        probs = sorted_logits.softmax(dim=-1)

        # Remove tokens with probability less than top_a*(max(probs))^2 (token with 0 are kept)
        probs_max = probs[..., 0, None]
        sorted_indices_to_remove = probs < probs_max * probs_max * self.top_a

        if self.min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep
            sorted_indices_to_remove[..., : self.min_tokens_to_keep] = 0

        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        scores = scores.masked_fill(indices_to_remove, self.filter_value)
        return scores


class MirostatLogitsWarper(LogitsWarper):
    def __init__(self, mirostat_mode: int, mirostat_tau: float, mirostat_eta: float, filter_value: float = -float("Inf"), min_tokens_to_keep: int = 1):
        if mirostat_mode not in [2]:
            raise ValueError(f"`mirostat` has to be a an integer 2, but is {mirostat_mode}")
        self.mirostat_mode = mirostat_mode
        self.mirostat_eta = mirostat_eta
        self.mirostat_tau = mirostat_tau
        self.filter_value = filter_value
        self.min_tokens_to_keep = min_tokens_to_keep
        self.mu = 2 * self.mirostat_tau
        self.e = 0

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        logits = scores[0]
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        prob_original = torch.softmax(sorted_logits, dim=-1).tolist()  # candidates

        # Truncate the words with surprise values greater than mu
        for i, candidate in enumerate(prob_original):
            if candidate > 0 and -math.log2(candidate) > self.mu:
                if (i == 0):
                    sorted_logits = sorted_logits[:1]
                else:
                    sorted_logits = sorted_logits[:i]
                break

        # Normalize the probabilities of the remaining words
        prob_topk = torch.softmax(sorted_logits, dim=0)

        prev_i = torch.multinomial(prob_topk, num_samples=1, replacement=True).to('cuda')

        observed_surprise = -math.log2(prob_topk[prev_i])
        self.e = observed_surprise - self.mirostat_tau

        # Update mu using the learning rate and error
        self.mu -= self.mirostat_eta * self.e

        sorted_indices_to_remove = torch.ones_like(scores[0], dtype=torch.bool)
        sorted_indices_to_remove[prev_i] = False

        indices_to_remove = sorted_indices_to_remove.unsqueeze(0).scatter(1, sorted_indices.unsqueeze(0), sorted_indices_to_remove.unsqueeze(0))
        scores = scores.masked_fill(indices_to_remove, self.filter_value)
        return scores


def get_logits_warper_patch(self, generation_config):
    warpers = self._get_logits_warper_old(generation_config)
    warpers_to_add = LogitsProcessorList()
    min_tokens_to_keep = 2 if generation_config.num_beams > 1 else 1

    if generation_config.mirostat_mode is not None and generation_config.mirostat_mode == 2:
        warpers_to_add.append(MirostatLogitsWarper(mirostat_mode=generation_config.mirostat_mode, mirostat_eta=generation_config.mirostat_eta, mirostat_tau=generation_config.mirostat_tau, min_tokens_to_keep=min_tokens_to_keep))
        # We need to disable samplers other than temperature
        for warper in warpers:
            if not isinstance(warper, TemperatureLogitsWarper):
                warpers.remove(warper)
    else:
        if generation_config.tfs is not None and 0.0 <= generation_config.tfs <= 1.0:
            warpers_to_add.append(TailFreeLogitsWarper(tfs=generation_config.tfs, min_tokens_to_keep=min_tokens_to_keep))
        if generation_config.top_a is not None and 0.0 <= generation_config.top_a <= 1.0:
            warpers_to_add.append(TopALogitsWarper(top_a=generation_config.top_a, min_tokens_to_keep=min_tokens_to_keep))

    if warpers and isinstance(warpers[-1], LogitNormalization):
        warpers = warpers[:-1] + warpers_to_add + [warpers[-1]]
    else:
        warpers += warpers_to_add

    return warpers


def generation_config_init_patch(self, **kwargs):
    self.__init___old(**kwargs)
    self.tfs = kwargs.pop("tfs", 1.0)
    self.top_a = kwargs.pop("top_a", 0.0)
    self.mirostat_mode = kwargs.pop("mirostat_mode", 0)
    self.mirostat_eta = kwargs.pop("mirostat_eta", 0.1)
    self.mirostat_tau = kwargs.pop("mirostat_tau", 5)


def hijack_samplers():
    transformers.GenerationMixin._get_logits_warper_old = transformers.GenerationMixin._get_logits_warper
    transformers.GenerationMixin._get_logits_warper = get_logits_warper_patch

    transformers.GenerationConfig.__init___old = transformers.GenerationConfig.__init__
    transformers.GenerationConfig.__init__ = generation_config_init_patch



class Stream(transformers.StoppingCriteria):
    def __init__(self, callback_func=None):
        self.callback_func = callback_func

    def __call__(self, input_ids, scores) -> bool:
        if self.callback_func is not None:
            self.callback_func(input_ids[0])
        return False



class Iteratorize:

    """
    Transforms a function that takes a callback
    into a lazy iterator (generator).

    Adapted from: https://stackoverflow.com/a/9969000
    """

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
            except:
                traceback.print_exc()
                pass

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