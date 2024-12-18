import torch
import intel_extension_for_pytorch as ipex
from transformers import GPT2Tokenizer, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, LlamaTokenizer, LlamaForCausalLM
def batch_encode(prompts, tokenizer, prompt_len=512):
        input_tokens = tokenizer.batch_encode_plus(prompts, return_tensors="pt", padding="max_length", max_length=len(prompts))
        for t in input_tokens:
            if torch.is_tensor(input_tokens[t]):
                input_tokens[t] = input_tokens[t].to("xpu")
                #input_tokens[t] = input_tokens[t].to(torch.cuda.current_device())
        return input_tokens


def generate_prompt(model, tokenizer, prompts):
    
    input_tokens = batch_encode(prompts, tokenizer)
    print(input_tokens)
    generate_kwargs = dict(max_new_tokens=30, do_sample=False)
    output_ids = model.generate(**input_tokens, **generate_kwargs)
    print(output_ids)
    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)

    return outputs

if __name__ == '__main__':
     
    model = LlamaForCausalLM.from_pretrained("/flare/Aurora_deployment/vsastry/hf_new_cp/") 
    model.to("xpu") # model.cuda()
    model.seqlen = 4096
    
    # get llama tokenizer
    tokenizer = LlamaTokenizer.from_pretrained("/flare/Aurora_deployment/AuroraGPT/datasets/dolma/utils/tokenizer.model") 
    tokenizer.pad_token = tokenizer.eos_token
    output = generate_prompt(model, tokenizer, prompts=["What is the language spoken in Mexico ?"])
    print(output)
