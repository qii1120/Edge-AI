import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm.auto import tqdm
from datasets import load_dataset
import random
import numpy as np

from hqq_utils import AutoHQQHFModel, get_size_of_model
from hqq.utils.patching import recommended_inductor_config_setter

from quant_cfg import get_quant_config_slm

#####################################################################
# === SPEC NOTICE ===
# Only "load model" and "generate" function selection can be modified.
# DO NOT change PPL calculation, timing, or throughput logic.
#####################################################################

# === (Optional) Define your own custom generate function. ===
# This is useful if you want full control over KV cache and generation steps.
# You can modify this function to suit your needs.
# By default, we use model.generate() for simplicity and general use.
def generate(model, input_ids, past_key_values, max_new_tokens):
    # 確保 initial_input_ids 已經在正確的 device
    initial_input_len = input_ids.shape[1]
    # 預分配一個足夠大的張量來存儲結果，避免多次 cat
    # 假設 batch_size 為 1
    generated_ids = torch.empty(
        1, initial_input_len + max_new_tokens,
        dtype=input_ids.dtype,
        device=input_ids.device
    )
    generated_ids[:, :initial_input_len] = input_ids

    with torch.no_grad():
        # Prefill
        outputs = model.prefill_forward(
            input_ids, # 這裡仍然傳入原始 input_ids
            past_key_values=past_key_values,
            position_ids=None,
            attention_mask=None,
            cache_position=None,
            logits_to_keep=1
        )
        past_key_values = outputs.past_key_values
        next_token = torch.argmax(outputs.logits, dim=-1)

        # 就地更新預分配的張量
        generated_ids[:, initial_input_len] = next_token

        # Token-by-token Decoding
        for i in range(max_new_tokens):
            pos = initial_input_len + i # 當前 token 的位置
            cache_position = torch.tensor([pos], device=input_ids.device, dtype=torch.long) # 對於 batch_size=1

            outputs = model(
                next_token, # 這裡還是只傳入下一個 token
                past_key_values=past_key_values,
                position_ids=cache_position.unsqueeze(0),
                cache_position=cache_position
            )
            logits = outputs.logits
            next_token = torch.argmax(logits, dim=-1)

            # 就地更新
            if i < max_new_tokens -1: # 避免最後一次循環寫入超出範圍
                generated_ids[:, initial_input_len + i + 1] = next_token
            past_key_values = outputs.past_key_values

    # 返回實際生成的內容 (如果需要裁剪，則裁剪到生成的實際長度)
    # 這裡可能需要根據實際生成長度來截取，但為了與原始邏輯兼容，先返回完整長度
    return generated_ids
def evaluate_ppl(model, tokenizer, device="cuda:0"):
    test_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    
    test_enc = tokenizer("\n\n".join(test_dataset["text"]), return_tensors="pt")
    model.seqlen = 2048
    test_enc = test_enc.input_ids.to(device)
    
    nsamples = test_enc.numel() // model.seqlen
    nlls = []  
    for i in tqdm(range(nsamples), desc="Evaluating..."):
        batch = test_enc[:, (i * model.seqlen):((i + 1) * model.seqlen)]
        
        with torch.no_grad():
            lm_logits = model(batch).logits

        shift_logits = lm_logits[:, :-1, :].contiguous().float()
        shift_labels = test_enc[:, (i * model.seqlen):((i + 1) * model.seqlen)][:, 1:]

        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)

    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    
    return ppl.item()

def main():
    ############## Set Up ##############
    torch.manual_seed(0)
    random.seed(0)
    recommended_inductor_config_setter()
    
    max_new_tokens = 256    # Number of new tokens to generate
    device = 'cuda:0'
    backend = 'gemlite'
    
    ### === TODO: Load your model (you may change this part) ===
    model_name = "qii1120/final-model-group17"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map=device,
    )
    #####################################
    
    model.eval() 
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # === (Optional) Uncomment the following lines if using the custom generate() function. ===
    model.prefill_forward = model.forward
    model.forward = torch.compile(model.forward, mode='max-autotune', dynamic=False, fullgraph=True)

    # Quantization
    quant_config = get_quant_config_slm(model)
    
    AutoHQQHFModel.quantize_model(model, quant_config=quant_config, compute_dtype=torch.float16, device=device)

    from hqq.utils.patching import prepare_for_inference
    prepare_for_inference(model, backend=backend) 

    warmup_prompt = "Explain what AI is."
    inputs = tokenizer(warmup_prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    
    # === (Optional) Set up StaticCache for manual KV cache management ===
    from transformers import StaticCache
    past_key_values = StaticCache(
        config=model.config, 
        max_batch_size=1, 
        max_cache_len=max_new_tokens + 16, 
        device=model.device, 
        dtype=torch.float16
    )
    ####################################################################
    
    for i in tqdm(range(5), desc="Warm Up..."):
        #  === Default: use model.generate() for end-to-end warm-up === 
        # _ = model.generate(
        #     input_ids=input_ids,
        #     attention_mask=attention_mask,
        #     max_new_tokens=max_new_tokens,
        #     pad_token_id=tokenizer.eos_token_id,
        # )
        
        # === (Optional) Use custom generate() if uncommented ===
        generated = generate(model, input_ids, past_key_values, max_new_tokens)
        past_key_values.reset()
        
    prompt = "How to learn a new language?"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    tputs = []
    time_record = []
    for _ in tqdm(range(10), desc="Test Inference"):
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()

        # === Default: Use model.generate() for end-to-end timing === 
        # generated = model.generate(
        #     input_ids=input_ids,
        #     attention_mask=attention_mask,
        #     max_new_tokens=max_new_tokens,
        #     pad_token_id=tokenizer.eos_token_id,
        # )
        
        # === Optional: Use custom generate() if uncommented ===
        generated = generate(model, input_ids, past_key_values, max_new_tokens)
        past_key_values.reset()

        end.record()
        torch.cuda.synchronize()
        elapsed_ms = start.elapsed_time(end)
        tput = tput = generated[0][input_ids.shape[1]:].shape[0]/(elapsed_ms / 1000)
        time_record.append(elapsed_ms / 1000)
        tputs.append(tput)
        
    response = tokenizer.decode(generated[0][input_ids.shape[1]:], skip_special_tokens=True)
    sorted_tputs = np.sort(tputs)[2:-2]
    org_tput = np.mean(sorted_tputs)
    print(f'Prompt: {prompt}\nResponse: {response}\n')
    
    print(f'Time Record: {time_record}')
    print(f'Throughput Record: {tputs} toks/s\n')

    ### Your final throughput result ###
    print(f'Throughput: {org_tput} toks/s')
    ppl = evaluate_ppl(model, tokenizer, device)
    print(f"Perplexity (PPL): {ppl}")
    
    # model.save_pretrained("qii_model")
    # tokenizer.save_pretrained("qii_model")
    
    # output_hub_model_id = "qii1120/Final-Model-Group17-v2"

    # print(f"Pushing quantized model to Hugging Face Hub: {output_hub_model_id}...")
    # try:
    #     model.push_to_hub(output_hub_model_id, commit_message="Final model group17")
    #     tokenizer.push_to_hub(output_hub_model_id, commit_message="Final model group17")
    #     print("Final model successfully pushed to Hugging Face Hub!")
    # except Exception as e:
    #     print(f"Failed to push model to Hugging Face Hub: {e}")
    #     print("Please ensure you have logged in using `huggingface-cli login` and your token has 'write' access.")


    # Save results to CSV
    import csv
    rounded_tput = round(org_tput, 1)
    ppl = round(ppl, 2)

    with open("result.csv", mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Id", "value"])
        writer.writerow([0, ppl])
        writer.writerow([1, rounded_tput])

    # print(f"\nSaving model to huggingface hub...")
    # try:
    #     # merged_model = model.merge_and_unload()
    #     model.push_to_hub("qii1120/Llama-3.2-3B-Instruct-model_v2", private=True)
    #     tokenizer.push_to_hub("qii1120/Llama-3.2-3B-Instruct-model_v2", private=True)
    #     print(f"Model and tokenizer successfully saved to huggingface hub.")
    # except Exception as e:
    #     print(f"Failed to save model to huggingface hub: {e}")
        
if __name__ == '__main__':
    main()
