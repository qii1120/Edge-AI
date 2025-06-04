import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, DataCollatorForLanguageModeling # Added DataCollatorForLanguageModeling
from peft import PeftModel, LoraConfig, get_peft_model # Ensure LoraConfig, get_peft_model are imported
from tqdm.auto import tqdm
from datasets import load_dataset
import random
import numpy as np
from torch.optim import AdamW # For the optimizer
from transformers import get_linear_schedule_with_warmup # Optional: for learning rate scheduling

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
def generate(model, input_ids_prompt, past_key_values_cache, max_new_tokens_to_generate):
    # input_ids_prompt: The initial prompt tokens, e.g., shape [batch_size, prompt_seq_len]
    # past_key_values_cache: The StaticCache object.
    # max_new_tokens_to_generate: The total number of new tokens to generate.

    if max_new_tokens_to_generate == 0:
        return input_ids_prompt.clone()

    _input_ids_internal = input_ids_prompt.clone() # Internal tensor to accumulate tokens

    # --- Prefill Phase ---
    with torch.no_grad():
        outputs_prefill = model.prefill_forward(
            _input_ids_internal, # The prompt
            past_key_values=past_key_values_cache,
            position_ids=None,
            attention_mask=None,
            cache_position=None, # StaticCache usually handles this internally for prefill
            logits_to_keep=1     # As in your run_slm_stu.py
        )
        # Assuming StaticCache is updated in-place by prefill_forward,
        # but re-assigning is safer if it returns a new/updated cache object.
        past_key_values_cache = outputs_prefill.past_key_values

        # First newly generated token (T0)
        logits_t0 = outputs_prefill.logits
        if logits_t0.ndim == 3 and logits_t0.shape[1] == 1: # Ensure [B, V]
            logits_t0 = logits_t0.squeeze(1)
        
        token_t0 = torch.argmax(logits_t0, dim=-1) # Shape [B]
        if token_t0.ndim == 1:
            token_t0 = token_t0.unsqueeze(-1) # Shape [B, 1] for concatenation

        _input_ids_internal = torch.cat([_input_ids_internal, token_t0], dim=-1)
        
        # This `token_t0` will be the first `next_token_for_loop`
        next_token_for_loop = token_t0

    # --- Decoding Loop Phase ---
    # We've generated 1 token (t0). Need to generate (max_new_tokens_to_generate - 1) more.
    with torch.no_grad():
        for _ in range(max_new_tokens_to_generate - 1):
            # `pos` is the current length of `_input_ids_internal` which includes `next_token_for_loop` (the token from previous step)
            pos = _input_ids_internal.shape[1]
            
            # This cache_position logic is taken directly from your run_slm_stu.py's generate loop.
            # It implies that `position_ids` for `next_token_for_loop` (at actual index `pos-1`) is `pos`.
            loop_decode_cache_position = torch.arange(pos, pos + 1, device=_input_ids_internal.device, dtype=torch.long)

            outputs_loop = model( # This model.forward is the compiled one for decoding
                next_token_for_loop, # Token from previous step, shape [B, 1]
                past_key_values=past_key_values_cache,
                position_ids=loop_decode_cache_position.unsqueeze(0), # Shape [B,1] or [1,1]
                cache_position=loop_decode_cache_position  # Shape [B] or [1]
            )
            # Again, assume StaticCache updated in-place, but re-assign if needed.
            past_key_values_cache = outputs_loop.past_key_values

            logits_loop = outputs_loop.logits # Shape [B, 1, V]
            
            predicted_token_in_loop = torch.argmax(logits_loop, dim=-1) # Shape [B, 1]
            
            _input_ids_internal = torch.cat([_input_ids_internal, predicted_token_in_loop], dim=-1)
            next_token_for_loop = predicted_token_in_loop # For the next iteration

    return _input_ids_internal

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

    #recommended_inductor_config_setter()
    #backend = 'gemlite' 

    max_new_tokens = 256    # Number of new tokens to generate
    device = 'cuda:0'
    
    # --- 修改開始：載入你的蒸餾 QLoRA 適配器 ---
    # 基礎學生模型 ID (與 distillation.py 中 student_model_id 一致)
    base_model_name = "meta-llama/Llama-3.2-1B-Instruct"
    # 你在 Hugging Face Hub 上的適配器 ID
    #qlora_adapter_hub_id = "qii1120/llama1B-qlora-distilled-wikitext_S2_R64_MLP_E8_S6k_LR1.5e-5_TGM80_Tkd1.5_KDR0.6_GC"
    #commit_id = "3e0b6b938ff3ae620a3ec55e237198aa843b1896"
    qlora_adapter_hub_id = "fuku13001/llama1B-qlora-distilled-wikitext_S2_R128_MLP_E12_S15k_LR1e-5_TGM80_KDR0.6_TKD1.5_GC"
    commit_id =  "469ad153a9cfe0d361546dd8de40bd5df2afed74"
    output_hub_model_id = f"qii1120/final-model-group17" 
    local_adapter_save_dir = f"qlora_student_adapter-final" # Local dir can still use underscore if preferred

    # === 在此處初始化 Tokenizer ===
    print(f"INFO: Loading tokenizer for '{base_model_name}'...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token # 非常重要，確保 pad_token 被設定
        print(f"INFO: Tokenizer pad_token set to eos_token (ID: {tokenizer.eos_token_id}).")
    # === Tokenizer 初始化結束 ===


    # 1. 以 FP16 精度加載基礎學生模型
    print(f"INFO: Loading base student model '{base_model_name}' in FP16 for LoRA merging...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,   # 直接加載為 FP16
        device_map=device,           # 將模型直接加載到目標設備
        trust_remote_code=True,
        attn_implementation="sdpa"   # 保持這個優化
    )
    print(f"INFO: FP16 Base student model loaded on device: {model.device}, dtype: {model.dtype}. Attention implementation: {model.config._attn_implementation}")

    # 2. 從 Hugging Face Hub 將訓練好的 LoRA 適配器加載到 FP16 基礎模型上
    print(f"INFO: Loading LoRA adapter from '{qlora_adapter_hub_id}' onto FP16 model...")
    try:
        model = PeftModel.from_pretrained(model, qlora_adapter_hub_id, is_trainable=False, revision=commit_id)
    except Exception as e:
        print(f"ERROR: Failed to load adapter from Hub ID '{qlora_adapter_hub_id}'. Error: {e}")
        print("Please ensure the Hub ID is correct and the model/adapter is public or you have access.")
        return # Or handle error as appropriate
    print(f"INFO: LoRA adapter loaded. Model is PeftModel on device: {model.device}.")

    # 3. 合併 LoRA 適配器到基礎模型中
    print("INFO: Merging LoRA adapter into the FP16 base model...")
    model = model.merge_and_unload()
    print(f"INFO: LoRA adapter merged. Model is now {type(model)}.")
    
    # 確認模型仍在正確的設備上且為 FP16
    if str(model.device) != device:
        print(f"WARNING: Model may have moved from {model.device} after merge. Moving back to {device}...")
        model = model.to(device)
    if model.dtype != torch.float16:
        print(f"WARNING: Model dtype is {model.dtype} after merge. Casting to torch.float16...")
        model = model.to(torch.float16)
        
    print(f"INFO: Merged model is on device: {model.device} with dtype: {model.dtype}")
    # --- 修改結束 ---
    
    # === 第二階段 LoRA 微調超參數 === baseline: 11.97 (12.28, 11.88, 12.40(2048), )
    ENABLE_SECOND_STAGE_FINETUNING = True
    SECOND_LORA_R = 64  # default 64 (單獨調而已, alpha沒跟著改) 64 128 差不多, 32高0.01但較快一點 ， (R, alpha同改) 32差約0.05
    SECOND_LORA_ALPHA = 128 # default 128 (單獨調) 256變差
    SECOND_LORA_DROPOUT = 0.05 # 0.5=12.01, 0.2=11.99, 0.1=11.98, 0.05=11.97, 0.04 = 11.98, 0.03 = 11.97, 0.02 = 11.97
    SECOND_LORA_TARGET_MODULES = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "down_proj",
        "up_proj"
    ]
    # --- MODIFIED LINES START ---
    LEARNING_RATE_STAGE2 = 1.5e-6
    # (batch=1, grad=8) 1e-6: 12.48, 2e-6: 11.97, 2.5e-6: 11.97 (12.28, 11.88, 12.40), 3e-6: 12.04, 4e-6: (, , 12.42)
    # (batch=2, grad=8) 2.5e-6: (12.43), 4e-6: (12.36), 5e-6: (12.46)
    # --- MODIFIED LINES END ---
    NUM_EPOCHS_STAGE2 = 1
    BATCH_SIZE_STAGE2 = 1
    MAX_SEQ_LENGTH_STAGE2 = 512 # 保持序列長度
    GRAD_ACCUMULATION_STEPS_STAGE2 = 8
    NUM_WARMUP_STEPS_RATIO_STAGE2 = 0.1 # 保持預熱比例
    # === (結束) 第二階段 LoRA 微調超參數 ===
    
    # vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
    # === 在此處插入新的第二階段微LAMBDA 微調代碼 ===

    if ENABLE_SECOND_STAGE_FINETUNING:
        print("\nINFO: === Starting Second Stage LoRA Fine-tuning ===")

        # 確保 tokenizer 已經載入 (應該在你的代碼前面已經有了)
        # tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token # 重要：確保 pad_token
            print("INFO: Set tokenizer.pad_token to tokenizer.eos_token for second stage fine-tuning.")

        # 1. 準備數據集 (WikiText-2 訓練集)
        print("INFO: Loading WikiText-2 train dataset for second stage fine-tuning...")
        try:
            train_dataset_stage2 = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        except Exception as e:
            print(f"ERROR: Could not load wikitext-2-raw-v1 train split: {e}")
            print("INFO: Skipping second stage fine-tuning.")
            ENABLE_SECOND_STAGE_FINETUNING = False # 發生錯誤則跳過

        if ENABLE_SECOND_STAGE_FINETUNING:
            # 過濾掉空文本或過短文本
            train_dataset_stage2 = train_dataset_stage2.filter(lambda example: len(example['text'].strip()) > 10)

            def tokenize_function_stage2(examples):
                # 將文本塊連接起來再分塊，而不是單獨編碼每個小文本
                # 這有助於模型學習更長的上下文
                concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
                total_length = len(concatenated_examples[list(examples.keys())[0]])
                # 我們丟棄最後一小塊，但你可以選擇填充它
                total_length = (total_length // MAX_SEQ_LENGTH_STAGE2) * MAX_SEQ_LENGTH_STAGE2

                result = {
                    k: [t[i : i + MAX_SEQ_LENGTH_STAGE2] for i in range(0, total_length, MAX_SEQ_LENGTH_STAGE2)]
                    for k, t in concatenated_examples.items()
                }
                # 創建 labels，對於 Causal LM，labels 通常與 input_ids 相同
                result["labels"] = result["input_ids"].copy()
                return result

            print("INFO: Tokenizing dataset for second stage fine-tuning...")
            # 先對所有文本進行初步的 tokenizer 轉換，但不進行 padding 或 truncation
            def pre_tokenize_function(examples):
                return tokenizer(examples['text'], add_special_tokens=False) # add_special_tokens=False 避免在連接時產生過多eos

            tokenized_dataset_stage2_intermediate = train_dataset_stage2.map(
                pre_tokenize_function,
                batched=True,
                remove_columns=train_dataset_stage2.column_names
            )

            processed_dataset_stage2 = tokenized_dataset_stage2_intermediate.map(
                tokenize_function_stage2,
                batched=True,
            )
            print(f"INFO: Second stage fine-tuning dataset prepared with {len(processed_dataset_stage2)} samples of length {MAX_SEQ_LENGTH_STAGE2}.")


            # 2. 配置第二個 LoRA
            print("INFO: Configuring LoRA for the second stage fine-tuning...")
            second_lora_config = LoraConfig(
                r=SECOND_LORA_R,
                lora_alpha=SECOND_LORA_ALPHA,
                lora_dropout=SECOND_LORA_DROPOUT,
                target_modules=SECOND_LORA_TARGET_MODULES,
                bias="none",
                task_type="CAUSAL_LM"
            )

            # model 此時是 FP16 的基礎模型 (已合併第一次 LoRA)
            model.train() # 將模型設置為訓練模式
            model = get_peft_model(model, second_lora_config)
            print("INFO: Second stage LoRA model configured. Trainable parameters:")
            model.print_trainable_parameters()

            # 3. 訓練迴圈
            optimizer_stage2 = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE_STAGE2)

            # --- MODIFIED/UNCOMMENTED LINES START ---
            # 計算總訓練步數和預熱步數
            # 注意：len(processed_dataset_stage2) 會因 MAX_SEQ_LENGTH_STAGE2 變化而變化
            # 如果 MAX_SEQ_LENGTH_STAGE2 變大，樣本數會減少
            num_optimizer_steps_per_epoch_stage2 = (len(processed_dataset_stage2) + BATCH_SIZE_STAGE2 * GRAD_ACCUMULATION_STEPS_STAGE2 - 1) // (BATCH_SIZE_STAGE2 * GRAD_ACCUMULATION_STEPS_STAGE2)
            num_training_steps_stage2 = num_optimizer_steps_per_epoch_stage2 * NUM_EPOCHS_STAGE2
            num_warmup_steps_stage2 = int(NUM_WARMUP_STEPS_RATIO_STAGE2 * num_training_steps_stage2)

            print(f"INFO: Total training steps for stage 2: {num_training_steps_stage2}")
            print(f"INFO: Warmup steps for stage 2: {num_warmup_steps_stage2}")

            # 啟用學習率排程器
            scheduler_stage2 = get_linear_schedule_with_warmup(
                optimizer_stage2,
                num_warmup_steps=num_warmup_steps_stage2,
                num_training_steps=num_training_steps_stage2
            )
            # --- MODIFIED/UNCOMMENTED LINES END ---

            print(f"INFO: Starting second stage LoRA fine-tuning for {NUM_EPOCHS_STAGE2} epoch(s)...")
            global_step = 0
            for epoch in range(NUM_EPOCHS_STAGE2):
                model.train()
                epoch_loss = 0
                num_samples = len(processed_dataset_stage2)
                # 調整 tqdm 的 total，使其顯示的是 optimizer steps 的數量
                progress_bar = tqdm(range(num_optimizer_steps_per_epoch_stage2), desc=f"Epoch {epoch+1}/{NUM_EPOCHS_STAGE2}")

                # 修改訓練迭代邏輯以配合 optimizer steps 的 progress_bar
                batch_idx_counter = 0 
                for i in range(0, num_samples, BATCH_SIZE_STAGE2):
                    batch_indices = list(range(i, min(i + BATCH_SIZE_STAGE2, num_samples)))
                    if not batch_indices: continue

                    input_ids_list = [torch.tensor(processed_dataset_stage2[j]['input_ids']) for j in batch_indices]
                    labels_list = [torch.tensor(processed_dataset_stage2[j]['labels']) for j in batch_indices]
                    input_ids = torch.stack(input_ids_list).to(device)
                    labels = torch.stack(labels_list).to(device)
                    attention_mask = torch.ones_like(input_ids).to(device)

                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss / GRAD_ACCUMULATION_STEPS_STAGE2
                    loss.backward()
                    epoch_loss += loss.item() * GRAD_ACCUMULATION_STEPS_STAGE2

                    if (batch_idx_counter + 1) % GRAD_ACCUMULATION_STEPS_STAGE2 == 0 or (i + BATCH_SIZE_STAGE2) >= num_samples:
                        torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), 1.0)
                        optimizer_stage2.step()
                        scheduler_stage2.step() # <<< 在 optimizer step 後調用 scheduler step
                        optimizer_stage2.zero_grad()
                        global_step +=1
                        progress_bar.update(1) # 更新 progress_bar
                        if global_step % 10 == 0: # 每 10 個 optimizer step 打印一次 loss
                             current_lr = scheduler_stage2.get_last_lr()[0] if scheduler_stage2 else LEARNING_RATE_STAGE2
                             print(f"Epoch {epoch+1}, OptStep {global_step}, BatchGroup ~{ (batch_idx_counter + 1) // GRAD_ACCUMULATION_STEPS_STAGE2 }: Avg Batch Loss: {loss.item() * GRAD_ACCUMULATION_STEPS_STAGE2 :.4f}, LR: {current_lr:.2e}")
                    
                    batch_idx_counter +=1
                
                progress_bar.close()
                avg_epoch_loss = epoch_loss / (num_samples / BATCH_SIZE_STAGE2) if (num_samples / BATCH_SIZE_STAGE2) > 0 else 0
                print(f"INFO: Epoch {epoch+1} completed. Average training loss: {avg_epoch_loss :.4f}")

            # 4. 合併第二階段的 LoRA 適配器
            print("INFO: Merging second stage LoRA adapter...")
            model = model.merge_and_unload()
            print("INFO: Second stage LoRA adapter merged.")
            model.eval() # 確保模型回到評估模式

        print("INFO: === Second Stage LoRA Fine-tuning Finished (or skipped) ===")
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    # === (結束) 新的第二階段 LoLAMBDA 微調代碼 ===


    model.eval()
    # tokenizer = AutoTokenizer.from_pretrained(base_model_name) # Tokenizer from base
    
    # --- 先 HQQ 量化 ---
    #print("Starting HQQ quantization...")
    #quant_config = get_quant_config_slm(model) 
    #AutoHQQHFModel.quantize_model(model, quant_config=quant_config, compute_dtype=torch.float16, device=device)
    #print("HQQ quantization finished.")

    #print("Preparing model for inference with HQQ backend...")
    #from hqq.utils.patching import prepare_for_inference
    #prepare_for_inference(model, backend=backend) 
    #print("Model preparation for inference finished.")
    
    # --- 為自定義 generate 函數設置 prefill_forward 和編譯 decoding_forward ---
    # 此時 model.forward 是 HQQ 量化後且 prepare_for_inference處理過的版本
    # 將此版本賦予 prefill_forward (因為自定義 generate 的 prefill 也應使用量化模型)
    model.prefill_forward = model.forward
    print("Assigned current (quantized, prepared) model.forward to model.prefill_forward.")

    # 現在編譯 model.forward 給自定義 generate 的解碼循環使用
    #print("Compiling model.forward (for decoding loop) with torch.compile...")
    model.forward = torch.compile(model.forward, mode='max-autotune', dynamic=False, fullgraph=True)
    #print("Model compilation for decoding finished.")

    torch.cuda.empty_cache()
    # ------------------------------------------------------


    warmup_prompt = "Explain what AI is."
    inputs = tokenizer(warmup_prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    
    #kv_cache_max_len = model.config.max_position_embeddings
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
        #_ = model.generate(
        #    input_ids=input_ids,
        #    attention_mask=attention_mask,
        #    max_new_tokens=max_new_tokens,
        #    pad_token_id=tokenizer.eos_token_id,
        #)
        
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
        tput = generated[0][input_ids.shape[1]:].shape[0]/(elapsed_ms / 1000)
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
    
    print(f"Saving quantized model to local directory: {local_adapter_save_dir}...")
    model.save_pretrained(local_adapter_save_dir)
    tokenizer.save_pretrained(local_adapter_save_dir)
    
    print(f"Pushing quantized model to Hugging Face Hub: {output_hub_model_id}...")
    try:
        model.push_to_hub(output_hub_model_id, commit_message="Final model group17")
        tokenizer.push_to_hub(output_hub_model_id, commit_message="Final model group17")
        print("Quantized model successfully pushed to Hugging Face Hub!")
    except Exception as e:
        print(f"Failed to push model to Hugging Face Hub: {e}")
        print("Please ensure you have logged in using `huggingface-cli login` and your token has 'write' access.")


    # Save results to CSV
    import csv
    rounded_tput = round(org_tput, 1)
    ppl = round(ppl, 2)

    with open("result.csv", mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Id", "value"])
        writer.writerow([0, ppl])
        writer.writerow([1, rounded_tput])
        
if __name__ == '__main__':
    main()