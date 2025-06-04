import torch
import torch.nn.functional as F
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling, # 雖然可能未使用，但保留以匹配您之前的腳本
    get_cosine_schedule_with_warmup  # 確保導入 cosine scheduler
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training
)
from datasets import load_dataset
from tqdm.auto import tqdm
import random # 保留導入
import numpy as np
from huggingface_hub import create_repo, HfApi # 保留導入

# --- QLoRA 和蒸餾相關的配置 ---
# <<< MODIFIED: 進一步提升 LoRA 容量 >>>
LORA_R = 128 # 原為 64
LORA_ALPHA = 256 # 原為 128 (通常是 LORA_R 的兩倍)
LORA_DROPOUT = 0.05 # 保持不變
LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj"
]

BNB_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

def run_qlora_distillation_refined_s2(): # <<< 函數名加入標識 S2 (Stage 2 attempt)
    # --- 1. 環境與模型設定 ---
    device = 'cuda:1' # <<< MODIFIED: 嚴格使用 cuda:1
    teacher_model_id = "meta-llama/Llama-3.2-3B-Instruct"
    student_model_id = "meta-llama/Llama-3.2-1B-Instruct"
    
    # <<< MODIFIED: 新的實驗標籤，反映參數調整 >>>
    experiment_tag = "wikitext_S2_R128_MLP_E12_S15k_LR1e-5_TGM80_KDR0.6_TKD1.5_GC"
    local_adapter_save_dir = f"qlora_student_adapter_{experiment_tag}"
    
    # <<< 請務必將 YOUR_HF_USERNAME 替換為您的 Hugging Face 用戶名 >>>
    # output_hub_model_id = f"fuku13001/llama1B-qlora-distilled-{experiment_tag}" 
    output_hub_model_id = f"fuku13001/llama1B-qlora-distilled-{experiment_tag}"


    # --- 蒸餾超參數 (應用推薦的調整) ---
    distill_learning_rate = 1e-5      # <<< MODIFIED: 嘗試更低的學習率 (原 1.5e-5)
    distill_num_epochs = 12           # <<< MODIFIED: 增加訓練週期 (原 8)
    teacher_gen_max_new_tokens = 80   # <<< MODIFIED: 增加教師生成長度 (原 60)
    kd_ratio = 0.6                    # <<< MODIFIED: 略微增加 KD Loss 的權重 (原 0.5)
    T_kd = 1.5                        # <<< MODIFIED: 略微降低溫度 (原 2.0)，使目標分佈更尖銳一點
    
    teacher_gen_temperature = 0.7     # 略微調整 (原 0.75)

    # QLoRA 腳本原有設定
    gradient_accumulation_steps = 4 
    max_seq_length = 512      
    MIN_PROMPT_LEN_FOR_DISTILL_CHARS = 80 
    NUM_WIKITEXT_SAMPLES_TO_USE = 15000 # <<< MODIFIED: 增加每個 epoch 的樣本數 (原 6000)
    warmup_ratio = 0.05 # 保持

    # <<< MODIFIED: 啟用梯度檢查點以節省顯存 >>>
    USE_GRADIENT_CHECKPOINTING = True

    print(f"--- Configuration ({experiment_tag}) ---")
    print(f"Device: {device}")
    print(f"Teacher Model: {teacher_model_id}, Student Model: {student_model_id}")
    print(f"Local Adapter Save Dir: {local_adapter_save_dir}")
    print(f"Hub Model ID: {output_hub_model_id}")
    print(f"Training: {NUM_WIKITEXT_SAMPLES_TO_USE} samples/epoch, for {distill_num_epochs} epochs.")
    print(f"QLoRA: R={LORA_R}, Alpha={LORA_ALPHA}, Dropout={LORA_DROPOUT}, TargetModules={LORA_TARGET_MODULES}")
    print(f"BNB_CONFIG: Compute DType={BNB_CONFIG.bnb_4bit_compute_dtype}")
    print(f"Distill Params: LR={distill_learning_rate}, KD Weight={kd_ratio}, T_kd={T_kd}")
    print(f"Teacher Gen: MaxTokens={teacher_gen_max_new_tokens}, Temp={teacher_gen_temperature}")
    print(f"Training Details: GradAccum={gradient_accumulation_steps}, MaxSeqLen={max_seq_length}, WarmupRatio={warmup_ratio}")
    print(f"Gradient Checkpointing: {USE_GRADIENT_CHECKPOINTING}")
    print(f"--- End Configuration ---")

    tokenizer = AutoTokenizer.from_pretrained(teacher_model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("Tokenizer pad_token set to eos_token.")

    print("Loading Teacher Model...")
    teacher_model = AutoModelForCausalLM.from_pretrained(
        teacher_model_id,
        torch_dtype=torch.float16,
        device_map={"": device} 
    ).eval()
    print("Teacher Model Loaded.")

    print("Loading and Configuring QLoRA Student Model...")
    student_model = AutoModelForCausalLM.from_pretrained(
        student_model_id,
        quantization_config=BNB_CONFIG, 
        device_map={"": device} # 模型會被完整加載到指定 device
    )
    
    student_model = prepare_model_for_kbit_training(
        student_model, use_gradient_checkpointing=USE_GRADIENT_CHECKPOINTING
    ) 

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM"
    )
    student_model = get_peft_model(student_model, lora_config)
    student_model.train() 
    print("QLoRA Student Model Configured. Trainable parameters:")
    student_model.print_trainable_parameters()

    print("Preparing wikitext dataset for distillation...")
    try:
        raw_dataset_wikitext = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        distill_texts = []
        print(f"Filtering up to {NUM_WIKITEXT_SAMPLES_TO_USE} valid samples from WikiText-2...")
        count = 0
        # Shuffle once and take top N to ensure variety if NUM_WIKITEXT_SAMPLES_TO_USE is less than dataset size
        # If NUM_WIKITEXT_SAMPLES_TO_USE is greater, it will use all valid samples up to that number.
        shuffled_wikitext = raw_dataset_wikitext.shuffle(seed=42) 
        
        for sample in tqdm(shuffled_wikitext, desc="Filtering WikiText Samples", total=min(len(shuffled_wikitext), NUM_WIKITEXT_SAMPLES_TO_USE)):
            text = sample['text'].strip()
            if len(text) >= MIN_PROMPT_LEN_FOR_DISTILL_CHARS and \
               not text.startswith(" = ") and not text.endswith(" = ") and \
               len(text.split()) > 10: # Basic filter for number of words
                distill_texts.append(text)
                count += 1
            if count >= NUM_WIKITEXT_SAMPLES_TO_USE: # Stop if we have enough samples
                break
        
        if not distill_texts:
            raise ValueError("No suitable samples found in wikitext after filtering and no fallback defined.")
        if count < NUM_WIKITEXT_SAMPLES_TO_USE:
             print(f"Warning: Could only filter {len(distill_texts)} samples from WikiText-2 (target: {NUM_WIKITEXT_SAMPLES_TO_USE}). Using available samples.")
        
        print(f"Distillation dataset prepared with {len(distill_texts)} samples.")
    except Exception as e:
        print(f"Failed to load or process wikitext dataset: {e}")
        print("Please ensure 'datasets' library is installed and you have internet access.")
        return 

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, student_model.parameters()), 
        lr=distill_learning_rate
    )
    
    num_samples_per_epoch = len(distill_texts)
    # Effective batch size is 1 due to per-sample processing loop
    # Optimizer steps per epoch depends on gradient_accumulation_steps
    steps_per_epoch = (num_samples_per_epoch + gradient_accumulation_steps - 1) // gradient_accumulation_steps
    total_optimizer_steps = steps_per_epoch * distill_num_epochs
    num_warmup_steps = int(warmup_ratio * total_optimizer_steps)
    
    print(f"Number of samples per epoch: {num_samples_per_epoch}")
    print(f"Optimizer steps per epoch: {steps_per_epoch}")
    print(f"Total effective optimizer steps: {total_optimizer_steps}")
    print(f"Warmup steps: {num_warmup_steps}")

    scheduler = get_cosine_schedule_with_warmup( 
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=total_optimizer_steps
    )
    print("Using Cosine learning rate scheduler.")

    ce_loss_fn = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    kd_loss_fn = torch.nn.KLDivLoss(reduction="batchmean")

    try:
        print(f"Ensuring Hugging Face Hub repository '{output_hub_model_id}' exists...")
        # Ensure you are logged in: huggingface-cli login
        create_repo(output_hub_model_id, private=True, exist_ok=True) # Set private=False if you want it public
        print(f"Hub repository '{output_hub_model_id}' ensured.")
    except Exception as e:
        print(f"Warning: Could not create or ensure Hub repo {output_hub_model_id} beforehand: {e}")
        print("Please ensure you are logged in to Hugging Face CLI ('huggingface-cli login') and the repo name is valid.")


    print(f"\n--- Starting QLoRA-assisted Dataset Distillation ({distill_num_epochs} epochs) ---")
    global_optimizer_step_counter = 0 

    for epoch in range(distill_num_epochs):
        print(f"\nEpoch {epoch+1}/{distill_num_epochs}")
        student_model.train() 
        optimizer.zero_grad() 
        
        epoch_total_loss_sum = 0.0
        epoch_ce_loss_sum = 0.0
        epoch_kd_loss_sum = 0.0
        num_samples_processed_in_epoch = 0

        # Shuffling texts for each epoch to ensure different order of samples if NUM_WIKITEXT_SAMPLES_TO_USE is less than total
        # If using all filtered samples, this internal shuffle isn't strictly necessary as dataset was shuffled once.
        # random.shuffle(distill_texts) # Optional: shuffle data for each epoch

        progress_bar_samples = tqdm(distill_texts, desc=f"Epoch {epoch+1} Samples")
        for text_idx, input_text in enumerate(progress_bar_samples):
            # Ensure prompt_max_len calculation is robust
            # Max length for prompt tokens = total_seq_len - max_new_tokens_by_teacher - safety_buffer_for_special_tokens
            prompt_max_len = max_seq_length - teacher_gen_max_new_tokens - 15 # Increased buffer slightly
            if prompt_max_len <=0:
                print(f"Error: prompt_max_len ({prompt_max_len}) is too small. Check max_seq_length and teacher_gen_max_new_tokens.")
                return

            inputs = tokenizer(
                input_text, return_tensors="pt", max_length=prompt_max_len, 
                truncation=True, padding=False # No padding for variable length prompts
            ).to(device)
            current_input_ids = inputs.input_ids
            current_attention_mask = inputs.attention_mask

            if current_input_ids.shape[1] == 0: 
                print(f"Warning: Skipped empty tokenized prompt for sample index {text_idx}. Original text: '{input_text[:50]}...'")
                continue
            
            # --- A. Teacher Generates Targets ---
            with torch.no_grad():
                teacher_model.eval() 
                # Teacher generates full sequence (prompt + completion)
                teacher_generation_ids = teacher_model.generate(
                    current_input_ids, attention_mask=current_attention_mask,
                    max_new_tokens=teacher_gen_max_new_tokens, pad_token_id=tokenizer.eos_token_id,
                    do_sample=True, temperature=teacher_gen_temperature, top_k=50, top_p=0.95,
                    # Ensure generated sequence does not exceed max_seq_length
                    # max_length = max_seq_length # This can also be used if preferred
                )
                
                # Prepare inputs for teacher to get logits (full generated sequence except last token)
                # Ensure the sequence does not exceed max_seq_length for teacher logit generation.
                # This is important if teacher_generation_ids could become > max_seq_length
                teacher_logits_input_ids = teacher_generation_ids[:, :max_seq_length-1]
                teacher_logits_attention_mask = torch.ones_like(teacher_logits_input_ids, device=device)

                teacher_outputs_for_kd = teacher_model(
                    teacher_logits_input_ids, attention_mask=teacher_logits_attention_mask
                )
                teacher_distill_logits = teacher_outputs_for_kd.logits # Shape: [1, seq_len_teacher_kd, vocab_size]

            # --- B. Student Predicts ---
            # Student input is the same as teacher's input for logit generation
            student_input_ids_for_train = teacher_logits_input_ids # Shape: [1, seq_len_teacher_kd]
            student_attention_mask_for_train = teacher_logits_attention_mask
            
            student_model.train() 
            student_outputs = student_model(
                student_input_ids_for_train, attention_mask=student_attention_mask_for_train
            )
            student_distill_logits = student_outputs.logits # Shape: [1, seq_len_teacher_kd, vocab_size]
            
            # --- C. Calculate Loss ---
            # Labels are the teacher_generation_ids shifted left (from 2nd token to end)
            # We need to ensure labels match the length of logits (seq_len_teacher_kd)
            labels = teacher_generation_ids[:, 1:teacher_distill_logits.shape[1]+1].contiguous()
            
            # Defensive length check/slice, though ideally logits and labels should align from construction
            current_seq_len = min(student_distill_logits.size(1), labels.size(1))
            if student_distill_logits.size(1) != current_seq_len:
                student_distill_logits = student_distill_logits[:, :current_seq_len, :]
            if teacher_distill_logits.size(1) != current_seq_len: # Should be same as student's
                 teacher_distill_logits = teacher_distill_logits[:, :current_seq_len, :]
            if labels.size(1) != current_seq_len:
                labels = labels[:, :current_seq_len]

            flat_student_logits = student_distill_logits.reshape(-1, student_distill_logits.size(-1))
            flat_labels = labels.reshape(-1)
            flat_teacher_distill_logits = teacher_distill_logits.reshape(-1, teacher_distill_logits.size(-1))
            
            if flat_student_logits.shape[0] == 0: # Skip if sequence length becomes zero after slicing
                print(f"Warning: Skipped sample {text_idx} due to zero effective sequence length after alignment.")
                continue

            loss_ce = ce_loss_fn(flat_student_logits, flat_labels)
            loss_kd = kd_loss_fn(
                F.log_softmax(flat_student_logits / T_kd, dim=-1),
                F.softmax(flat_teacher_distill_logits / T_kd, dim=-1)
            ) * (T_kd ** 2) # Scale KD loss by T^2
            
            current_loss = (1.0 - kd_ratio) * loss_ce + kd_ratio * loss_kd
            
            epoch_total_loss_sum += current_loss.item()
            epoch_ce_loss_sum += loss_ce.item()
            epoch_kd_loss_sum += loss_kd.item()
            num_samples_processed_in_epoch +=1

            loss_for_backward = current_loss / gradient_accumulation_steps 
            loss_for_backward.backward()
            
            # --- D. Optimizer Step ---
            if (text_idx + 1) % gradient_accumulation_steps == 0 or \
               (text_idx + 1) == len(distill_texts): # Last batch in epoch
                torch.nn.utils.clip_grad_norm_(filter(lambda p:p.requires_grad, student_model.parameters()), 1.0) 
                optimizer.step()
                scheduler.step() 
                optimizer.zero_grad()
                global_optimizer_step_counter +=1

                if global_optimizer_step_counter % 20 == 0 : # Log every 20 optimizer steps
                    progress_bar_samples.set_postfix({
                        "OptStep": global_optimizer_step_counter,
                        "LossNow": f"{current_loss.item():.4f}",
                        "CE": f"{loss_ce.item():.4f}",
                        "KD": f"{loss_kd.item():.4f}",
                        "LR": f"{scheduler.get_last_lr()[0]:.2e}"
                    })
        
        avg_epoch_total_loss = epoch_total_loss_sum / num_samples_processed_in_epoch if num_samples_processed_in_epoch > 0 else 0
        avg_epoch_ce_loss = epoch_ce_loss_sum / num_samples_processed_in_epoch if num_samples_processed_in_epoch > 0 else 0
        avg_epoch_kd_loss = epoch_kd_loss_sum / num_samples_processed_in_epoch if num_samples_processed_in_epoch > 0 else 0
        print(f"--- Epoch {epoch+1}/{distill_num_epochs} Summary ---")
        print(f"Average Total Loss: {avg_epoch_total_loss:.4f}")
        print(f"Average CE Loss: {avg_epoch_ce_loss:.4f}")
        print(f"Average KD Loss: {avg_epoch_kd_loss:.4f}")
        print(f"Final LR for epoch: {scheduler.get_last_lr()[0]:.2e}")

        # Per-Epoch Save and Upload
        print(f"\n--- Saving and Uploading for Epoch {epoch+1}/{distill_num_epochs} ---")
        print(f"Saving adapter and tokenizer locally to '{local_adapter_save_dir}' for epoch {epoch+1}...")
        student_model.save_pretrained(local_adapter_save_dir)
        tokenizer.save_pretrained(local_adapter_save_dir)
        print(f"Local save for epoch {epoch+1} complete.")
        
        if output_hub_model_id and "YOUR_HF_USERNAME" not in output_hub_model_id:
            print(f"Uploading to Hugging Face Hub: {output_hub_model_id} for epoch {epoch+1}...")
            try:
                epoch_commit_msg = (
                    f"Epoch {epoch+1}/{distill_num_epochs} update ({experiment_tag}). "
                    f"AvgLoss={avg_epoch_total_loss:.4f}. LR={scheduler.get_last_lr()[0]:.2e}. "
                )
                # student_model.push_to_hub is for PeftModel. For full model, use HfApi or save_pretrained then upload.
                # Since student_model is a PeftModel, this is correct.
                student_model.push_to_hub(output_hub_model_id, commit_message=epoch_commit_msg, private=True) # or private=False
                tokenizer.push_to_hub(output_hub_model_id, commit_message=f"Upload tokenizer - Epoch {epoch+1} ({experiment_tag})", private=True) # or private=False
                print(f"Successfully uploaded adapter and tokenizer for epoch {epoch+1} to Hub.")
            except Exception as e:
                print(f"Error uploading to Hugging Face Hub for epoch {epoch+1}: {e}")
        else:
            print("Skipping Hugging Face Hub upload as output_hub_model_id is not set or contains placeholder.")


    print("QLoRA-assisted knowledge distillation training completed.")

if __name__ == '__main__':
    if not torch.cuda.is_available():
        print("錯誤：沒有可用的 CUDA 裝置。此腳本需要 CUDA。")
        exit()
    else:
        num_devices = torch.cuda.device_count()
        # <<< MODIFIED: 嚴格指定腳本目標設備為 cuda:1 >>>
        script_target_device_str = 'cuda:1' 
        
        print(f"可用的 CUDA 裝置數量: {num_devices}")
        for i in range(num_devices):
            print(f"  裝置 {i}: {torch.cuda.get_device_name(i)}")
        
        try:
            target_device_idx = int(script_target_device_str.split(':')[1])
        except (IndexError, ValueError):
            print(f"錯誤：腳本中 device 字串 '{script_target_device_str}' 格式不正確。應為 'cuda:X'。")
            exit()

        if target_device_idx >= num_devices:
            print(f"錯誤：目標裝置 {script_target_device_str} 不可用。可用的最高索引為 cuda:{num_devices-1 if num_devices > 0 else 'N/A'}。")
            print(f"請確保裝置 cuda:1 存在且可用。")
            exit()
        
        print(f"腳本將嘗試在設定的 device='{script_target_device_str}' ({torch.cuda.get_device_name(target_device_idx)}) 上運行。")
        
        try:
            run_qlora_distillation_refined_s2() # <<< 修改函數調用名
        except Exception as e:
            print(f"執行過程中發生嚴重錯誤: {e}")
            import traceback
            traceback.print_exc()