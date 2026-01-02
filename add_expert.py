import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

# --- CONFIGURATION ---
BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.1"
NEW_ADAPTER_NAME = "my-medical-expert"

# 1. Load & Format Dataset
print("Loading and formatting dataset...")
dataset = load_dataset("lavita/ChatDoctor-HealthCareMagic-100k", split="train[:500]")

def format_row(row):
    # Standard Mistral format: <s>[INST] Question [/INST] Answer </s>
    prompt = f"<s>[INST] {row['input']} [/INST] {row['output']} </s>"
    return {"text": prompt}

dataset = dataset.map(format_row)

# 2. Quantization Config (FORCE FLOAT16)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,  # <--- SAFEGUARD 1: Strict Float16
)

# 3. Load Model
print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    torch_dtype=torch.float16,             # <--- SAFEGUARD 2: Force model dtype
    device_map="auto"
)
model.config.use_cache = False
model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# 4. LoRA Config
peft_config = LoraConfig(
    r=16,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "v_proj"]
)

# 5. SFTConfig (The Modern Setup)
sft_config = SFTConfig(
    output_dir="./results",
    dataset_text_field="text",
    # max_seq_length=512,
    num_train_epochs=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=1,
    learning_rate=2e-4,
    logging_steps=10,
    save_strategy="no",
    max_steps=50,
    gradient_checkpointing=True,
    packing=False,
    
    # --- SAFEGUARD 3: STRICT DTYPE SETTINGS ---
    fp16=True,       # Enable Standard Float16
    bf16=False,      # EXPLICITLY Disable BFloat16 (Crash prevention)
)

# 6. Initialize Trainer
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    # tokenizer=tokenizer,
    args=sft_config,
    peft_config=peft_config,
)

# 7. Train
print("Starting training (T4 Compatible)...")
trainer.train()

# 8. Save
print(f"Saving expert to {NEW_ADAPTER_NAME}...")
trainer.model.save_pretrained(NEW_ADAPTER_NAME)
print("✅ DONE! Expert saved.")
