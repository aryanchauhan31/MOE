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
    return {"text": f"<s>[INST] {row['input']} [/INST] {row['output']} </s>"}

dataset = dataset.map(format_row)

# 2. Quantization Config (A100 OPTIMIZED)
# We use bfloat16 for computation. This matches the A100's native tensor cores.
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,  # <--- CHANGED to bfloat16
)

# 3. Load Model
print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    torch_dtype=torch.bfloat16,             # <--- CHANGED to bfloat16
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

# 5. SFTConfig
sft_config = SFTConfig(
    output_dir="./results",
    dataset_text_field="text",
    # max_seq_length=512,
    num_train_epochs=1,
    per_device_train_batch_size=8,          # Increased to 8 (A100 can handle it)
    gradient_accumulation_steps=1,
    learning_rate=2e-4,
    logging_steps=10,
    save_strategy="no",
    max_steps=50,
    gradient_checkpointing=True,
    packing=False,
    
    # --- A100 SETTINGS ---
    fp16=False,      # Turn OFF standard float16
    bf16=True,       # Turn ON Brain Float 16 (Native to A100)
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
print("Starting training (A100 Mode)...")
trainer.train()

# 8. Save
print(f"Saving expert to {NEW_ADAPTER_NAME}...")
trainer.model.save_pretrained(NEW_ADAPTER_NAME)
print("✅ DONE! Expert saved.")
