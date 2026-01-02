import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer, BitsAndBytesConfig
from peft import PeftModel
from sentence_transformers import SentenceTransformer, util


BASE_MODEL_ID = "mistralai/Mistral-7B-v0.1"

# We use real adapters found on Hugging Face.
EXPERTS_CONFIG = {
    "math": {
        "description": "Solves math problems.",
        # Real adapter trained on GSM8K (Math)
        "adapter_id": "vineetsharma/qlora-adapter-Mistral-7B-Instruct-v0.1-gsm8k", 
        "anchors": ["solve for x", "calculate derivative", "what is the sum", "geometry"]
    },
    "chat": {
        "description": "General conversation.",
        # Real adapter trained on high-quality chat data (Zephyr/Notus recipe)
        "adapter_id": "argilla/notus-7b-v1-lora-adapter",
        "anchors": ["tell me a story", "how are you", "write an email", "explain history"]
    },
    "code": {
        "description": "Programming and debugging.",
        # Using Notus again as it has strong coding capabilities compared to base
        "adapter_id": "argilla/notus-7b-v1-lora-adapter",
        "anchors": ["def function", "write python script", "fix bug", "console.log"]
    }
}

class SemanticRouter:
    def __init__(self, experts_config):
        print("[Router] Initializing Embedding Model...")
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.expert_names = []
        self.anchor_embeddings = []
        self.expert_indices = []
        
        for idx, (name, data) in enumerate(experts_config.items()):
            anchors = data['anchors']
            embeddings = self.encoder.encode(anchors)
            self.expert_names.append(name)
            self.anchor_embeddings.append(embeddings)
            self.expert_indices.extend([idx] * len(anchors))
            
        self.anchor_embeddings = np.vstack(self.anchor_embeddings)
        self.expert_indices = np.array(self.expert_indices)

    def route(self, query):
        query_embedding = self.encoder.encode(query)
        scores = util.cos_sim(query_embedding, self.anchor_embeddings)[0]
        best_match_idx = torch.argmax(scores).item()
        selected_expert_idx = self.expert_indices[best_match_idx]
        return self.expert_names[selected_expert_idx], scores[best_match_idx].item()


class MoELoRAManager:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # NOTE: We switched to the INSTRUCT base model.
        # This aligns better with the adapters and follows instructions much better.
        base_model_id = "mistralai/Mistral-7B-Instruct-v0.1"
        
        print(f"[System] Loading Base Model: {base_model_id}...")
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )

        self.tokenizer = AutoTokenizer.from_pretrained(base_model_id)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            quantization_config=bnb_config, 
            device_map="auto"
        )

        self.router = SemanticRouter(EXPERTS_CONFIG)
        self.loaded_adapters = set()

        print("[System] Loading Experts...")
        for name, data in EXPERTS_CONFIG.items():
            adapter_id = data['adapter_id']
            internal_name = f"{name}_expert"
            print(f"  -> Loading Expert: {name.upper()}...")
            try:
                self.model.load_adapter(adapter_id, adapter_name=internal_name)
                self.loaded_adapters.add(name)
            except Exception as e:
                print(f"  [WARNING] Failed to load {name}: {e}")

    def format_prompt(self, prompt):
        """
        Wraps the user input in the Mistral Instruction format.
        Without this, the model just 'continues' the text instead of answering.
        """
        return f"<s>[INST] {prompt} [/INST]"

    def generate(self, prompt, max_new_tokens=200):
        # 1. Route
        expert_name, confidence = self.router.route(prompt)
        print(f"\n--- USER: {prompt} ---")
        print(f"[Router] Selected Expert: '{expert_name.upper()}' (Conf: {confidence:.2f})")
        
        # 2. Swap Adapter
        if expert_name in self.loaded_adapters:
            internal_name = f"{expert_name}_expert"
            self.model.set_adapter(internal_name)
        else:
            self.model.disable_adapters()
        
        # 3. Apply Formatting (CRITICAL FIX)
        formatted_prompt = self.format_prompt(prompt)
        
        # 4. Generate with proper sampling
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.device)
        streamer = TextStreamer(self.tokenizer, skip_prompt=True)
        
        _ = self.model.generate(
            **inputs, 
            max_new_tokens=max_new_tokens,
            streamer=streamer,
            pad_token_id=self.tokenizer.eos_token_id,
            do_sample=True,      # Adds creativity so it doesn't loop
            temperature=0.7,     # Controls randomness
            top_p=0.9)

# --- EXECUTION ---
if __name__ == "__main__":
    try:
        # Initialize System
        moe_system = MoELoRAManager()
        
        # Test Loop
        test_prompts = [
            "Calculate the derivative of 2x^2 + 4x",
            "Write a Python function to sort a list",
            "Tell me a story about a dragon"
        ]
        
        for p in test_prompts:
            moe_system.generate(p)
            
    except Exception as e:
        print(f"\n[CRITICAL ERROR] {e}")
