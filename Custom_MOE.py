import torch
import torch.nn as nn 
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

class LoRARouter(nn.Module):
  def __init__(self, input_dim, num_experts):
    super().__init__()
    self.classifier = nn.Linear(input_dim, num_experts)
  
  def forward(self, hidden_states):
    pooled = hidden_states.max(dim=1).values
    logits = self.classifier(pooled)
    return torch.argmax(logits, dim=-1)

class CustomMoESystem:
  def __init__(self, base_model_name, adapters_dict):
    self.device = ('cuda' if torch.cuda.is_available() else 'cpu')
    # Base Model
    self.base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype = torch.float16,
        device_map = 'auto'
    )
    self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    self.adapter_names = list(adapter_dict.keys())
    for name, path in adapter_dict.items():
      print(f"Loading Expert : {name}")
    
    input_dim = self.base_model.config.hidden_size
    self.router = LoRARouter(input_dim, len(self.adapter_names)).to(self.device)
    self.router.eval() 
  
  def generate(self, prompt):
    inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

    with torch.no_grad():
      
