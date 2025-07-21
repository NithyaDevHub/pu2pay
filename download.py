#huggingface-cli download mistralai/Mistral-Small-3.1-24B-Base-2503 --local-dir ./downloaded_model --resume-download --repo-type=model

# download_qwen.py
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_id = "Qwen/Qwen2.5-32B-Instruct"
save_path = "./qwen2.5-32b-instruct"

# Load model (auto device placement, mixed precision)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Save to local directory
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
print(f"✅ Saved model and tokenizer to {save_path}")
