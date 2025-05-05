from transformers import LlamaTokenizer, LlamaForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType

MODEL_ID = "meta-llama/Llama-2-13b-hf"


# === 3. Quantization Config (BitsAndBytes) ===
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype="bfloat16"
)

# Load tokenizer and model
tokenizer = LlamaTokenizer.from_pretrained(MODEL_ID)
model = LlamaForCausalLM.from_pretrained(MODEL_ID, load_in_4bit=True, device_map="auto", quantization_config=quantization_config)

model.config.use_cache = False
model.config.pretraining_tp = 1

tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"