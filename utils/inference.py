from transformers import LlamaTokenizer, LlamaForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import torch

MODEL_ID = "meta-llama/Llama-2-13b-hf"
ADAPTER_PATH = "./katutubo-llm-alpaca-reddit-tuned/checkpoint-17952"


class KatutuboLLM:
    def __init__(self):
        self.tokenizer = self._load_tokenizer()
        self.model = self._load_model()
        self.model = self._load_adapter(self.model)
        self.model.eval()
        self.chat_history = []

        self.system_prompt = (
            "You are a friendly and helpful assistant who responds in Taglish. "
            "Keep your answers short, chill, and easy to understand — parang ka-chat lang."
        )

    def _load_tokenizer(self):
        tokenizer = LlamaTokenizer.from_pretrained(MODEL_ID)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        return tokenizer

    def _load_model(self):
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        return LlamaForCausalLM.from_pretrained(
            MODEL_ID,
            quantization_config=quant_config,
            device_map={"":0},
        )

    def _load_adapter(self, base_model):
        return PeftModel.from_pretrained(base_model, ADAPTER_PATH)

    #def _build_prompt(self, current_question, context=None):
    #    prompt = f"<<SYS>> {self.system_prompt} <</SYS>>\n"

    #    for turn in self.chat_history:
    #        prompt += f"question: {turn['user']}\nanswer: {turn['bot']}\n"

    #   prompt += f"question: {current_question}\nanswer: "
    #   return prompt

    def _build_prompt(self, current_question, context=None):
        max_tokens = 512
        system_prefix = f"[{self.system_prompt}]\n"
        
        # If context is provided, skip chat history
        if context:
            context_block = f"\nHere's the relevant information:\n{context}\n"
            current_block = f"\nquestion: {current_question}\nanswer: "
            prompt = system_prefix + context_block + current_block
        else:
            # No context: include as much chat history as fits
            chat_turns = []
            for turn in self.chat_history:
                chat_turns.append(f"\nquestion: {turn['user']}\nanswer: {turn['assistant']}")
            current_block = f"\nquestion: {current_question}\nanswer: "
            base_prompt = system_prefix + "".join(chat_turns) + current_block

            # Truncate if exceeds token limit
            while True:
                token_count = len(self.tokenizer(base_prompt, return_tensors="pt")["input_ids"][0])
                if token_count <= max_tokens or not chat_turns:
                    break
                chat_turns.pop(0)  # Remove oldest turn
                base_prompt = system_prefix + "".join(chat_turns) + current_block

            prompt = base_prompt

        return prompt


    def infer(self, instruction, history, context=None):
        self.chat_history = history
        prompt = self._build_prompt(instruction, context)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.7,
                top_k=50,
                top_p=0.9,
                repetition_penalty=1.2,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id
            )

        decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = decoded[len(prompt):].strip()

        return response

    def reset(self):
        """Clear conversation history."""
        self.chat_history = []

if __name__ == "__main__":
    bot = KatutuboLLM()
    print(bot.infer("Anong symptoms ng dengue?"))
    print(bot.infer("Delikado ba ito?"))
