import platform
import torch
from transformers import MarianMTModel, MarianTokenizer, GenerationConfig
from huggingface_hub import login

class MarianFineTunedTranslator:

    def __init__(self, access_token=None):

        self.access_token = access_token
        if self.access_token:
            self.hf_login()
        else:
            print("Access token not provided. Skipping login.")

        self.detect_device()
        self.load_translation_model()

    def detect_device(self):
        if torch.cuda.is_available():
            self.device = "cuda"
        elif platform.machine() == "arm64" and torch.backends.mps.is_available():
            self.device = "mps"  # For Apple Silicon (M1, M2, M3)
        else:
            self.device = "cpu"

    def hf_login(self):
        print(f"Logging in with access token...........")
        login(self.access_token)

    def load_translation_model(self):

        self.model_name = "p06pratibha/fine-tuned-opus-mt-en-fr"
        self.tokenizer = MarianTokenizer.from_pretrained(self.model_name)
        self.model = MarianMTModel.from_pretrained(
            self.model_name, torch_dtype=torch.float32
        ).to(self.device)

    def translate_text(self, text):
        inputs = self.tokenizer.encode(
            text, return_tensors="pt", max_length=512, truncation=True
        ).to(self.device)
        # Define generation config explicitly
        generation_config = GenerationConfig(
            max_length=512,
            num_beams=5,
            early_stopping=True
        )
        translated_tokens = self.model.generate(
            inputs,
            generation_config=generation_config  # Pass generation config
        )
        translated_text = self.tokenizer.decode(
            translated_tokens[0], skip_special_tokens=True
        )
        return translated_text