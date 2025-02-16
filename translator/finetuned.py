import platform
import torch
import os
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from transformers import MarianMTModel, MarianTokenizer, GenerationConfig
from huggingface_hub import login
from helper import EvaluationHelper

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

        self.model_name = "DIPEN-SEN/opus-mt-en-fr-fine-tuned"
        #self.model_name = "Helsinki-NLP/opus-mt-en-fr"
        self.tokenizer = MarianTokenizer.from_pretrained(self.model_name)
        self.model = MarianMTModel.from_pretrained(
            self.model_name, torch_dtype=torch.float32
        ).to(self.device)

    def translate_text(self, text, reference_text=None):

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
        # Tokenize both translated and reference text
        if (reference_text):
            evaluation_helper = EvaluationHelper()
            evaluation_data = evaluation_helper.evaluate_translation(translated_text, reference_text)

        return translated_text, evaluation_data