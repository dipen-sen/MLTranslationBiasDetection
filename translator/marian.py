import platform

import evaluate
import torch
from transformers import MarianMTModel, MarianTokenizer, DataCollatorForSeq2Seq, Seq2SeqTrainer, EarlyStoppingCallback, \
    GenerationConfig, Seq2SeqTrainingArguments
from huggingface_hub import HfApi, login
import pandas as pd
from datasets import Dataset

class MarianTranslator:

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

        self.model_name = "Helsinki-NLP/opus-mt-en-fr"
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

    def tokenize_function(self, examples):
        return self.tokenizer(
            examples["english"],
            text_target=examples["french"],
            padding="max_length",
            truncation=True,
        )

    def preprocess(self, dataset_path):
        yield "Converting CSV to Dataframe..."
        df = pd.read_csv(dataset_path)
        dataset = Dataset.from_pandas(df)
        yield "Tokenizing dataset..."
        tokenized_dataset = dataset.map(self.tokenize_function, batched=True)
        yield "Splitting dataset into train and test sets..."
        train_test_split = tokenized_dataset.train_test_split(test_size=0.1)
        yield "Preprocessing complete."
        return train_test_split["train"], train_test_split["test"]

    def train_model(self, dataset_path, output_dir="./marian_trained", epochs=10, hub_repo=None, private_repo=False,
                    resume_from_checkpoint=None):
        yield "Starting preprocessing..."
        for log in self.preprocess(dataset_path):
            yield log

        train_dataset, eval_dataset = self.preprocess(dataset_path)
        data_collator = DataCollatorForSeq2Seq(self.tokenizer, model=self.model)
        bleu_metric = evaluate.load("sacrebleu")

        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            decoded_preds = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
            decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
            decoded_labels = [[label] for label in decoded_labels]
            bleu = bleu_metric.compute(predictions=decoded_preds, references=decoded_labels)
            return {"bleu": bleu["score"]}

        training_args = Seq2SeqTrainingArguments(
            output_dir=output_dir,
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=3,
            learning_rate=3e-5,
            warmup_steps=500,
            weight_decay=0.01,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            gradient_accumulation_steps=4,
            num_train_epochs=epochs,
            lr_scheduler_type="cosine",
            logging_dir="./logs",
            predict_with_generate=True,
            fp16=False,
            resume_from_checkpoint=resume_from_checkpoint,
            load_best_model_at_end=True,
            metric_for_best_model="bleu",
            greater_is_better=True,
            save_on_each_node=True
        )

        if hub_repo:
            training_args.set_push_to_hub(model_id=hub_repo, private_repo=private_repo)

        trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )

        yield "Starting model training..."
        trainer.train()
        yield "Training complete. Evaluating model..."
        trainer.evaluate()
        yield "Evaluation complete. Model training pipeline finished."