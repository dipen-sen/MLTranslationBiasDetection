import platform
import evaluate
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import MarianMTModel, MarianTokenizer, DataCollatorForSeq2Seq, Seq2SeqTrainer, EarlyStoppingCallback, \
    GenerationConfig, Seq2SeqTrainingArguments, AutoTokenizer
from huggingface_hub import HfApi, login
import pandas as pd
from datasets import Dataset
from sklearn.metrics import confusion_matrix
import itertools
import math
from translator import EvaluationHelper

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

    def translate_text(self, text, reference_text=None):

        inputs = self.tokenizer.encode(
            text, return_tensors="pt", max_length=512, truncation=True
        ).to(self.device)
        generation_config = GenerationConfig(
            max_length=512,
            num_beams=5,
            early_stopping=True
        )
        translated_tokens = self.model.generate(
            inputs,
            generation_config=generation_config
        )
        translated_text = self.tokenizer.decode(
            translated_tokens[0], skip_special_tokens=True
        )

        # Tokenize both translated and reference text
        if(reference_text):
            evaluation_helper = EvaluationHelper()
            evaluation_data = evaluation_helper.evaluate_translation(translated_text, reference_text)
        return translated_text , evaluation_data

    def tokenize_function(self, examples):
        return self.tokenizer(
            examples["english"],
            text_target=examples["french"],
            padding="max_length",
            truncation=True,
        )

    def preprocess(self, dataset_path):
        df = pd.read_csv(dataset_path)
        dataset = Dataset.from_pandas(df)
        tokenized_dataset = dataset.map(self.tokenize_function, batched=True)
        train_test_split = tokenized_dataset.train_test_split(test_size=0.1)
        return train_test_split["train"], train_test_split["test"]

    def plot_confusion_matrix(self, y_true, y_pred, title):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=set(y_true), yticklabels=set(y_true))
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title(title)
        plt.savefig(f"{title.replace(' ', '_').lower()}.png")
        plt.close()

    def compute_perplexity(self, loss):
        return math.exp(loss) if loss < 100 else float('inf')

    def train_model(self, dataset_path, output_dir="./marian_trained", epochs=10, hub_repo=None, private_repo=False,
                    resume_from_checkpoint=None):
        train_dataset, eval_dataset = self.preprocess(dataset_path)
        data_collator = DataCollatorForSeq2Seq(self.tokenizer, model=self.model)
        bleu_metric = evaluate.load("sacrebleu")

        losses, bleu_scores, perplexities = [], [], []

        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            decoded_preds = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
            decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
            decoded_labels = [[label] for label in decoded_labels]
            bleu = bleu_metric.compute(predictions=decoded_preds, references=decoded_labels)
            return {"bleu": bleu["score"]}

        training_args = Seq2SeqTrainingArguments(
            output_dir=output_dir,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=3,
            learning_rate=3e-5,
            warmup_steps=500,
            weight_decay=0.01,
            #per_device_train_batch_size=4,
            #per_device_eval_batch_size=4,
            gradient_accumulation_steps=4,
            num_train_epochs=epochs,
            lr_scheduler_type="cosine",
            logging_dir="./logs",
            predict_with_generate=True,
            fp16=False,
            resume_from_checkpoint=resume_from_checkpoint,
            load_best_model_at_end=True,
            metric_for_best_model="bleu",
            greater_is_better=True
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

        train_result = trainer.train()
        print(train_result)
        self.tokenizer.save_pretrained(training_args.output_dir)
        #if os.path.isdir(training_args.output_dir):
        #    checkpoints = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
        #    if checkpoints:
        #        last_checkpoint = os.path.join(output_dir, sorted(checkpoints, key=lambda x: int(x.split('-')[-1]))[-1])
        #        print(f"Last checkpoint found: {last_checkpoint}")
        #        tokenizer = AutoTokenizer.from_pretrained(last_checkpoint)
        #        tokenizer.save_pretrained(training_args.output_dir)
        #    else:
        #        last_checkpoint = None
        #        print("No checkpoint found.")
        #else:
        #    last_checkpoint = None
        #    print("Output directory does not exist.")





        trainer.push_to_hub()
        #training_logs = trainer.train()
        #for log in training_logs.metrics:
        #    loss = log.get("loss", None)
        #    bleu = log.get("eval_bleu", None)
        #    if loss is not None:
        #        losses.append(loss)
        #        perplexities.append(self.compute_perplexity(loss))
        #    if bleu is not None:
        #        bleu_scores.append(bleu)

        yield "Training complete. Evaluating model..."
        evaluation_result = trainer.evaluate()

        train_loss = train_result.training_loss
        perplexity = torch.exp(torch.tensor(train_loss)).item()
        epochs_range = range(epochs)

        training_loss_values = [log["loss"] for log in trainer.state.log_history if "loss" in log]
        bleu_scores = [log["eval_bleu"] for log in trainer.state.log_history if "eval_bleu" in log]

        # Generate plots
        plt.figure()
        plt.plot(losses, label='Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss Over Time')
        plt.legend()
        plt.savefig('training_loss.png')
        plt.close()

        plt.figure()
        plt.plot(bleu_scores, label='BLEU Score')
        plt.xlabel('Epoch')
        plt.ylabel('BLEU Score')
        plt.title('BLEU Score Over Time')
        plt.legend()
        plt.savefig('bleu_score.png')
        plt.close()

        plt.figure()
        plt.plot(perplexities, label='Perplexity')
        plt.xlabel('Epoch')
        plt.ylabel('Perplexity')
        plt.title('Perplexity Over Time')
        plt.legend()
        plt.savefig('perplexity.png')
        plt.close()

        yield "Evaluation complete. Model training pipeline finished."
        return "training_loss.png", "bleu_score.png", "perplexity.png"
