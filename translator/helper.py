import nltk
import sacrebleu
from nltk.translate.meteor_score import meteor_score
from bert_score import score
import math
from fractions import Fraction
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

class EvaluationHelper:
    def __init__(self):
        """Initialize Helper class and download necessary resources."""
        nltk.download('punkt', quiet=True)
        # Download nltk resources
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)

    def calculate_bleu(self, reference_text, candidate_text):
        """Calculate the BLEU score between reference and candidate sentences in French."""

        # Tokenize input sentences
        reference_tokens = [[nltk.word_tokenize(reference_text.lower(), language="french")]]  # ✅ Correct format
        candidate_tokens = nltk.word_tokenize(candidate_text.lower(), language="french")

        # Compute BLEU score with better smoothing
        smoothing = SmoothingFunction().method4
        bleu_score = sentence_bleu(reference_tokens, candidate_tokens, smoothing_function=smoothing)

        return bleu_score

    def bleu_score(self, reference_text, translated_text):
        biased_tokens = [nltk.word_tokenize(sent.lower(), language="french") for sent in translated_text]
        mitigated_tokens = [nltk.word_tokenize(sent.lower(), language="french") for sent in reference_text]
        # Compute BLEU score
        smoothing = SmoothingFunction().method1  # Smoothing to handle short sentences
        bleu_score = sentence_bleu(biased_tokens, mitigated_tokens[0], smoothing_function=smoothing)
        print(f"BLEU Score: {bleu_score}")
        return bleu_score

    def evaluate_translation(self, candidate, reference):
        """
        Evaluates a candidate translation against a reference translation
        using BLEU, METEOR, chrF, and BERTScore.
        """

        # BLEU Score
        bleu = sacrebleu.sentence_bleu(candidate, [reference]).score

        # METEOR Score (using nltk)
        meteor = meteor_score([reference.split()], candidate.split()) * 100  # Scale to 100

        # chrF Score (uses character-level n-grams)
        chrf = sacrebleu.CHRF().sentence_score(candidate, [reference]).score

        # BERTScore (semantic similarity)
        P, R, F1 = score([candidate], [reference], lang="fr")  # 'fr' for French
        bert_f1 = F1.item() * 100  # Convert tensor to float and scale

        # Print Results
        print(f"BLEU Score: {bleu:.2f}")
        print(f"METEOR Score: {meteor:.2f}")
        print(f"chrF Score: {chrf:.2f}")
        print(f"BERTScore (F1): {bert_f1:.2f}")

        return bleu , meteor, chrf, bert_f1