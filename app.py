import gradio as gr
import nltk
import pandas as pd
import os
from dotenv import load_dotenv

from translator import MarianTranslator, MarianFineTunedTranslator, TranslationAnalyzer

load_dotenv()
# Instantiate classes
pretrained_translator = MarianTranslator(os.getenv("HUGGINGFACE_TOKEN"))
finetuned_translator = MarianFineTunedTranslator(os.getenv("HUGGINGFACE_TOKEN"))
analyzer = TranslationAnalyzer()
print(f"NLTK Version {nltk.__version__}")
# Function to handle translation and analysis
def perform_translation(input_text, reference_text):
    pretrained_translation , pretrained_scores = pretrained_translator.translate_text(input_text, reference_text)
    finetuned_translation, finetuned_scores = finetuned_translator.translate_text(input_text, reference_text)
    pretrained_bias_analysis = analyzer.evaluate_translation(input_text, pretrained_translation)
    finetuned_bias_analysis = analyzer.evaluate_translation(input_text, finetuned_translation)
    final_verdict = analyzer.compare_translation(input_text, pretrained_translation, finetuned_translation)

    pretrained_eval_data = pd.DataFrame({
        "BLEU Score": [pretrained_scores[0]],
        "METEOR Score": [pretrained_scores[1]],
        "chrF Score": [pretrained_scores[2]],
        "BERTScore (F1)": [pretrained_scores[3]]
    })

    finetuned_eval_data = pd.DataFrame({
        "BLEU Score": [finetuned_scores[0]],
        "METEOR Score": [finetuned_scores[1]],
        "chrF Score": [finetuned_scores[2]],
        "BERTScore (F1)": [finetuned_scores[3]]
    })



    return pretrained_translation, pretrained_eval_data, pretrained_bias_analysis, finetuned_translation, finetuned_eval_data, finetuned_bias_analysis, final_verdict

# Function to handle training
def train_model(csv_file):
    total_steps = 10  # Adjust based on actual training steps
    progress = 0

    yield "Initializing training...", progress, None, None, None  # Initial UI update

    # Iterate over logs & progress from train_model
    for i, data in enumerate(pretrained_translator.train_model(
            csv_file, output_dir="./marian_trained", epochs=10,
            hub_repo="DIPEN-SEN/opus-mt-en-fr-fine-tuned", private_repo=True
    )):
        if isinstance(data, tuple) and len(data) == 2:
            log, progress_step = data
            progress = int((i / total_steps) * 100)  # Convert step count to percentage
            yield log, progress, None, None, None  # Update UI with log and progress

    # Retrieve plots after training completes
    training_loss, bleu_score, perplexity = (
        "./training_loss.png",
        "./bleu_score.png",
        "./perplexity.png"
    )

    yield "Training complete! ✅", 100, training_loss, bleu_score, perplexity  # Final UI update

# Initially empty DataFrame
pretrained_empty_df = pd.DataFrame(columns=["Score", "Value"])
# Initially empty DataFrame
finetuned_empty_df = pd.DataFrame(columns=["Score", "Value"])

# Build the Gradio Interface
# Ensure flagged data directory exists
flagged_data_path = "flagged_logs.csv"


# Function to handle flagging
def flag_translation(input_text, reference_text, pretrained_translation, pretrained_scores, pretrained_bias,
                     finetuned_translation, finetuned_scores, finetuned_bias, final_verdict):
    flagged_entry = pd.DataFrame(
        [[input_text, reference_text, pretrained_translation, pretrained_scores, pretrained_bias, finetuned_translation,
          finetuned_scores, finetuned_bias, final_verdict]],
        columns=["Input Text", "Reference Text", "Pretrained Translation", "Pretrained Scores", "Pretrained Bias",
                 "Finetuned Translation", "Finetuned Scores", "Finetuned Bias", "Final Verdict"]
    )

    if not os.path.exists(flagged_data_path):
        flagged_entry.to_csv(flagged_data_path, index=False)
    else:
        flagged_entry.to_csv(flagged_data_path, mode="a", header=False, index=False)

    #return "✅ Flagged successfully!"
    return gr.update(visible=True)

# Function to hide flag notification on translation
def hide_flag_notification(*args):
    return gr.update(visible=False)  # Hide pop-up notification


with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.HTML("<h1 style='text-align: center;'>🌍 AI-Powered Translation</h1>")

    with gr.Group():
        gr.Markdown("### 📝 Enter Text to Translate")
        with gr.Row():
            input_text = gr.Textbox(label="English", placeholder="Type here...", lines=2, scale=5)
            translate_button = gr.Button("🚀 Translate", scale=1)
        gr.Markdown("### 📝 Enter Reference Text to calculate Bleu score")
        with gr.Row():
            reference_text = gr.Textbox(label="English", placeholder="Type here...", lines=2, scale=5)
            # Added for flagging
            #flag_button = gr.Button("🚩 Flag Translation")
            with gr.Group():
                flag_button = gr.Button("🚩 Flag Translation")  # Flag Button
                notification = gr.Markdown("✅ **Flagged Successfully!**", visible=False)  # Hidden pop-up message

    gr.HTML("<hr style='border:1px solid #ddd; margin:20px 0;'>")
    with gr.Row():
        with gr.Group():
            gr.Markdown("## 🤖 Pretrained Model")
            pretrained_translation = gr.Textbox(label="French Translation", interactive=False)
            pretrained_evaluation_table = gr.DataFrame(value=pretrained_empty_df, interactive=False,
                                                       label="Evaluation Scores", headers=["Metric", "Score"])
            pretrained_bias = gr.Textbox(label="Gender Bias Analysis", interactive=False)

        with gr.Group():
            gr.Markdown("## 🔧 Finetuned Model")
            finetuned_translation = gr.Textbox(label="French Translation", interactive=False)
            finetuned_evaluation_table = gr.DataFrame(value=finetuned_empty_df, interactive=False,
                                                      label="Evaluation Scores", headers=["Metric", "Score"])
            finetuned_bias = gr.Textbox(label="Gender Bias Analysis", interactive=False)

    # New Section for Final Verdict
    with gr.Group():
        gr.Markdown("### 📝 Compare Translation")
        final_verdict = gr.Textbox(label="Analysis", interactive=False, lines=10, elem_id="final-verdict")



    # Connect translation function
    translate_button.click(
        perform_translation,
        inputs=[input_text, reference_text],
        outputs=[pretrained_translation, pretrained_evaluation_table, pretrained_bias, finetuned_translation,
                 finetuned_evaluation_table, finetuned_bias, final_verdict]
    ).then(
        fn=hide_flag_notification,  # Hide flagging pop-up after translation
        inputs=[],
        outputs=[notification]
    )

    # Connect flagging function (shows pop-up notification)
    flag_button.click(
        flag_translation,
        inputs=[input_text, reference_text, pretrained_translation, pretrained_evaluation_table, pretrained_bias,
                finetuned_translation, finetuned_evaluation_table, finetuned_bias, final_verdict],
        outputs=[notification]
    )

# Launch the app
demo.launch()