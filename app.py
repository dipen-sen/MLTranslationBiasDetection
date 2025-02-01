import gradio as gr
import time
#from translator.marian import MarianTranslator
#from translator.finetuned import MarianFineTunedTranslator
#from translator.analyzer import TranslationAnalyzer

from translator import MarianTranslator, MarianFineTunedTranslator, TranslationAnalyzer

# Instantiate classes
pretrained_translator = MarianTranslator("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")
finetuned_translator = MarianFineTunedTranslator("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")
analyzer = TranslationAnalyzer()

# Function to handle translation and analysis
def perform_translation(input_text):
    pretrained_translation = pretrained_translator.translate_text(input_text)
    finetuned_translation = finetuned_translator.translate_text(input_text)
    pretrained_bias_analysis = analyzer.evaluate_translation(input_text, pretrained_translation)
    finetuned_bias_analysis = analyzer.evaluate_translation(input_text, finetuned_translation)
    final_verdict = analyzer.compare_translation(input_text, pretrained_translation, finetuned_translation)
    return pretrained_translation, pretrained_bias_analysis, finetuned_translation, finetuned_bias_analysis, final_verdict

# Function to handle training
def train_model(csv_file):
    total_steps = 10  # Example: Adjust based on training epochs or dataset size
    progress = 0

    yield "Initializing training...", progress  # Show initial status

    for i, log in enumerate(pretrained_translator.train_model(
            csv_file, output_dir="./marian_trained", epochs=10,
            hub_repo="DIPEN-SEN/opus-mt-en-fr-finetuned", private_repo=True
    )):
        progress = int((i / total_steps) * 100)  # Convert step count to percentage
        yield log, progress  # Send log + updated progress to UI

    yield "Training complete! ✅", 100  # Mark completion


# Build the Gradio Interface
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.HTML("<h1 style='text-align: center;'>🌍 AI-Powered Translation</h1>")

    with gr.Group():
        gr.Markdown("### 📝 Enter Text to Translate")
        with gr.Row():
            input_text = gr.Textbox(label="English", placeholder="Type here...", lines=2, scale=5)
            translate_button = gr.Button("🚀 Translate", scale=1)

    gr.HTML("<hr style='border:1px solid #ddd; margin:20px 0;'>")

    with gr.Row():
        with gr.Group():
            gr.Markdown("## 🤖 Pretrained Model")
            pretrained_translation = gr.Textbox(label="French Translation", interactive=False)
            pretrained_bias = gr.Textbox(label="Gender Bias Analysis", interactive=False)

        with gr.Group():
            gr.Markdown("## 🔧 Finetuned Model")
            finetuned_translation = gr.Textbox(label="French Translation", interactive=False)
            finetuned_bias = gr.Textbox(label="Gender Bias Analysis", interactive=False)

    # New Section for Final Verdict
    with gr.Group():
        gr.Markdown("### 📝 Compare Translation")
        final_verdict = gr.Textbox(label="Analysis", interactive=False, lines=10, elem_id="final-verdict")

    translate_button.click(
        perform_translation,
        inputs=[input_text],
        outputs=[pretrained_translation, pretrained_bias, finetuned_translation, finetuned_bias, final_verdict]
    )




    gr.HTML("<hr style='border:1px solid #ddd; margin:20px 0;'>")

    # New Section for training model
    with gr.Row():  # Grouping CSV upload and train button in the same row
        gr.Markdown("### 📝 Upload CSV for Fine-Tuning")
        with gr.Column(scale=2):  # CSV upload and button on the left
            csv_upload = gr.File(label="Upload CSV", type="filepath")
        with gr.Column(scale=1):  # Train button on the right
            train_button = gr.Button("🚀 Train", scale=1)

    # Modify UI to include Progress Bar & Status
    progress_slider = gr.Slider(0, 100, value=0, label="Training Progress", interactive=False)
    status_text = gr.Textbox(label="Status", interactive=False)
    # Console logs (wide text box)
    training_logs = gr.Textbox(label="Training Console Logs", interactive=False, lines=10, elem_id="console-output")

    train_button.click(
        fn=train_model,
        inputs=[csv_upload],
        outputs=[training_logs, progress_slider]  # Now updating logs + progress bar
    )

# Launch the app
demo.launch()