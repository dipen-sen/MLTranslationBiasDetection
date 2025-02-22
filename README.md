# 🌍 AI-Powered Translation App

## 📌 Project Overview

This project is a **Neural Machine Translation (NMT) application** that translates English text to French using the **Helsinki-NLP/opus-mt-en-fr** model. The app also provides **fine-tuning capabilities** to improve translation quality and includes **bias analysis** to compare translations.

The user interface is built with **Gradio**, making it easy to interact with the translation models and visualize training progress.

---

## 🚀 Features

✅ **English to French Translation** using a **pretrained** and **fine-tuned** model\
✅ **Translation Quality Analysis** with BLEU score evaluation\
✅ **Bias Analysis** for comparing gender biases in translations\
✅ **Fine-Tuning Support** to train the model on custom datasets\
✅ **Live Training Logs & Progress Bar** in the Gradio UI\
✅ **Deployable Locally** on macOS (M1, M2, M3) and Windows

---

## 📂 Project Structure

```
📦 ai-powered-translation
├── 📁 translator
│   ├── marian.py               # Pretrained Translator Class
│   ├── finetuned.py            # Fine-Tuned Translator Class
│   ├── analyzer.py             # Translation Analysis Class
├── app.py                      # Gradio App Interface
├── requirements.txt            # Python Dependencies
├── README.md                   # Project Documentation
```

---

## 🛠 Installation & Setup

### **1️⃣ Install Dependencies**

Ensure you have **Python 3.8+** installed, then run:

```bash
pip install -r requirements.txt
```

### **2️⃣ Run the Gradio App**

```bash
python app.py
```

This will launch a local **Gradio UI** where you can input text and interact with the translator.

---

## 🔧 Usage Guide

### **🔹 Translate Text**

1. Enter text in English.
2. Click "🚀 Translate" to get:
   - **Pretrained Translation**
   - **Fine-Tuned Translation**
   - **Bias Analysis**
   - **Comparison of Both Translations**

### **🔹 Train the Model**

1. Upload a **CSV file** containing English-French text pairs.
2. Click "🚀 Train" to fine-tune the model.
3. View **live logs** and a **progress bar** while training.

---

## 🖥️ Model Training Details

The training pipeline uses:

- **Hugging Face Transformers** for model fine-tuning
- **BLEU Score Evaluation** for quality assessment
- **Early Stopping** for optimized performance
- **Apple MPS (if available)** for faster training on macOS

---

## 📌 Requirements

- Python 3.8+
- PyTorch
- Hugging Face Transformers
- Gradio
- Datasets

Install all dependencies using:

```bash
pip install -r requirements.txt
```

---

## 🤖 Future Improvements

- ✅ Add support for more languages (e.g., Spanish, German)
- ✅ Improve bias detection with advanced NLP techniques
- ✅ Deploy as a web-based API for wider accessibility

---

## 📝 License

This project is open-source under the **MIT License**.

---

### ✨ Developed by : 

