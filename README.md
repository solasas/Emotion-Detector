# Emotion Detector from Text

A web app that detects emotions from text using a pre-trained transformer model. Type any sentence and the app returns the detected emotion with confidence scores for all 7 emotion categories.

![Python](https://img.shields.io/badge/Python-3.8+-blue) ![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow) ![Gradio](https://img.shields.io/badge/UI-Gradio-orange)

---

## Demo

**Single sentence mode** — type one sentence, get the top emotion + confidence bar chart.

**Multi sentence mode** — type a paragraph (sentences separated by `.`) and get per-sentence emotion breakdown.

---

## Emotions Detected

| Emotion | Emoji |
|---------|-------|
| Joy | 😄 |
| Anger | 😡 |
| Sadness | 😢 |
| Fear | 😨 |
| Surprise | 😲 |
| Disgust | 🤢 |
| Neutral | 😐 |

---

## Tech Stack

- **Python** — core language
- **HuggingFace Transformers** — loads and runs the pre-trained model
- **PyTorch** — the engine that runs the model's math under the hood
- **Gradio** — builds the web UI in pure Python

**Model:** [`j-hartmann/emotion-english-distilroberta-base`](https://huggingface.co/j-hartmann/emotion-english-distilroberta-base)

---

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/emotion-detector.git
cd emotion-detector
```

### 2. Create and activate a virtual environment
```bash
python3 -m venv venv
source venv/bin/activate       # Mac/Linux
venv\Scripts\activate          # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the app
```bash
python app.py
```

Open the URL shown in the terminal (usually `http://127.0.0.1:7860`) in your browser.

> The first run will download the model (~250MB). After that it loads from cache instantly.

---

## Project Structure

```
emotion-detector/
├── app.py           # Gradio UI + emotion detection logic
├── predict.py       # Standalone prediction script (for testing)
├── verify.py        # Verifies all dependencies are installed
├── requirements.txt # Python dependencies
└── README.md
```

---

## Concepts Covered

### Virtual Environments
Isolated Python environments that keep project dependencies separate from the system Python. Prevents version conflicts across projects.

### Pre-trained Models
Instead of training from scratch (which requires huge datasets and compute), we use a model already trained by researchers and hosted on the HuggingFace Hub. We just download and use it.

### HuggingFace Pipeline
A high-level abstraction that wraps the full inference process — tokenization (text → numbers), model forward pass, and output decoding (numbers → labels) — into a single function call.

### Tokenization
Models don't understand raw text. Tokenization breaks text into sub-word pieces (tokens) and converts them to numbers the model can process. This is why the model only works reliably on English — it was trained on English tokens.

### Confidence Scores
The model outputs a probability distribution across all 7 emotions. All scores sum to 1.0. The highest score is the predicted emotion, but the full distribution shows how certain (or uncertain) the model is.

### Output Formatting
Raw model output is a nested list of dicts. We unwrap, sort, and restructure it into a clean format the UI can display — a common post-processing pattern in ML pipelines.

### Gradio Interface
`gr.Interface` wires a Python function to a web UI. Inputs and outputs map directly to the function's arguments and return values. `gr.Label` renders a score dictionary as a bar chart automatically.

---

## Known Limitations

- **English only** — the model was trained on English text. Other languages return unreliable results (often "neutral").
- **Sentence splitting** — the multi-sentence mode splits on `.` only. Sentences ending in `?` or `!` are treated as one chunk.
- **Sarcasm** — the model struggles with sarcastic or ironic text (e.g., "Oh great, another Monday.").

---

## Possible Improvements

- Add language detection and warn users if input is not English
- Use `nltk.sent_tokenize` for smarter sentence splitting
- Compare with another model and show side-by-side results
- Add emoji display directly on the confidence bar chart