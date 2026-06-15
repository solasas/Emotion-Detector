import gradio as gr
from transformers import pipeline

classifier = pipeline(
    task="text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    top_k=None
)


EMOJI_MAP = {
    "joy": "joy 😄",
    "anger": "anger 😡",
    "sadness": "sadness 😢",
    "fear": "fear 😨",
    "surprise": "surprise 😲",
    "disgust": "disgust 🤢",
    "neutral": "neutral 😐"
}


def detect_emotion(text):
    if not text.strip():
        return "Please enter some text.", {}

    raw = classifier(text)
    emotions = raw[0]
    emotions = sorted(emotions, key=lambda x: x["score"], reverse=True)
    scores = {item["label"]: round(item["score"], 4) for item in emotions}
    top_emotion = EMOJI_MAP[emotions[0]["label"]]
    return top_emotion, scores


def detect_multi(text):
    sentences = [s.strip() for s in text.split(".") if s.strip()]
    if not sentences:
        return "Please enter some text.", {}

    results = []
    for sentence in sentences:
        top, _ = detect_emotion(sentence)
        results.append(f"{sentence}  →  {top}")

    return "\n".join(results)


single = gr.Interface(
    fn=detect_emotion,
    inputs=gr.Textbox(placeholder="Type a sentence and press Submit..."),
    outputs=[
        gr.Textbox(label="Top Emotion"),
        gr.Label(label="Confidence Scores")
    ],
    title="Single Sentence",
    examples=[
        ["I just got promoted at work!"],
        ["I miss my dog so much."],
        ["I can't believe they did that to me!"],
        ["Oh great, another Monday."]
    ]
)

multi = gr.Interface(
    fn=detect_multi,
    inputs=gr.Textbox(placeholder="Type multiple sentences separated by periods...", lines=4),
    outputs=gr.Textbox(label="Emotion per Sentence"),
    title="Multi Sentence",
    examples=[
        ["I woke up happy. Then I saw the news. Now I feel nothing."]
    ]
)

demo = gr.TabbedInterface(
    [single, multi],
    ["Single Sentence", "Multi Sentence"]
)

demo.launch()