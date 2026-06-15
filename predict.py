from transformers import pipeline

print("Loading model...")

classifier = pipeline(
    task="text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    top_k=None
)

print("Model loaded!")


def detect_emotion(text):
    raw = classifier(text)
    emotions = raw[0]  # unwrap the outer list

    # sort by score, highest first
    emotions = sorted(emotions, key=lambda x: x["score"], reverse=True)

    # build a clean dictionary: {"joy": 0.85, "sadness": 0.05, ...}
    scores = {item["label"]: round(item["score"], 4) for item in emotions}

    top_emotion = emotions[0]["label"]

    return top_emotion, scores


# test it
top, scores = detect_emotion("I just got promoted at work!")

print("Top emotion:", top)
print("All scores:")
for emotion, score in scores.items():
    print(f"  {emotion}: {score}")