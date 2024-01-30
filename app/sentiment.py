import logging
import os

import numpy as np
from scipy.special import softmax
from transformers import (AutoConfig, AutoModelForSequenceClassification,
                          AutoTokenizer, TFAutoModelForSequenceClassification)

from app.constants import MODEL_DIRECTORY, MODEL

logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)

def preprocess(text):
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)

def get_sentiment(text):
    if not os.path.exists(MODEL_DIRECTORY):
        model = AutoModelForSequenceClassification.from_pretrained(MODEL)
        model.save_pretrained(MODEL)
        tokenizer = AutoTokenizer.from_pretrained(MODEL)
        
    else:
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIRECTORY)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIRECTORY)
    
    text = preprocess(text)
    encoded_input = tokenizer(text, return_tensors='pt')
    output = model(**encoded_input)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)

    return {
        "negative": float(scores[0]),
        "neutral": float(scores[1]),
        "positive": float(scores[2]),
    }
