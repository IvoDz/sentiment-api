from typing import Union

from fastapi import FastAPI

from app.sentiment import get_sentiment

app = FastAPI()


@app.get("/")
def read_root():
    return {"Welcome": "Go to /sentiment/<text> to get sentiment scores of <text>"}

@app.get("/sentiment/{text}")
def sentiment(text: str):
    return get_sentiment(text)

