from transformers import pipeline
_ = pipeline("zero-shot-classification", "facebook/bart-large-mnli")