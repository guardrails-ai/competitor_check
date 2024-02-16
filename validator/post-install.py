import nltk
import spacy

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")


model = "en_core_web_trf"
if not spacy.util.is_package(model):
    print(
        f"Spacy model {model} not installed. "
        "Download should start now and take a few minutes."
    )
    spacy.cli.download(model)  # type: ignore