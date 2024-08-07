import nltk
from typing import Any, Dict
import spacy
class InferlessPythonModel:

    def initialize(self):
        spacy.require_gpu()
        spacy.cli.download("en_core_web_trf")
        self.nlp = spacy.load("en_core_web_trf")
        nltk.download('punkt')
        
    def infer(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        text = inputs["text"]
        competitors = inputs["competitors"]

        doc = self.nlp(text)
        anonymized_text = text
        for ent in doc.ents:
            if ent.text in competitors:
                anonymized_text = anonymized_text.replace(ent.text, "[COMPETITOR]")
        return {"result": anonymized_text}
        
    def finalize(self):
        pass
    
    
