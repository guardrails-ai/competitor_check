from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Union
import spacy
import nltk

app = FastAPI()

# Initialize the SpaCy model with GPU support and download necessary NLTK data
spacy.cli.download("en_core_web_trf")
spacy_model_name = "en_core_web_trf"
nlp = spacy.load(spacy_model_name)
nltk.download('punkt')

class InferenceData(BaseModel):
    name: str
    shape: List[int]
    data: Union[List[str], List[float]]
    datatype: str

class InputRequest(BaseModel):
    inputs: List[InferenceData]

class OutputResponse(BaseModel):
    modelname: str
    modelversion: str
    outputs: List[InferenceData]

@app.post("/validate", response_model=OutputResponse)
async def check_competitors(input_request: InputRequest):
    text = None
    competitors = None
    
    for inp in input_request.inputs:
        if inp.name == "text":
            text = inp.data[0]
        elif inp.name == "competitors":
            competitors = inp.data
    
    if text is None or competitors is None:
        raise HTTPException(status_code=400, detail="Invalid input format")
    
    # Perform NER and anonymization
    doc = nlp(text)
    anonymized_text = text
    for ent in doc.ents:
        if ent.text in competitors:
            anonymized_text = anonymized_text.replace(ent.text, "[COMPETITOR]")
    
    output_data = OutputResponse(
        modelname="CompetitorCheckModel",
        modelversion="1",
        outputs=[
            InferenceData(
                name="result",
                datatype="BYTES",
                shape=[1],
                data=[anonymized_text]
            )
        ]
    )
    
    print(f"Output data: {output_data}")
    return output_data

# Run the app with uvicorn
# Save this script as app.py and run with: uvicorn app:app --reload
