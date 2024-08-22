from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import spacy
import os

app = FastAPI()
# Initialize the detoxify model once
env = os.environ.get("env", "dev")

if env == "prod":
    spacy.require_gpu()

class InferenceData(BaseModel):
    name: str
    shape: List[int]
    data: List
    datatype: str


class InputRequest(BaseModel):
    inputs: List[InferenceData]


class OutputResponse(BaseModel):
    modelname: str
    modelversion: str
    outputs: List[InferenceData]

class CompetitorCheck:
    model_name = "en_core_web_trf"
    nlp = spacy.load(model_name)

    def infer(text_vals, competitors):
        outputs = []
        for idx, text in enumerate(text_vals):
            doc = CompetitorCheck.nlp(text)

            located_competitors = []
            for ent in doc.ents:
                if ent.text in competitors:
                    located_competitors.append(ent.text)

            outputs.append(
                InferenceData(
                    name=f"result{idx}",
                    datatype="BYTES",
                    shape=[1],
                    data=[located_competitors],
                )
            )

        output_data = OutputResponse(
            modelname=CompetitorCheck.model_name, modelversion="1", outputs=outputs
        )

        return output_data.model_dump()

@app.post("/validate", response_model=OutputResponse)
async def competitor_check(input_request: InputRequest):
    competitors = []
    for inp in input_request.inputs:
        if inp.name == "text":
            text_vals = inp.data
        elif inp.name == "competitors":
            competitors = inp.data

    if text_vals is None or competitors is None:
        raise HTTPException(status_code=400, detail="Invalid input format")

    return CompetitorCheck.infer(text_vals, competitors)

# Sagemaker specific endpoints
@app.get("/ping")
async def healtchcheck():
    return {"status": "ok"}

@app.post("/invocations", response_model=OutputResponse)
async def competitor_check_sagemaker(input_request: InputRequest):
    return await competitor_check(input_request)
