import asyncio
import json
import os
import re
from typing import Callable, Dict, List, Optional

import nltk
import requests
import spacy
import validators
from guardrails.logger import logger
from guardrails.validators import (
    FailResult,
    PassResult,
    ValidationResult,
    Validator,
    register_validator,
)
from spacy_download import load_spacy


@register_validator(name="guardrails/competitor_check", data_type="string")
class CompetitorCheck(Validator):
    """Validates that LLM-generated text is not naming any competitors from a
    given list.

    In order to use this validator you need to provide an extensive list of the
    competitors you want to avoid naming including all common variations.

    Args:
        competitors (List[str]): List of competitors you want to avoid naming
        on_fail (Optional[Callable]): Callback function to be executed on validation failure
        api_endpoint (Optional[str]): API endpoint for external NER service
        api_key (Optional[str]): API key for external NER service
    """

    def __init__(
        self,
        competitors: List[str],
        api_endpoint: str = None,
        on_fail: Optional[Callable] = None,
    ):
        super().__init__(competitors=competitors, on_fail=on_fail)
        self._competitors = competitors
        self.model = "en_core_web_trf"
        self.api_endpoint = api_endpoint

        print("competitors")
        print(competitors)
        
        self.nlp = self._load_nlp_model(self.model)

    def _load_nlp_model(self, model: str) -> spacy.language.Language:
        """Loads either a local spaCy model or uses an external API endpoint for NER.

        Args:
            model (str): The model name to load
            api_endpoint (Optional[str]): If given an endpoint, returns a callable that
            sends a request to the given endpoint

        Returns:
            spacy.language.Language: The returned model or a callable that sends a
            request to the given API endpoint.
        """
        if self.api_endpoint:
            if validators.url(self.api_endpoint):
                return self.query
            else:
                logger.WARNING(
                    f"api_endpoint {self.api_endpoint} supplied to",
                    "CompetitorCheck Validator but the address was not valid",
                )

        else:
            return load_spacy(model)

    def exact_match(self, text: str, competitors: List[str]) -> List[str]:
        """Performs exact match to find competitors from a list in a given
        text.

        Args:
            text (str): The text to search for competitors.
            competitors (list): A list of competitor entities to match.

        Returns:
            list: A list of matched entities.
        """

        found_entities = []
        for entity in competitors:
            pattern = rf"\b{re.escape(entity)}\b"
            match = re.search(pattern.lower(), text.lower())
            if match:
                found_entities.append(entity)
        return found_entities

    def perform_ner(self, text: str) -> List[str]:
        """Performs named entity recognition on text using a provided NLP
        model.

        Args:
            text (str): The text to perform named entity recognition on.

        Returns:
            entities: A list of entities found.
        """

        doc = self.nlp(text)
        print(doc)
        entities = []
        # Running local model

        if hasattr(doc, "ents"):
            for ent in doc.ents:
                entities.append(ent.text)
            return entities
        else:
            # After parsing local model or recieving from api endpoint, we can just return a list
            return doc

    def is_entity_in_list(self, entities: List[str], competitors: List[str]) -> List:
        """Checks if any entity from a list is present in a given list of
        competitors.

        Args:
            entities (list): A list of entities to check
            competitors (list): A list of competitor names to match

        Returns:
            List: List of found competitors
        """

        found_competitors = []
        for entity in entities:
            for item in competitors:
                pattern = rf"\b{re.escape(item)}\b"
                match = re.search(pattern.lower(), entity.lower())
                if match:
                    found_competitors.append(item)
        return found_competitors

    def validate(self, value: str, metadata=Dict) -> ValidationResult:
        """Checks a text to find competitors' names in it.

        While running, store sentences naming competitors and generate a fixed output
        filtering out all flagged sentences.

        Args:
            value (str): The value to be validated.
            metadata (Dict, optional): Additional metadata. Defaults to empty dict.

        Returns:
            ValidationResult: The validation result.
        """

        sentences = nltk.sent_tokenize(value)
        last_sentence = sentences[-1]
        if last_sentence.endswith((".", "?", "!")):
            flagged_sentences = []
            filtered_sentences = []
            list_of_competitors_found = []

            # Get the last word, and check if its a competitor. Because we are streaming
            # last_word = sentences[-1].split(" ")[-1]
            entities = self.exact_match(last_sentence, self._competitors)
            print("entities")
            print(entities)
            if entities:
                ner_entities = self.perform_ner(last_sentence)
                found_competitors = self.is_entity_in_list(ner_entities, entities)
                print('found_competitors')
                print(found_competitors)
                if found_competitors:
                    flagged_sentences.append((found_competitors, last_sentence))
                    list_of_competitors_found += found_competitors
                    logger.debug(f"Found: {found_competitors} named in '{last_sentence}'")
                else:
                    filtered_sentences.append(last_sentence)

            else:
                filtered_sentences.append(last_sentence)
            filtered_output = " ".join(filtered_sentences)

            if len(flagged_sentences):
                list_of_competitors_found = list(set(list_of_competitors_found))
                return FailResult(
                    fix_value=filtered_output,
                    error_message=json.dumps(
                        {
                            "match_string": list_of_competitors_found,
                            "violation": "CompetitorCheck",
                            "error_msg": "A competitor was mentioned.",
                        }
                    ),
                )
            return PassResult()
        return PassResult()

    def query(self, query_str: str) -> list[str]:
        """Sends a request to the supplied API endpoint using a raw post request.

        Args:
            query_str (str): The string to query the 'en_core_web_trf' with.

        Returns:
            list[str]: The resulting output from the 'en_core_web_trf' model
        """
        headers = {"Authorization": f"Bearer {os.environ['HF_API_KEY']}"}

        def query(payload):
            response = requests.post(self.api_endpoint, headers=headers, json=payload)
            return response.json()

        return query(str({"inputs": query_str}))
