import re
from typing import Callable, Dict, List, Optional

import nltk
import requests
import spacy
from guardrails.logger import logger
from guardrails.validators import (
    FailResult,
    PassResult,
    ValidationResult,
    Validator,
    register_validator,
)


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
        on_fail: Optional[Callable] = None,
        api_endpoint: str = None,
        api_key: str = None,
    ):
        super().__init__(competitors=competitors, on_fail=on_fail)
        self._competitors = competitors
        model = "en_core_web_trf"

        if api_endpoint and api_key:
            self.nlp = self.query
        self.nlp = spacy.load(model)

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

    def perform_ner(self, text: str, nlp) -> List[str]:
        """Performs named entity recognition on text using a provided NLP
        model.

        Args:
            text (str): The text to perform named entity recognition on.
            nlp: The NLP model to use for entity recognition.

        Returns:
            entities: A list of entities found.
        """

        doc = nlp(text)
        entities = []
        for ent in doc.ents:
            entities.append(ent.text)
        return entities

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
        flagged_sentences = []
        filtered_sentences = []
        list_of_competitors_found = []

        for sentence in sentences:
            entities = self.exact_match(sentence, self._competitors)
            if entities:
                ner_entities = self.perform_ner(sentence, self.nlp)
                found_competitors = self.is_entity_in_list(ner_entities, entities)

                if found_competitors:
                    flagged_sentences.append((found_competitors, sentence))
                    list_of_competitors_found.append(found_competitors)
                    logger.debug(f"Found: {found_competitors} named in '{sentence}'")
                else:
                    filtered_sentences.append(sentence)

            else:
                filtered_sentences.append(sentence)

        filtered_output = " ".join(filtered_sentences)

        if len(flagged_sentences):
            return FailResult(
                error_message=(
                    f"Found the following competitors: {list_of_competitors_found}. "
                    "Please avoid naming those competitors next time"
                ),
                fix_value=filtered_output,
            )
        else:
            return PassResult()

    def query(self, query_str: str) -> list[str]:
        """Sends a request to the supplied API endpoint using a raw post request.

        Args:
            query_str (str): The string to query the 'en_core_web_trf' with.

        Returns:
            list[str]: The resulting output from the 'en_core_web_trf' model
        """
        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = str(
            {
                "inputs": query_str,
            }
        )
        response = requests.post(self.api_url, headers=headers, json=payload)
        return response.json()
