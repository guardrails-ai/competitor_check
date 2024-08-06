import json
import re
from typing import Any, Callable, Dict, List, Optional

import nltk
import spacy
from guardrails.logger import logger
from guardrails.validator_base import ErrorSpan
from guardrails.validators import (
    FailResult,
    PassResult,
    ValidationResult,
    Validator,
    register_validator,
)


@register_validator(
    name="guardrails/competitor_check", data_type="string", has_guardrails_endpoint=True
)
class CompetitorCheck(Validator):
    """Validates that LLM-generated text is not naming any competitors from a
    given list.

    In order to use this validator you need to provide an extensive list of the
    competitors you want to avoid naming including all common variations.

    Args:
        competitors (List[str]): List of competitors you want to avoid naming
    """

    def chunking_function(self, chunk: str):
        """
        Use a sentence tokenizer to split the chunk into sentences.

        Because using the tokenizer is expensive, we only use it if there
        is a period present in the chunk.
        """
        # using the sentence tokenizer is expensive
        # we check for a . to avoid wastefully calling the tokenizer
        if "." not in chunk:
            return []
        sentences = nltk.sent_tokenize(chunk)
        if len(sentences) == 0:
            return []
        if len(sentences) == 1:
            sentence = sentences[0].strip()
            # this can still fail if the last chunk ends on the . in an email address
            if sentence[-1] == ".":
                return [sentence, ""]
            else:
                return []

        # return the sentence
        # then the remaining chunks that aren't finished accumulating
        return [sentences[0], "".join(sentences[1:])]

    def __init__(
        self,
        competitors: List[str],
        on_fail: Optional[Callable] = None,
        **kwargs,
    ):
        super().__init__(
            competitors=competitors,
            on_fail=on_fail,
            **kwargs,
        )
        self._competitors = competitors
        model = "en_core_web_trf"
        if self.use_local:
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

    def perform_ner(self, text: str) -> List[str]:
        """Performs named entity recognition on text using a provided NLP
        model.

        Args:
            text (str): The text to perform named entity recognition on.

        Returns:
            entities: A list of entities found.
        """

        doc = self.nlp(text)
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
        error_spans: List[ErrorSpan] = []
        list_of_competitors_found = []
        start_ind = 0
        for sentence in sentences:
            entities = self.exact_match(sentence, self._competitors)
            if entities:
                ner_entities = self._inference(
                    {"text": sentence, "competitors": self._competitors}
                )
                if isinstance(ner_entities, str):
                    ner_entities = [ner_entities]
                found_competitors = self.is_entity_in_list(ner_entities, entities)
                if found_competitors:
                    flagged_sentences.append((found_competitors, sentence))
                    list_of_competitors_found.append(found_competitors)
                    logger.debug(f"Found: {found_competitors} named in '{sentence}'")
                else:
                    filtered_sentences.append(sentence)

            else:
                filtered_sentences.append(sentence)
            start_ind += len(sentence)

        filtered_output = " ".join(filtered_sentences)
        found_entities = []
        for tup in flagged_sentences:
            for entity in tup[0]:
                found_entities.append(entity)

        def find_all(a_str, sub):
            start = 0
            while True:
                start = a_str.find(sub, start)
                if start == -1:
                    return
                yield start
                start += len(sub)  # use start += 1 to find overlapping matches

        error_spans = []
        for entity in found_entities:
            starts = list(find_all(value, entity))
            for start in starts:
                error_spans.append(
                    ErrorSpan(
                        start=start,
                        end=start + len(entity),
                        reason=f"Competitor found: {value[start:start+len(entity)]}",
                    )
                )

        if len(flagged_sentences):
            return FailResult(
                error_message=(
                    f"Found the following competitors: {list_of_competitors_found}. "
                    "Please avoid naming those competitors next time"
                ),
                fix_value=filtered_output,
                error_spans=error_spans,
            )
        else:
            return PassResult()

    def _inference_local(self, model_input: Any) -> str:
        """Local inference method to detect and anonymize competitor names."""
        text = model_input["text"]
        competitors = model_input["competitors"]

        doc = self.nlp(text)
        anonymized_text = text
        for ent in doc.ents:
            if ent.text in competitors:
                anonymized_text = anonymized_text.replace(ent.text, "[COMPETITOR]")
        return anonymized_text

    def _inference_remote(self, model_input: Any) -> str:
        """Remote inference method for a hosted ML endpoint."""
        request_body = {
            "inputs": [
                {
                    "name": "text",
                    "shape": [1],
                    "data": [model_input["text"]],
                    "datatype": "BYTES"
                },
                {
                    "name": "competitors",
                    "shape": [len(model_input["competitors"])],
                    "data": model_input["competitors"],
                    "datatype": "BYTES"
                }
            ]
        }
        response = self._hub_inference_request(json.dumps(request_body), self.validation_endpoint)

        if not response or "outputs" not in response:
            raise ValueError("Invalid response from remote inference", response)

        outputs = response["outputs"][0]["data"][0]

        return outputs