import json
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from gliner import GLiNER
from guardrails.validator_base import ErrorSpan
from guardrails.validators import (
    FailResult,
    PassResult,
    ValidationResult,
    Validator,
    register_validator,
)
from sentence_splitter import SentenceSplitter


sentence_splitter_supported_languages = {
    "cs": "Czech",
    "da": "Danish",
    "nl": "Dutch",
    "en": "English",
    "fi": "Finnish",
    "fr": "French",
    "de": "German",
    "el": "Greek",
    "hu": "Hungarian",
    "is": "Icelandic(",
    "it": "Italian",
    "lv": "Latvian",
    "lt": "Lithuanian",
    "no": "Norwegian(Bokmål)",
    "pl": "Polish",
    "pt": "Portuguese",
    "ro": "Romanian",
    "ru": "Russian",
    "sk": "Slovak",
    "sl": "Slovene",
    "es": "Spanish",
    "sv": "Swedish",
    "tr": "Turkish",
}


@dataclass
class NERDetection:
    name: str
    entity_type: str
    start: int
    end: int
    probability: float


@register_validator(
    name="guardrails/competitor_check", data_type="string", has_guardrails_endpoint=True
)
class CompetitorCheck(Validator):
    """Validates that LLM-generated text is not naming any competitors from a
    given list.

    In order to use this validator you need to provide an extensive list of the
    competitors you want to avoid naming including all common variations.

    Args:
        competitors: (list[str]): A list of competitors to be matched in the output.
        Variations on spelling are handled by the ML model, but the precheck
        (if run) is case-insensitive.  It is advisable to use the most common
        capitalization: i.e., "US Postal Service" instead of "us Postal sErViCe".

        enable_direct_match_prefilter (bool): Defaults to False. If true, will
        perform a case-insensitive search across the text for each of the
        competitors before attempting to use the ML model.  For example, the text,
        "It's a lovely day." does not have the substring 'Guardrails', so there's
        no need to perform a model-based check.  This results in greatly increased
        speed in the case where a competitor is mentioned, but comes at the cost
        of a higher false negative rate.  This value should be set to false if a
        competitor has multiple nuanced spellings or variations in spacing, for
        example: "USPS" and "US Postal Service" and "United States Postal Service" or
        if the competitor is an extremely common term like 'Apple'.

        named_entity_types: (list[str]): A list of competitor 'types' to be detected.
        By default, this is simply ['organization', 'brand', 'company'].  If desired,
        it is possible to specify, for example, ['product', 'location', 'group',] or
        ['bank',]. These will change the sensitivity and sensitibilities of the model
        and may require commensurate changes in the threshold.

    """

    # This is only used to filter out noise. The threshold defined in the init is what
    # determines the actual pass/fail.
    DETECTION_THRESHOLD = 0.01

    def __init__(
            self,
            competitors: List[str],
            threshold: float = 0.5,
            enable_direct_match_prefilter: bool = False,
            sentence_splitter_language: str = "en",
            competitor_types: Optional[List[str]] = None,  # Default: ["organization",].
            on_fail: Optional[Callable] = None,
            **kwargs,
    ):
        super().__init__(
            competitors=competitors,
            threshold=threshold,
            on_fail=on_fail,
            **kwargs,
        )

        if competitor_types is None:
            competitor_types = ["organization",]

        if sentence_splitter_language not in sentence_splitter_supported_languages:
            valid_languages = ", ".join([
                f"'{k}': {v}" for k, v in sentence_splitter_supported_languages.items()
            ])
            raise ValueError(f"The specified language '{sentence_splitter_language}' is"
                             f"not supported.  Valid languages are {valid_languages}")
        if sentence_splitter_language != "en":
            print("WARNING: The primary model is designed for English, this may not "
                  "behave as expected.")

        if not (0.0 < threshold < 1.0):
            raise ValueError(f"Got an unexpected value for threshold: {threshold}."
                             f"A threshold of <0 will always trip. A threshold >1 will"
                             f"never trip.")

        self.threshold = threshold
        self.competitors = competitors
        self.competitor_types = competitor_types
        self.prefilter = enable_direct_match_prefilter
        self.sentence_splitter = SentenceSplitter(language=sentence_splitter_language)
        self.model = GLiNER.from_pretrained("urchade/gliner_medium-v2.1")

    def _chunking_function(self, chunk: str) -> List[str]:
        """
        Use a sentence tokenizer to split the chunk into sentences.

        Because using the tokenizer is expensive, we only use it if there
        is a period present in the chunk.
        """
        # Rough heuristic:
        if "." not in chunk:
            return []
        sentences = self.sentence_splitter.split(text=chunk)
        if len(sentences) == 0:
            return []
        if len(sentences) == 1:
            sentence = sentences[0].strip()
            # this can still fail if the last chunk ends on the . in an email address
            if sentence[-1] == ".":
                # We only found one sentence, but it's complete:
                return [sentence, ""]
            else:
                # We've obtained a partial sentence and it's not over, so wait for more:
                return []

        # Pull the sentence we were able to find off of the chunk and return the rest.
        # Do NOT just " ".join because that doesn't preserve spacing.
        return [sentences[0], chunk.lstrip(sentences[0])]

    @staticmethod
    def exact_match(texts: list[str], competitors: List[str]) -> List[bool]:
        """Performs exact match to find competitors from a list in a given
        text.

        Args:
            texts (list[str]): An array of sentences to check for competitors.
            competitors (list): A list of competitor entities to match.

        Returns:
            A list of booleans, true when a sentence may contain a competitor.
        """
        competitor_regex = re.compile(
            r"|".join([re.escape(c.lower()) for c in competitors])
        )
        found_entities = []
        for t in texts:
            match = re.search(competitor_regex, t.lower())
            found_entities.append(bool(match))
        return found_entities

    def validate(
            self,
            value: Union[List[str], str],
            metadata: Optional[dict] = None
    ) -> ValidationResult:
        """Checks a text to find competitors' names in it.

        While running, store sentences naming competitors and generate a fixed output
        filtering out all flagged sentences.

        Args:
            value (str|list[str]): The value to be validated.
            metadata (Dict, optional): Additional metadata. Defaults to empty dict.
            'competitor_types', 'competitors', and 'threshold' can be overridden.

        Returns:
            ValidationResult: The validation result.
        """
        if metadata is None:
            metadata = dict()

        if isinstance(value, str):
            single_string = True
            sentences = [value,]
        else:
            single_string = False
            sentences = value

        if self.prefilter:
            candidate_competitors = list()
            potential_matches = self.exact_match(sentences, self.competitors)
            for needs_checking, sentence in zip(potential_matches, sentences):
                if not needs_checking:
                    candidate_competitors.append([])
                else:
                    candidate_competitors.append(self._inference({
                        "texts": [sentence,],
                        "competitor_types": metadata.get(
                            "competitor_types",
                            self.competitor_types
                        )
                    })[0])
        else:
            candidate_competitors = self._inference({
                "texts": sentences,
                "competitor_types": metadata.get(
                    "competitor_types",
                    self.competitor_types
                )
            })

        detected_competitors, error_spans = CompetitorCheck.compute_error_spans(
            sentences,
            competitors=metadata.get("competitors", self.competitors),
            detected_competitors=candidate_competitors,
            threshold=metadata.get("threshold", self.threshold),
        )

        # Error spans is a list of lists.  If any is NOT empty, then we have an error.
        if any([e for e in error_spans]):
            filtered_output = self.compute_filtered_output(sentences, error_spans)
            competitors_found = ", ".join(detected_competitors)

            if single_string:
                # We got a single string passed into the validator, so it makes sense
                # for us to flatten the filtered output.
                assert len(filtered_output) == 1
                filtered_output = filtered_output[0]

            # Couldn't we do sum(error_spans)?
            combined_error_spans = []
            for e in error_spans:
                combined_error_spans.extend(e)

            return FailResult(
                error_message=(
                    f"Found the following competitors: {competitors_found}. "
                    "Please avoid naming those competitors next time."
                ),
                fix_value=filtered_output,
                error_spans=combined_error_spans,
            )
        else:
            return PassResult()

    def _inference_local(self, model_input: Dict) -> List[List[NERDetection]]:
        """Local inference method to detect and anonymize competitor names.
        Returns a list per sentence with NER detections."""
        texts = model_input["texts"]
        competitor_types = model_input["competitor_types"]

        if isinstance(texts, str):
            texts = [texts,]

        all_scores = []
        for sentence in texts:
            detections = self.model.predict_entities(
                sentence,
                competitor_types,
                threshold=CompetitorCheck.DETECTION_THRESHOLD
            )
            sentence_scores = list()
            for d in detections:
                sentence_scores.append(NERDetection(
                    name=d['text'],
                    entity_type=d['label'],
                    start=d['start'],
                    end=d['end'],
                    probability=d['score']
                ))
            all_scores.append(sentence_scores)
        return all_scores

    def _inference_remote(self, model_input: Dict) -> List[List[NERDetection]]:
        """Remote inference method for a hosted ML endpoint."""
        
        texts = model_input["texts"]
        competitor_types = model_input["competitor_types"]

        if isinstance(texts, str):
            texts = [texts,]

        request_body = {
            "inputs": [
                {
                    "name": "text",
                    "shape": [len(texts)],
                    "data": texts,
                    "datatype": "BYTES"
                },
                {
                    "name": "competitor_types",
                    "shape": [len(competitor_types)],
                    "data": competitor_types,
                    "datatype": "BYTES"
                }
            ]
        }
        response = self._hub_inference_request(json.dumps(request_body), self.validation_endpoint)

        if not response or "outputs" not in response:
            raise ValueError("Invalid response from remote inference", response)

        ner_entities = []
        for output in response["outputs"]:
            ner_entities.append(output["data"][0])

        return ner_entities

    @staticmethod
    def compute_error_spans(
            sentences: List[str],
            competitors: List[str],
            detected_competitors: List[List[NERDetection]],
            threshold: float,
    ) -> Tuple[List[str], List[List[ErrorSpan]]]:
        """Given a list of sentences and the probability of each competitor,
        generate a list of error spans for each sentence.
        """
        assert len(sentences) == len(detected_competitors)

        detections = set()

        competitors = set([c.lower() for c in competitors])

        all_error_spans: List[List[ErrorSpan]] = []
        for sentence, ner_detections in zip(sentences, detected_competitors):
            error_spans: List[ErrorSpan] = list()
            for detection in ner_detections:
                if detection.probability < threshold:
                    continue
                if detection.name.lower() not in competitors:
                    continue
                detections.add(detection.name)
                error_spans.append(ErrorSpan(
                    start=detection.start,
                    end=detection.end,
                    reason=f"Found '{detection.name}' (Confidence: {detection.probability})",
                ))
            all_error_spans.append(error_spans)
        return list(detections), all_error_spans

    # Replace mentions of competitors with [COMPETITOR]
    @staticmethod
    def compute_filtered_output(
            sentences: List[str],
            error_spans: List[List[ErrorSpan]]
    ) -> List[str]:
        assert len(sentences) == len(error_spans)

        filtered_outputs = list()
        for sentence, errors in zip(sentences, error_spans):
            if not errors:
                filtered_outputs.append(sentence)
            else:
                # We need to sort the error spans from last to first so we don't mess
                # up the indexing of the errors.
                for e in sorted(errors, key=lambda err: -err.start):
                    sentence = sentence[:e.start] + "[COMPETITOR]" + sentence[e.end:]
                filtered_outputs.append(sentence)
        return filtered_outputs
