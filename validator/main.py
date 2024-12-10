import json
import re
from typing import Any, Callable, Dict, List, Optional, Union

from guardrails.validator_base import ErrorSpan
from guardrails.validators import (
    FailResult,
    PassResult,
    ValidationResult,
    Validator,
    register_validator,
)

from sentence_splitter import SentenceSplitter
from transformers import pipeline


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

        enable_direct_match_prefilter (bool): Defaults to True. If true, will
        perform a case-insensitive search across the text for each of the
        competitors before attempting to use the ML model.  For example, the text,
        "It's a lovely day." does not have the substring 'Guardrails', so there's
        no need to perform a model-based check.  This results in greatly increased
        speed in the case where a competitor is mentioned, but comes at the cost
        of a higher false negative rate.  This value should be set to false if a
        competitor has multiple nuanced spellings or variations in spacing, for
        example: "USPS" and "US Postal Service" and "United States Postal Service".

        sentence_splitter_language: Defaults to "en" and should generally not be
        changed.  The underlying model is focused on English but may work, at least
        partially, on other languages.  Languages outside of English are untested and
        unsupported.
    """

    def __init__(
            self,
            competitors: List[str],
            threshold: float = 0.5,
            enable_direct_match_prefilter: bool = True,
            sentence_splitter_language: str = "en",
            on_fail: Optional[Callable] = None,
            **kwargs,
    ):
        super().__init__(
            competitors=competitors,
            threshold=threshold,
            on_fail=on_fail,
            **kwargs,
        )

        if sentence_splitter_language not in sentence_splitter_supported_languages:
            valid_languages = ", ".join([
                f"'{k}': {v}" for k,v in sentence_splitter_supported_languages.items()
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
        self.prefilter = enable_direct_match_prefilter
        self.sentence_splitter = SentenceSplitter(language=sentence_splitter_language)
        self.model = pipeline(
            "zero-shot-classification",
            "facebook/bart-large-mnli"
        )

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

    def validate(
            self,
            value: Union[List[str], str],
            metadata: Optional[dict] = None
    ) -> ValidationResult:
        """Checks a text to find competitors' names in it.

        While running, store sentences naming competitors and generate a fixed output
        filtering out all flagged sentences.

        Args:
            value (str): The value to be validated.
            metadata (Dict, optional): Additional metadata. Defaults to empty dict.

        Returns:
            ValidationResult: The validation result.
        """
        if metadata is None:
            metadata = dict()

        # We can expect to get full sentences, but we might get paragraphs, too.
        sentences = self.sentence_splitter.split(value)

        detected_competitors = self._inference({
            "text": sentences,
            "competitors": metadata.get("competitors", self.competitors)
        })

        error_spans = self.compute_error_spans(sentences, detected_competitors)

        # Error spans is a list of lists.  If any is NOT empty, then we have an error.
        if any([e for e in error_spans]):
            filtered_output = self.compute_filtered_output(sentences, error_spans)
            competitors_found = ", ".join(self.flatten_competitors(detected_competitors))

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

    def _inference_local(self, model_input: Dict) -> List[Dict[str, float]]:
        """Local inference method to detect and anonymize competitor names.
        Returns an array of dictionaries that map a competitor name to a probability."""
        text = model_input["text"]
        competitors = model_input["competitors"]

        if isinstance(text, str):
            text = [text,]

        # Frustratingly, `TypeError: TextInputSequence must be str` appears when we try
        # to use either an array of text OR an array of hypotheses.  They should both
        # be supported according to the tokenizer documentation.  When we use Bart
        # Tokenizer and the paired sentence approach we instead get an invalid
        # tokenization, so we need to do a bunch of parallel calls.
        # tokens = self.tokenizer.encode( ... ) for each candidate and sentence.
        # That's NOT performant, so we instead use a pipeline which lets us do a
        # parallel call at the cost of not being able to tweak the hypothesis.
        # See https://github.com/huggingface/transformers/issues/7735
        predictions = self.model(text, competitors, multi_label=True)

        competitor_scores = []  # An array of dictionaries, one per sentence.
        for p in predictions:
            competitor_scores.append({k: v for k, v in zip(p['labels'], p['scores'])})
        return competitor_scores

    def _inference_remote(self, model_input: Dict) -> List[Dict[str, float]]:
        """Remote inference method for a hosted ML endpoint."""
        
        text = model_input["text"]
        competitors = model_input["competitors"]

        if isinstance(text, str):
            text = [text]

        request_body = {
            "inputs": [
                {
                    "name": "text",
                    "shape": [len(text)],
                    "data": text,
                    "datatype": "BYTES"
                },
                {
                    "name": "competitors",
                    "shape": [len(competitors)],
                    "data": competitors,
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

    def compute_error_spans(
            self,
            sentences: List[str],
            detected_competitors: List[Dict[str, float]]
    ) -> List[List[ErrorSpan]]:
        """Given a list of sentences and the probability of each competitor,
        generate a list of error spans for each sentence.  If a sentence is thought to
        have a competitor but it can't be exactly matched to a word, the error span
        will highlight the whole sentence.
        """
        assert len(sentences) == len(detected_competitors)

        all_error_spans: List[List[ErrorSpan]] = []
        for idx, sentence in enumerate(sentences):
            dc = detected_competitors[idx]
            error_spans: List[ErrorSpan] = list()
            for competitor, probability in dc.items():
                if probability < self.threshold:
                    continue
                instances_found = 0
                start_idx = 0
                while sentence.find(competitor, start_idx) > -1:
                    instances_found += 1
                    start_idx = sentence.find(competitor, start_idx)
                    end_idx = start_idx + len(competitor)
                    error_spans.append(ErrorSpan(
                        start=start_idx,
                        end=end_idx,
                        reason=f"Found '{competitor}' (Confidence: {probability})",
                    ))
                    start_idx = end_idx
                if instances_found == 0:
                    # We have evidence that there's a mention in this sentence but can't
                    # find and replace the exact position.
                    error_spans.append(ErrorSpan(
                        start=0,
                        end=len(sentence),
                        reason=f"Detected possible mention of '{competitor}' "
                               f"(Confidence: {probability})",
                    ))
            all_error_spans.append(error_spans)
        return all_error_spans

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
    
    def flatten_competitors(
            self,
            detected_competitors: List[Dict[str, float]]
    ) -> List[str]:
        """Given the detections from every sentence, make a list of the unique
        competitors detected.

        This _does_ do another iteration over the probabilities that we could remove if
        we tracked the data separately, but it's not an expensive operation and makes
        for cleaner code."""
        list_of_competitors_found = set()
        for sentence_dc in detected_competitors:
            for competitor, probability in sentence_dc.items():
                if probability > self.threshold:
                    list_of_competitors_found.add(competitor)
        return list(list_of_competitors_found)
