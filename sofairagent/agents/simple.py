import re
import sys
from typing import Optional, Union, Literal

import json_repair
from classconfig import ConfigurableSubclassFactory, ConfigurableValue, ListOfConfigurableSubclassFactoryAttributes
from pydantic import BaseModel, Field, ValidationError
from ruamel.yaml.scalarstring import LiteralScalarString

from sofairagent.agents.base import Agent, Mention, MentionAdditionalInfo
from sofairagent.agents.search_agent import Candidate
from sofairagent.api.base import StructuredResponseFormatFaker
from sofairagent.utils.template import Template, TemplateTransformer


class MentionsResponse(BaseModel):
    """
    Represents the response from the find candidates request.
    """
    mentions: list[Mention] = Field([], description="List of extracted software names.")


class FindCandidatesResponse(BaseModel):
    """
    Represents the response from the find candidates request.
    """
    candidates: list[str] = Field([], description="List of extracted software mentions.")


class SimpleAgent(Agent):
    """
    Simple LLM agent extracting software mentions from text with single turn.
    """

    use_reasoning: Optional[Union[bool, Literal['low', 'medium', 'high']]] = ConfigurableValue(
        user_default=False,
        desc="Whether to use reasoning in the verification step."
    )

    system_prompt: Template = ConfigurableValue(
        user_default=LiteralScalarString(f"""You are a software mention extraction agent. Your task is to find software mentions in the text from scientific papers. 
Extract any named entity that resembles a software name, even if it is not a known software. If you are unsure, it is better to include a candidate than to miss one.
"""),
        desc="System prompt to find software mentions candidates in the text. ",
        transform=TemplateTransformer()
    )
    few_shot_examples: Optional[list[tuple[str, str]]] = ConfigurableValue(
        desc="Few-shot examples",
        user_default=[
            (
                "user",
                LiteralScalarString("""The text is:

We used the Ollama inference API [https://ollama.com/docs/api] to run the model locally.

Please extract all software mentions from the text.""")
            ),
            (
                "assistant",
                """{
    "candidates": ["Ollama"]
}
"""
            ),
            (
                "user",
                LiteralScalarString("""The text is:

Based on list of known public managers, we used the snowballing technique (Myers & Newman, 2007) to recruit new respondents. We first made contact with them by telephone, informing them about the study goals; shortly thereafter, we sent them a personalized link to the software Q-software with specific instructions on how to do the sorting exercise online.

Please extract all software mentions from the text.""")
            ),
            (
                "assistant",
                """{
    "candidates": ["Q-software"]
}
"""
            ),
            (
                "user",
                LiteralScalarString("""The text is:

We used the open-source Transformers library developed by Hugging Face (Wolf et al., 2020) to implement and train our models. Our training code MLTrainer is available at https://example.com, the whole code base is implemented using Python.

Please extract all software mentions from the text.""")
            ),
            (
                "assistant",
                """{
    "candidates": ["Transformers", "MLTrainer"]
}
"""
            )
        ]
    )
    prompt: Template = ConfigurableValue(
        user_default=LiteralScalarString("""The text is:

{{text}}

Please extract all software mentions from the text.
"""),
        desc="Jinja2 template for the prompt to find software mentions candidates in the text. ",
        transform=TemplateTransformer()
    )

    fake_structured_format: Optional[StructuredResponseFormatFaker] = ConfigurableSubclassFactory(
        StructuredResponseFormatFaker,
        "If set, it will be used to fake the structured response format for the model.",
        user_default=None,
        voluntary=True
    )

    context_window_for_database: int = ConfigurableValue(
        user_default=100,
        desc="Number of characters to use as context around the software mention."
    )

    def __post_init__(self):
        self.request_factory = self.model_api.get_request_factory()

    def __call__(self, text: str) -> list[Mention]:
        """
        Processes the input text and returns a list of software mentions.

        :param text: The input text to process.
        :return: A list of software mentions found in the text.
        """
        messages = [{"role": "system", "content": self.system_prompt.render({"text": text})}]

        if self.few_shot_examples:
            for role, content in self.few_shot_examples:
                messages.append({"role": role, "content": content})
        messages.append({"role": "user", "content": self.prompt.render({"text": text})})
        request = self.request_factory(
            custom_id="mentions_extraction",
            model=self.model,
            message=messages,
            options=self.requests_options,
            response_format=FindCandidatesResponse,
            fake_structured=self.fake_structured_format
        )
        raw_response = self.model_api.process_single_request(request).response.get_raw_content()
        response = json_repair.loads(raw_response)

        try:
            response = FindCandidatesResponse.model_validate(response)
        except ValidationError as e:
            print("Validation error:", e)
            return []

        return self.convert_candidates_to_mentions(text, response.candidates)

    def convert_candidates_to_mentions(self, text: str, candidates: list[str]) -> list[Mention]:
        """
        Converts a list of candidates to a list of mentions by finding their start offsets in the text.

        :param text: original text from which the candidates were extracted
        :param candidates: list of candidates to convert
        :return: list of mentions
        """
        mentions = []
        for c in candidates:
            try:
                mention = self.convert_candidate_to_mention(text, c)
                mentions.extend(mention)
            except ValueError as e:
                print(f"Warning: {e}", file=sys.stderr)
        return mentions

    @staticmethod
    def get_context_window(text: str, start_offset: int, surface_form: str, window: int) -> str:
        """
        Extracts a context window around the surface form in the text.

        :param text: original text from which the candidate was extracted
        :param start_offset: start offset of the surface form in the text
        :param surface_form: exact form of the mention in the text
        :param window: number of characters to include before and after the surface form
        :return: context window around the surface form
        """
        context_start = max(0, start_offset - window)
        context_end = min(len(text), start_offset + len(surface_form) + window)
        return text[context_start:context_end]

    def convert_candidate_to_mention(self, text: str, candidate: str) -> list[Mention]:
        """
        Converts a candidate to a mention by finding its start offset in the text.

        :param text: original text from which the candidate was extracted
        :param candidate: candidate to convert
        :return: mention
        """

        # find all occurrences of the candidate in the text
        res = []
        for surface_form_start in [m.start() for m in re.finditer(re.escape(candidate), text)]:
            # ensure that the candidate is not part of a larger word
            if (surface_form_start > 0 and text[surface_form_start - 1].isalnum()) or (surface_form_start + len(candidate) < len(text) and text[surface_form_start + len(candidate)].isalnum()):
                continue
            res.append(Mention(
                surface_form=candidate,
                context=self.get_context_window(text, surface_form_start, candidate, self.context_window_for_database),
                start_offset=surface_form_start,
                version=None,
                publisher=[],
                url=[],
                language=[]
            ))

        return res

    def call_second_pass(self, text: str, all_document_mentions: list[Mention], text_start_offset: int) -> list[Mention]:
        return []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

