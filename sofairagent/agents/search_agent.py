import logging
import logging
import re
import sys
from collections import defaultdict
from typing import Optional, Union, Literal

import json_repair
import validators
from classconfig import ConfigurableSubclassFactory, ConfigurableValue, ConfigurableFactory
from pydantic import BaseModel, Field, ValidationError
from ruamel.yaml.scalarstring import LiteralScalarString

from sofairagent.agents.base import Agent, Mention, MentionAdditionalInfo
from sofairagent.api.base import StructuredResponseFormatFaker
from sofairagent.search import DDGSSearcher, Searcher
from sofairagent.software_database import SoftwareDatabase
from sofairagent.utils.template import Template, TemplateTransformer
from sofairagent.verifier.verifier import Verifier


class AdditionalInfo(BaseModel):
    surface_form: str = Field(description="Exact form of the mention in the text")
    context: str = Field(description="Context of the mention exactly as it appears in the text.")


class Candidate(BaseModel):
    """
    Represents a candidate software mention extracted from the text.
    """
    id: int | str = Field(
        description="Unique identifier of the candidate. It can be just a number starting from 0 and incremented by 1 for each candidate")
    surface_form: str = Field(description="Mention name exactly as it appears in the text")
    context: str = Field(description="Context of the mention exactly as it appears in the text. ")
    version: Optional[AdditionalInfo] = Field(
        description="Exact form of the version in the text and its context. It can be null if no version was extracted.")
    publisher: list[AdditionalInfo] = Field(description="List of publishers extracted from the text. ")
    url: list[AdditionalInfo] = Field(description="List of URLs extracted from the text.")
    language: list[AdditionalInfo] = Field(description="List of programming languages extracted from the text.")


class FindCandidatesResponse(BaseModel):
    """
    Represents the response from the find candidates request.
    """
    candidates: list[str] = Field([], description="List of extracted software mentions candidates.")


class IncludeCheckList(BaseModel):
    """
    Represents a checklist of verification points for software to be included.
    """
    commercial_and_general_purpose_software: bool = Field(
        description="Whether the software is commercial and general-purpose software, e.g. Excel, Photoshop, CorelDraw.")
    research_software: bool = Field(description="Whether the software is developed in academic or research context.")
    software_environment: bool = Field(
        description="Whether the software is a programming environment, e.g. R, Matlab, SAS.")
    named_software_component_package: bool = Field(description="e.g. R package pROC, Python package NetworkX.")
    implicit_software_mentions: bool = Field(
        description="Generic words like “program”, “script”, “code”, “package” if they clearly indicate executable software.")
    workflow: bool = Field(
        description="Whether the software is a high-level specifications: in data-intensive scientific domains, the complexity of data processing has led to the common definiton and usage of workflows associated to a scientific experiments, e.g. Galaxy, Kepler, Apache Taverna.")
    api: bool = Field(description="When referred to as executable/shared software.")
    operating_system: bool = Field(
        description="(Windows, Linux, macOS) if the OS itself is being referenced (not just “running on Windows”).")
    devices_with_embedded_software: bool = Field(description="If the mention clearly refers to the software part.")


class ExcludeCheckList(BaseModel):
    """
    Represents a checklist of verification points for software to be excluded.
    """
    algorithm: bool = Field(
        description="Whether the mention is an algorithm. Unless the context shows their implementation was run as software.")
    model: bool = Field(
        description="Whether the mention is only a model (machine learning models, simulation models). Unless the mention refers to the software implementing the model (e.g. BERT library vs. BERT model)")
    database: bool = Field(
        description="Whether the mention is only a database (if it is not clear that the reference is made to the software part). Unless explicitly mentioned as software tools (compiler, interpreter, IDE).")
    programming_language: bool = Field(
        description="Whether the mention is only a programming language (e.g. written in BASIC, FORTRAN, etc.).")
    operating_system: bool = Field(
        description="Whether the mention is only an operating system when only used as an attribute of other software (e.g. “SPSS for Windows” → OS not annotated).")
    bibliographic_references: bool = Field(
        description="Whether the mention is only a bibliographic reference to software (e.g. “R Development Core Team, 2020” in references).")


class AmbiguousCheckList(BaseModel):
    """
    Represents a checklist of verification points for ambiguous software mentions.
    """
    algorithm_vs_software: bool = Field(
        description="Whether the mention is an algorithm name used to refer to the software implementing the algorithm. If so, it should be annotated as software.")
    model_vs_software: bool = Field(
        description="Whether the mention is a model name used to refer to the software implementing/running the model. If so, it should be annotated as software.")
    database_vs_software: bool = Field(
        description="Whether the mention is a database name used to refer to the software providing access to the database. If so, it should be annotated as software.")
    device_vs_software: bool = Field(
        description="Whether the mention is a device name used to refer to the software part of the device. If so, it should be annotated as software.")


class VerifyCandidateResponse(BaseModel):
    """
    Represents the response from the verify candidate request.
    """
    """
    include_check_list: IncludeCheckList = Field(
        description="Checklist of verification points for software to be included.")
    exclude_check_list: ExcludeCheckList = Field(
        description="Checklist of verification points for software to be excluded.")
    
    reason: str = Field(description="Short reason for the verification decision.")
    """
    confidence: Literal["low", "medium", "high"] = Field(description="Confidence level of the verification decision.")
    is_software: bool = Field(description="Whether the candidate was verified as a software mention or not.")


class VerifyMentionResponse(BaseModel):
    """
    Represents the response from the verify candidate request.
    """
    include_check_list: IncludeCheckList = Field(
        description="Checklist of verification points for software to be included.")
    exclude_check_list: ExcludeCheckList = Field(
        description="Checklist of verification points for software to be excluded.")
    reason: str = Field(description="Short reason for the verification decision.")
    is_software: bool = Field(description="Whether the candidate was verified as a software mention or not.")
    verified: Optional[Mention] = Field(
        description="Verified software mention. Is null if the mention was not verified as a software mention.")


class FillAdditionalInfoResponse(BaseModel):
    """
    Represents the response from the fill additional info request.
    """
    version: Optional[AdditionalInfo] = Field(
        description="Exact form of the version in the text and its context. It can be null if no version was extracted.")
    publisher: list[AdditionalInfo] = Field(description="List of publishers extracted from the text. ")
    url: list[AdditionalInfo] = Field(description="List of URLs extracted from the text.")
    language: list[AdditionalInfo] = Field(description="List of languages extracted from the text.")


class URLRepairResponse(BaseModel):
    """
    Represents the response from the URL repair request.
    """
    repaired_url: str = Field(description="Repaired URL. It can be an empty string if the URL could not be repaired.")


class SearchQueryResponse(BaseModel):
    """
    Represents a LLM response for creating a search query.
    """
    query: str = Field(description="Search query to use for searching the software.")

class SearchAgent(Agent):
    """
    An agent that is able to utilize a search engine to find software mentions in the text.
    """

    search: Searcher = ConfigurableSubclassFactory(
        Searcher,
        "Searcher to use for the agent.",
        user_default=DDGSSearcher
    )

    software_database: SoftwareDatabase = ConfigurableFactory(
        SoftwareDatabase,
        "Software database to use for checking existing software mentions.",
    )

    max_repairs: int = ConfigurableValue(
        user_default=3,
        desc="Maximum number of trials to repair the candidates contexts."
    )

    use_reasoning: Optional[Union[bool, Literal['low', 'medium', 'high']]] = ConfigurableValue(
        user_default=False,
        desc="Whether to use reasoning in the verification step."
    )
    verify: bool = ConfigurableValue(
        user_default=True,
        desc="Whether to verify the candidates after finding them."
    )
    fill: bool = ConfigurableValue(
        user_default=True,
        desc="Whether to fill additional information for the verified mentions.",
        voluntary=True
    )

    find_candidates_system_prompt: Template = ConfigurableValue(
        user_default=LiteralScalarString(f"""Your task is to find software mentions in the text from scientific papers. 
Extract any named entity that resembles a software name, even if it is not a known software. If you are unsure, it is better to include a candidate than to miss one.
"""),
        desc="System prompt to find software mentions candidates in the text. ",
        transform=TemplateTransformer()
    )
    find_candidates_few_shot: Optional[list[tuple[str, str]]] = ConfigurableValue(
        desc="Few-shot examples for finding software mentions candidates.",
        user_default=[
            (
                "user",
                LiteralScalarString("""The text is:

We used the Ollama inference API [https://ollama.com/docs/api] to run the model locally.

Please extract all software mentions from the text.""")
            ),
            (
                "assistant",
                LiteralScalarString("""{
    "candidates": ["Ollama"]
}
""")
            ),
            (
                "user",
                LiteralScalarString("""The text is:

Based on list of known public managers, we used the snowballing technique (Myers & Newman, 2007) to recruit new respondents. We first made contact with them by telephone, informing them about the study goals; shortly thereafter, we sent them a personalized link to the software Q-software with specific instructions on how to do the sorting exercise online.

Please extract all software mentions from the text.""")
            ),
            (
                "assistant",
                LiteralScalarString("""{
    "candidates": ["Q-software"]
}
""")
            ),
            (
                "user",
                LiteralScalarString("""The text is:

We used the open-source Transformers library developed by Hugging Face (Wolf et al., 2020) to implement and train our models. Our training code MLTrainer is available at https://example.com, the whole code base is implemented using Python.

Please extract all software mentions from the text.""")
            ),
            (
                "assistant",
                LiteralScalarString("""{
    "candidates": ["Transformers", "MLTrainer"]
}
""")
            )
        ]
    )
    find_candidates_prompt: Template = ConfigurableValue(
        user_default=LiteralScalarString("""The text is:

{{text}}

Please extract all software mentions from the text."""),
        desc="Jinja2 template for the prompt to find software mentions candidates in the text. ",
        transform=TemplateTransformer()
    )

    url_repair_system_prompt: Template = ConfigurableValue(
        user_default=LiteralScalarString("""You are a URL repair agent. Your task is to repair the given URL extracted from the text from a scientific paper.
You will be provided with text from which the URL was extracted and the URL itself.
Make sure that the URL is not just valid, but that it points to the correct resource, i.e., the resource that was mentioned in the text
Try to repair it if it is not a valid URL. If you cannot repair it, return an empty string."""),
        desc="System prompt to repair URLs in the candidates. ",
        transform=TemplateTransformer()
    )

    url_repair_prompt: Template = ConfigurableValue(
        user_default=LiteralScalarString("""The text is:

{{text}}

The URL is: '{{url}}'
Please repair the URL if it is not valid or return an empty string."""),
        desc="Jinja2 template for the prompt to repair URLs in the candidates. ",
        transform=TemplateTransformer()
    )
    repair_problems_prompt_system_prompt: Template = ConfigurableValue(
        user_default=LiteralScalarString("""You are a software mention extraction agent. Your task is to find software mentions in the text from scientific papers. 
You have previously extracted software mentions candidates from the text, but some of them have problems. 
Please fix them using the text from which the software mention candidate was extracted."""),
        desc="System prompt to repair candidates with identified problems. ",
        transform=TemplateTransformer()
    )
    repair_problems_prompt: Template = ConfigurableValue(
        user_default=LiteralScalarString("""The following problems were identified for the given candidate.
Text:

{{text}}

Problematic candidate:
{{candidate | model_dump_json }}

Problems:
{% for p in problems %}
        - {{p}}
{% endfor %}

Please fix all problems of the candidate."""),
        desc="Jinja2 template for the prompt to repair candidates with identified problems. ",
        transform=TemplateTransformer()
    )

    obtain_search_query_system_prompt: Template = ConfigurableValue(
        user_default=LiteralScalarString("""You are a software mention extraction agent. You are about to search for a software mention in a web search engine.
To get the best search results, you need to create a good search query for which you will obtain the best results that will help you to verify whether given mention is a software or not.
You will be provided with the text from which the mention was extracted and the mention itself.
Please create a search query that will help you to find out whether the mention is a software or not.
Make sure that the query is not too broad, but also not too specific.
If you think that the context of the mention will help with the search, include it in the query."""),
        desc="System prompt to create a search query for the search engine. ",
        transform=TemplateTransformer()
    )

    obtain_search_query_prompt: Template = ConfigurableValue(
        user_default=LiteralScalarString("""The text is:

{{text}}

The mention to search for is: 

{{candidate | model_dump_json }}

Please create a search query for the mention."""),
        desc="Jinja2 template for the prompt to create a search query for the search engine. ",
        transform=TemplateTransformer()
    )

    context_window_for_database: int = ConfigurableValue(
        user_default=100,
        desc="Number of characters to use as context around the software mention for storing validate software to the software database."
    )
    accepted_confidences_for_database: float = ConfigurableValue(
        user_default=1.0,
        desc="The confidence must be higher to store the verified software mention to the database."
    )

    fake_structured_format: Optional[StructuredResponseFormatFaker] = ConfigurableSubclassFactory(
        StructuredResponseFormatFaker,
        "If set, it will be used to fake the structured response format for the model.",
        user_default=None,
        voluntary=True
    )

    paragraph_separator: str = ConfigurableValue(
        user_default="\n",
        desc="Separator to use between paragraphs when splitting the text into smaller chunks.",
        voluntary=True
    )

    verify_open_tag: str = ConfigurableValue(
        desc="String to mark the beginning of a software mention in the text.",
        user_default=" <verify> "
    )
    verify_close_tag: str = ConfigurableValue(
        desc="String to mark the end of a software mention in the text.",
        user_default=" </verify> "
    )
    mention_open_tag: str = ConfigurableValue(
        desc="String to mark the beginning of a software mention in the text.",
        user_default=" <mention> "
    )
    mention_close_tag: str = ConfigurableValue(
        desc="String to mark the end of a software mention in the text.",
        user_default=" </mention> "
    )

    fill_mention_system_prompt: Template = ConfigurableValue(
        user_default=LiteralScalarString("""Your task is to extract additional information about the software mention from the text. The additional information are:
"version" -  It can be a number, an identifier or a date. It is expected that a mentioned software has only one version.
"publisher" - It is usually the organization or the company owning the software or having developed the software. It is expected that a mentioned software has only one publisher in the same mention context a most, but several is possible.
"url" - The URL can link to the code repository, to the software project page, to its documentation, etc. Although very rare, it is possible to have several url component for a software mention.
"language" - We only consider here the language when used to indicate how the source code is written, not the language as a broader reference to the programming environment used to develop the mentioned software.

You will be provided with the text from which the mention was extracted and the mention itself. The mention will be marked in the text with <mention> and </mention> tags.
Please extract the additional information about the software mention from the text.
"""),
        desc="System prompt to fill the software mention with additional information. ",
        transform=TemplateTransformer()
    )
    fill_mention_few_shot: Optional[list[tuple[str, str]]] = ConfigurableValue(
        desc="Few-shot examples for filling the software mention with additional information.",
        user_default=[
            (
                "user",
                LiteralScalarString("""Extract additional information about the software mention from the text.
The text is:

We used the open-source <mention> Transformers </mention> library developed by Hugging Face (Wolf et al., 2020) to implement and train our models. Our training code MLTrainer is available at https://example.com, the whole code base is implemented using Python.

Mention to extract additional information about:
surface form: Transformers
context: open-source Transformers library
""")
            ),
            (
                "assistant",
                LiteralScalarString("""{
    "version": null,
    "publisher": [
        {
            "surface_form": "Hugging Face",
            "context": "developed by Hugging Face (Wolf",
        }
    ],
    "url": [],
    "language": [
        {
            "surface_form": "Python",
            "context": "implemented using Python.",
        }
    ]
}""")
            ),
            (
                "user",
                LiteralScalarString("""Extract additional information about the software mention from the text.
The text is:

Our solution CryptoSeek is open-source solution for secure communication. The <mention> CryptoSeek </mention> version 1.2.3 developed by SecureSoft is available at https://securesoft.com/cryptoseek. We used Rust to implement it.

Mention to extract additional information about:
surface form: CryptoSeek
context: The CryptoSeek version 1.2.3 developed
""")
            ),
            (
                "assistant",
                LiteralScalarString("""{
    "version": {
        "surface_form": "1.2.3",
        "context": "version 1.2.3"
    },
    "publisher": [
        {
            "surface_form": "SecureSoft",
            "context": "developed by SecureSoft"
        }
    ],
    "url": [
        {
            "surface_form": "https://securesoft.com/cryptoseek",
            "context": "available at https://securesoft.com/cryptoseek"
        }
    ],
    "language": [
        {
            "surface_form": "Rust",
            "context": "We used Rust to implement it."
        }
    ]
}""")
            ),
            (
                "user",
                LiteralScalarString("""Extract additional information about the software mention from the text.
The text is:

In our experiments we focused on high recall methods. The data analysis was performed. We specifically used the tidyverse package to manipulate and visualize the data. 

Mention to extract additional information about:
surface form: tidyverse
context: used the tidyverse package to
""")
            ),
            (
                "assistant",
                LiteralScalarString("""{
    "version": null,
    "publisher": [],
    "url": [],
    "language": []
}""")
            )
        ]
    )

    fill_mention_prompt: Template = ConfigurableValue(
        user_default=LiteralScalarString("""Extract additional information about the software mention from the text.
The text is:

{{marked_text}}

Mention to extract additional information about:
surface form: {{mention.surface_form}}"""),
        desc="Jinja2 template for the prompt to fill the software mention with additional information. ",
        transform=TemplateTransformer()
    )

    verifier: Verifier = ConfigurableFactory(
        Verifier,
        "Verifier to use for verifying the candidates.",
    )

    def __post_init__(self):
        self.tools = [
            {
                'type': 'function',
                'function': {
                    'name': 'search_engine',
                    'description': 'Web search engine. Use this tool to search for software mentions in the text. The search engine will return a list of search results that can be used to verify software mentions. It returns a list of search results with title, URL, and snippet.',
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'query': {
                                'type': 'string',
                                'description': 'The search query to use for the search engine. It should be the name of the software mention to search for. Use the context of the mention if you think it will help with the search.',
                            },
                        },
                        'required': ['query'],
                    },
                },
            }
        ]
        self.request_factory = self.model_api.get_request_factory()

    def __enter__(self):
        self.search.__enter__()
        self.software_database.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.search.__exit__(exc_type, exc_val, exc_tb)
        self.software_database.__exit__(exc_type, exc_val, exc_tb)

    def __call__(self, text: str) -> list[Mention]:
        """
        Processes the input text and returns a list of software mentions.

        :param text: The input text to process.
        :return: A list of software mentions found in the text.
        """
        # find candidates, we focus on high recall here
        paragraphs = self.split_text_into_paragraphs(text)
        candidates = []
        for _, p in paragraphs:
            candidates.extend(self.find_candidates_llm(p))

        candidates.extend([surface_form for _, surface_form in self.software_database.known_surface_forms_in_text(text)])
        candidates = list(set(candidates))  # remove duplicates

        # convert candidate names to mentions, this can even blow up the number of mentions as we find all occurrences of the candidate in the text
        mentions = self.convert_candidates_to_mentions(text, candidates)
        mentions = self.filter_overlapping_mentions(mentions)

        logging.info(f"Found {len(mentions)} mentions in the text.")

        verified_candidates = []
        for p_start_offset, p in paragraphs:
            mentions_in_p = [m for m in mentions if p_start_offset <= m.start_offset < p_start_offset + len(p)]
            if self.verify:
                mentions_in_p = self.verify_candidates(p, mentions_in_p, p_start_offset)
            if self.fill:
                self.fill_mentions(p, mentions_in_p)
            verified_candidates.extend(mentions_in_p)

        self.add_mentions_to_database(text, verified_candidates)
        return verified_candidates

    def filter_overlapping_mentions(self, mentions: list[Mention]) -> list[Mention]:
        """
        Filters out overlapping mentions, keeping only the longest mention in case of overlaps.

        :param mentions: The list of mentions to filter.
        :return: The list of filtered mentions.
        """
        # sort mentions by start_offset and then by length (longest first)
        mentions = sorted(mentions, key=lambda m: (m.start_offset, -len(m.surface_form)))
        filtered_mentions = []
        last_end = -1
        for m in mentions:
            if m.start_offset >= last_end:
                filtered_mentions.append(m)
                last_end = m.start_offset + len(m.surface_form)
        return filtered_mentions

    def fill_mentions(self, text: str, mentions: list[Mention]):
        """
        Fills the mentions with additional information such as version, publisher, URL, and language.

        :param text: The input text from which the mentions were extracted.
        :param mentions: The list of mentions to fill with additional information.
        :return: The list of mentions filled with additional information.
        """
        for m in mentions:
            try:
                self.fill_mention(text, m)
            except ValueError as e:
                print(f"Warning: {e}", file=sys.stderr)

    def fill_mention(self, text: str, mention: Mention):
        """
        Fills the mention with additional information such as version, publisher, URL, and language.

        :param text: The input text from which the mention was extracted.
        :param mention: The mention to fill with additional information.
        :return: The mention filled with additional information.
        """

        marked_text = self.mark_orig_text(text, mention, self.mention_open_tag, self.mention_close_tag, 0)

        messages = [
            {"role": "system", "content": self.fill_mention_system_prompt.render(
                data={"text": text, "marked_text": marked_text, "mention": mention})},
        ]
        if self.fill_mention_few_shot is not None:
            for role, content in self.fill_mention_few_shot:
                messages.append({"role": role, "content": content})

        messages.append(
            {"role": "user", "content": self.fill_mention_prompt.render(
                data={"text": text, "marked_text": marked_text, "mention": mention})},
        )
        request = self.request_factory(
            custom_id="fill_additional_info",
            model=self.model,
            message=messages,
            options=self.requests_options,
            response_format=FillAdditionalInfoResponse,
            reasoning=self.use_reasoning,
            fake_structured=self.fake_structured_format
        )
        api_output = self.model_api.process_single_request(request)
        # get the tool calls from structured output, as it is not currently supported by ollama to use tool calling and structured output together
        reply = api_output.response.get_raw_content()
        reply = json_repair.loads(reply)
        try:
            reply = FillAdditionalInfoResponse.model_validate(reply)
        except ValidationError as e:
            print(
                f"Warning: Could not validate the output of the model for filling additional information for mention '{mention.surface_form}': {e}",
                file=sys.stderr)
            return

        # convert reply to candidate
        candidate = Candidate(
            id="1",
            surface_form=mention.surface_form,
            context=mention.context,
            version=reply.version,
            publisher=reply.publisher,
            url=reply.url,
            language=reply.language
        )
        self.repair(text, candidate)

        self.fill_mention_with_additional_info(text, mention, candidate)


    def fill_mention_with_additional_info(self, text: str, mention: Mention, additional_info: Candidate):
        """
        Fills the mention with additional information such as version, publisher, URL, and language.

        :param text: The input text from which the mention was extracted.
        :param mention: The mention to fill with additional information.
        :param additional_info: The additional information to fill the mention with.
        """

        def assemble_additional_info(context: str, surface_form: str, type_name: str) -> MentionAdditionalInfo:
            start_offset = self.get_start_offset(text, context, surface_form, type_name)
            return MentionAdditionalInfo(
                surface_form=surface_form,
                context=context,
                start_offset=start_offset
            )
        if additional_info.version is not None:
            try:
                mention.version = assemble_additional_info(additional_info.version.context, additional_info.version.surface_form, "Version")
            except ValueError as e:
                print(f"Warning: {e}", file=sys.stderr)

        for p in additional_info.publisher:
            try:
                mention.publisher.append(assemble_additional_info(p.context, p.surface_form, "Publisher"))
            except ValueError as e:
                print(f"Warning: {e}", file=sys.stderr)
        for u in additional_info.url:
            try:
                mention.url.append(assemble_additional_info(u.context, u.surface_form, "URL"))
            except ValueError as e:
                print(f"Warning: {e}", file=sys.stderr)
        for l in additional_info.language:
            try:
                mention.language.append(assemble_additional_info(l.context, l.surface_form, "Language"))
            except ValueError as e:
                print(f"Warning: {e}", file=sys.stderr)

    def find_candidates_llm(self, text: str) -> list[str]:
        """
        Finds software mentions in the text using the search engine.

        :param text: The input text to search for software mentions.
        :return: A list of software mentions found in the text.
        """
        messages = [{"role": "system", "content": self.find_candidates_system_prompt.render({"text": text})}]

        if self.find_candidates_few_shot:
            for role, content in self.find_candidates_few_shot:
                messages.append({"role": role, "content": content})
        messages.append({"role": "user", "content": self.find_candidates_prompt.render({"text": text})})
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

        return response.candidates

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
                confidence=None,
                version=None,
                publisher=[],
                url=[],
                language=[]
            ))

        return res

    def add_mentions_to_database(self, text: str, mentions: list[Mention]):
        """
        Adds the given mentions to the software database.

        :param text: original text from which the mentions were extracted
        :param mentions: list of mentions to add to the database
        """
        for m in mentions:
            if m.confidence > self.accepted_confidences_for_database:
                context = self.get_context_window(text, m.start_offset, m.surface_form, self.context_window_for_database)
                self.software_database.add_software(m.surface_form, context)

    @staticmethod
    def get_start_offset(text: str, context: str, surface_form: str, type_name: str = "Software") -> int:
        """
        Finds the start offset of the surface form in the context within the text.

        :param text: original text from which the candidate was extracted
        :param context: context of the mention
        :param surface_form: exact form of the mention in the text
        :param type_name: type of the mention (used for error messages)
        :return: start offset of the surface form in the text
        :raise ValueError: if the context or surface form is not found in the text
        """
        start_offset = text.find(context)
        if start_offset == -1:
            raise ValueError(f"Context '{context}' not found in text.")
        start_offset += context.find(surface_form)
        if start_offset == -1:
            raise ValueError(f"{type_name} surface form '{surface_form}' not found in {type_name} context '{context}'.")
        return start_offset

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

    def verify_candidates(self, text: str, candidates: list[Mention], text_offset: int) -> list[Mention]:
        """
        Verifies software mentions candidates in the text.

        :param text: The input text to verify software mentions.
        :param candidates: A list of software mentions candidates to verify.
        :param text_offset: Start offset of the text in the document.
        :return: A list of verified software mentions.
        """

        to_verify = []
        for c in candidates:
            search_query = self.obtain_search_query(text, c)
            search_results = self.search.search(search_query).results
            marked_text = self.mark_orig_text(text, c, self.verify_open_tag, self.verify_close_tag, text_offset)
            to_verify.append({
                "marked_input_text": marked_text,
                "search_results": search_results,
            })

        res = []
        for c, (ver_res, prob) in zip(candidates, self.verifier(to_verify)):
            if ver_res:
                c.confidence = prob
                res.append(c)
        return res

    def verify_candidate(self, text: str, candidates: list[Mention], target: Mention, text_offset: int) -> bool:
        """
        Verifies a single software mention candidate in the text.

        :param text: The input text to verify software mentions.
        :param candidates: A list of software mentions candidates to verify.
        :param target: The software mention candidate to verify.
        :param text_offset: Start offset of the text in the document.
        :return: True if the software mention is verified, False otherwise.
        """

        search_query = self.obtain_search_query(text, target)
        search_results = self.search.search(search_query).results

        marked_text = self.mark_orig_text(text, target, self.verify_open_tag, self.verify_close_tag, text_offset)

        messages = [
            {"role": "system", "content": self.verification_system_prompt.render(
                data={"text": text, "marked_text": marked_text, "candidates": candidates, "target": target, "search_results": search_results, })},
        ]
        if self.verification_few_shot is not None:
            for role, content in self.verification_few_shot:
                messages.append({"role": role, "content": content})

        messages.append(
            {"role": "user", "content": self.verification_prompt.render(
                data={"text": text, "marked_text": marked_text, "candidates": candidates, "target": target, "search_results": search_results})},
        )
        logging.info(f"verify_candidate | message: {messages[-1]['content']} | search_query: {search_query}")

        request = self.request_factory(
            custom_id="verify_candidate",
            model=self.model,
            message=messages,
            options=self.requests_options,
            response_format=VerifyCandidateResponse,
            reasoning=self.use_reasoning,
            fake_structured=self.fake_structured_format
        )
        api_output = self.model_api.process_single_request(request)
        # get the tool calls from structured output, as it is not currently supported by ollama to use tool calling and structured output together
        reply = api_output.response.get_raw_content()
        reply = json_repair.loads(reply)
        try:
            reply = VerifyCandidateResponse.model_validate(reply)
        except ValidationError as e:
            print(
                f"Warning: Could not validate repaired output in verify_candidate for candidate {target.surface_form}. Error: {e}",
                file=sys.stderr)
            return False

        target.confidence = reply.confidence
        logging.info(f"verify_candidate | confidence: {reply.confidence} | is_software: {reply.is_software}")
        return reply.is_software

    def mark_orig_text(self, orig_text: str, candidate: Mention, open_tag: str, close_tag: str, align_offset: int = 0) -> str:
        """
        Marks the candidate surface form in the original text with specified tags.

        :param orig_text: Original text
        :param candidate: Candidate mention to be marked
        :param open_tag: Tag to mark the beginning of the candidate surface form
        :param close_tag: Tag to mark the end of the candidate surface form
        :param align_offset: Offset to align the candidate start offset with the original text (used when processing per paragraph)
        :return: Text with marked candidate surface form
        """
        before = orig_text[:candidate.start_offset - align_offset]
        after = orig_text[candidate.start_offset + len(candidate.surface_form) - align_offset:]
        return f"{before}{open_tag}{candidate.surface_form}{close_tag}{after}"

    def obtain_search_query(self, text: str, candidate: Mention) -> str:
        """
        Obtains a search query for the given candidate.

        :param text: The input text to process.
        :param candidate: The candidate for which to obtain the search query.
        :return: The search query.
        """
        request = self.request_factory(
            custom_id="obtain_search_query",
            model=self.model,
            message=[
                {"role": "system",
                 "content": self.obtain_search_query_system_prompt.render(data={"text": text, "candidate": candidate})},
                {"role": "user",
                 "content": self.obtain_search_query_prompt.render(data={"text": text, "candidate": candidate})}
            ],
            options=self.requests_options,
            response_format=SearchQueryResponse,
            fake_structured=self.fake_structured_format
        )
        raw_response = self.model_api.process_single_request(request).response.get_raw_content()
        response = json_repair.loads(raw_response)
        response = SearchQueryResponse.model_validate(response)
        return response.query.strip()

    def split_text_into_paragraphs(self, text: str) -> list[tuple[int, str]]:
        """
        Splits the text into paragraphs and returns a list of tuples with the start offset and the paragraph text.

        :param text: Text to split.
        :return: List of tuples with the start offset and the paragraph text.
        """

        paragraphs = []
        offset = 0
        for paragraph in text.split(self.paragraph_separator):
            if len(paragraph) > 0:
                paragraphs.append((offset, paragraph))
            offset += len(paragraph) + 1

        return paragraphs

    def repair(self, text: str, candidate: Candidate):
        """
        Repairs context problems in candidates.

        :param text: original text from which the candidates were extracted
        :param candidate: candidate to repair
        """
        cnt = 0
        while True and cnt < self.max_repairs:
            cnt += 1
            flag = False
            try:
                for f in [
                    self.repair_non_existing_context,
                    self.repair_surface_form_not_in_context, self.repair_same_context_candidates,
                    self.repair_ambiguous_contexts, self.repair_urls
                ]:
                    flag = flag | f(text, candidate)

                if not flag:
                    # no repairs were made
                    break
            except ValidationError as e:
                # LLM output was invalid, we skip this repair iteration
                continue

    def send_repair_problem_request(self, text: str, candidate: Candidate,
                                    problems: list[str]) -> Candidate:
        """
        Sends a request to the model to repair a candidate with identified problems.

        :param text: original text from which the candidates were extracted
        :param candidate: candidate to repair
        :param problems: list of problems identified for the candidate
        :return: repaired candidate
        """
        request = self.request_factory(
            custom_id="repair_problems",
            model=self.model,
            message=[
                {"role": "system", "content": self.repair_problems_prompt_system_prompt.render(
                    data={"text": text, "candidate": candidate, "problems": problems})},
                {"role": "user", "content": self.repair_problems_prompt.render(
                    data={"text": text, "candidate": candidate, "problems": problems})}
            ],
            options=self.requests_options,
            response_format=Candidate,
            fake_structured=self.fake_structured_format
        )
        response = self.model_api.process_single_request(request).response.get_raw_content()
        response = json_repair.loads(response)
        response = Candidate.model_validate(response)
        return response

    def repair_non_existing_context(self, text: str, c: Candidate) -> bool:
        """
        Finds candidates with non-existing context.

        :param text: original text from which the candidates were extracted
        :param c: candidate to check
        :return: True if there was an attempt to repair, False otherwise
        """
        flag = False
        problems = []

        if c.version is not None and c.version.context not in text:
            problems.append(f"Version context '{c.version.context}' not found in text.")
        for p in c.publisher:
            if p.context not in text:
                problems.append(f"Publisher context '{p.context}' not found in text.")
        for u in c.url:
            if u.context not in text:
                problems.append(f"URL context '{u.context}' not found in text.")
        for l in c.language:
            if l.context not in text:
                problems.append(f"Language context '{l.context}' not found in text.")

        if len(problems) > 0:
            flag = True
            rep_c = self.send_repair_problem_request(text, c, problems)
            c.version = rep_c.version
            c.publisher = rep_c.publisher
            c.url = rep_c.url
            c.language = rep_c.language

        return flag

    def repair_surface_form_not_in_context(self, text: str, c: Candidate) -> bool:
        """
        Finds candidates with the surface form not in the context.

        :param text: original text from which the candidates were extracted
        :param c: candidate to check
        :return: True if there was an attempt to repair, False otherwise
        """
        flag = False
        problems = []

        if c.version is not None and c.version.surface_form not in c.version.context:
            problems.append(
                f"Version surface form '{c.version.surface_form}' not in version context '{c.version.context}'. Take on mind that the search is case-sensitive.")
        elif c.version is not None and c.version.context.count(c.version.surface_form) > 1:
            problems.append(
                f"Version surface form '{c.version.surface_form}' appears more than once in version context '{c.version.context}'")

        for p in c.publisher:
            if p.surface_form not in p.context:
                problems.append(
                    f"Publisher surface form '{p.surface_form}' not in publisher context '{p.context}'. Take on mind that the search is case-sensitive.")
            elif p.context.count(p.surface_form) > 1:
                problems.append(
                    f"Publisher surface form '{p.surface_form}' appears more than once in publisher context '{p.context}'")

        for u in c.url:
            if u.surface_form not in u.context:
                problems.append(
                    f"URL surface form '{u.surface_form}' not in URL context '{u.context}'. Take on mind that the search is case-sensitive.")
            elif u.context.count(u.surface_form) > 1:
                problems.append(
                    f"URL surface form '{u.surface_form}' appears more than once in URL context '{u.context}'")

        for l in c.language:
            if l.surface_form not in l.context:
                problems.append(
                    f"Language surface form '{l.surface_form}' not in language context '{l.context}'. Take on mind that the search is case-sensitive.")
            elif l.context.count(l.surface_form) > 1:
                problems.append(
                    f"Language surface form '{l.surface_form}' appears more than once in language context '{l.context}'")

        if len(problems) > 0:
            flag = True
            rep_c = self.send_repair_problem_request(text, c, problems)
            c.version = rep_c.version
            c.publisher = rep_c.publisher
            c.url = rep_c.url
            c.language = rep_c.language

        return flag

    def repair_same_context_candidates(self, text: str, candidate: Candidate) -> bool:
        """
        Repairs candidates with the same context by asking the model to differentiate them.

        :param text: original text from which the candidates were extracted
        :param candidate: condidate to repair/check
        :return: flag whether repair was done
        """

        flag = False
        if len(same_context := self.same_context_candidates(candidate)) > 0:
            flag = True
            for (surface_form, context), problem_candidates in same_context.items():
                # we use only the first candidate
                candidate, annot_type = problem_candidates[0]
                problems = [
                    f"The tuple of surface form '{surface_form}' and context '{context}' is used for {len(problem_candidates)} candidates. Please differentiate to make the tuple unique. For following candidate it is used as {annot_type}."
                ]
                res = self.send_repair_problem_request(text, candidate, problems)

                candidate.version = res.version
                candidate.publisher = res.publisher
                candidate.url = res.url
                candidate.language = res.language

        return flag

    def same_context_candidates(self, c: Candidate) -> dict[
        tuple[str, str], list[tuple[Candidate, str]]]:
        """
        Finds surface form and context pairs that are not distinguishable.

        :param c: list of candidates to check
        :return: dictionary mapping (surface form, context) to list of candidates having this context and description of context type
        """
        surface_forms = defaultdict(list)
        surface_forms[(c.surface_form, c.context)].append((c, "software name"))
        if c.version is not None:
            surface_forms[(c.version.surface_form, c.version.context)].append((c, "version"))
        for p in c.publisher:
            surface_forms[(p.surface_form, p.context)].append((c, "publisher"))
        for u in c.url:
            surface_forms[(u.surface_form, u.context)].append((c, "url"))
        for l in c.language:
            surface_forms[(l.surface_form, l.context)].append((c, "language"))

        res = {}
        for k, lst in surface_forms.items():
            if len(lst) > 1:
                res[k] = lst
        return res

    def repair_ambiguous_contexts(self, text: str, candidate: Candidate) -> bool:
        """
        Repairs candidates with ambiguous contexts by asking the model to differentiate them.

        :param text: original text from which the candidates were extracted
        :param candidate: candidate to repair/check
        :return: flag whether repair was done
        """
        flag = False
        if len(ambiguous_contexts := self.ambiguous_contexts_candidates(text, candidate)) > 0:
            flag = True
            for context, (candidates_with_annot_type, start_offsets) in ambiguous_contexts.items():
                for candidate, annot_type in candidates_with_annot_type:
                    problems = [
                        f"The context '{context}' appears {len(start_offsets)} times in the text, at offsets {start_offsets}. Please differentiate the context to make it unambiguous. The context is used for the candidate as {annot_type}."
                    ]
                    res = self.send_repair_problem_request(text, candidate, problems)

                    candidate.version = res.version
                    candidate.publisher = res.publisher
                    candidate.url = res.url
                    candidate.language = res.language

        return flag

    def ambiguous_contexts_candidates(self, text: str, c: Candidate) -> dict[
        str, tuple[list[tuple[Candidate, str]], list[int]]]:
        """
        Finds ambiguous contexts. A context is considered ambiguous if it appears in more than one place in the text.

        :param text: original text from which the candidates were extracted
        :param c: list of candidates to check
        :return: dictionary mapping context to (list of candidates with description of context type, list of start offsets in the text)
        """

        res = {}

        def proc_context(ctx: str, ctype: str, can: Candidate):
            matches = text.count(ctx)
            if matches > 1:
                if ctx not in res:
                    all_start_offsets = [m.start() for m in re.finditer(re.escape(ctx), text)]
                    res[ctx] = ([(can, ctype)], all_start_offsets)
                else:
                    res[ctx][0].append((can, ctype))
        if c.version is not None:
            proc_context(c.version.context, "version", c)
        for p in c.publisher:
            proc_context(p.context, "publisher", c)
        for u in c.url:
            proc_context(u.context, "url", c)
        for l in c.language:
            proc_context(l.context, "language", c)
        return res

    def repair_urls(self, text: str, c: Candidate) -> bool:
        """
        Repairs URLs in candidates by checking if they are valid URLs.
        Works in place.

        :param text: original text from which the candidates were extracted
        :param candidates: list of candidates to repair
        :return: flag whether any URL was repaired
        """
        flag = False
        for c_url in c.url:
            if not validators.url(c_url.surface_form):
                repaired_url = self.repair_url(text, c_url.surface_form, c_url.context)
                if repaired_url in text:  # only accept the repaired URL if it is in the text
                    flag = True
                    c_url.surface_form = repaired_url

        return flag

    def repair_url(self, text: str, url: str, context: str) -> str:
        """
        Repairs a URL in a candidate .

        :param text: text from which url was extracted
        :param url: URL to repair
        :param context: context from which url was extracted
        :return: repaired URL
        """

        # is there valid url inside?
        match = re.search(r'(https?://[^\s]+)', context)
        if match and validators.url(match.group(1)):
            return match.group(1)

        # missing scheme?
        if validators.url("https://" + url) and "https://" + url in context:
            return "https://" + url
        elif validators.url("http://" + url) and "http://" + url in context:
            return "http://" + url

        # ask LLM to repair
        request = self.request_factory(
            custom_id="repair_url",
            model=self.model,
            message=[
                {"role": "system", "content": self.url_repair_system_prompt.render(data={"text": text, "url": url})},
                {"role": "user", "content": self.url_repair_prompt.render(data={"text": text, "url": url})}
            ],
            options=self.requests_options,
            response_format=URLRepairResponse,
            fake_structured=self.fake_structured_format
        )
        response = self.model_api.process_single_request(request).response.get_raw_content()
        response = json_repair.loads(response)
        response = URLRepairResponse.model_validate(response)
        response = response.repaired_url.strip()
        if validators.url(response):
            return response
        return url
