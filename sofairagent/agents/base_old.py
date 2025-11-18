from abc import ABC, abstractmethod
from typing import Optional, Literal

from classconfig import ConfigurableSubclassFactory, ConfigurableMixin, ConfigurableFactory, ConfigurableValue
from pydantic import BaseModel

from sofairagent.api import API, OllamaAPI
from sofairagent.api.base import RequestOptions


class MentionAdditionalInfo(BaseModel):
    """
    Represents additional information about a software mention.
    """
    surface_form: str
    context: str
    start_offset: int


class Mention(BaseModel):
    """
    Represents a software mention in the text.
    """

    surface_form: str
    context: str
    start_offset: int
    version: Optional[MentionAdditionalInfo]
    publisher: list[MentionAdditionalInfo]
    url: list[MentionAdditionalInfo]
    language: list[MentionAdditionalInfo]


class Agent(ABC, ConfigurableMixin):
    model_api: API = ConfigurableSubclassFactory(API, "API to use for the model.", user_default=OllamaAPI)
    model: str = ConfigurableValue(
        "Model name to use for the API.", user_default="llama3.2:latest"
    )
    requests_options: RequestOptions = ConfigurableFactory(
        RequestOptions, "Request options for the model. Not all options are available for all APIs.",

    )

    @abstractmethod
    def __call__(self, text: str) -> list[Mention]:
        """
        Processes the input text and returns a list of software mentions.

        :param text: The input text to process.
        :return: A list of software mentions found in the text.
        """
        ...

    @abstractmethod
    def call_second_pass(self, text: str, all_document_mentions: list[Mention], text_start_offset: int) -> list[Mention]:
        """
        Calls the agent a second time to find mentions that could have been missed in the first pass.

        :param text: The input text to process.
        :param all_document_mentions: All mentions found in the document in the first pass.
        :param text_start_offset: Start offset of the text in the document.
        :return: Additional mentions found in the second pass.
        """
        ...

    @abstractmethod
    def __enter__(self):
        ...

    @abstractmethod
    def __exit__(self, exc_type, exc_val, exc_tb):
        ...
