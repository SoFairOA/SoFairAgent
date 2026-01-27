import json
from abc import ABC, abstractmethod
from typing import Optional, Literal, Union, Type, Mapping, Any

import json_repair
from classconfig import ConfigurableValue, ConfigurableMixin
from classconfig.validators import StringValidator, MinValueIntegerValidator
from ollama import ChatResponse
from openai import NotGiven, NOT_GIVEN
from openai.types.chat import ChatCompletion
from openai.types.responses import Response
from pydantic import BaseModel, Field
from ruamel.yaml.scalarstring import LiteralScalarString

from sofairagent.utils.template import Template, TemplateTransformer


class APIConfigMixin(ConfigurableMixin):
    """
    To have a consistent API configuration (one configuration file from user perspective),
    we use this mixin to define the API configuration.
    """

    api_key: str = ConfigurableValue(desc="API key.", validator=StringValidator())
    base_url: Optional[str] = ConfigurableValue(desc="Base URL for API.", user_default=None, voluntary=True)
    pool_interval: Optional[int] = ConfigurableValue(
        desc="Interval in seconds for checking the status of the batch request.",
        user_default=300,
        voluntary=True,
        validator=lambda x: x is None or x > 0)
    process_requests_interval: Optional[int] = ConfigurableValue(
        desc="Interval in seconds between sending requests when processed synchronously.",
        user_default=1,
        voluntary=True,
        validator=lambda x: x is None or x >= 0)


class APIRequestBody(ABC, BaseModel):
    """
    Represents the body of an API request.
    """
    model: str  # Model to use for the request
    messages: list[dict]  # List of messages in the request

    @property
    @abstractmethod
    def structured(self) -> bool:
        """
        Indicates whether the response is expected to be structured.
        """
        ...


class OpenAIAPIRequestBody(APIRequestBody):
    """
    Represents the body of an OpenAI API request.
    """
    type: Literal["openai"] = "openai"  # Type of the API request
    temperature: Optional[float]  # Temperature for the model
    logprobs: Optional[bool]  # Whether to return log probabilities
    max_completion_tokens: Optional[int]  # Maximum number of tokens to generate
    response_format: dict | Type[BaseModel] | None  # Format of the response, if any
    tools: Optional[list[dict]]  # Defines tools for the model, if any

    @property
    def structured(self) -> bool:
        return self.response_format is not None


class OllamaAPIRequestBody(APIRequestBody):
    """
    Represents the body of an OpenAI API request.
    """
    type: Literal["ollama"] = "ollama"  # Type of the API request
    options: dict  # Options for the model, such as temperature, max tokens, etc.
    format: Optional[dict] = None  # Format of the response, if any
    tools: Optional[list[dict]] = None  # Defines tools for the model, if any
    think: Optional[Union[bool, Literal['low', 'medium', 'high']]] = None  # Whether to enable "think" mode for the model

    @property
    def structured(self) -> bool:
        return self.format is not None


class APIRequest(BaseModel):
    """
    Represents a request to an API.
    """
    custom_id: str  # Custom ID for the request
    body: Union[OpenAIAPIRequestBody, OllamaAPIRequestBody] = Field(discriminator='type')


class FunctionCall(BaseModel):
    name: str
    arguments: Mapping[str, Any]


class APIResponse(BaseModel, ABC):
    """
    Represents the response from an API call.
    """
    structured: bool

    @abstractmethod
    def get_raw_content(self) -> str:
        """
        Returns the raw content of the response.

        """
        ...

    @abstractmethod
    def get_function_calls(self) -> list[FunctionCall]:
        """
        Returns the function calls from the response if any.

        :return: List of function call dictionaries.
        """
        ...

    @abstractmethod
    def get_reasoning_trace(self) -> Optional[str]:
        """
        Returns the reasoning trace from the response if any.

        :return: Reasoning trace string or None.
        """
        ...

    @abstractmethod
    def to_message_dict(self) -> dict:
        """
        Converts the response to a message dictionary.

        :return: Message dictionary with 'role', 'content' and other fields as needed.
        """
        ...


class APIResponseOpenAI(APIResponse):
    type: Literal["openai"] = "openai"
    body: ChatCompletion

    def get_raw_content(self) -> str:
        return self.body.choices[0].message.content

    def get_function_calls(self) -> list[FunctionCall]:
        raise NotImplementedError("Function call extraction is not implemented for OpenAI responses.")

    def get_reasoning_trace(self) -> Optional[str]:
        raise NotImplementedError("Reasoning trace extraction is not implemented for OpenAI responses.")

    def to_message_dict(self) -> dict:
        raise NotImplementedError("Conversion to message dict is not implemented for OpenAI responses.")


class APIResponseOllama(APIResponse):
    type: Literal["ollama"] = "ollama"
    body: ChatResponse

    def get_raw_content(self, choice: Optional[int] = None) -> str:
        if choice is not None:
            raise ValueError("Ollama API does not support multiple choices.")

        return self.body.message.content

    def get_function_calls(self, choice: Optional[int] = None) -> list[FunctionCall]:
        if self.body.message.tool_calls:
            return [FunctionCall(name=call.function.name, arguments=call.function.arguments) for call in self.body.message.tool_calls]
        return []

    def get_reasoning_trace(self) -> Optional[str]:
        return self.body.message.thinking if self.body.message.thinking else None

    def to_message_dict(self) -> dict:
        return self.body.message.model_dump()


class APIOutput(BaseModel):
    """
    Represents the output of an API call.
    """
    custom_id: str
    response: Optional[Union[APIResponseOpenAI, APIResponseOllama]] = Field(None, discriminator='type')
    error: Optional[str] = None


class APIBase(ABC, APIConfigMixin):
    """
    Base class for API implementations.
    """
    ...


class RequestOptions(ConfigurableMixin):
    temperature: Optional[float] = ConfigurableValue(
        desc="Temperature of the model. Controls randomness in the output.",
        user_default=1.0, voluntary=True
    )
    logprobs: Optional[bool] = ConfigurableValue(
        desc="Whether to return log probabilities of the output tokens or not. If true, returns the log probabilities of each output token returned in the content of message.",
        user_default=False, voluntary=True
    )
    max_completion_tokens: Optional[int] = ConfigurableValue(
        desc="Maximum number of tokens generated.",
        user_default=1024, voluntary=True, validator=MinValueIntegerValidator(1)
    )
    num_ctx: Optional[int] = ConfigurableValue(
        desc="Maximum number of context tokens to use.",
        user_default=2048, voluntary=True, validator=MinValueIntegerValidator(1)
    )


class StructuredResponseFormatFaker(ConfigurableMixin):
    template: Template = ConfigurableValue(
        user_default=LiteralScalarString("""{{prompt}}
Provide the response in the following JSON format. Do not include any other text.

{{json_schema}}"""),
        desc="Template to use for faking structured response format. The template should contain two placeholders: "
             "`{{prompt}}` for the original prompt and `{{json_schema}}` for the JSON schema of the response format.",
        transform=TemplateTransformer()
    )

    def __call__(self, prompt: str, model: Type[BaseModel]) -> str:
        return self.template.render({"prompt": prompt, "json_schema": json.dumps(model.model_json_schema(), indent=2)})


class RequestFactory(ABC):
    """
    Abstract factory class for creating API requests.
    """

    @abstractmethod
    def __call__(self, custom_id: str, model: str, message: list[dict[str, str]], options: RequestOptions,
                 response_format: Optional[Type[BaseModel]] = None, tools: Optional[list[dict]] = None,
                 reasoning: Optional[Union[bool, Literal['low', 'medium', 'high']]] = None,
                 fake_structured: Optional[StructuredResponseFormatFaker] = None) -> APIRequest:
        """
        Creates an API request.

        :param custom_id: Custom ID for the request.
        :param model: Model to use for the request.
        :param message: List of messages in the request. In form of a list of dictionaries with 'role' and 'content'.
        :param options: Optional options for the request.
        :param response_format: Optional format for the structured response (should be a Pydantic model).
        :param tools: Optional tools for the model to call.
            see https://github.com/ollama/ollama/blob/main/docs/api.md#chat-request-with-tools for details.
        :param reasoning: Whether to enable reasoning
        :param fake_structured: If true, the request will not use the structured response format, but it will add the
            json format to the prompt. This is useful for APIs that do not support structured response formats.
            WARNING: it is allways applied to the last message in the list.
        :return: APIRequest object.
        """
        ...


class OpenAIRequestFactory(RequestFactory):
    """
    Factory for creating OpenAI API requests.
    """

    def __call__(self, custom_id: str, model: str, message: list[dict[str, str]], options: RequestOptions,
                 response_format: Optional[Type[BaseModel]] = None, tools: Optional[list[dict]] = None,
                 reasoning: Optional[Union[bool, Literal['low', 'medium', 'high']]] = None,
                 fake_structured: Optional[StructuredResponseFormatFaker] = None) -> APIRequest:

        if fake_structured is not None:
            message[-1]['content'] = fake_structured(prompt=message[-1]['content'], model=response_format)
            response_format = None

        body = OpenAIAPIRequestBody(
            model=model,
            messages=message,
            temperature=NOT_GIVEN if options.temperature is None else options.temperature,
            logprobs=NOT_GIVEN if options.logprobs is None else options.logprobs,
            max_completion_tokens=NOT_GIVEN if options.max_completion_tokens is None else options.max_completion_tokens,
            response_format=response_format,
            tools=tools
        )
        return APIRequest(custom_id=custom_id, body=body)


class OllamaRequestFactory(RequestFactory):
    """
    Factory for creating Ollama API requests.
    """

    def __call__(self, custom_id: str, model: str, message: list[dict[str, str]], options: RequestOptions,
                 response_format: Optional[Type[BaseModel]] = None, tools: Optional[list[dict]] = None,
                 reasoning: Optional[Union[bool, Literal['low', 'medium', 'high']]] = None,
                 fake_structured: Optional[StructuredResponseFormatFaker] = None) -> APIRequest:
        req_options = {}

        if options.temperature is not None:
            req_options['temperature'] = options.temperature

        if options.max_completion_tokens is not None:
            req_options['num_predict'] = options.max_completion_tokens

        if options.num_ctx is not None:
            req_options['num_ctx'] = options.num_ctx

        if fake_structured is not None:
            message[-1]['content'] = fake_structured(prompt=message[-1]['content'], model=response_format)
            response_format = None

        body = OllamaAPIRequestBody(
            model=model,
            messages=message,
            options=req_options,
            format=response_format.model_json_schema() if response_format else None,
            tools=tools,
            think=reasoning
        )
        return APIRequest(custom_id=custom_id, body=body)

