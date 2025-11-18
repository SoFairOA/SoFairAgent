import sys
import time
import traceback
from abc import abstractmethod
from collections.abc import Iterable
from typing import Generator

from classconfig import ConfigurableSubclassFactory
from ollama import Client as OllamaClient
from openai import OpenAI, APIError, RateLimitError

from sofairagent.api import APIOutput, APIResponseOpenAI, APIResponseOllama
from sofairagent.api.base import APIBase, APIRequest, RequestFactory, OllamaRequestFactory, OpenAIRequestFactory
from sofairagent.api.logger import APILogger, APISQLiteLogger


class API(APIBase):
    """
    Handles requests to the API.
    """
    logger: APILogger | None = ConfigurableSubclassFactory(
        APILogger,
        "Logger for API requests and responses.",
        voluntary=True,
        user_default=APISQLiteLogger
    )

    @abstractmethod
    def process_single_request(self, request: APIRequest) -> APIOutput:
        """
        Processes a single request.

        :param request: Request dictionary.
        :return: Processed request
        """
        ...

    def process_requests(self, requests: Iterable[APIRequest]) -> Generator[APIOutput, None, None]:
        """
        Processes a list of requests.

        :param requests: Iterable of request dictionaries.
        :return: Processed requests
        """
        for i, request in enumerate(requests):
            if i > 0 and self.process_requests_interval > 0:
                time.sleep(self.process_requests_interval)
            yield self.process_single_request(request)

    @classmethod
    @abstractmethod
    def get_request_factory(cls) -> RequestFactory:
        """
        Returns the request factory for this API.

        :return: Request factory class.
        """
        ...


class OpenAPI(API):
    """
    Handles requests to the OpenAI API.
    """

    def __post_init__(self):
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def process_single_request(self, request: APIRequest) -> APIOutput:
        try:
            while True:
                try:
                    response = self.client.responses.create(**request.body.model_dump(exclude={"type"}))
                    break
                except RateLimitError:
                    print(f"Rate limit reached. Waiting for {self.pool_interval} seconds.", flush=True,
                          file=sys.stderr)
                    time.sleep(self.pool_interval)

            return APIOutput(
                custom_id=request.custom_id,
                response=APIResponseOpenAI(
                    body=response.model_dump(),
                    structured=request.body.structured
                ),
                error=None
            )
        except APIError as e:
            return APIOutput(
                custom_id=request.custom_id,
                response=None,
                error=str(e)
            )

    @classmethod
    def get_request_factory(cls) -> RequestFactory:
        return OpenAIRequestFactory()


class OllamaAPI(API):

    def __post_init__(self):
        self.client = OllamaClient(host=self.base_url)

    def process_single_request(self, request: APIRequest) -> APIOutput:
        try:
            if self.logger is not None:
                self.logger.log_request(request)
            response = self.client.chat(**request.body.model_dump(exclude={"type"}))
            res = APIOutput(
                custom_id=request.custom_id,
                response=APIResponseOllama(
                    body=response,
                    structured=request.body.structured
                ),
                error=None
            )
            if self.logger is not None:
                self.logger.log_response(res)
            return res
        except Exception as e:
            print(traceback.format_exc(), file=sys.stderr, flush=True)
            return APIOutput(
                custom_id=request.custom_id,
                response=None,
                error=str(e)
            )

    @classmethod
    def get_request_factory(cls) -> RequestFactory:
        return OllamaRequestFactory()
