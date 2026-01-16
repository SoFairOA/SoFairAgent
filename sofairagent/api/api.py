import logging
import sys
import time
import traceback
from abc import abstractmethod
from collections.abc import Iterable
from typing import Generator

from classconfig import ConfigurableSubclassFactory, ConfigurableValue
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
    n_tries: int = ConfigurableValue(
        "Number of tries for each request in case of failure.",
        user_default=3,
        voluntary=True
    )

    @abstractmethod
    def _process_single_request(self, request: APIRequest) -> APIOutput:
        """
        Processes a single request.

        :param request: Request dictionary.
        :return: Processed request
        """
        ...

    def process_single_request(self, request: APIRequest) -> APIOutput:
        """
        Processes a single request.

        :param request: Request dictionary.
        :return: Processed request
        """
        exception = None
        for _attempt in range(self.n_tries):
            try:
                return self._process_single_request(request)
            except Exception as e:
                logging.info(traceback.format_exc())
                exception = e
                continue

        return APIOutput(
            custom_id=request.custom_id,
            response=None,
            error=str(exception)
        )

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

    def _process_single_request(self, request: APIRequest) -> APIOutput:
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

    @classmethod
    def get_request_factory(cls) -> RequestFactory:
        return OpenAIRequestFactory()


class OllamaAPI(API):

    def __post_init__(self):
        self.client = OllamaClient(host=self.base_url)

    def _process_single_request(self, request: APIRequest) -> APIOutput:
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

    @classmethod
    def get_request_factory(cls) -> RequestFactory:
        return OllamaRequestFactory()
