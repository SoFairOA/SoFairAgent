import random
import sqlite3
import time
from abc import ABC, abstractmethod
from os import PathLike
from typing import Optional
from classconfig import ConfigurableValue, RelativePathTransformer, ConfigurableSubclassFactory, ConfigurableMixin
from ddgs import DDGS
from ddgs.exceptions import DDGSException
from pydantic import BaseModel


class SearchResult(BaseModel):
    """
    Represents a search result.
    """
    title: str
    link: str
    snippet: str


class SearchOutput(BaseModel):
    """
    Represents a collection of search results.
    """
    results: list[SearchResult]


class SearchCache(ABC):
    """
    Base class for search caching functionality.
    """

    @abstractmethod
    def __len__(self):
        """
        Returns the number of items in the cache.
        """
        ...

    @abstractmethod
    def __getitem__(self, query: str) -> SearchOutput:
        """
        Retrieves search results from the cache for the given query.
        :param query: The search query.
        :return: A list of search results.
        :raises KeyError: If no search results were found for the given query.
        """
        ...

    @abstractmethod
    def __setitem__(self, query: str, search_output: SearchOutput):
        """
        Stores search results in the cache for the given query.
        :param query: The search query.
        :param search_output: The search results.
        """
        ...

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

class NoSearchCache(SearchCache):
    """
    Implements a no-op cache that does not store any search results.
    """

    def __len__(self):
        return 0

    def __getitem__(self, query: str) -> SearchOutput:
        raise KeyError(f"No search results found for query: {query}")

    def __setitem__(self, query: str, search_output: SearchOutput):
        pass

class LFUInMemorySearchCache(SearchCache):
    """
    Implements the Least Frequently Used (LFU) in-memory cache for search results.
    """

    max_size: int = ConfigurableValue(
        "Maximum size of the cache. If exceeded, the least frequently used item will be removed. Use 0 for unlimited size.",
        user_default=1000,
        voluntary=True,
        validator=lambda x: x >= 0
    )

    def __init__(self, max_size: int = 1000):
        """
        Initializes the LFUInMemorySearchCache with an optional maximum size.
        """
        self.cache = {}
        self.frequency = {}
        self.max_size = max_size

    def __len__(self):
        return len(self.cache)

    def __getitem__(self, query: str) -> SearchOutput:
        if query not in self.cache:
            raise KeyError(f"No search results found for query: {query}")
        self.frequency[query] += 1
        return self.cache[query]

    def __setitem__(self, query: str, search_output: SearchOutput):
        self.cache[query] = search_output
        self.frequency[query] = self.frequency.get(query, 0) + 1

        if len(self.cache) > self.max_size > 0:
            # Remove the least frequently used item
            lfu_query = min(self.frequency, key=self.frequency.get)
            del self.cache[lfu_query]
            del self.frequency[lfu_query]


class SQLiteSearchCache(SearchCache):
    """
    Implements a SQLite-based search cache.
    """

    db_path: str = ConfigurableValue(
        "Path to the SQLite database file for caching search results.",
        transform=RelativePathTransformer()
    )

    def __init__(self, db_path: str | PathLike[str]):
        """
        Initializes the SQLiteSearchCache with the specified database path.
        """

        self.db_path = db_path
        self.connection = None
        self.cursor = None

    def __len__(self):
        self.cursor.execute("SELECT COUNT(*) FROM search_cache")
        return self.cursor.fetchone()[0]

    def __getitem__(self, query: str) -> SearchOutput:
        self.cursor.execute("SELECT results FROM search_cache WHERE query = ?", (query,))
        row = self.cursor.fetchone()
        if row is None:
            raise KeyError(f"No search results found for query: {query}")
        results = row[0]
        return SearchOutput.model_validate_json(results)

    def __setitem__(self, query: str, search_output: SearchOutput):
        results_json = search_output.model_dump_json()
        self.cursor.execute(
            "INSERT OR REPLACE INTO search_cache (query, results) VALUES (?, ?)",
            (query, results_json)
        )
        self.connection.commit()

    def __enter__(self):
        if self.connection is not None:
            return self
        self.connection = sqlite3.connect(self.db_path)
        self.cursor = self.connection.cursor()
        self.cursor.execute(
            "CREATE TABLE IF NOT EXISTS search_cache (query TEXT PRIMARY KEY, results TEXT)"
        )
        self.connection.commit()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        if self.connection:
            self.connection.close()
            self.connection = None
            self.cursor = None


class Searcher(ABC, ConfigurableMixin):
    """
    Base class for search functionality.
    """

    cache: SearchCache = ConfigurableSubclassFactory(
        SearchCache,
        "Cache for search results.",
        user_default=SQLiteSearchCache
    )
    max_results: int = ConfigurableValue(
        "Maximum number of results to return for a search query.",
        user_default=3,
        voluntary=True,
        validator=lambda x: x > 0
    )
    wait_between_requests: int = ConfigurableValue(
        "Time in seconds to wait between requests to avoid rate limiting.",
        user_default=0,
        voluntary=True,
        validator=lambda x: x >= 0
    )

    def __post_init__(self):
        self.last_request_time = 0

    def search(self, query: str) -> SearchOutput:
        """
        Searches for the given query and returns a list of results.

        :param query: The search query.
        :return: A list of search results.
        """

        try:
            results = self.cache[query]
        except KeyError:
            results = self.no_cache_search(query)
            self.cache[query] = results
        return results

    def _wait_guard(self):
        """
        Ensures that the time between requests is respected.
        If the last request was made less than wait_between_requests seconds ago, it waits.
        """
        if self.wait_between_requests > 0:
            elapsed_time = time.time() - self.last_request_time
            if elapsed_time < self.wait_between_requests:
                time.sleep(self.wait_between_requests - elapsed_time)
        self.last_request_time = time.time()

    def no_cache_search(self, query: str) -> SearchOutput:
        """
        Performs a search without caching results.

        :param query: The search query.
        :return: A list of search results.
        """
        self._wait_guard()
        return self._no_cache_search(query)

    @abstractmethod
    def _no_cache_search(self, query: str) -> SearchOutput:
        """
        Abstract method to perform a search without caching results.
        This method should be implemented by subclasses.

        :param query: The search query.
        :return: A list of search results.
        """
        ...

    def __enter__(self):
        self.cache.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cache.__exit__(exc_type, exc_val, exc_tb)


class DDGSSearcher(Searcher):
    """
    Implements a searcher using DDGS library.
    """
    backend: str = ConfigurableValue(
        "The search engine/ backend to use for DDGS search. You can use comma separated backends to randomly choose from them. It will provide all backends to ddgs, but in random order.",
        user_default="auto",
        voluntary=True,
    )

    def __post_init__(self):
        super().__post_init__()
        self.searcher = DDGS()
        self.backend = [x.strip() for x in self.backend.split(",")] if "," in self.backend else self.backend

    def _no_cache_search(self, query: str) -> SearchOutput:
        """
        Performs a search using DuckDuckGo.

        :param query: The search query.
        :return: A list of search results.
        """

        try:
            backend = self.backend if isinstance(self.backend, str) else ", ".join(random.sample(self.backend, len(self.backend)))
            results = self.searcher.text(query, max_results=self.max_results, backend=backend)
        except DDGSException:
            results = []

        return SearchOutput(
            results=[
                SearchResult(
                    title=result.get("title", ""),
                    link=result.get("href", ""),
                    snippet=result.get("body", "")
                )
                for result in results
            ]
        )