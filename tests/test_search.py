import shutil
from pathlib import Path
from unittest import TestCase
from unittest.mock import MagicMock

from cryptography.hazmat.backends.openssl import backend

from sofairagent.search import LFUInMemorySearchCache, SQLiteSearchCache, SearchOutput, SearchResult, DDGSSearcher

SCRIPT_DIR = Path(__file__).parent
TMP_DIR = SCRIPT_DIR / "tmp"


class TestLFUInMemorySearchCache(TestCase):
    def setUp(self):
        self.cache = LFUInMemorySearchCache(max_size=3)

    def test__len__(self):
        self.assertEqual(len(self.cache), 0)
        self.cache['query1'] = SearchOutput(results=[SearchResult(title="title1", link="url1", snippet="snippet1")])
        self.assertEqual(len(self.cache), 1)
        self.cache['query2'] = SearchOutput(results=[SearchResult(title="title2", link="url2", snippet="snippet2")])
        self.assertEqual(len(self.cache), 2)
        self.cache['query3'] = SearchOutput(results=[SearchResult(title="title3", link="url3", snippet="snippet3")])
        self.assertEqual(len(self.cache), 3)
        self.cache['query4'] = SearchOutput(results=[SearchResult(title="title4", link="url4", snippet="snippet4")])  # This should evict the least frequently used item
        self.assertEqual(len(self.cache), 3)

    def test_eviction_policy(self):
        self.cache['query1'] = SearchOutput(results=[SearchResult(title="title1", link="url1", snippet="snippet1")])
        self.cache['query2'] = SearchOutput(results=[SearchResult(title="title2", link="url2", snippet="snippet2")])
        self.cache['query3'] = SearchOutput(results=[SearchResult(title="title3", link="url3", snippet="snippet3")])

        # Access query1 twice, query2 once, query3 not at all
        _ = self.cache['query1']
        _ = self.cache['query1']
        _ = self.cache['query2']

        # Adding a new item should evict query3 (least frequently used)
        self.cache['query4'] = SearchOutput(results=[SearchResult(title="title4", link="url4", snippet="snippet4")])

        with self.assertRaises(KeyError):
            _ = self.cache['query3']

        # Ensure other items are still accessible
        self.assertEqual(self.cache['query1'], SearchOutput(results=[SearchResult(title="title1", link="url1", snippet="snippet1")]))
        self.assertEqual(self.cache['query2'], SearchOutput(results=[SearchResult(title="title2", link="url2", snippet="snippet2")]))
        self.assertEqual(self.cache['query4'], SearchOutput(results=[SearchResult(title="title4", link="url4", snippet="snippet4")]))


DB_PATH = TMP_DIR / "test.db"


def create_tmp():
    if TMP_DIR.exists():
        shutil.rmtree(TMP_DIR)
    TMP_DIR.mkdir(parents=True, exist_ok=True)


class TestSQLiteSearchCache(TestCase):
    def setUp(self):
        create_tmp()
        self.cache = SQLiteSearchCache(DB_PATH)
        self.cache.__enter__()

    def tearDown(self):
        self.cache.__exit__(None, None, None)
        if TMP_DIR.exists():
            shutil.rmtree(TMP_DIR)

    def test__len__(self):
        self.assertEqual(len(self.cache), 0)
        self.cache['query1'] = SearchOutput(results=[SearchResult(title="title1", link="url1", snippet="snippet1")])
        self.assertEqual(len(self.cache), 1)
        self.cache['query2'] = SearchOutput(results=[SearchResult(title="title2", link="url2", snippet="snippet2")])
        self.assertEqual(len(self.cache), 2)
        self.cache['query3'] = SearchOutput(results=[SearchResult(title="title3", link="url3", snippet="snippet3")])
        self.assertEqual(len(self.cache), 3)

    def test_set_and_get_item(self):
        self.cache['query1'] = SearchOutput(results=[SearchResult(title="title1", link="url1", snippet="snippet1")])
        self.cache['query2'] = SearchOutput(results=[SearchResult(title="title2", link="url2", snippet="snippet2")])
        self.cache['query3'] = SearchOutput(results=[SearchResult(title="title3", link="url3", snippet="snippet3")])

        self.assertEqual(self.cache['query1'], SearchOutput(results=[SearchResult(title="title1", link="url1", snippet="snippet1")]))
        self.assertEqual(self.cache['query2'], SearchOutput(results=[SearchResult(title="title2", link="url2", snippet="snippet2")]))
        self.assertEqual(self.cache['query3'], SearchOutput(results=[SearchResult(title="title3", link="url3", snippet="snippet3")]))

        with self.assertRaises(KeyError):
            _ = self.cache['nonexistent_query']


class TestDDGSSearch(TestCase):
    def setUp(self):
        create_tmp()
        self.search = DDGSSearcher(
            cache=SQLiteSearchCache(DB_PATH),
            max_results=3,
            wait_between_requests=1,
            backend="duckduckgo"
        )
        self.mock_duckduckgo = MagicMock()
        self.search.searcher = self.mock_duckduckgo

        self.search.__enter__()

    def tearDown(self):
        self.search.__exit__(None, None, None)
        if TMP_DIR.exists():
            shutil.rmtree(TMP_DIR)

    def test_search(self):
        mock_results = [
            {"title": "title1", "href": "url1", "body": "snippet1"},
            {"title": "title2", "href": "url2", "body": "snippet2"},
            {"title": "title3", "href": "url3", "body": "snippet3"}
        ]
        self.mock_duckduckgo.text.return_value = mock_results

        query = "test query"
        results = self.search.search(query)

        self.assertEqual(results, SearchOutput(results=[
            SearchResult(title="title1", link="url1", snippet="snippet1"),
            SearchResult(title="title2", link="url2", snippet="snippet2"),
            SearchResult(title="title3", link="url3", snippet="snippet3")
        ]))

        # the next call should hit the cache
        self.mock_duckduckgo.text.assert_called_once_with(query, max_results=3, backend="duckduckgo")

        results = self.search.search(query)
        self.assertEqual(results, SearchOutput(results=[
            SearchResult(title="title1", link="url1", snippet="snippet1"),
            SearchResult(title="title2", link="url2", snippet="snippet2"),
            SearchResult(title="title3", link="url3", snippet="snippet3")
        ]))

        self.mock_duckduckgo.text.assert_called_once_with(query, max_results=3, backend="duckduckgo")

