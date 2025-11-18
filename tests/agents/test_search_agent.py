from re import search
from unittest import TestCase
from unittest.mock import MagicMock

from sofairagent.agents.search_agent import SearchAgent, Candidate, AdditionalInfo, URLRepairResponse
from sofairagent.api.base import RequestOptions, FunctionCall
from sofairagent.search import SearchOutput, SearchResult
from sofairagent.software_database import SoftwareDatabase


class TestSearchAgent(TestCase):
    def setUp(self):
        self.searcher = MagicMock()
        self.software_database = MagicMock()
        self.agent = SearchAgent(
            search=self.searcher,
            requests_options=RequestOptions(),
            software_database=self.software_database,
        )
        self.maxDiff = None

    def test_convert_candidate_to_mention(self):
        text = "This is a test text mentioning SoftwareX version 1.2.3 by ExampleCorp. More info at https://example.com. It is written in Python."
        candidate = Candidate(
            id=1,
            surface_form="SoftwareX",
            context="mentioning SoftwareX version",
            version=AdditionalInfo(
                surface_form="1.2.3",
                context="version 1.2.3 by"
            ),
            publisher=[
                AdditionalInfo(
                    surface_form="ExampleCorp",
                    context="by ExampleCorp."
                )
            ],
            url=[
                AdditionalInfo(
                    surface_form="https://example.com",
                    context="More info at https://example.com."
                )
            ],
            language=[
                AdditionalInfo(
                    surface_form="Python",
                    context="It is written in Python."
                )
            ]
        )

        mention = self.agent.convert_candidate_to_mention(text, candidate)

        self.assertEqual("SoftwareX", mention.surface_form)
        self.assertEqual("mentioning SoftwareX version", mention.context)
        self.assertEqual(search("SoftwareX", text).start(), mention.start_offset)
        self.assertEqual("1.2.3", mention.version.surface_form)
        self.assertEqual("version 1.2.3 by", mention.version.context)
        self.assertEqual(search("1.2.3", text).start(), mention.version.start_offset)
        self.assertEqual(1, len(mention.publisher))
        self.assertEqual("ExampleCorp", mention.publisher[0].surface_form)
        self.assertEqual("by ExampleCorp.", mention.publisher[0].context)
        self.assertEqual(search("ExampleCorp", text).start(), mention.publisher[0].start_offset)
        self.assertEqual(1, len(mention.url))
        self.assertEqual("https://example.com", mention.url[0].surface_form)
        self.assertEqual("More info at https://example.com.", mention.url[0].context)
        self.assertEqual(search("https://example.com", text).start(), mention.url[0].start_offset)
        self.assertEqual(1, len(mention.language))
        self.assertEqual("Python", mention.language[0].surface_form)
        self.assertEqual("It is written in Python.", mention.language[0].context)
        self.assertEqual(search("Python", text).start(), mention.language[0].start_offset)

    def test_convert_candidates_to_mentions(self):
        text = "This is a test text mentioning SoftwareX and SoftwareY."
        candidates = [
            Candidate(
                id=1,
                surface_form="SoftwareX",
                context="mentioning SoftwareX and",
                version=None,
                publisher=[],
                url=[],
                language=[]
            ),
            Candidate(
                id=2,
                surface_form="SoftwareY",
                context="and SoftwareY.",
                version=None,
                publisher=[],
                url=[],
                language=[]
            )
        ]

        mentions = self.agent.convert_candidates_to_mentions(text, candidates)

        self.assertEqual(2, len(mentions))

        self.assertEqual("SoftwareX", mentions[0].surface_form)
        self.assertEqual("mentioning SoftwareX and", mentions[0].context)
        self.assertEqual(search("SoftwareX", text).start(), mentions[0].start_offset)

        self.assertEqual("SoftwareY", mentions[1].surface_form)
        self.assertEqual("and SoftwareY.", mentions[1].context)
        self.assertEqual(search("SoftwareY", text).start(), mentions[1].start_offset)

    def test_software_surface_form_ok(self):
        candidate = Candidate(
            id=1,
            surface_form="SoftwareX",
            context="mentioning SoftwareX.",
            version=AdditionalInfo(
                surface_form="1.2.3",
                context="version 1.2.3 by"
            ),
            publisher=[
                AdditionalInfo(
                    surface_form="ExampleCorp",
                    context="by ExampleCorp."
                )
            ],
            url=[
                AdditionalInfo(
                    surface_form="https://example.com",
                    context="More info at https://example.com."
                )
            ],
            language=[
                AdditionalInfo(
                    surface_form="Python",
                    context="It is written in Python."
                )
            ]
        )

        flag = self.agent.repair_surface_form_not_in_context("This is a test text mentioning SoftwareX version 1.2.3 by ExampleCorp. More info at https://example.com. It is written in Python.", [candidate])

        self.assertFalse(flag)

    def test_software_surface_form_not_in_context(self):
        candidate = Candidate(
            id=1,
            surface_form="SoftwareX",
            context="mentioning version",
            version=None,
            publisher=[],
            url=[],
            language=[]
        )

        self.agent.send_repair_problem_request = MagicMock()
        flag = self.agent.repair_surface_form_not_in_context("This is a test text mentioning SoftwareX version 1.2.3 by ExampleCorp. More info at https://example.com. It is written in Python.", [candidate])
        self.assertTrue(flag)
        self.agent.send_repair_problem_request.assert_called_once()
        self.assertSequenceEqual(
            ["Surface form 'SoftwareX' not in context 'mentioning version'. Take on mind that the search is case-sensitive."],
            self.agent.send_repair_problem_request.call_args[0][-1])

    def test_version_surface_form_version_not_in_context(self):
        candidate = Candidate(
            id=1,
            surface_form="SoftwareX",
            context="mentioning SoftwareX version",
            version=AdditionalInfo(
                surface_form="1.2.3",
                context="by ExampleCorp"
            ),
            publisher=[],
            url=[],
            language=[]
        )

        self.agent.send_repair_problem_request = MagicMock()
        flag = self.agent.repair_surface_form_not_in_context("This is a test text mentioning SoftwareX version 1.2.3 by ExampleCorp. More info at https://example.com. It is written in Python.", [candidate])
        self.assertTrue(flag)
        self.agent.send_repair_problem_request.assert_called_once()
        self.assertSequenceEqual(
            ["Version surface form '1.2.3' not in version context 'by ExampleCorp'. Take on mind that the search is case-sensitive."],
            self.agent.send_repair_problem_request.call_args[0][-1])

    def test_publisher_surface_form_publisher_not_in_context(self):
        candidate = Candidate(
            id=1,
            surface_form="SoftwareX",
            context="mentioning SoftwareX version",
            version=None,
            publisher=[
                AdditionalInfo(
                    surface_form="ExampleCorp",
                    context="by"
                )
            ],
            url=[],
            language=[]
        )

        self.agent.send_repair_problem_request = MagicMock()
        flag = self.agent.repair_surface_form_not_in_context("This is a test text mentioning SoftwareX version 1.2.3 by ExampleCorp. More info at https://example.com. It is written in Python.", [candidate])
        self.assertTrue(flag)
        self.agent.send_repair_problem_request.assert_called_once()
        self.assertSequenceEqual(
            ["Publisher surface form 'ExampleCorp' not in publisher context 'by'. Take on mind that the search is case-sensitive."],
            self.agent.send_repair_problem_request.call_args[0][-1])

    def test_url_surface_form_url_not_in_context(self):
        candidate = Candidate(
            id=1,
            surface_form="SoftwareX",
            context="mentioning SoftwareX version",
            version=None,
            publisher=[],
            url=[
                AdditionalInfo(
                    surface_form="https://example.com",
                    context="More info at"
                )
            ],
            language=[]
        )

        self.agent.send_repair_problem_request = MagicMock()
        flag = self.agent.repair_surface_form_not_in_context("This is a test text mentioning SoftwareX version 1.2.3 by ExampleCorp. More info at https://example.com. It is written in Python.", [candidate])
        self.assertTrue(flag)
        self.agent.send_repair_problem_request.assert_called_once()
        self.assertSequenceEqual(["URL surface form 'https://example.com' not in URL context 'More info at'. Take on mind that the search is case-sensitive."], self.agent.send_repair_problem_request.call_args[0][-1])

    def test_language_surface_form_language_not_in_context(self):
        candidate = Candidate(
            id=1,
            surface_form="SoftwareX",
            context="mentioning SoftwareX version",
            version=None,
            publisher=[],
            url=[],
            language=[
                AdditionalInfo(
                    surface_form="Python",
                    context="It is written in"
                )
            ]
        )

        self.agent.send_repair_problem_request = MagicMock()
        flag = self.agent.repair_surface_form_not_in_context("This is a test text mentioning SoftwareX version 1.2.3 by ExampleCorp. More info at https://example.com. It is written in Python.", [candidate])
        self.assertTrue(flag)
        self.agent.send_repair_problem_request.assert_called_once()
        self.assertSequenceEqual(
            ["Language surface form 'Python' not in language context 'It is written in'. Take on mind that the search is case-sensitive."],
            self.agent.send_repair_problem_request.call_args[0][-1])

    def test_same_context_candidates_eok(self):
        candidates = [
                Candidate(
                    id=1,
                    surface_form="SoftwareX",
                    context="mentioning SoftwareX.",
                    version=AdditionalInfo(
                        surface_form="1.2.3",
                        context="version 1.2.3 by"
                    ),
                    publisher=[
                        AdditionalInfo(
                            surface_form="ExampleCorp",
                            context="by ExampleCorp."
                        )
                    ],
                    url=[
                        AdditionalInfo(
                            surface_form="https://example.com",
                            context="More info at https://example.com."
                        )
                    ],
                    language=[
                        AdditionalInfo(
                            surface_form="Python",
                            context="It is written in Python."
                        )
                    ]
                ),
                Candidate(
                    id=2,
                    surface_form="BestSoft",
                    context="mentioning BestSoft.",
                    version=AdditionalInfo(
                        surface_form="2.3.4",
                        context="version 2.3.4 by"
                    ),
                    publisher=[
                        AdditionalInfo(
                            surface_form="BestCorp",
                            context="by BestCorp."
                        )
                    ],
                    url=[
                        AdditionalInfo(
                            surface_form="https://best.com",
                            context="More info at https://best.com."
                        )
                    ],
                    language=[
                        AdditionalInfo(
                            surface_form="Java",
                            context="It is written in Java."
                        )
                    ]
                )
            ]

        problems = self.agent.same_context_candidates(candidates)
        self.assertEqual(0, len(problems))

    def test_same_context_candidates_all_problems(self):
        candidates = [
                Candidate(
                    id=1,
                    surface_form="SoftwareX",
                    context="mentioning SoftwareX.",
                    version=AdditionalInfo(
                        surface_form="1.2.3",
                        context="version 1.2.3 by"
                    ),
                    publisher=[
                        AdditionalInfo(
                            surface_form="ExampleCorp",
                            context="by ExampleCorp."
                        )
                    ],
                    url=[
                        AdditionalInfo(
                            surface_form="https://example.com",
                            context="More info at https://example.com."
                        )
                    ],
                    language=[
                        AdditionalInfo(
                            surface_form="Python",
                            context="It is written in Python."
                        )
                    ]
                ),
                Candidate(
                    id=2,
                    surface_form="SoftwareX",
                    context="mentioning SoftwareX.",
                    version=AdditionalInfo(
                        surface_form="1.2.3",
                        context="version 1.2.3 by"
                    ),
                    publisher=[
                        AdditionalInfo(
                            surface_form="ExampleCorp",
                            context="by ExampleCorp."
                        )
                    ],
                    url=[
                        AdditionalInfo(
                            surface_form="https://example.com",
                            context="More info at https://example.com."
                        )
                    ],
                    language=[
                        AdditionalInfo(
                            surface_form="Python",
                            context="It is written in Python."
                        )
                    ]
                ),
            ]

        problems = self.agent.same_context_candidates(candidates)
        self.assertEqual(5, len(problems))
        self.assertDictEqual({
            ("SoftwareX", "mentioning SoftwareX."): [(candidates[0], "software name"), (candidates[1], "software name")],
            ("1.2.3", "version 1.2.3 by"): [(candidates[0], "version"), (candidates[1], "version")],
            ("ExampleCorp", "by ExampleCorp."): [(candidates[0], "publisher"), (candidates[1], "publisher")],
            ("https://example.com", "More info at https://example.com."): [(candidates[0], "url"), (candidates[1], "url")],
            ("Python", "It is written in Python."): [(candidates[0], "language"), (candidates[1], "language")],
        }, problems)

    def test_same_context_candidates_some_problems(self):
        candidates = [
                Candidate(
                    id=1,
                    surface_form="SoftwareX",
                    context="mentioning SoftwareX.",
                    version=AdditionalInfo(
                        surface_form="1.2.3",
                        context="version 1.2.3 by"
                    ),
                    publisher=[
                        AdditionalInfo(
                            surface_form="Java",
                            context="It is Java."
                        )
                    ],
                    url=[
                        AdditionalInfo(
                            surface_form="https://example.com",
                            context="More info at https://example.com."
                        )
                    ],
                    language=[
                        AdditionalInfo(
                            surface_form="Python",
                            context="It is written in Python."
                        )
                    ]
                ),
                Candidate(
                    id=2,
                    surface_form="BestSoft",
                    context="mentioning BestSoft.",
                    version=AdditionalInfo(
                        surface_form="1.2.3",
                        context="version 1.2.3 by"
                    ),
                    publisher=[
                        AdditionalInfo(
                            surface_form="BestCorp",
                            context="by BestCorp."
                        )
                    ],
                    url=[
                        AdditionalInfo(
                            surface_form="https://best.com",
                            context="More info at https://best.com."
                        )
                    ],
                    language=[
                        AdditionalInfo(
                            surface_form="Java",
                            context="It is Java."
                        )
                    ]
                ),
            ]

        problems = self.agent.same_context_candidates(candidates)
        self.assertEqual(2, len(problems))
        self.assertDictEqual({
            ("1.2.3", "version 1.2.3 by"): [(candidates[0], "version"), (candidates[1], "version")],
            ("Java", "It is Java."): [(candidates[0], "publisher"), (candidates[1], "language")],
        }, problems)

    def test_ambiguous_contexts_candidates_eok(self):
        text = "This text mentions SoftwareX version 1.2.3 by ExampleCorp see more at https://example.com. It is written in Python. It also mentions SoftwareX version 2.0.0 by ExampleCorp see also more at https://example.com. It is also written in Python."
        candidates = [
                Candidate(
                    id=1,
                    surface_form="SoftwareX",
                    context="mentions SoftwareX.",
                    version=AdditionalInfo(
                        surface_form="1.2.3",
                        context="version 1.2.3 by"
                    ),
                    publisher=[
                        AdditionalInfo(
                            surface_form="ExampleCorp",
                            context="1.2.3 by ExampleCorp."
                        )
                    ],
                    url=[
                        AdditionalInfo(
                            surface_form="https://example.com",
                            context="see more at https://example.com."
                        )
                    ],
                    language=[
                        AdditionalInfo(
                            surface_form="Python",
                            context="It is written in Python."
                        )
                    ]
                ),
                Candidate(
                    id=2,
                    surface_form="SoftwareX",
                    context="also mentions SoftwareX.",
                    version=AdditionalInfo(
                        surface_form="2.0.0",
                        context="version 2.0.0 by"
                    ),
                    publisher=[
                        AdditionalInfo(
                            surface_form="ExampleCorp",
                            context="2.0.0 by ExampleCorp."
                        )
                    ],
                    url=[
                        AdditionalInfo(
                            surface_form="https://example.com",
                            context="see also more at https://example.com."
                        )
                    ],
                    language=[
                        AdditionalInfo(
                            surface_form="Python",
                            context="It is also written in Python."
                        )
                    ]
                )
            ]

        problems = self.agent.ambiguous_contexts_candidates(text, candidates)
        self.assertEqual(0, len(problems))

    def test_ambiguous_contexts_candidates_problems(self):
        text = "This text mentions SoftwareX version 1.0.0 by ExampleCorp see more at https://example.com. It is written in Python. It also mentions SoftwareX version 2.0.0 by ExampleCorp see also more at https://example.com. It is also written in Python."
        candidates = [
            Candidate(
                id=1,
                surface_form="SoftwareX",
                context="mentions SoftwareX",
                version=AdditionalInfo(
                    surface_form="0.0",
                    context="0.0 by"
                ),
                publisher=[
                    AdditionalInfo(
                        surface_form="ExampleCorp",
                        context="by ExampleCorp"
                    )
                ],
                url=[
                    AdditionalInfo(
                        surface_form="https://example.com",
                        context="more at https://example.com."
                    )
                ],
                language=[
                    AdditionalInfo(
                        surface_form="Python",
                        context="written in Python."
                    )
                ]
            ),
            Candidate(
                id=2,
                surface_form="SoftwareX",
                context="mentions SoftwareX",
                version=AdditionalInfo(
                    surface_form="0.0",
                    context="0.0 by"
                ),
                publisher=[
                    AdditionalInfo(
                        surface_form="ExampleCorp",
                        context="by ExampleCorp"
                    )
                ],
                url=[
                    AdditionalInfo(
                        surface_form="https://example.com",
                        context="more at https://example.com."
                    )
                ],
                language=[
                    AdditionalInfo(
                        surface_form="Python",
                        context="written in Python."
                    )
                ]
            )
        ]

        problems = self.agent.ambiguous_contexts_candidates(text, candidates)
        self.assertEqual(5, len(problems))
        self.assertSequenceEqual(
            (
                [
                    (candidates[0], "software name"), (candidates[1], "software name")
                ],
                [10, 124]
            ),
            problems["mentions SoftwareX"]
        )
        self.assertSequenceEqual(
            (
                [
                    (candidates[0], "version"), (candidates[1], "version")
                ],
                [39, 153]
            ),
            problems["0.0 by"]
        )
        self.assertSequenceEqual(
            (
                [
                    (candidates[0], "publisher"), (candidates[1], "publisher")
                ],
                [43, 157]
            ),
            problems["by ExampleCorp"]
        )
        self.assertSequenceEqual(
            (
                [
                    (candidates[0], "url"), (candidates[1], "url")
                ],
                [62, 181]
            ),
            problems["more at https://example.com."]
        )
        self.assertSequenceEqual(
            (
                [
                    (candidates[0], "language"), (candidates[1], "language")
                ],
                [97, 221]
            ),
            problems["written in Python."]
        )

    def test_eval_function_calls(self):
        self.searcher.search.return_value = SearchOutput(
            results=[
                SearchResult(
                    title="Title 1",
                    link="https://example.com",
                    snippet="Snipppet 1"
                )
            ]
        )
        function_calls = [
            FunctionCall(
                name="search_engine",
                arguments={
                    "query": "Search query"
                }
            ),
            FunctionCall(
                name="unknown_function",
                arguments={
                    "arg1": "dummy"
                }
            )
        ]

        res = self.agent.eval_function_calls(function_calls)

        self.searcher.search.assert_called_once_with(query="Search query")
        self.assertSequenceEqual(
            [
                (False, self.searcher.search.return_value.model_dump_json()),
                (True, "Unknown function call: unknown_function")
            ],
            res
        )

    def test_repair_url(self):
        self.agent.model_api = MagicMock()
        self.assertEqual("https://github.com/example/example", self.agent.repair_url("Please see our github page at https://github.com/example/example for more info.", "github page at https://github.com/example/example", "at https://github.com/example/example for more"))
        self.agent.model_api.process_single_request.assert_not_called()
        self.assertEqual("https://example.com", self.agent.repair_url("More info at https://example.com.", "example.com", "at https://example.com."))
        self.agent.model_api.process_single_request.assert_not_called()

        self.agent.model_api.process_single_request = MagicMock()
        self.agent.model_api.process_single_request.return_value.response.get_raw_content.return_value = URLRepairResponse(repaired_url="https://example.com").model_dump_json()

        self.assertEqual("https://example.com", self.agent.repair_url("More info at https://example.com.", "://example.com.", "at https://example.com."))
        self.agent.model_api.process_single_request.assert_called_once()

