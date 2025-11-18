import json
from contextlib import redirect_stdout
from io import StringIO
from unittest import TestCase
from unittest.mock import MagicMock

from intervaltree import IntervalTree, Interval

from sofairagent.__main__ import ExtractDatasetWorkflow
from sofairagent.agents.base import Mention, MentionAdditionalInfo


class TestExtractDatasetWorkflow(TestCase):
    def setUp(self):
        self.agent = MagicMock()
        self.workflow = ExtractDatasetWorkflow(
            agent=self.agent
        )

    def test_extract_mentions_from_text(self):
        text = "This is a test text mentioning Python and TensorFlow."
        self.agent.return_value = [
            Mention(surface_form="Python", context="mentioning Python", start_offset=31, version=None, publisher=[], url=[], language=[]),
            Mention(surface_form="TensorFlow", context="and TensorFlow.", start_offset=42, version=None, publisher=[], url=[], language=[])
        ]
        mentions = self.workflow.extract_mentions_from_text(text)
        self.assertEqual(len(mentions), 2)
        self.assertEqual(mentions[0].surface_form, "Python")
        self.assertEqual(mentions[1].surface_form, "TensorFlow")
        self.agent.assert_called_once_with(text)

    def test_extract_mentions_from_text_per_paragraph(self):
        text = "This is the first paragraph.\nThis is the second paragraph mentioning PyTorch."
        self.agent.side_effect = [
            [Mention(surface_form="first", context="the first paragraph", start_offset=12, version=None, publisher=[], url=[], language=[])],
            [Mention(surface_form="PyTorch", context="mentioning PyTorch", start_offset=67, version=None, publisher=[], url=[], language=[])]
        ]
        mentions = self.workflow.extract_mentions_from_text(text)
        self.assertEqual(len(mentions), 2)
        self.assertEqual(mentions[0].surface_form, "first")
        self.assertEqual(mentions[1].surface_form, "PyTorch")
        self.assertEqual(self.agent.call_count, 2)
        self.agent.assert_any_call("This is the first paragraph.")
        self.agent.assert_any_call("This is the second paragraph mentioning PyTorch.")

    def test_convert_mentions_to_bio(self):
        # This text mentions SoftwareX version 1.0.0 by ExampleCorp see more at https://example.com. It is written in Python.\nIt also mentions SoftwareY version 2.0.0 by SoftCorp see also more at https://soft.com. It is written in C++.
        start_offsets = [0, 5, 10, 19, 29, 37, 43, 46, 58, 62, 67, 70, 91, 94, 97, 105, 108, 116, 119, 124, 133, 143, 151, 157, 160, 169, 173, 178, 183, 186, 204, 207, 210, 218, 221]
        mentions = [
            Mention(
                surface_form="SoftwareX",
                context="mentions SoftwareX version",
                start_offset=19,
                version=MentionAdditionalInfo(
                    surface_form="1.0.0",
                    context="version 1.0.0",
                    start_offset=37
                ),
                publisher=[MentionAdditionalInfo(surface_form="ExampleCorp", context="by ExampleCorp", start_offset=46)],
                url=[MentionAdditionalInfo(surface_form="https://example.com", context="at https://example.com.", start_offset=70)],
                language=[MentionAdditionalInfo(surface_form="Python", context="written in Python.", start_offset=108)]
            ),
            Mention(
                surface_form="SoftwareY",
                context="mentions SoftwareY version",
                start_offset=133,
                version=MentionAdditionalInfo(
                    surface_form="2.0.0",
                    context="version 2.0.0",
                    start_offset=151
                ),
                publisher=[MentionAdditionalInfo(surface_form="SoftCorp", context="by SoftCorp", start_offset=160)],
                url=[MentionAdditionalInfo(surface_form="https://soft.com", context="at https://soft.com.", start_offset=186)],
                language=[MentionAdditionalInfo(surface_form="C++", context="written in C++.", start_offset=221)]
            )
        ]
        bio = self.workflow.convert_mentions_to_bio(mentions, start_offsets)
        expected_bio = [0, 0, 0, 1, 0, 3, 0, 5, 0, 0, 0, 7, 0, 0, 0, 0, 9, 0, 0, 0, 1, 0, 3, 0, 5, 0, 0, 0, 0, 7, 0, 0, 0, 0, 9]

        self.assertSequenceEqual(expected_bio, bio)

    def test_search_and_mark(self):
        labels = [0, 0, 0, 0, 0]
        token_search = IntervalTree(Interval(i*10, i*10 + 10, i) for i in range(5))
        self.workflow.search_and_mark(token_search, labels, 12, 32, "SOFTWARE")
        expected_labels = [0, 1, 2, 2, 0]
        self.assertSequenceEqual(expected_labels, labels)

    def test_mark(self):
        labels = [0, 0, 0, 0, 0]
        self.workflow.mark(labels, 1, 4, "SOFTWARE")
        expected_labels = [0, 1, 2, 2, 0]
        self.assertSequenceEqual(expected_labels, labels)

    def test_call(self):
        mentions = [
            Mention(surface_form="Python", context="mentioning Python", start_offset=31, version=None, publisher=[],
                    url=[], language=[]),
            Mention(surface_form="TensorFlow", context="mentioning TensorFlow", start_offset=29, version=None,
                    publisher=[], url=[], language=[])
        ]
        self.agent.side_effect = [[mentions[0]], [mentions[1]]]

        # capture the stdout
        res = StringIO()
        with redirect_stdout(res):
            self.workflow(
                sample_ids=["sample1", "sample2"],
                texts=["This is a test text mentioning Python.", "Another text mentioning TensorFlow."],
                start_offsets=None
            )

        output = res.getvalue().strip().split("\n")
        self.assertEqual(len(output), 2)
        record1 = json.loads(output[0])
        record2 = json.loads(output[1])

        self.assertEqual(record1["id"], "sample1")
        self.assertEqual(record2["id"], "sample2")

        self.assertEqual(len(record1["mentions"]), 1)
        self.assertEqual(mentions[0], Mention.model_validate(record1["mentions"][0]))
        self.assertEqual(len(record2["mentions"]), 1)
        self.assertEqual(mentions[1], Mention.model_validate(record2["mentions"][0]))

    def test_call_bio(self):
        self.workflow.bio_output = True
        mentions = [
            Mention(surface_form="Python", context="mentioning Python", start_offset=31, version=None, publisher=[],
                    url=[], language=[]),
            Mention(surface_form="TensorFlow", context="mentioning TensorFlow", start_offset=24, version=None,
                    publisher=[], url=[], language=[])
        ]
        self.agent.side_effect = [[mentions[0]], [mentions[1]]]

        # capture the stdout
        res = StringIO()
        with redirect_stdout(res):
            self.workflow(
                sample_ids=["sample1", "sample2"],
                texts=["This is a test text mentioning Python.", "Another text mentioning TensorFlow."],
                start_offsets=[[0, 5, 8, 10, 15, 20, 31], [0, 8, 13, 24]]
            )

        output = res.getvalue().strip().split("\n")
        self.assertEqual(len(output), 2)
        record1 = json.loads(output[0])
        record2 = json.loads(output[1])

        self.assertEqual(record1["id"], "sample1")
        self.assertEqual(record2["id"], "sample2")

        expected_bio1 = [0, 0, 0, 0, 0, 0, 1]
        expected_bio2 = [0, 0, 0, 0, 0, 1]
        self.assertEqual([0, 0, 0, 0, 0, 0, 1], expected_bio1)
        self.assertEqual([0, 0, 0, 0, 0, 1], expected_bio2)

    def test_split_text_into_paragraphs(self):
        text = "This is the first paragraph.\n\nThis is the second paragraph.\nThis is still the third paragraph.\n\nThis is the fourth paragraph."
        paragraphs = self.workflow.split_text_into_paragraphs(text)
        expected_paragraphs = [
            (0, "This is the first paragraph."),
            (30, "This is the second paragraph."),
            (60, "This is still the third paragraph."),
            (96, "This is the fourth paragraph.")
        ]

        self.assertEqual(expected_paragraphs, paragraphs)
