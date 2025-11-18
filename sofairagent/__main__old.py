import argparse
import copy
import json
import logging
import math
import random
import sys
from argparse import ArgumentParser
from typing import Sequence, Optional

from classconfig import Config, ConfigurableFactory, ConfigurableMixin, ConfigurableSubclassFactory, ConfigurableValue, \
    YAML
from intervaltree import IntervalTree, Interval
from tqdm import tqdm
from whitespacetokenizer import whitespace_tokenizer

from sofairagent.agents import Agent, Mention, SearchAgent
from sofairagent.etl.dataset import DatasetFactory
from sofairagent.verifier.create_train_dataset import CreateTrainDatasetWorkflow


class ExtractDatasetWorkflow(ConfigurableMixin):
    """
    Extracts software mentions from text.
    """

    agent: Agent = ConfigurableSubclassFactory(Agent, "Agent for extraction.", user_default=SearchAgent)
    per_paragraph: bool = ConfigurableValue(
        "Whether to extract mentions per paragraph or from the whole text at once.", user_default=True
    )
    bio_dict: Optional[dict[str, int]] = ConfigurableValue(
        "Dictionary mapping BIO tags to integers. Required if bio_output is True.",
        user_default={
            "O": 0,
            "B-SOFTWARE": 1,
            "I-SOFTWARE": 2,
            "B-VERSION": 3,
            "I-VERSION": 4,
            "B-PUBLISHER": 5,
            "I-PUBLISHER": 6,
            "B-URL": 7,
            "I-URL": 8,
            "B-LANGUAGE": 9,
            "I-LANGUAGE": 10
        },
        voluntary=True
    )

    def __enter__(self):
        self.agent.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.agent.__exit__(exc_type, exc_val, exc_tb)

    def __call__(self, sample_ids: Sequence[str], texts: Sequence[str],
                 bio_output: bool = False, start_offsets: Optional[Sequence[Sequence[int]]] = None):
        """
        Runs the workflow.

        :param sample_ids: ID of the sample.
        :param texts: Text from which to extract software mentions.
        :param bio_output: If True, outputs BIO tags for each token in the text.
        :param start_offsets: Start offsets of tokens in the text. If not provided, whitespace tokenization is used.
        :raises ValueError: If bio_output is True and start_offsets is not provided.
        """
        if bio_output and start_offsets is None:
            raise ValueError("Start offsets must be provided if bio_output is True.")

        for i, (sample_id, text) in tqdm(enumerate(zip(sample_ids, texts)), total=len(texts),
                                         desc="Extracting mentions"):
            try:
                mentions = self.extract_mentions_from_text(text)
            except Exception as e:
                print(f"Failed to extract mentions from sample {sample_id}: {e}", flush=True, file=sys.stderr)
                mentions = []

            res = {
                "id": sample_id,
                "mentions": [m.model_dump() for m in mentions]
            }
            if bio_output:
                if start_offsets is None:
                    res["start_offsets"] = [s for _, s, _ in whitespace_tokenizer(text)]
                else:
                    res["start_offsets"] = start_offsets[i]

                labels = self.convert_mentions_to_bio(mentions, res["start_offsets"])
                res["labels"] = labels

            print(json.dumps(res, ensure_ascii=False), flush=True)

    @staticmethod
    def split_text_into_paragraphs(text: str) -> list[tuple[int, str]]:
        """
        Splits the text into paragraphs and returns a list of tuples with the start offset and the paragraph text.

        :param text: Text to split.
        :return: List of tuples with the start offset and the paragraph text.
        """

        paragraphs = []
        offset = 0
        for paragraph in text.split("\n"):
            if len(paragraph) > 0:
                paragraphs.append((offset, paragraph))
            offset += len(paragraph) + 1

        return paragraphs

    @staticmethod
    def align_offsets(start_offset: int, mentions: list[Mention]):
        """
        Shifts the offsets of the mentions by the given text start offset to fit to the whole document.

        :param start_offset: Start offset of the text from which the mentions were extracted.
        :param mentions: List of mentions to shift. The mentions are modified in-place.
        """
        for mention in mentions:
            mention.start_offset += start_offset
            if mention.version:
                mention.version.start_offset += start_offset
            for publisher in mention.publisher:
                publisher.start_offset += start_offset
            for url in mention.url:
                url.start_offset += start_offset
            for language in mention.language:
                language.start_offset += start_offset

    def extract_mentions_from_text(self, text: str) -> list:
        """
        Extracts software mentions from the given text.

        :param text: Text from which to extract software mentions.
        :return: List of extracted software mentions.
        """

        mentions = []
        if self.per_paragraph:
            paragraphs = self.split_text_into_paragraphs(text)

            # First pass
            for (start_offset, paragraph) in paragraphs:
                try:
                    m = self.agent(paragraph)
                except Exception as e:
                    print(f"WARNING: Failed to extract mentions from paragraph starting at offset {start_offset}: {e}",
                          flush=True, file=sys.stderr)
                    continue
                self.align_offsets(start_offset, m)
                mentions.extend(m)

            if len(mentions) > 0:
                # Second pass
                for (start_offset, paragraph) in paragraphs:
                    try:
                        m = self.agent.call_second_pass(paragraph, copy.deepcopy(mentions), start_offset)
                    except Exception as e:
                        print(f"WARNING: During second pass failed to extract mentions from paragraph starting at offset {start_offset}: {e}",
                              flush=True, file=sys.stderr)
                        continue
                    self.align_offsets(start_offset, m)
                    mentions.extend(m)
        else:
            mentions = self.agent(text)

        return mentions

    def convert_mentions_to_bio(self, mentions: list[Mention], start_offsets: Sequence[int]) -> list[int]:
        """
        Converts the extracted mentions to BIO tags.

        :param mentions: List of extracted software mentions.
        :param start_offsets: Start offsets of tokens in the text.
        :return: List of BIO tags for each token in sequence
        """

        labels = [self.bio_dict["O"]] * len(start_offsets)
        token_search = IntervalTree(
            Interval(s, start_offsets[i + 1] if i + 1 < len(start_offsets) else math.inf, i) for i, s in
            enumerate(start_offsets))

        for m in mentions:
            self.search_and_mark(token_search, labels, m.start_offset, m.start_offset + len(m.surface_form), "SOFTWARE")

            if m.version:
                self.search_and_mark(token_search, labels, m.version.start_offset,
                                     m.version.start_offset + len(m.version.surface_form), "VERSION")
            for publisher in m.publisher:
                self.search_and_mark(token_search, labels, publisher.start_offset,
                                     publisher.start_offset + len(publisher.surface_form), "PUBLISHER")
            for url in m.url:
                self.search_and_mark(token_search, labels, url.start_offset, url.start_offset + len(url.surface_form),
                                     "URL")
            for language in m.language:
                self.search_and_mark(token_search, labels, language.start_offset,
                                     language.start_offset + len(language.surface_form), "LANGUAGE")

        return labels

    def search_and_mark(self, token_search: IntervalTree, labels: list[int], start_offset: int, end_offset: int,
                        t: str):
        """
        Searches for the tokens overlapping with the given character offsets and marks them with the given BIO tags.

        :param token_search: IntervalTree with token start and end offsets and their indices
        :param labels: sequence of labels that will be modified in-place
        :param start_offset: character start offset
        :param end_offset: character end offset (exclusive)
        :param t: type of the entity USE UPPERCASE (e.g., SOFTWARE, VERSION, URL, etc.)
        :raises ValueError: If token_search is invalid
        """
        intervals = sorted(token_search.overlap(start_offset, end_offset), key=lambda x: x[-1])
        if not intervals:
            logging.warning(f"Could not find tokens for entity {t} with character offsets {start_offset}-{end_offset}.")
            return

        token_start_offset = intervals[0][-1]
        token_end_offset = intervals[-1][-1] + 1  # exclusive
        self.mark(labels, token_start_offset, token_end_offset, t)

    def mark(self, labels: list[int], start_offset: int, end_offset: int, t: str):
        """
        Marks the tokens with the given BIO tags.

        :param labels: sequence of labels that will be modified in-place
        :param start_offset: token start offset
        :param end_offset: token end offset (exclusive)
        :param t: type of the entity USE UPPERCASE (e.g., SOFTWARE, VERSION, URL, etc.)
        """
        labels[start_offset] = self.bio_dict[f"B-{t}"]
        for i in range(start_offset + 1, end_offset):
            labels[i] = self.bio_dict[f"I-{t}"]

    @classmethod
    def load(cls, path_to_config: Optional[str]) -> "ExtractDatasetWorkflow":
        """
        Loads this workflow from the configuration file.
        :param path_to_config: Path to the configuration file.
        """

        config = Config(cls).load(path_to_config)

        factory = ConfigurableFactory(cls)
        return factory.create(config)


def extract_dataset(args):
    """
    Extracts software mentions from dataset.

    :param args: Parsed arguments.
    """

    with ExtractDatasetWorkflow.load(args.config) as workflow:
        dataset = DatasetFactory(
            path=args.dataset,
            name=args.subset,
        ).create(args.split)
        workflow(
            sample_ids=dataset[args.id_field],
            texts=dataset[args.text_field],
            bio_output=args.bio,
            start_offsets=dataset[args.start_offsets] if args.start_offsets in dataset.column_names else None
        )


def test_extract(args: argparse.Namespace):
    """
    Method for testing extraction on custom text.

    :param args: User arguments.
    """
    with ExtractDatasetWorkflow.load(args.config) as workflow:
        print("Enter text (empty line to finish):")
        lines = []
        while True:
            line = input()
            if line == "":
                break
            lines.append(line)
        text = "\n".join(lines)

        print("Extracting mentions...")
        mentions = workflow.extract_mentions_from_text(text)
        print("Extracted mentions:")
        for m in mentions:
            print(m.model_dump_json(indent=2))


def create_train_dataset_verifier(args: argparse.Namespace):
    with CreateTrainDatasetWorkflow.load(args.config) as workflow:
        workflow(args.output)


def split_verifier_dataset(args: argparse.Namespace):
    assert args.val_size + args.test_size < 1.0, "Validation and test size must be less than 1.0"
    assert args.val_size > 0.0, "Validation size must be greater than 0.0"
    assert args.test_size > 0.0, "Test size must be greater than 0.0"

    with open(args.dataset) as f, open(args.train, 'w') as ftrain, open(args.validation, 'w') as fval, open(args.test, 'w') as ftest:
        orig_ids = []
        for line in f:
            record = json.loads(line)
            orig_ids.append(record["orig_id"])

        unique_orig_ids = list(set(orig_ids))
        unique_orig_ids.sort()
        random.seed(args.seed)
        random.shuffle(unique_orig_ids)
        n = len(unique_orig_ids)
        val_cutoff = int(n * args.val_size)
        test_cutoff = int(n * (args.val_size + args.test_size))
        val_ids = set(unique_orig_ids[:val_cutoff])
        test_ids = set(unique_orig_ids[val_cutoff:test_cutoff])
        train_ids = set(unique_orig_ids[test_cutoff:])

        f.seek(0)
        for line in f:
            line = line.strip()
            record = json.loads(line)
            if record["orig_id"] in train_ids:
                print(line, file=ftrain)
            elif record["orig_id"] in val_ids:
                print(line, file=fval)
            elif record["orig_id"] in test_ids:
                print(line, file=ftest)
            else:
                raise ValueError(f"orig_id {record['orig_id']} not found in any split")


def create_config(args: argparse.Namespace):
    """
    Method for generating configuration for workflow.

    :param args: User arguments.
    """
    config = Config(ExtractDatasetWorkflow)
    print(YAML().dumps(config.generate_yaml_config()))


def create_config_create_train_dataset_verifier(args: argparse.Namespace):
    """
    Method for generating configuration for workflow.

    :param args: User arguments.
    """
    config = Config(CreateTrainDatasetWorkflow)
    print(YAML().dumps(config.generate_yaml_config()))


def main():
    logging.basicConfig(format='%(process)d: %(levelname)s : %(asctime)s : %(message)s', level=logging.WARNING)

    parser = ArgumentParser(description="Agentic LLM for software mention extraction.")
    subparsers = parser.add_subparsers()

    extract_dataset_parser = subparsers.add_parser("extract_dataset", help="Extracts software mentions from text.")
    extract_dataset_parser.add_argument("dataset",
                                        help="Name/path of Hugging Face dataset.",
                                        default="SoFairOA/sofair_dataset")
    extract_dataset_parser.add_argument("-i", "--id_field",
                                        help="Field name with sample ids in the dataset.",
                                        default="id")
    extract_dataset_parser.add_argument("-t", "--text_field",
                                        help="Field name with text in the dataset. The text should be plain text with single paragraphs per line.",
                                        default="text")
    extract_dataset_parser.add_argument("-o", "--start_offsets",
                                        help="Field name with start_offsets of tokens in the text. If not provided the BIO output will be generated for whitespace tokenization.",
                                        default="start_offsets")
    extract_dataset_parser.add_argument("-u", "--subset",
                                        help="Subset of the dataset to use. If not provided, the default one is used",
                                        default=None)
    extract_dataset_parser.add_argument("-s", "--split", help="Split of the gold dataset.", default="test")
    extract_dataset_parser.add_argument("-c", "--config", help="Path to the configuration file.")
    extract_dataset_parser.add_argument("-b", "--bio", help="If set, outputs BIO tags for each token in the text.",
                                        action="store_true")
    extract_dataset_parser.set_defaults(func=extract_dataset)

    test_extract_parser = subparsers.add_parser("test_extract", help="Allows to test extraction on custom text.")
    test_extract_parser.add_argument("-c", "--config", help="Path to the configuration file.")
    test_extract_parser.set_defaults(func=test_extract)

    train_dataset_verifier_parser = subparsers.add_parser("create_train_dataset_verifier", help="Creates a dataset for training a verifier model.")
    train_dataset_verifier_parser.add_argument("config", help="Path to the configuration file.")
    train_dataset_verifier_parser.add_argument("output", help="Path to save the dataset.")
    train_dataset_verifier_parser.set_defaults(func=create_train_dataset_verifier)

    split_verifier_dataset_parser = subparsers.add_parser("split_verifier_dataset",
                                                          help="Splits a dataset for training a verifier model into train, validation, and test sets. The split is done in a way that documents are not mixed between the sets according to orig_id.")
    split_verifier_dataset_parser.add_argument("dataset", help="Path to the dataset to split.")
    split_verifier_dataset_parser.add_argument("train", help="Path to save the training set.")
    split_verifier_dataset_parser.add_argument("validation", help="Path to save the validation set.")
    split_verifier_dataset_parser.add_argument("test", help="Path to save the test set.")
    split_verifier_dataset_parser.add_argument("--val_size", type=float, default=0.1, help="Fraction of the dataset to use for validation.")
    split_verifier_dataset_parser.add_argument("--test_size", type=float, default=0.1, help="Fraction of the dataset to use for testing.")
    split_verifier_dataset_parser.add_argument("-s", "--seed", type=int, default=42, help="Random seed for shuffling the documents before splitting.")
    split_verifier_dataset_parser.set_defaults(func=split_verifier_dataset)

    create_config_parser = subparsers.add_parser("create_config", help="Creates configuration.")
    create_config_parser.set_defaults(func=create_config)

    create_config_create_train_dataset_verifier_parser = subparsers.add_parser("create_config_create_train_dataset_verifier", help="Creates configuration for create_train_dataset_verifier workflow.")
    create_config_create_train_dataset_verifier_parser.set_defaults(func=create_config_create_train_dataset_verifier)

    args = parser.parse_args()

    if args is not None:
        args.func(args)
    else:
        exit(1)


if __name__ == '__main__':
    main()
