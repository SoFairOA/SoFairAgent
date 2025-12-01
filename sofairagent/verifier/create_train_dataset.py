import json
import sys
from typing import Sequence, Optional

import json_repair
from classconfig import ConfigurableFactory, ConfigurableValue, ConfigurableSubclassFactory, ConfigurableMixin, Config
from classconfig.transforms import EnumTransformer
from intervaltree import IntervalTree, Interval
from pydantic import BaseModel, parse_obj_as, TypeAdapter, ValidationError
from ruamel.yaml.scalarstring import LiteralScalarString
from tqdm import tqdm

from sofairagent.agents import Mention
from sofairagent.agents.search_agent import SearchQueryResponse
from sofairagent.api import API, OllamaAPI
from sofairagent.api.base import RequestOptions, StructuredResponseFormatFaker
from sofairagent.etl.dataset import DatasetFactory, DatasetSplit
from sofairagent.search import Searcher, SearchResult, DDGSSearcher
from sofairagent.utils.template import Template, TemplateTransformer


class CreateTrainDatasetWorkflow(ConfigurableMixin):
    original_dataset: DatasetFactory = ConfigurableFactory(DatasetFactory, desc="Original dataset that was used for extracting candidates.")
    original_dataset_split: DatasetSplit = ConfigurableValue(desc="Split of the original dataset that was used for extracting candidates.", user_default=DatasetSplit.TRAIN.name, transform=EnumTransformer(DatasetSplit))
    original_dataset_token_start_offset_field: str = ConfigurableValue(desc="Field in the original dataset that contains the list of token start offsets.", user_default="start_offsets")
    original_dataset_bio_tags_field: str = ConfigurableValue(desc="Field in the original dataset that contains the list of BIO tags.", user_default="labels")
    original_dataset_text_field: str = ConfigurableValue(desc="Field in the original dataset that contains the original text.", user_default="text")

    candidate_dataset: DatasetFactory = ConfigurableFactory(DatasetFactory, desc="Dataset with candidates to be verified. Samples must be in the same order as in the original dataset.")
    candidate_dataset_split: DatasetSplit = ConfigurableValue(desc="Split of the candidate dataset that will be used for verification.", user_default=DatasetSplit.TRAIN.name, transform=EnumTransformer(DatasetSplit))
    candidates_field: str = ConfigurableValue(desc="Field in the candidate dataset that contains the list of candidates in form of json", user_default="mentions")
    search: Searcher = ConfigurableSubclassFactory(
        Searcher,
        "Searcher that will be used to obtain additional information about candidates.",
        user_default=DDGSSearcher
    )
    model_api: API = ConfigurableSubclassFactory(API, "API to use for creating search queries", user_default=OllamaAPI)
    model: str = ConfigurableValue(
        "Model name to use for the API.",
        user_default="llama3.2:latest"
    )

    requests_options: RequestOptions = ConfigurableFactory(
        RequestOptions, "Request options for the model. Not all options are available for all APIs.",

    )
    fake_structured_format: Optional[StructuredResponseFormatFaker] = ConfigurableSubclassFactory(
        StructuredResponseFormatFaker,
        "If set, it will be used to fake the structured response format for the model.",
        user_default=None,
        voluntary=True
    )
    input_template: Template = ConfigurableValue(
        desc="Jinja2 template for the input to the model. You can use any field from the original (original_ prefix) and candidate (candidate_) datasets. Also the candidate itself is available as 'candidate' and the marked input text as 'marked_input_text'. Additionally, the boolean 'is_software_mention' is available to indicate whether the candidate is indeed a software mention according to the original dataset.",
        user_default=LiteralScalarString("""{{marked_input_text}}

Search results for the target candidate are:
{% for r in search_results %}
{{r | model_dump_json}}
{%endfor %}
"""),
        transform=TemplateTransformer()
    )
    software_bio_begin_tag: int = ConfigurableValue(
        desc="Integer value representing the beginning of a software mention in BIO tagging scheme.",
        user_default=1
    )
    software_bio_inside_tag: int = ConfigurableValue(
        desc="Integer value representing the inside of a software mention in BIO tagging scheme.",
        user_default=2
    )
    mark_open_tag: str = ConfigurableValue(
        desc="String to mark the beginning of a software mention in the text.",
        user_default=" <verify> "
    )
    mark_close_tag: str = ConfigurableValue(
        desc="String to mark the end of a software mention in the text.",
        user_default=" </verify> "
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
    per_paragraph: bool = ConfigurableValue(
        desc="Whether to create the dataset per paragraph. If False, the dataset will be created per document.",
        user_default=True
    )
    paragraph_separator: str = ConfigurableValue(
        desc="String that separates paragraphs in the text. Only used if per_paragraph is True.",
        user_default="\n"
    )
    id_field: str = ConfigurableValue(
        desc="Field name to use for the sample ID in the input.",
        user_default="id",
        voluntary=True
    )
    orig_id_field: str = ConfigurableValue(
        desc="Field name to use for the original sample ID in the output.",
        user_default="orig_id",
        voluntary=True
    )
    add_gt_candidates: bool = ConfigurableValue(
        desc="Whether to add all ground truth software mentions from the original dataset as candidates to be verified.",
        user_default=True,
        voluntary=True
    )

    context_window: int = ConfigurableValue(
        desc="Number of characters of context to include before and after the software mention when creating ground truth candidates.",
        user_default=50,
        voluntary=True
    )
    world_size: int = ConfigurableValue(
        desc="World size for distributed processing. If 1, no distributed processing is used.",
        user_default=1,
        voluntary=True
    )
    rank: int = ConfigurableValue(
        desc="Rank of the current process for distributed processing.",
        user_default=0
    )
    def __post_init__(self):
        self.original_ds = self.original_dataset.get_split(self.original_dataset_split)
        self.candidate_ds = self.candidate_dataset.get_split(self.candidate_dataset_split)
        if self.world_size > 1:
            self.candidate_ds = self.candidate_ds.shard(self.world_size, self.rank)

        self.request_factory = self.model_api.get_request_factory()

    def __enter__(self):
        self.search.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.search.__exit__(exc_type, exc_val, exc_tb)

    def obtain_software_char_intervals(self, token_start_offset: Sequence[int], bio: Sequence[int]) -> IntervalTree:
        """
        Obtains character intervals for software mentions based on BIO tags and token start offsets.

        :param token_start_offset: sequence of token start offsets
        :param bio: sequence of BIO tags for each token
        :return: IntervalTree with character intervals for software mentions
        """

        software_char_intervals = IntervalTree()
        current_software_start = None
        for i, tag in enumerate(bio):
            if tag == self.software_bio_begin_tag:
                if current_software_start is not None:
                    # Close previous software mention
                    software_char_intervals.addi(current_software_start, token_start_offset[i])
                current_software_start = token_start_offset[i]
            elif tag != self.software_bio_inside_tag:
                if current_software_start is not None:
                    # Close previous software mention
                    software_char_intervals.addi(current_software_start, token_start_offset[i])
                    current_software_start = None
        if current_software_start is not None:
            # Close last software mention if it goes till the end
            software_char_intervals.addi(current_software_start, token_start_offset[-1] + 1)
        return software_char_intervals

    def mark_orig_text(self, orig_text: str, candidate: Mention, align_offset: int = 0) -> str:
        """
        Marks the candidate surface form in the original text with specified tags.

        :param orig_text: Original text
        :param candidate: Candidate mention to be marked
        :param align_offset: Offset to align the candidate start offset with the original text (used when processing per paragraph)
        :return: Text with marked candidate surface form
        """
        before = orig_text[:candidate.start_offset + align_offset]
        after = orig_text[candidate.start_offset + len(candidate.surface_form) + align_offset:]
        return f"{before}{self.mark_open_tag}{candidate.surface_form}{self.mark_close_tag}{after}"

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
        for _ in range(3):
            try:
                raw_response = self.model_api.process_single_request(request).response.get_raw_content()
                break
            except Exception as e:
                print(f"Error obtaining search query: {e}. Retrying...", file=sys.stderr)
        else:
            print(f"Failed to obtain search query after 3 attempts. Returning surface form as query.", file=sys.stderr)
            return candidate.surface_form

        try:
            response = json_repair.loads(raw_response)
        except RecursionError:
            response = raw_response

        response = SearchQueryResponse.model_validate(response)
        return response.query.strip()

    def obtain_search_results(self, text: str, candidate: Mention) -> tuple[Optional[str], list[SearchResult]]:
        """
        Obtains search results for the given candidate.

        :param text: The input text to process.
        :param candidate: The candidate for which to obtain the search results.
        :return: The search results.
            Empty list if no query could be obtained.
        """
        try:
            query = self.obtain_search_query(text, candidate)
        except ValidationError:
            return None, []

        if not query:
            return None, []
        return query, self.search.search(query).results

    def paragraphs_intervals(self, text: str) -> IntervalTree:
        """
        Obtains character intervals for paragraphs in the text. Paragraphs are identified by new lines.

        :param text: input text
        :return: IntervalTree with character intervals for paragraphs and associated paragraph text
        """
        paragraph_intervals = IntervalTree()
        current_paragraph_start = 0
        for i, char in enumerate(text):
            if text[i:i+len(self.paragraph_separator)] == self.paragraph_separator:
                if current_paragraph_start < i:
                    paragraph_intervals.addi(current_paragraph_start, i, text[current_paragraph_start:i])
                current_paragraph_start = i + 1
        if current_paragraph_start < len(text):
            paragraph_intervals.addi(current_paragraph_start, len(text), text[current_paragraph_start:])
        return paragraph_intervals

    def __call__(self, output_path: str):
        with open(output_path, "w", encoding="utf-8") as out_f:
            # sort datasets by id to ensure they are in the same order
            sorted_original_ds = self.original_ds.sort(self.id_field)
            sorted_candidate_ds = self.candidate_ds.sort(self.id_field)

            original_ids = set(s[self.id_field] for s in sorted_original_ds)
            candidate_ids = set(s[self.id_field] for s in sorted_candidate_ds)
            if len(self.original_ds) > len(self.candidate_ds) and candidate_ids.issubset(original_ids):
                print("The candidate dataset is subset of the original dataset. Proceeding with the intersection.", file=sys.stderr)
                sorted_original_ds = sorted_original_ds.filter(lambda x: x[self.id_field] in candidate_ids)

            for orig_sample, cand_sample in tqdm(zip(sorted_original_ds, sorted_candidate_ds), total=len(self.candidate_ds), desc="Creating training dataset"):
                assert orig_sample[self.id_field] == cand_sample[self.id_field], f"Original sample ID {orig_sample['id']} does not match candidate sample ID {cand_sample['id']}"
                software_char_intervals: IntervalTree = self.obtain_software_char_intervals(orig_sample[self.original_dataset_token_start_offset_field], orig_sample[self.original_dataset_bio_tags_field])
                candidates = TypeAdapter(list[Mention]).validate_python(cand_sample[self.candidates_field])

                if self.add_gt_candidates:
                    # add all ground truth software mentions from the original dataset as candidates
                    gt_candidates = []
                    for interval in software_char_intervals:
                        surface_form = orig_sample[self.original_dataset_text_field][interval.begin:interval.end]
                        gt_candidates.append(Mention(
                            surface_form=surface_form,
                            context=orig_sample[self.original_dataset_text_field][max(0, interval.begin - self.context_window):min(len(orig_sample[self.original_dataset_text_field]), interval.end + self.context_window)],
                            start_offset=interval.begin,
                            confidence=None,
                            version=None,
                            publisher=[],
                            url=[],
                            language=[]
                        ))
                    # merge candidates
                    existing_intervals = IntervalTree(
                        Interval(cand.start_offset, cand.start_offset + len(cand.surface_form)) for cand in candidates
                    )
                    for gt_cand in gt_candidates:
                        gt_interval = (gt_cand.start_offset, gt_cand.start_offset + len(gt_cand.surface_form))
                        if not existing_intervals.overlap(gt_interval[0], gt_interval[1]):
                            candidates.append(gt_cand)

                text = orig_sample[self.original_dataset_text_field]
                paragraphs = self.paragraphs_intervals(text)
                align_offset = 0
                for cand in tqdm(candidates, desc="Processing candidates", leave=False):
                    if self.per_paragraph:
                        in_paragraph = list(paragraphs[cand.start_offset])
                        if len(in_paragraph) != 1:
                            raise ValueError(f"Candidate {cand} is not fully contained in a single paragraph. Number of containing paragraphs: {len(in_paragraph)}")
                        text = in_paragraph[0].data
                        align_offset = -in_paragraph[0].begin

                    is_software_mention = software_char_intervals.overlaps(cand.start_offset, cand.start_offset + len(cand.surface_form))
                    marked_input_text = self.mark_orig_text(text, cand, align_offset)

                    query, search_results = self.obtain_search_results(orig_sample[self.original_dataset_text_field], cand)

                    template_data = {
                        **{f"original_{k}": v for k, v in orig_sample.items()},
                        **{f"candidate_{k}": v for k, v in cand.model_dump().items()},
                        "candidate": cand,
                        "text": text,
                        "marked_input_text": marked_input_text,
                        "is_software_mention": is_software_mention,
                        "search_results": search_results,
                        "search_query": query
                    }
                    model_input = self.input_template.render(template_data)

                    res = {
                        "orig_id": orig_sample[self.orig_id_field],
                        "input": model_input,
                        "is_software_mention": is_software_mention,
                        "marked_input_text": marked_input_text,
                        "text": text,
                        "candidate": cand.model_dump_json(),
                        "search_results": [r.model_dump_json() for r in search_results],
                        "search_query": query,
                    }
                    print(json.dumps(res, ensure_ascii=False), file=out_f, flush=True)

    @classmethod
    def load(cls, path_to_config: Optional[str]) -> "CreateTrainDatasetWorkflow":
        """
        Loads this workflow from the configuration file.
        :param path_to_config: Path to the configuration file.
        """

        config = Config(cls).load(path_to_config)

        factory = ConfigurableFactory(cls)
        return factory.create(config)

