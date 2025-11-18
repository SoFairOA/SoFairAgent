from typing import Optional, Generator, Iterable

import sys
import torch
from classconfig import ConfigurableValue, ConfigurableFactory, ConfigurableSubclassFactory
from ruamel.yaml.scalarstring import LiteralScalarString
from transformers import AutoTokenizer

from sofairagent.utils.template import Template, TemplateTransformer
from sofairagent.verifier.model_factory import SequenceClassificationModelFactory
from sofairagent.verifier.tokenizer_factory import TokenizerFactory


class Verifier:
    """
    A class that identifies candidate documents for software mention extraction.
    """

    model_factory: SequenceClassificationModelFactory = ConfigurableFactory(SequenceClassificationModelFactory, "Model configuration.")
    tokenizer_factory: Optional[TokenizerFactory] = ConfigurableSubclassFactory(TokenizerFactory,
                                                                            desc="Hugging Face tokenizer for the model. Leave empty if you wish to initialize it from the model.",
                                                                            name="tokenizer",
                                                                            voluntary=True)
    threshold: Optional[float] = ConfigurableValue(
        "The threshold for the model's confidence probability. Documents with a probability below this threshold will be filtered out. By default, no threshold is applied and a class with the highest probability is selected.",
        user_default=None,
        voluntary=True
    )
    batch_size: int = ConfigurableValue("Batch size for processing documents.", user_default=8)
    input_template: Template = ConfigurableValue(user_default=LiteralScalarString("""
{{marked_input_text}}

{% if search_results | length > 0 %}
Search results for the target candidate are:
{% for r in search_results %}
{{r | model_dump_json}}
{%endfor %}
{% endif %}
"""),
                                                 desc="Jinja2 template for model input.",
                                                 transform=TemplateTransformer()
                                                 )

    def __init__(self,
                 model_factory: SequenceClassificationModelFactory,
                 tokenizer_factory: Optional[TokenizerFactory] = None,
                 threshold: Optional[float] = None,
                 batch_size: int = 32,
                 input_template: Template = Template("""
{{marked_input_text}}

{% if search_results | length > 0 %}
Search results for the target candidate are:
{% for r in search_results %}
{{r | model_dump_json}}
{%endfor %}
{% endif %}
""")):
        """
        Initializes the filter with a model and an optional threshold.
        """
        self.model_factory = model_factory
        self.tokenizer_factory = tokenizer_factory

        self.threshold = threshold
        self.batch_size = batch_size

        self.model = model_factory.create()

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_factory.model_path, use_fast=True,
                                                       cache_dir=self.model_factory.cache_dir) \
            if self.tokenizer_factory is None else self.tokenizer_factory.create()
        self.input_template = input_template

    @torch.no_grad()
    def process_batch(self, batch: Iterable[dict]) -> list[tuple[bool, float]]:
        """
        Verifies the documents based on the model's predictions.

        :param batch: A sequence of dicts with information for verification. Each dict must contain all keys used in the input_template.
        :return: A list of tuples (is_software_mention, probability).
        """
        input_strs = [self.input_template.render(p) for p in batch]
        inputs = self.tokenizer(input_strs, return_tensors="pt", padding=True, truncation=True)

        if self.model_factory.device is not None:
            inputs = {k: v.to(self.model_factory.device) for k, v in inputs.items()}

        outputs = self.model(**inputs)

        probabilities = outputs.logits.softmax(dim=-1)
        if self.threshold is None:
            # Select the class with the highest probability
            verification = probabilities.argmax(dim=-1).flatten().tolist()
        else:
            verification = (probabilities[:, 1] >= self.threshold).flatten().tolist()

        probabilities = probabilities[:, 1].flatten().tolist()
        results = []
        for v, p, s in zip(verification, probabilities, input_strs):
            print(f"Input: {s}\nIs software mention: {bool(v)}, Probability: {p}\n", flush=True, file=sys.stderr)
            results.append((bool(v), p))

        return results

    def __call__(self, verify: Iterable[dict]) -> Generator[tuple[bool, float], None, None]:
        """
        Filters software mentions based on the model's predictions.

        :param verify: A sequence of dicts with information for verification. Each dict must contain all keys used in the input_template.
        :return: A generator of tuples (is_software_mention, probability).
        """
        batch = []

        for i, p in enumerate(verify):
            batch.append(p)

            if (i + 1) % self.batch_size == 0:
                res = self.process_batch(batch)
                for r in res:
                    yield r

                batch = []

        if len(batch) > 0:
            res = self.process_batch(batch)
            for r in res:
                yield r
