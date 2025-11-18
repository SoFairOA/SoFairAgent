from abc import ABC
from typing import Optional

from classconfig import ConfigurableMixin, RelativePathTransformer, ConfigurableFactory
from classconfig import ConfigurableValue
from classconfig.validators import BoolValidator, FloatValidator, AnyValidator, ListOfTypesValidator, IsNoneValidator, \
    StringValidator
from transformers import BitsAndBytesConfig
from transformers import PreTrainedModel, AutoModel, PretrainedConfig, AutoConfig, AutoModelForTokenClassification, \
    AutoModelForSequenceClassification


class ConfigurableBitsAndBytesFactory:
    """
    Configuration factory for bits and bytes quantization.
    """

    load_in_8bit: bool = ConfigurableValue("This flag is used to enable 8-bit quantization with LLM.int8().",
                                           user_default=False,
                                           validator=BoolValidator(),
                                           voluntary=True
                                           )
    load_in_4bit: bool = ConfigurableValue(
        "This flag is used to enable 4-bit quantization by replacing the Linear layers with FP4/NF4 layers from `bitsandbytes`.",
        user_default=False,
        validator=BoolValidator(),
        voluntary=True
    )
    llm_int8_threshold: float = ConfigurableValue(
        "This corresponds to the outlier threshold for outlier detection as described in `LLM.int8() : 8-bit Matrix Multiplication for Transformers at Scale` paper: https://arxiv.org/abs/2208.07339 Any hidden states value that is above this threshold will be considered an outlier and the operation on those values will be done in fp16. Values are usually normally distributed, that is, most values are in the range [-3.5, 3.5], but there are some exceptional systematic outliers that are very differently distributed for large models. These outliers are often in the interval [-60, -6] or [6, 60]. Int8 quantization works well for values of magnitude ~5, but beyond that, there is a significant performance penalty. A good default threshold is 6, but a lower threshold might be needed for more unstable models (small models, fine-tuning).",
        user_default=6.0,
        validator=FloatValidator(),
        voluntary=True
    )
    llm_int8_skip_modules: Optional[list[str]] = ConfigurableValue(
        "An explicit list of the modules that we do not want to convert in 8-bit. This is useful for models such as Jukebox that has several heads in different places and not necessarily at the last position. For example for `CausalLM` models, the last `lm_head` is kept in its original `dtype`.",
        validator=AnyValidator([ListOfTypesValidator(str), IsNoneValidator()]),
        voluntary=True
    )
    llm_int8_enable_fp32_cpu_offload: bool = ConfigurableValue(
        "This flag is used for advanced use cases and users that are aware of this feature. If you want to split your model in different parts and run some parts in int8 on GPU and some parts in fp32 on CPU, you can use this flag. This is useful for offloading large models such as `google/flan-t5-xxl`. Note that the int8 operations will not be run on CPU.",
        user_default=False,
        validator=BoolValidator(),
        voluntary=True
    )
    llm_int8_has_fp16_weight: bool = ConfigurableValue(
        "This flag runs LLM.int8() with 16-bit main weights. This is useful for fine-tuning as the weights do not have to be converted back and forth for the backward pass.",
        user_default=False,
        validator=BoolValidator(),
        voluntary=True)
    bnb_4bit_compute_dtype: Optional[str] = ConfigurableValue(
        "This sets the computational type which might be different than the input type. For example, inputs might be fp32, but computation can be set to bf16 for speedups.",
        user_default=None,
        voluntary=True,
        validator=AnyValidator([StringValidator(), IsNoneValidator()])
    )
    bnb_4bit_quant_type: str = ConfigurableValue(
        "This sets the quantization data type in the bnb.nn.Linear4Bit layers. Options are FP4 and NF4 data types which are specified by `fp4` or `nf4`.",
        user_default="fp4",
        voluntary=True,
        validator=StringValidator()
    )
    bnb_4bit_use_double_quant: bool = ConfigurableValue(
        "This flag is used for nested quantization where the quantization constants from the first quantization are quantized again.",
        user_default=False,
        validator=BoolValidator(),
        voluntary=True
    )
    bnb_4bit_quant_storage: Optional[str] = ConfigurableValue(
        "This sets the storage type to pack the quanitzed 4-bit prarams.",
        user_default=None,
        voluntary=True,
        validator=AnyValidator([StringValidator(), IsNoneValidator()])
    )

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def create(self) -> BitsAndBytesConfig:
        return BitsAndBytesConfig(**self.kwargs)


class HFModelFactory(ConfigurableMixin, ABC):
    """
    Base factory for creating HF models.
    """
    LOADER = AutoModel
    CONFIG_LOADER = AutoConfig

    model_path: str = ConfigurableValue("Name or path to the model.",
                                        transform=RelativePathTransformer(force_relative_prefix=True))

    attn_implementation: Optional[str] = ConfigurableValue(
        'The attention implementation to use in the model (if relevant). Can be any of "eager" (manual implementation of the attention), "sdpa" (using F.scaled_dot_product_attention), or "flash_attention_2" (using Dao-AILab/flash-attention). By default, if available, SDPA will be used for torch>=2.1.1. The default is otherwise the manual "eager" implementation.',
        user_default="flash_attention_2",
        voluntary=True)

    cache_dir: Optional[str] = ConfigurableValue("Path to Hugging Face cache directory.", user_default=None,
                                                 voluntary=True)

    quantization: Optional[ConfigurableBitsAndBytesFactory] = ConfigurableFactory(ConfigurableBitsAndBytesFactory,
                                                                                  "Configuration for bits and bytes quantization.",
                                                                                  voluntary=True)
    torch_dtype: Optional[str] = ConfigurableValue(
        "Override the default torch.dtype and load the model under a specific dtype",
        user_default=None,
        voluntary=True)
    trust_remote_code: Optional[bool] = ConfigurableValue(
        "Whether to trust remote code.",
        user_default=False,
        voluntary=True
    )
    device: Optional[str] = ConfigurableValue(
        "Which device to use for the model. If not specified, the model will be loaded on the CPU.",
        user_default="cpu",
        voluntary=True
    )

    config: Optional[PretrainedConfig] = ConfigurableValue("Configuration for the model.", user_default=None,
                                                           voluntary=True)

    def __init__(self, **kwargs):
        """
        Initializes the factory.

        :param kwargs: Factory configuration.
        """
        ConfigurableMixin.__init__(self, **kwargs)
        self.was_quantized = False

    def get_model_args(self, **kwargs) -> dict:
        """
        Returns arguments for model creation.

        :param kwargs: Additional arguments that will be used to update the model arguments obtained from the factory configuration.
        :return: model arguments.
        """
        additional_args_for_model = {}
        if isinstance(self.quantization, ConfigurableBitsAndBytesFactory):
            q = self.quantization.create()
            self.was_quantized = q.load_in_4bit or q.load_in_8bit
            if self.was_quantized:
                additional_args_for_model["quantization_config"] = q

        if self.torch_dtype is not None:
            additional_args_for_model["dtype"] = self.torch_dtype

        if self.attn_implementation is not None:
            additional_args_for_model["attn_implementation"] = self.attn_implementation

        if self.cache_dir is not None:
            additional_args_for_model["cache_dir"] = self.cache_dir

        if self.trust_remote_code is not None:
            additional_args_for_model["trust_remote_code"] = self.trust_remote_code

        additional_args_for_model.update(kwargs)
        return additional_args_for_model

    def create(self, **kwargs) -> PreTrainedModel:
        """
        Creates the model.

        :return: Model.
        """

        if self.config is not None:
            all_args = self.get_model_args(**kwargs)
            all_args.update(self.config)
            config, unused_kwargs = self.CONFIG_LOADER.from_pretrained(
                self.model_path,
                return_unused_kwargs=True,
                **all_args
            )
            model = self.LOADER.from_pretrained(self.model_path, config=config, **unused_kwargs)
        else:
            model = self.LOADER.from_pretrained(self.model_path, **self.get_model_args(**kwargs))

        if self.device is not None and not self.was_quantized:
            model.to(self.device)
        return model


class TokenClassificationModelFactory(HFModelFactory):
    """
    Factory for transformers token classification model.
    """
    LOADER = AutoModelForTokenClassification
    labels: Optional[list[str]] = ConfigurableValue(
        desc="Classification labels, the position is specifying label id. Leave empty for automatic detection of labels from dataset or using labels from model configuration.",
        validator=AnyValidator([IsNoneValidator(), ListOfTypesValidator(str, allow_empty=True)])
    )

    def get_model_args(self, **kwargs) -> dict:
        """
        Returns arguments for model creation.

        :param kwargs: Additional arguments that will be used to update the model arguments obtained from the factory configuration.
        :return: model arguments.
        """

        labels = self.labels

        if labels:  # if labels are specified, update the model arguments
            kwargs["num_labels"] = len(labels)
            kwargs["id2label"] = {i: label for i, label in enumerate(labels)}
            kwargs["label2id"] = {label: i for i, label in enumerate(labels)}
        additional_args_for_model = super().get_model_args(**kwargs)
        return additional_args_for_model

    def create(self, **kwargs) -> PreTrainedModel:
        """
        Creates the model.

        :return: initialized model
        :raises ValueError: if labels are not specified and dataset is not provided.
        """

        return super().create(**kwargs)


class SequenceClassificationModelFactory(TokenClassificationModelFactory):
    """
    Factory for transformers token classification model.
    """
    LOADER = AutoModelForSequenceClassification

