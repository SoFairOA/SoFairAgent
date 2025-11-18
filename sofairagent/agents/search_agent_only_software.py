import json
import re
import sys

import validators
from collections import defaultdict
from typing import Optional, Union, Literal

import json_repair
from classconfig import ConfigurableSubclassFactory, ConfigurableValue, ConfigurableFactory
from intervaltree import IntervalTree
from pydantic import BaseModel, Field, ValidationError
from ruamel.yaml.scalarstring import LiteralScalarString

from sofairagent.agents.base import Agent, Mention, MentionAdditionalInfo
from sofairagent.api.base import FunctionCall, StructuredResponseFormatFaker
from sofairagent.search import DDGSSearcher, Searcher
from sofairagent.software_database import SoftwareDatabase
from sofairagent.utils.template import Template, TemplateTransformer


class AdditionalInfo(BaseModel):
    surface_form: str = Field(description="Exact form of the mention in the text")
    context: str = Field(description="Context of the mention exactly as it appears in the text.")


class Candidate(BaseModel):
    """
    Represents a candidate software mention extracted from the text.
    """
    id: int | str = Field(
        description="Unique identifier of the candidate. It can be just a number starting from 0 and incremented by 1 for each candidate")
    surface_form: str = Field(description="Mention name exactly as it appears in the text")
    context: str = Field(description="Context of the mention exactly as it appears in the text. ")


class FindCandidatesResponse(BaseModel):
    """
    Represents the response from the find candidates request.
    """
    candidates: list[Candidate] = Field([], description="List of extracted software mentions candidates.")


class IncludeCheckList(BaseModel):
    """
    Represents a checklist of verification points for software to be included.
    """
    commercial_and_general_purpose_software: bool = Field(
        description="Whether the software is commercial and general-purpose software, e.g. Excel, Photoshop, CorelDraw.")
    research_software: bool = Field(description="Whether the software is developed in academic or research context.")
    software_environment: bool = Field(
        description="Whether the software is a programming environment, e.g. R, Matlab, SAS.")
    named_software_component_package: bool = Field(description="e.g. R package pROC, Python package NetworkX.")
    implicit_software_mentions: bool = Field(
        description="Generic words like “program”, “script”, “code”, “package” if they clearly indicate executable software.")
    workflow: bool = Field(
        description="Whether the software is a high-level specifications: in data-intensive scientific domains, the complexity of data processing has led to the common definiton and usage of workflows associated to a scientific experiments, e.g. Galaxy, Kepler, Apache Taverna.")
    api: bool = Field(description="When referred to as executable/shared software.")
    operating_system: bool = Field(
        description="(Windows, Linux, macOS) if the OS itself is being referenced (not just “running on Windows”).")
    devices_with_embedded_software: bool = Field(description="If the mention clearly refers to the software part.")


class ExcludeCheckList(BaseModel):
    """
    Represents a checklist of verification points for software to be excluded.
    """
    algorithm: bool = Field(
        description="Whether the mention is an algorithm. Unless the context shows their implementation was run as software.")
    model: bool = Field(
        description="Whether the mention is only a model (machine learning models, simulation models). Unless the mention refers to the software implementing the model (e.g. BERT library vs. BERT model)")
    database: bool = Field(
        description="Whether the mention is only a database (if it is not clear that the reference is made to the software part). Unless explicitly mentioned as software tools (compiler, interpreter, IDE).")
    programming_language: bool = Field(
        description="Whether the mention is only a programming language (e.g. written in BASIC, FORTRAN, etc.).")
    operating_system: bool = Field(
        description="Whether the mention is only an operating system when only used as an attribute of other software (e.g. “SPSS for Windows” → OS not annotated).")
    bibliographic_references: bool = Field(
        description="Whether the mention is only a bibliographic reference to software (e.g. “R Development Core Team, 2020” in references).")


class AmbiguousCheckList(BaseModel):
    """
    Represents a checklist of verification points for ambiguous software mentions.
    """
    algorithm_vs_software: bool = Field(
        description="Whether the mention is an algorithm name used to refer to the software implementing the algorithm. If so, it should be annotated as software.")
    model_vs_software: bool = Field(
        description="Whether the mention is a model name used to refer to the software implementing/running the model. If so, it should be annotated as software.")
    database_vs_software: bool = Field(
        description="Whether the mention is a database name used to refer to the software providing access to the database. If so, it should be annotated as software.")
    device_vs_software: bool = Field(
        description="Whether the mention is a device name used to refer to the software part of the device. If so, it should be annotated as software.")


class VerifyCandidateResponse(BaseModel):
    """
    Represents the response from the verify candidate request.
    """
    include_check_list: IncludeCheckList = Field(
        description="Checklist of verification points for software to be included.")
    exclude_check_list: ExcludeCheckList = Field(
        description="Checklist of verification points for software to be excluded.")
    ambiguous_check_list: AmbiguousCheckList = Field(
        description="Checklist of verification points for ambiguous software mentions.")
    reason: str = Field(description="Reason for the verification decision.")
    is_software: bool = Field(description="Whether the candidate was verified as a software mention or not.")
    verified: Optional[Candidate] = Field(
        description="Verified software mention candidate. Is null if the candidate was not verified as a software mention.")


class VerifyMentionResponse(BaseModel):
    """
    Represents the response from the verify candidate request.
    """
    include_check_list: IncludeCheckList = Field(
        description="Checklist of verification points for software to be included.")
    exclude_check_list: ExcludeCheckList = Field(
        description="Checklist of verification points for software to be excluded.")
    ambiguous_check_list: AmbiguousCheckList = Field(
        description="Checklist of verification points for ambiguous software mentions.")
    reason: str = Field(description="Reason for the verification decision.")
    is_software: bool = Field(description="Whether the candidate was verified as a software mention or not.")
    verified: Optional[Mention] = Field(
        description="Verified software mention. Is null if the mention was not verified as a software mention.")


class URLRepairResponse(BaseModel):
    """
    Represents the response from the URL repair request.
    """
    repaired_url: str = Field(description="Repaired URL. It can be an empty string if the URL could not be repaired.")


class SearchQueryResponse(BaseModel):
    """
    Represents a LLM response for creating a search query.
    """
    query: str = Field(description="Search query to use for searching the software.")


guidelines = """      
# Definition/scope for software annotation

## 1) Annotating all software, not just "research software"

We consider mentions of software in scientific literature without limitation to "research software". We found the notion of "research software" unclear. 

From the point of view of software sharing, "research software" is usually understood as software produced by researchers or by research software engineers. However, mainstream commercial software are very broadly used in science and mentioned in scholar papers when describing research activities. Examples of very commonly mentioned general purposes and mainstream software are Excel, Photoshop or PostgresQL. Such general software can also be the object of a research study. So, from the point of view of software citation, any software mentioned in scholar literature is relevant - they are "software of interest" for research and should be annotated.  

## 2) What should be considered as a "software" entity?

Software products correspond in practice to various artefacts, which are not always clear to consider as "software". This is particularly challenging from the point of view of software citation, but this remains an issue even when identifying software sharing. 

A standard definition of software is "a collection of computer programs that provides the instructions for telling a computer what to do and how to do it" (Wikipedia). Everything that can provide processing instructions to a computer, whatever its form, can therefore be seen as software. This relatively broad definition of software covers a large variety of software products, for instance from macro and formula part of an Excel sheet to large software project with multiple components, hundred thousand lines of source code and binary packages for a variety of computing environments. 

Any of these software products have a potential research interest, for reuse or reproducibility purposes, and could be therefore valuable to share. Monitoring software in research supposes to be able to identify any mentions of a software product independently from the scale of the software production and independently from its form. 

The types/formats of software depend a lot on the technical domain and the used programing framework. 

**We propose to cover the notion of software in general independently from any particular distribution forms.** 

- **Software** products typically can be published as standalone applications or libraries/plugins, either as executable code (binaries), package (e.g. R package, combining script and binaries), as a more comprehensive open source project (program, script, data resources, documentation, build scripts, etc.), script program or macro to be interpreted and exectuted within a particular software environment, source code that require manual building, small standalone script (e.g. "gist"), plaform (including data, data management software and service software), web services, images to be executed as containers, or software embedded in an hardware device.

All these software distribution formats are considered as software to be annotated for the present annotations guidelines. 

- **Algorithm** versus software: as a general guideline, algorithm mention are not considered as software mention and are not be annotated. However, it is quite frequent that the name of an algorithm and its implementation (so as software) are used in papers in an interchangeable manner. While it is clear that we want to exclude "algorithm names" from software entities, they can be used to refer to the implementation. This is one of the most frequent ambiguity we have identified in the Softcite dataset and this was similarly reported for the [SoMeSci dataset](https://doi.org/10.1145/3459637.3482017). The distinction could sometime be done in context, but a guideline is necessary when the usage of the name is general and ambiguous on purpose. 

Examples: [10.1038/ng.2007.13](https://pubmed.ncbi.nlm.nih.gov/17952075/)

```
  Finally, we applied the EIGENSTRAT method [46], which relies on patterns of correlation 
  between individuals to detect stratification, to our Icelandic discovery sample. 
```

*EIGENSTRAT* here is the name of the method and of the software implementing the method. As the context describes the application of the method of the algorithm on actual data, it refers to the use of the software implementation and it should therefore be annotated as a software mention.

[10.1038/bjc.2016.25](https://www.nature.com/articles/bjc201625)

```
  Messenger RNA expression was normalised to household gene expression (GAPDH and RPL13A 
  for BON-1; HPRT and YWAZ for QGP-1) according to the geNorm algorithm (Mestdagh et 
  al, 2009). 
```

*geNorm* is an algorithm and referenced as such above, but it is software too - and the software is actually used for the normalization in the described research. It should therefore be annotated as a software mention.

As a general guidelines regarding an algorithm name and its implementation used in an interchangeable manner in a scholar publication: in case the reference is made to an implemented algorithm with the algorithm name, we consider it as software mention if the context indicates that the implemented software aspect is involved in the statement. 

- The notion of **models** (machine learning models, simulation models) versus software is often unclear. Models encode data processing and executable action/prediction process. They are however in a format closer to data, because the "instructions" are only digital transformations. Models themselves should be run in a software environment. Despite their "executable" nature, models are usually not considered as software and have dedicated distinct sharing infrastructure (e.g. the [CoMSES Net](https://www.comses.net)). 

So as a general guideline, standalone models are **not** to consider as software product. 

However, like algorithms, we observe that it can be relatively frequent (in the machine learning area for example) to use the same name to refer both for a model and a software product for implementing/running a model. For example, `BERT` is a python software project (https://github.com/google-research/bert), a model, a family of models (retrained on different domains), or a ML approach (a Deep Learning architecture and methodology for training it):

[10.48550/arXiv.2103.11943](https://arxiv.org/pdf/2103.11943.pdf)

```
  The representation of the BERT system allows it to be used as a basis for measuring the 
  similarity of sentences in natural languages
```

Similarly as for algorithm, we need to identify whether the mention refers to the model product, the approach/method or the software to decide if the mention shall be considered as software mention or not. In case the reference is made in general to the whole framework, including the software, we would consider it as software mention. 

- **Database** versus software: in scientific publications, it is quite frequent to mention a database name as a general service covering the data and  software to access/search the data (including web services and database management software, e.g. PostgresQL). 

Example from PMC4863732

```
  Scientific articles were obtained through PubMed MEDLINE
```

MEDLINE is at the same time a large metadata collection and a database service to query this catalogue. 

Example from 10.1002/pam.22030

```
  Data come from the Integrated Public Use Microdata Series (IPUMS) database
```

Integrated Public Use Microdata Series (IPUMS) is a database and an online platform.

The related guideline for the Softcite corpus is as follow: 

```
  "The relevant distinction should be whether the text is referring to a data collection/dataset 
  (ie the data in the database) or to the software that provides access to a dataset. If it is 
  clear that they are referring to the data inside, it is not a reference to a software." 
```

The guideline thus also means that when it is not clear that we refer to the data inside the database, it should be considered as software too. 

- Very common is life science, **scientific devices** are used in most of the experiments. They usually includes software, embedded or to be install on a PC to control the device, process the aquired data, export the data, etc.. 

Example: [PMC4644012](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4644012/)

```
  The Gram-negative coccobacilli were initially identified as Pasturella pneumotropica by the 
  VITEK 2 system, software version 06.01 (BioMerieux, France) using the GN card, with bionumber 
  0001010210040001 and an excellent identification (probability 99%).
```

The [VITEK 2 system](https://www.biomerieux-usa.com/clinical/vitek-2-healthcare) embeds software performing predictions. 

Given the variety of software embodiments, what is mentioned is often larger system or devices including software. It is thus important to decide in context to which part the authors are referring to, and if the statement refers to the software part of the device. 

- **Software components** of a more complete infrastructure: A reference is made to a general infrastructure, including some software components. For example in [10.20955/r.2018.1-16](http://herosolutions.com.pk/breera/foundation/images/whitepaper.pdf) "Bitcoin wallet". We consider that we are refering to a software environment, and thus annotate this as a software mention. 

- Reference to a **programming languages**. For example: [10.1257/jep.4.1.99](https://www.aeaweb.org/articles?id=10.1257/jep.4.1.99)

```
  It is written in BASIC, a close analogue to FORTRAN.
```

We consider that software languages (written in BASIC, in FORTRAN, ...) are not software per se, because they are specifications (a grammar), similar to a protocol specification. When used together with a specific software mention, programing language are considered as "attributes" of this software (e.g. written in R). They are not annotated as software but with the mark-up `<rs type="language">`, which identifies in context the programming language of the mentioned software. 

Software tools for implementing a software language (like a C compiler, a Java virtual machine, an Integrated Development Environment like R-Studio, etc.) are software, they are annotated as a software mention. 

- **Operating system** (OS): when used together with a specific software mention, they are considered as "attributes" of this software (e.g. "running on Windows"). The reference to the OS here is just a further specification about the main software that is discussed. In this case, OS are not annotated as additional software mention. 

However, OS can also be referenced as such when the mention refers specifically to the OS implementation and not to some software using them. In this case, the OS is annotated as software. 

- Non-named usage of a programming environment. Software was produced on the environment (some code was written), but it is implicit, not shared, nor reusable. 

Example: [10.1136/gut.2011.238386](https://gut.bmj.com/content/gutjnl/61/1/69.full.pdf)

```
  Multiple imputation was conducted in R 2.11." 
```

The programming environment here is clearly a software and should be annotated as such. In addition, the non-named usage corresponding to the written code is also a software, implicit, running in the R environment, and should be annotated as a software mention. 

- **Workflow** as high-level specifications: in data-intensive scientific domains, the complexity of data processing has led to the common definiton and usage of workflows associated to a scientific experiments. Examples of such workflow systems are [Galaxy](https://galaxyproject.org) (life science), [Kepler](https://kepler-project.org) (physics and environment sciences), [Apache Taverna](https://incubator.apache.org/projects/taverna.html) (bioinformatics, astronomy, biodiversity - now retired), or [KNIME](https://www.knime.com). As workflows are formal instructions similar to high level scripting language, interpreted in a computer environment, and actually shared for reproducibility and reuse purposes. Therefore, we consider such executable workflows as software products. 

- **API**: An API is an intermediary product between documentation and executable software. It is challenging to decide if an API should be considered as software, because it requires an implementation to be executable. On the other hand, an API corresponds to instructions that can be executed when used on an environment or with other software components implementing the API, like other software depending on other software components. Given that it is the nature of an API to be shared and used for collaborative work in software, we consider API product as software too.
"""


class SearchAgentOnlySoftware(Agent):
    """
    An agent that is able to utilize a search engine to find software mentions in the text.
    """

    search: Searcher = ConfigurableSubclassFactory(
        Searcher,
        "Searcher to use for the agent.",
        user_default=DDGSSearcher
    )

    software_database: SoftwareDatabase = ConfigurableFactory(
        SoftwareDatabase,
        "Software database to use for checking existing software mentions.",
    )

    max_repairs: int = ConfigurableValue(
        user_default=3,
        desc="Maximum number of trials to repair the candidates contexts."
    )

    use_reasoning: Optional[Union[bool, Literal['low', 'medium', 'high']]] = ConfigurableValue(
        user_default=False,
        desc="Whether to use reasoning in the verification step."
    )

    find_candidates_system_prompt: Template = ConfigurableValue(
        user_default=LiteralScalarString(f"""You are a software mention extraction agent. Your task is to find software mentions in the text from scientific papers. 
This is the first step of the process, you should extract as many software mentions candidates as possible. Meaning, you can extract mentions even if you are not quite sure that they are mentions of software. Do not extract names of general concepts, algorithms, and methods.

Contexts shouldn't be longer than a couple of words before and after the surface_form (including surface_form), so that it can be used to identify the mention position in the text. \n
If there are no software mentions candidates in the text, return an empty list.
"""),
        desc="System prompt to find software mentions candidates in the text. ",
        transform=TemplateTransformer()
    )
    find_candidates_few_shot: Optional[list[tuple[str, str]]] = ConfigurableValue(
        desc="Few-shot examples for finding software mentions candidates.",
        user_default=[
            (
                "user",
                LiteralScalarString("""The text is:

We used the Ollama inference API [https://ollama.com/docs/api] to run the model locally.

Please extract all software mentions from the text.""")
            ),
            (
                "assistant",
                LiteralScalarString("""[
    {
        "id": 1,
        "surface_form": "Ollama",
        "context": "used the Ollama inference",
    }
]""")
            ),
            (
                "user",
                LiteralScalarString("""The text is:

Based on list of known public managers, we used the snowballing technique (Myers & Newman, 2007) to recruit new respondents. We first made contact with them by telephone, informing them about the study goals; shortly thereafter, we sent them a personalized link to the software Q-software with specific instructions on how to do the sorting exercise online.

Please extract all software mentions from the text.""")
            ),
            (
                "assistant",
                LiteralScalarString("""[
    {
        "id": 1,
        "surface_form": "Q-software",
        "context": "personalized link to the software Q-software with specific instructions",
    }
]""")
            ),
            (
                "user",
                LiteralScalarString("""The text is:

We used the open-source Transformers library developed by Hugging Face (Wolf et al., 2020) to implement and train our models. Our training code MLTrainer is available at https://example.com, the whole code base is implemented using Python.

The following surface forms were previously verified as software in different documents and they appear in the provided text. Please check whether they are used as software mentions in the provided text too.
Known surface forms:
transformers

Please extract all software mentions candidates from the text. Do not concentrate only on the known surface forms.""")
            ),
            (
                "assistant",
                LiteralScalarString("""[
    {
        "id": 1,
        "surface_form": "Transformers",
        "context": "open-source Transformers library",
    },
    {
        "id": 2,
        "surface_form": "MLTrainer",
        "context": "training code MLTrainer is available",
    }
]""")
            )
        ]
    )
    find_candidates_prompt: Template = ConfigurableValue(
        user_default=LiteralScalarString("""The text is:

{{text}}

{% if known_surface_forms | length > 0 %}
The following surface forms were previously verified as software in different documents and they appear in the provided text. Please check whether they are used as software mentions in the provided text too.
Known surface forms:
{% for k in known_surface_forms %}
{{k}}
{% endfor %}

Please extract all software mentions candidates from the text. Do not concentrate only on the known surface forms.
{% else %}
Please extract all software mentions candidates from the text.
{% endif %}"""),
        desc="Jinja2 template for the prompt to find software mentions candidates in the text. ",
        transform=TemplateTransformer()
    )

    find_candidates_repair_output_prompt: Template = ConfigurableValue(
        user_default=LiteralScalarString(
            """The json you provided is not valid. Please fix it. Here is the error: {{error}}"""),
        desc="Jinja2 template for the prompt to repair the output of the find candidates step. ",
        transform=TemplateTransformer()
    )

    verification_system_prompt: Template = ConfigurableValue(
        user_default=LiteralScalarString(f"""You are a software mention verification agent. Your task is to verify if the given mention is a software or not. Fix any mistakes in the candidate context if needed.
You will be given a text from a scientific paper from which you have previously extracted software mentions candidates, candidates themself, and the mention you should verify.
Reason about the mention and use provided search engine results to help you with the verification. Use search engine results when you are not sure or not familiar with given software and you need additional information.

When you are done, return verified mention if it is truly a software, otherwise return null.
"""),
        desc="System prompt to verify software mentions in the text. ",
        transform=TemplateTransformer()
    )
    verification_few_shot: Optional[list[tuple[str, str]]] = ConfigurableValue(
        desc="Few-shot examples for verifying software mentions.",
        user_default=[
            (
                "user",
                LiteralScalarString("""From the text below, verify if the given candidate is a software mention or not:

We used the open-source Transformers library developed by Hugging Face (Wolf et al., 2020) to implement and train our models. Our training code MLTrainer is available at https://example.com, the whole code base is implemented using Python.

All software mentions candidates are:
{
    "id": 1,
    "surface_form": "Transformers",
    "context": "open-source Transformers library",
}
{
    "id": 2,
    "surface_form": "MLTrainer",
    "context": "training code MLTrainer is available",
}
    
Search results for the target candidate are:
{
    "title": "Transformers",
    "snippet": "An ancient struggle between two Cybertronian races, the heroic Autobots and the evil Decepticons, comes to Earth, with a clue to the ultimate power held by a teenager.",
    "link": "https://www.imdb.com/title/tt0418279/"
}
{
    "title": "Transformers",
    "snippet": "Transformers provides thousands of pretrained models to perform tasks on texts such as classification, information extraction, question answering, summarization, translation, text generation, etc in 100+ languages.",
    "link": "https://huggingface.co/docs/transformers/index"
}
{
    "title": "Electronic Transformers",
    "snippet": "Electronic Transformers, Inc. (ETI) is a leading manufacturer of custom magnetics products including transformers, inductors, chokes, and coils.",
    "link": "https://www.electronictransformers.com/"
}

Please decide whether this candidate is a software mention or not:
{
    "id": 1,
    "surface_form": "Transformers",
    "context": "open-source Transformers library",
}
""")
            ),
            (
                "assistant",
                LiteralScalarString("""{
    "include_check_list": {
        "commercial_and_general_purpose_software": false,
        "research_software": false,
        "software_environment": false,
        "named_software_component_package": true,
        "implicit_software_mentions": false,
        "workflow": false,
        "api": false,
        "operating_system": false,
        "devices_with_embedded_software": false
    },
    "exclude_check_list": {
        "algorithm": false,
        "model": false,
        "database": false,
        "programming_language": false,
        "operating_system": false,
        "bibliographic_references": false
    },
    "ambiguous_check_list": {
        "algorithm_vs_software": false,
        "model_vs_software": false,
        "database_vs_software": false,
        "device_vs_software": false
    },
    "reason": "The candidate 'Transformers' is verified as a software mention because it is a named software component/package. The context 'open-source Transformers library' indicates that it is a software library. The publisher 'Hugging Face' is a well-known organization in the field of machine learning and natural language processing, which further supports the identification of 'Transformers' as a software mention.",
    "is_software": true,
    "verified": {
        "id": 1,
        "surface_form": "Transformers",
        "context": "open-source Transformers library",
    }
}""")
            ),
            (
                "user",
                LiteralScalarString("""From the text below, verify if the given candidate is a software mention or not:
                
We used Binary Tree to implement our search. We obtained significant speed up over our previous implementation. The BinTreeSearch version 2.0 developed by our team FastSearch Inc. is available at https://example.com/bintreesearch. We used C++ to obtain the best performance.

All software mentions candidates are:
{
    "id": 1,
    "surface_form": "Binary Tree",
    "context": "used Binary Tree to implement",
}
{
    "id": 2,
    "surface_form": "BinTreeSearch",
    "context": "The BinTreeSearch version",
}

Search results for the target candidate are:
{
    "title": "Binary tree - Wikipedia",
    "snippet": "In computer science, a binary tree is a tree data structure in which each node has at most two children, which are referred to as the left child and the right child.",
    "link": "https://en.wikipedia.org/wiki/Binary_tree"
}
{
    "title": "Binary Tree - GeeksforGeeks",
    "snippet": "A binary tree is a tree data structure in which each node has at most two children, which are referred to as the left child and the right child.",
    "link": "https://www.geeksforgeeks.org/binary-tree-data-structure/"
}
{
    "title": "Binary Tree Package",
    "snippet": "Package for creating, traversing, and manipulating binary trees in Python.",
    "link": "https://github.com/example/binarytree"
}

The target candidate surface form was previously varified as software in different documents. Here are some contexts where it was seen as software:
Our package Binary Tree is available at https://github.com/example/binarytree.

Please decide whether this candidate is a software mention or not:
{
    "id": 1,
    "surface_form": "Binary Tree",
    "context": "used Binary Tree to implement",
}""")
            ),
            (
                "assistant",
                LiteralScalarString("""{
    "include_check_list": {
        "commercial_and_general_purpose_software": false,
        "research_software": false,
        "software_environment": false,
        "named_software_component_package": false,
        "implicit_software_mentions": false,
        "workflow": false,
        "api": false,
        "operating_system": false,
        "devices_with_embedded_software": false
    },
    "exclude_check_list": {
        "algorithm": true,
        "model": false,
        "database": false,
        "programming_language": false,
        "operating_system": false,
        "bibliographic_references": false
    },
    "ambiguous_check_list": {
        "algorithm_vs_software": false,
        "model_vs_software": false,
        "database_vs_software": false,
        "device_vs_software": false
    },
    "reason": "The search result suggests that there exists package named Binary Tree, however, the context 'used Binary Tree to implement' indicates that it is used as a general concept or data structure rather than a specific software product.",
    "is_software": false,
    "verified": null
}""")
            ),
            (
                "user",
                LiteralScalarString("""From the text below, verify if the given candidate is a software mention or not:

Our solution CryptoSeek is open-source solution for secure communication. The CryptoSeek version 1.2.3 developed by SecureSoft is available at https://securesoft.com/cryptoseek. We used Rust to implement it.

All software mentions candidates are:
{
    "id": 1,
    "surface_form": "CryptoSeek",
    "context": "solution CryptoSeek is open-source",
}
{
    "id": 2,
    "surface_form": "CryptoSeek",
    "context": "The CryptoSeek version",
}

Search results for the target candidate are:
{
    "title": "Coinbase",
    "snippet": "Coinbase is a secure online platform for buying, selling, transferring, and storing cryptocurrency.",
    "link": "https://www.coinbase.com/"
}
{
    "title": "CryptoSeek - Secure Communication Software",
    "snippet": "CryptoSeek is an open-source software solution for secure communication, developed by SecureSoft. It allows instant messaging, voice, and video calls with end-to-end encryption.",
    "link": "https://securesoft.com/cryptoseek"
}
{
    "title": "Binance",
    "snippet": "Binance is a global cryptocurrency exchange that provides a platform for trading various cryptocurrencies.",
    "link": "https://www.binance.com/"
}

Please decide whether this candidate is a software mention or not:
{
    "id": 2,
    "surface_form": "CryptoSeek",
    "context": "The CryptoSeek developed by",
}

"""),
            ),
            (
                "assistant",
                LiteralScalarString("""{
    "include_check_list": {
        "commercial_and_general_purpose_software": true,
        "research_software": false,
        "software_environment": false,
        "named_software_component_package": false,
        "implicit_software_mentions": false,
        "workflow": false,
        "api": false,
        "operating_system": false,
        "devices_with_embedded_software": false
    },
    "exclude_check_list": {
        "algorithm": false,
        "model": false,
        "database": false,
        "programming_language": false,
        "operating_system": false,
        "bibliographic_references": false
    },
    "ambiguous_check_list": {
        "algorithm_vs_software": false,
        "model_vs_software": false,
        "database_vs_software": false,
        "device_vs_software": false
    },
    "reason": "The candidate 'CryptoSeek' is verified as a software mention. The search results indicate that 'CryptoSeek' is an open-source software solution for secure communication. The mention context seems to be slightly off, but it is clear that it refers to the second occurrence of 'CryptoSeek' in the text.",
    "is_software": true,
    "verified": {
        "id": 2,
        "surface_form": "CryptoSeek",
        "context": "The CryptoSeek version",
    }""")
            )

        ])
    verification_prompt: Template = ConfigurableValue(
        user_default=LiteralScalarString("""From the text below, verify if the given candidate is a software mention or not:

{{text}}

All software mentions candidates are:
{% for c in candidates: %}
{{c | model_dump_json }}
{% endfor %}    

Search results for the target candidate are:
{% for r in search_results %}
{{r | model_dump_json }}
{% endfor %}
{% if seen_in_contexts | length > 3 %}

The target candidate surface form was previously varified as software in different documents. Here are some contexts where it was seen as software:
{% for c in seen_in_contexts[:3] %}
{{c}}
{% endfor %}
{% endif %}

Please decide whether this candidate is a software mention or not:
{{target | model_dump_json }}

"""),
        transform=TemplateTransformer()
    )

    verification_repair_output_prompt: Template = ConfigurableValue(
        user_default=LiteralScalarString(
            """The json you provided is not valid. Please fix it. Here is the error: {{error}}"""),
        desc="Jinja2 template for the prompt to repair the output of the verification step. ",
        transform=TemplateTransformer()
    )

    second_pass_verification_system_prompt: Template = ConfigurableValue(
        user_default=LiteralScalarString(f"""You are a software mention verification agent. In previous step you have extracted software mentions. 
However, you might have missed some mentions. It was found that surface forms of these mentions appear in the text, but you did not extract them. 
It is possible that in given context the surface form does not refer to a software, so you need to verify it. However, it is likely that it is a software, because the same surface form was already extracted as software mention in different context in the same document.
"""),
        desc="System prompt to verify software mentions in the text. ",
        transform=TemplateTransformer()
    )

    second_pass_few_shot: Optional[list[tuple[str, str]]] = ConfigurableValue(
        desc="Few-shot examples for verifying software mentions in the second pass.",
        user_default=[
            (
                "user",
                LiteralScalarString("""From the text below, verify if the given mention is a software or not and make sure extracted components (version, publisher, URL, language) are correct.

Our experiments proven that second method is faster and more reliable as shown on Figure 2. Due to large data, we used the Hadoop framework to implement our solution. Specifically, we used the Hadoop version 3.2.1.

Other candidates found in the text during second pass:
{
    "id": "second_pass_1",
    "surface_form": "Hadoop",
    "context": "used the Hadoop framework to",  
}

Search results for the target candidate are:
{
    "title": "Hadoop - Apache Software Foundation",
    "snippet": "Apache Hadoop is an open-source software framework used for distributed storage and processing of large data sets using the MapReduce programming model.",
    "link": "https://hadoop.apache.org/"
}
{
    "title": "What is Hadoop? - IBM",
    "snippet": "Hadoop is an open-source framework that allows for the distributed processing of large data sets across clusters of computers using simple programming models.",
    "link": "https://www.ibm.com/topics/hadoop"
}
{
    "title": "Hadoop Tutorial - W3Schools",
    "snippet": "Hadoop is an open-source software framework used for distributed storage and processing of large data sets using the MapReduce programming model.",
    "link": "https://www.w3schools.com/hadoop/"
}

Please verify the following mention:
{
    "id": "second_pass_2",
    "surface_form": "Hadoop",
    "context": "the Hadoop version",
}""")
            ),
            (
                "assistant",
                LiteralScalarString("""{
    "include_check_list": {
        "commercial_and_general_purpose_software": true,
        "research_software": false,
        "software_environment": false,
        "named_software_component_package": false,
        "implicit_software_mentions": false,
        "workflow": false,
        "api": false,
        "operating_system": false,
        "devices_with_embedded_software": false
    },
    "exclude_check_list": {
        "algorithm": false,
        "model": false,
        "database": false,
        "programming_language": false,
        "operating_system": false,
        "bibliographic_references": false
    },
    "ambiguous_check_list": {
        "algorithm_vs_software": false,
        "model_vs_software": false,
        "database_vs_software": false,
        "device_vs_software": false
    },
    "reason": "The candidate 'Hadoop' is verified as a software mention. The search results indicate that 'Hadoop' is an open-source software and mention with the same surface form was already extracted as software mention in the first pass.",
    "is_software": true,
    "verified": {
        "id": "second_pass_2",
        "surface_form": "Hadoop",
        "context": "the Hadoop version",
    }""")
            ),
            (
                "user",
                LiteralScalarString("""From the text below, verify if the given mention is a software or not and make sure extracted components (version, publisher, URL, language) are correct.
                
The state of the art in self driving cars is constantly evolving. The latest software Self Driving by team from Cornell University is available at https://example.com/selfdriving. It significantly pushes the SOTA forward. The implementation is done in c++.

Search results for the target candidate are:
{
    "title": "Self Driving - Wikipedia",
    "snippet": "Self-driving cars, also known as autonomous cars, driver-less cars, or robotic cars, are vehicles that are capable of sensing their environment and operating without human involvement.",
    "link": "https://en.wikipedia.org/wiki/Self-driving_car"
}
{
    "title": "Self Driving Car Software - NVIDIA",
    "snippet": "NVIDIA DRIVE™ is an AI platform for autonomous vehicles that includes hardware, software, and development tools to enable self-driving cars.",
    "link": "https://www.nvidia.com/en-us/self-driving-cars/"
}
{
    "title": "Self Driving Cars - Tesla",
    "snippet": "Tesla's self-driving cars are equipped with advanced hardware and software to enable autonomous driving capabilities.",
    "link": "https://www.tesla.com/autopilot"
}

Please verify the following mention:
{
    "id": "second_pass_1",
    "surface_form": "self driving",
    "context": "in self driving cars is constantly",
}""")
            ),
            (
                "assistant",
                LiteralScalarString("""{
    "include_check_list": {
        "commercial_and_general_purpose_software": false,
        "research_software": false,
        "software_environment": false,
        "named_software_component_package": false,
        "implicit_software_mentions": false,
        "workflow": false,
        "api": false,
        "operating_system": false,
        "devices_with_embedded_software": false
    },
    "exclude_check_list": {
        "algorithm": false,
        "model": false,
        "database": false,
        "programming_language": false,
        "operating_system": false,
        "bibliographic_references": false
    },
    "ambiguous_check_list": {
        "algorithm_vs_software": false,
        "model_vs_software": false,
        "database_vs_software": false,
        "device_vs_software": false
    },
    "reason": "The candidate 'self driving' is not verified as a software mention. The search results indicate that 'self-driving' refers to the general concept of autonomous vehicles rather than a specific software.",
    "is_software": false,
    "verified": null
}""")
            ),
            (
                "user",
                LiteralScalarString("""From the text below, verify if the given mention is a software or not and make sure extracted components (version, publisher, URL, language) are correct.

The data analysis was performed using the R programming language. We specifically used the tidyverse package to manipulate and visualize the data. The tidyverse version 1.3.0 is available at https://www.tidyverse.org/. The package is developed by RStudio.

Other candidates found in the text during second pass:
{
    "id": "second_pass_1",
    "surface_form": "R",
    "context": "using the R programming language.",
}
{
    "id": "second_pass_2",
    "surface_form": "tidyverse",
    "context": "used the tidyverse package to",
}
{

    "id": "second_pass_4",
    "surface_form": "RStudio",
    "context": "developed by RStudio.",
}


Search results for the target candidate are:
{
    "title": "tidyverse: Easily Install and Load the 'Tidyverse'",
    "snippet": "The tidyverse is an opinionated collection of R packages designed for data science. All packages share an underlying design philosophy, grammar, and data structures.",
    "link": "https://www.tidyverse.org/"
}
{
    "title": "What is the Tidyverse? - RStudio",
    "snippet": "The tidyverse is a collection of R packages that share common data science tools and principles, making it easier to learn and use R for data analysis.",
    "link": "https://rstudio.com/resources/what-is-the-tidyverse/"
}
{
    "title": "R Programming Language - Overview, Features, and Resources",
    "snippet": "R is a programming language and free software environment for statistical computing and graphics supported by the R Foundation for Statistical Computing.",
    "link": "https://www.r-project.org/"
}

Please verify the following mention:
{
    "id": "second_pass_3",
    "surface_form": "tidyverse",
    "context": "The tidyverse version",
}""")
            ),
            (
                "assistant",
                LiteralScalarString("""{
    "include_check_list": {
        "commercial_and_general_purpose_software": false,
        "research_software": false,
        "software_environment": false,
        "named_software_component_package": true,
        "implicit_software_mentions": false,
        "workflow": false,
        "api": false,
        "operating_system": false,
        "devices_with_embedded_software": false
    },
    "exclude_check_list": {
        "algorithm": false,
        "model": false,
        "database": false,
        "programming_language": false,
        "operating_system": false,
        "bibliographic_references": false
    },
    "ambiguous_check_list": {
        "algorithm_vs_software": false,
        "model_vs_software": false,
        "database_vs_software": false,
        "device_vs_software": false
    },
    "reason": "The candidate 'tidyverse' is verified as a software mention. The search results indicate that 'tidyverse' is a collection of R packages designed for data science.",
    "is_software": true,
    "verified": {
        "id": "second_pass_3",
        "surface_form": "tidyverse",
        "context": "The tidyverse version",
    }""")
            )
        ]
    )

    second_pass_verification_prompt: Template = ConfigurableValue(
        user_default=LiteralScalarString("""From the text below, verify if the given mention is a software or not and make sure extracted components (version, publisher, URL, language) are correct.

{{text}}

Other candidates found in the text during second pass:
{% for c in second_pass_candidates %}
{{c | model_dump_json }}
{% endfor %}

Search results for the target candidate are:
{% for r in search_results %}
{{r | model_dump_json}}
{%endfor %}

Please verify the following mention:
{{verify | model_dump_json }}
"""),
        desc="Jinja2 template for the prompt to verify software mentions in the text. ",
        transform=TemplateTransformer()
    )

    url_repair_system_prompt: Template = ConfigurableValue(
        user_default=LiteralScalarString("""You are a URL repair agent. Your task is to repair the given URL extracted from the text from a scientific paper.
You will be provided with text from which the URL was extracted and the URL itself.
Make sure that the URL is not just valid, but that it points to the correct resource, i.e., the resource that was mentioned in the text
Try to repair it if it is not a valid URL. If you cannot repair it, return an empty string."""),
        desc="System prompt to repair URLs in the candidates. ",
        transform=TemplateTransformer()
    )

    url_repair_prompt: Template = ConfigurableValue(
        user_default=LiteralScalarString("""The text is:

{{text}}

The URL is: '{{url}}'
Please repair the URL if it is not valid or return an empty string."""),
        desc="Jinja2 template for the prompt to repair URLs in the candidates. ",
        transform=TemplateTransformer()
    )
    repair_problems_prompt_system_prompt: Template = ConfigurableValue(
        user_default=LiteralScalarString("""You are a software mention extraction agent. Your task is to find software mentions in the text from scientific papers. 
You have previously extracted software mentions candidates from the text, but some of them have problems. 
Please fix them using the text from which the software mention candidate was extracted and also the other candidates if necessary."""),
        desc="System prompt to repair candidates with identified problems. ",
        transform=TemplateTransformer()
    )
    repair_problems_prompt: Template = ConfigurableValue(
        user_default=LiteralScalarString("""The following problems were identified for the given candidate.
Text:

{{text}}

All candidates:
{% for c in candidates %}
      {{c | model_dump_json }}
{% endfor %}

Problematic candidate:
{{candidate | model_dump_json }}

Problems:
{% for p in problems %}
        - {{p}}
{% endfor %}

Please fix all problems of the candidate."""),
        desc="Jinja2 template for the prompt to repair candidates with identified problems. ",
        transform=TemplateTransformer()
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

    context_window_for_database: int = ConfigurableValue(
        user_default=100,
        desc="Number of characters to use as context around the software mention for storing validate software to the software database."
    )

    fake_structured_format: Optional[StructuredResponseFormatFaker] = ConfigurableSubclassFactory(
        StructuredResponseFormatFaker,
        "If set, it will be used to fake the structured response format for the model.",
        user_default=None,
        voluntary=True
    )

    def __post_init__(self):
        self.tools = [
            {
                'type': 'function',
                'function': {
                    'name': 'search_engine',
                    'description': 'Web search engine. Use this tool to search for software mentions in the text. The search engine will return a list of search results that can be used to verify software mentions. It returns a list of search results with title, URL, and snippet.',
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'query': {
                                'type': 'string',
                                'description': 'The search query to use for the search engine. It should be the name of the software mention to search for. Use the context of the mention if you think it will help with the search.',
                            },
                        },
                        'required': ['query'],
                    },
                },
            }
        ]
        self.request_factory = self.model_api.get_request_factory()

    def __enter__(self):
        self.search.__enter__()
        self.software_database.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.search.__exit__(exc_type, exc_val, exc_tb)
        self.software_database.__exit__(exc_type, exc_val, exc_tb)

    def __call__(self, text: str) -> list[Mention]:
        """
        Processes the input text and returns a list of software mentions.

        :param text: The input text to process.
        :return: A list of software mentions found in the text.
        """

        candidates = self.find_candidates(text)
        verified_candidates = self.verify_candidates(text, candidates)
        repaired = self.repair(text, verified_candidates)
        mentions = self.convert_candidates_to_mentions(text, repaired)
        self.add_mentions_to_database(text, mentions)
        return mentions

    def call_second_pass(self, text: str, all_document_mentions: list[Mention], text_start_offset: int) -> list[
        Mention]:
        """
        Calls the agent a second time to find mentions that could have been missed in the first pass.

        :param text: The input text to process.
        :param all_document_mentions: All mentions found in the document in the first pass.
        :param text_start_offset: Start offset of the text in the document.
        :return: Additional mentions found in the second pass.
        """

        input_text_mentions = self.filter_and_align_mentions(text, text_start_offset, all_document_mentions)
        searched_candidates = self.back_search_candidates(text, input_text_mentions, all_document_mentions)
        verified = self.verify_second_pass_mentions(text, searched_candidates, input_text_mentions,
                                                    all_document_mentions)

        repaired = self.repair(text, verified)
        mentions = self.convert_candidates_to_mentions(text, repaired)
        self.add_mentions_to_database(text, mentions)
        return mentions

    def verify_second_pass_mentions(self, text: str, candidates: list[Candidate], input_text_mentions: list[Mention],
                                    all_document_mentions: list[Mention]) -> list[Candidate]:
        """
        Verifies the mentions found in the second pass.

        :param text: The input text\
        :param candidates: Candidates found in the second pass.
        :param input_text_mentions: Mentions already found in the input text.
        :param all_document_mentions: All mentions found in the document in the first pass.
        :return: Verified mentions from the second pass.
        """
        verified_mentions = []
        for i, m in enumerate(candidates):
            cur_m = self.verify_second_pass_mention(text, m, candidates[:i] + candidates[i + 1:], input_text_mentions,
                                                    all_document_mentions)
            if cur_m:
                verified_mentions.append(cur_m)
        return verified_mentions

    def verify_second_pass_mention(self, text: str, verify: Candidate, mentions: list[Candidate],
                                   input_text_mentions: list[Mention], all_document_mentions: list[Mention]) -> \
    Optional[Candidate]:
        """
        Verifies a single mention found in the second pass.

        :param text: The input text
        :param verify: Candidate to verify.
        :param mentions: Mentions found in the second pass without the one to verify.
        :param input_text_mentions: Mentions already found in the input text.
        :param all_document_mentions: All mentions found in the document in the first pass.
        :return: Verified candidate or None if the mention was not verified.
        """
        search_query = self.obtain_search_query(text, verify)
        search_results = self.search.search(search_query).results

        messages = [
            {"role": "system", "content": self.second_pass_verification_system_prompt.render(
                {"text": text, "verify": verify, "second_pass_candidates": mentions,
                 "input_text_mentions": input_text_mentions, "all_document_mentions": all_document_mentions, "search_results": search_results})}
        ]

        if self.second_pass_few_shot:
            for role, content in self.second_pass_few_shot:
                messages.append({"role": role, "content": content})

        messages.append(
            {"role": "user", "content": self.second_pass_verification_prompt.render(
                {"text": text, "verify": verify, "second_pass_candidates": mentions,
                 "input_text_mentions": input_text_mentions, "all_document_mentions": all_document_mentions,
                 "search_results": search_results})}
        )
        request = self.request_factory(
            custom_id="verify_candidate_second_pass",
            model=self.model,
            message=messages,
            options=self.requests_options,
            response_format=VerifyCandidateResponse,
            reasoning=self.use_reasoning,
            fake_structured=self.fake_structured_format
        )
        raw_response = self.model_api.process_single_request(request).response.get_raw_content()
        response = json_repair.loads(raw_response)
        response = VerifyCandidateResponse.model_validate(response)
        if not response.is_software:
            return None
        return response.verified

    def filter_and_align_mentions(self, text: str, text_start_offset: int, mentions: list[Mention]) -> list[Mention]:
        """
        Filters out mentions that are not part of the given text and aligns the start offsets of the mentions to the text.

        :param text: The input text
        :param text_start_offset: Start offset of the text in the document.
        :param mentions: List of mentions to filter and align.
        :return: Filtered and aligned list of mentions.
        """
        filtered_mentions = []
        text_end_offset = text_start_offset + len(text)
        for m in mentions:
            if text_start_offset <= m.start_offset < text_end_offset:
                m.start_offset -= text_start_offset
                filtered_mentions.append(m)
        return filtered_mentions

    def back_search_candidates(self, text: str, input_text_mentions: list[Mention],
                               all_document_mentions: list[Mention]) -> list[Candidate]:
        """
        Searches the text whether there exists mentions having the same surface form as the ones in all_document_mentions.

        :param text: The input text
        :param input_text_mentions: Mentions already found in the input text.
        :param all_document_mentions: All mentions found in the document in the first pass.
        :return: Additional candidates found by back searching the text.
        """

        found_mentions = []
        covered_intervals = IntervalTree()
        for m in input_text_mentions:
            covered_intervals.addi(m.start_offset, m.start_offset + len(m.surface_form))
        covered_intervals.merge_overlaps()

        text_ci = text.lower()
        for i, m in enumerate(all_document_mentions):
            try:
                start_index = text_ci.index(m.surface_form.lower())
                start_index, surface_form = self.obtain_word(text, m.surface_form, start_index)
                if len(covered_intervals.overlap(start_index, start_index + len(surface_form))) == 0:
                    found_mentions.append(Candidate(
                        id="second_pass_" + str(i),
                        surface_form=surface_form,
                        context=self.get_context_window(text, start_index, m.surface_form,
                                                        self.context_window_for_database)
                    ))
                    covered_intervals.addi(start_index, start_index + len(surface_form))

            except ValueError:
                continue
        return found_mentions

    @staticmethod
    def obtain_word(text: str, surface_form: str, start_offset: int) -> tuple[int, str]:
        """
        Obtains the whole word from the text given the surface form and its start offset.

        :param text: original text from which the candidate was extracted
        :param surface_form: exact form of the mention in the text
        :param start_offset: start offset of the surface form in the text
        :return: whole word containing the surface form in form of (start_offset, word)
        """
        left = start_offset
        right = start_offset + len(surface_form)

        while left > 0 and not text[left - 1].isspace():
            left -= 1
        while right < len(text) and not text[right].isspace():
            right += 1

        return left, text[left:right]

    def add_mentions_to_database(self, text: str, mentions: list[Mention]):
        """
        Adds the given mentions to the software database.

        :param text: original text from which the mentions were extracted
        :param mentions: list of mentions to add to the database
        """
        for m in mentions:
            context = self.get_context_window(text, m.start_offset, m.surface_form, self.context_window_for_database)
            self.software_database.add_software(m.surface_form, context)

    def convert_candidates_to_mentions(self, text: str, candidates: list[Candidate]) -> list[Mention]:
        """
        Converts a list of candidates to a list of mentions by finding their start offsets in the text.

        :param text: original text from which the candidates were extracted
        :param candidates: list of candidates to convert
        :return: list of mentions
        """
        mentions = []
        for c in candidates:
            try:
                mention = self.convert_candidate_to_mention(text, c)
                mentions.append(mention)
            except ValueError as e:
                print(f"Warning: {e}", file=sys.stderr)
        return mentions

    @staticmethod
    def get_start_offset(text: str, context: str, surface_form: str, type_name: str = "Software") -> int:
        """
        Finds the start offset of the surface form in the context within the text.

        :param text: original text from which the candidate was extracted
        :param context: context of the mention
        :param surface_form: exact form of the mention in the text
        :param type_name: type of the mention (used for error messages)
        :return: start offset of the surface form in the text
        :raise ValueError: if the context or surface form is not found in the text
        """
        start_offset = text.find(context)
        if start_offset == -1:
            raise ValueError(f"Context '{context}' not found in text.")
        start_offset += context.find(surface_form)
        if start_offset == -1:
            raise ValueError(f"{type_name} surface form '{surface_form}' not found in {type_name} context '{context}'.")
        return start_offset

    @staticmethod
    def get_context_window(text: str, start_offset: int, surface_form: str, window: int) -> str:
        """
        Extracts a context window around the surface form in the text.

        :param text: original text from which the candidate was extracted
        :param start_offset: start offset of the surface form in the text
        :param surface_form: exact form of the mention in the text
        :param window: number of characters to include before and after the surface form
        :return: context window around the surface form
        """
        context_start = max(0, start_offset - window)
        context_end = min(len(text), start_offset + len(surface_form) + window)
        return text[context_start:context_end]

    def convert_candidate_to_mention(self, text: str, candidate: Candidate) -> Mention:
        """
        Converts a candidate to a mention by finding its start offset in the text.

        :param text: original text from which the candidate was extracted
        :param candidate: candidate to convert
        :return: mention
        """

        def assemble_additional_info(context: str, surface_form: str, type_name: str) -> MentionAdditionalInfo:
            start_offset = self.get_start_offset(text, context, surface_form, type_name)
            return MentionAdditionalInfo(
                surface_form=surface_form,
                context=context,
                start_offset=start_offset
            )

        software_context_start = assemble_additional_info(text, candidate.surface_form, "Software").start_offset
        return Mention(
            surface_form=candidate.surface_form,
            context=candidate.context,
            start_offset=software_context_start,
            version=None,
            publisher=[],
            url=[],
            language=[]
        )

    def find_candidates(self, text: str) -> list[Candidate]:
        """
        Finds software mentions in the text using the search engine.

        :param text: The input text to search for software mentions.
        :return: A list of software mentions found in the text.
        """
        known_surface_forms = [f for _, f in self.software_database.known_surface_forms_in_text(text)]
        messages = [
            {"role": "system", "content": self.find_candidates_system_prompt.render(
                {"text": text, "known_surface_forms": known_surface_forms})}
        ]
        if self.find_candidates_few_shot is not None:
            for role, content in self.find_candidates_few_shot:
                messages.append({"role": role, "content": content})

        messages.append({"role": "user", "content": self.find_candidates_prompt.render(
                    {"text": text, "known_surface_forms": known_surface_forms})})

        request = self.request_factory(
            custom_id="find_candidates",
            model=self.model,
            message=messages,
            options=self.requests_options,
            response_format=FindCandidatesResponse,
            fake_structured=self.fake_structured_format
        )
        raw_response = self.model_api.process_single_request(request).response.get_raw_content()

        response = json_repair.loads(raw_response)

        try:
            response = FindCandidatesResponse.model_validate(response)
        except ValidationError as e:
            # try to repair the output
            repair_request = self.request_factory(
                custom_id="find_candidates_repair_output",
                model=self.model,
                message=[
                    {"role": "system", "content": self.find_candidates_system_prompt.render({"text": text})},
                    {"role": "user", "content": self.find_candidates_prompt.render({"text": text})},
                    {"role": "assistant", "content": raw_response},
                    {"role": "user",
                     "content": self.find_candidates_repair_output_prompt.render({"text": text, "error": str(e)})}
                ],
                options=self.requests_options,
                response_format=FindCandidatesResponse,
                fake_structured=self.fake_structured_format
            )
            response = self.model_api.process_single_request(repair_request).response.get_raw_content()
            response = json_repair.loads(response)
            response = FindCandidatesResponse.model_validate(response)
        return response.candidates

    def repair(self, text: str, candidates: list[Candidate]) -> list[Candidate]:
        """
        Repairs context problems in candidates.

        :param text: original text from which the candidates were extracted
        :param candidates: list of candidates to repair
        :return: list of candidates with repaired contexts
        """
        cnt = 0
        while True and cnt < self.max_repairs:
            cnt += 1
            flag = False
            try:
                for f in [
                    self.repair_non_existing_context,
                    self.repair_surface_form_not_in_context, self.repair_same_context_candidates,
                    self.repair_ambiguous_contexts, self.repair_overlapping_surface_forms
                ]:
                    flag = flag | f(text, candidates)

                if not flag:
                    # no repairs were made
                    break
            except ValidationError as e:
                # LLM output was invalid, we skip this repair iteration
                continue

        return candidates

    def repair_overlapping_surface_forms(self, text: str, candidates: list[Candidate]) -> bool:
        """
        Finds candidates with overlapping surface forms.

        :param text: original text from which the candidates were extracted
        :param candidates: list of candidates to check
        :return: True if there was an attempt to repair, False otherwise
        """
        flag = False
        intervals = IntervalTree()
        for i, c in enumerate(candidates):
            start = text.find(c.surface_form)
            if start == -1:
                continue
            end = start + len(c.surface_form)
            overlapping = intervals.overlap(start, end)
            if len(overlapping) > 0:
                problems = [f"Surface form '{c.surface_form}' overlaps with other surface forms. Please fix it."]
                flag = True
                rep_c = self.send_repair_problem_request(text, candidates, c, problems)
                candidates[i] = rep_c
            else:
                intervals.addi(start, end)
        return flag

    def send_repair_problem_request(self, text: str, candidates: list[Candidate], candidate: Candidate,
                                    problems: list[str]) -> Candidate:
        """
        Sends a request to the model to repair a candidate with identified problems.

        :param text: original text from which the candidates were extracted
        :param candidates: list of all candidates
        :param candidate: candidate to repair
        :param problems: list of problems identified for the candidate
        :return: repaired candidate
        """
        request = self.request_factory(
            custom_id="repair_problems",
            model=self.model,
            message=[
                {"role": "system", "content": self.repair_problems_prompt_system_prompt.render(
                    data={"text": text, "candidates": candidates, "candidate": candidate, "problems": problems})},
                {"role": "user", "content": self.repair_problems_prompt.render(
                    data={"text": text, "candidates": candidates, "candidate": candidate, "problems": problems})}
            ],
            options=self.requests_options,
            response_format=Candidate,
            fake_structured=self.fake_structured_format
        )
        response = self.model_api.process_single_request(request).response.get_raw_content()
        response = json_repair.loads(response)
        response = Candidate.model_validate(response)
        return response

    def repair_non_existing_context(self, text: str, candidates: list[Candidate]) -> bool:
        """
        Finds candidates with non-existing context.

        :param text: original text from which the candidates were extracted
        :param candidates: list of candidates to check
        :return: True if there was an attempt to repair, False otherwise
        """
        flag = False
        for i, c in enumerate(candidates):
            problems = []
            if c.context not in text:
                problems.append(f"Context '{c.context}' not found in text.")

            if len(problems) > 0:
                flag = True
                rep_c = self.send_repair_problem_request(text, candidates, c, problems)
                candidates[i] = rep_c

        return flag

    def repair_surface_form_not_in_context(self, text: str, candidates: list[Candidate]) -> bool:
        """
        Finds candidates with the surface form not in the context.

        :param text: original text from which the candidates were extracted
        :param candidates: list of candidates to check
        :return: True if there was an attempt to repair, False otherwise
        """
        flag = False
        for i, c in enumerate(candidates):
            problems = []
            if c.surface_form not in c.context:
                problems.append(
                    f"Surface form '{c.surface_form}' not in context '{c.context}'. Take on mind that the search is case-sensitive.")
            elif c.context.count(c.surface_form) > 1:
                problems.append(f"Surface form '{c.surface_form}' appears more than once in context '{c.context}'")

            if len(problems) > 0:
                flag = True
                rep_c = self.send_repair_problem_request(text, candidates, c, problems)
                candidates[i] = rep_c

        return flag

    def repair_same_context_candidates(self, text: str, candidates: list[Candidate]) -> bool:
        """
        Repairs candidates with the same context by asking the model to differentiate them.

        :param text: original text from which the candidates were extracted
        :param candidates: list of candidates to repair
        :return: flag whether repair was done
        """

        flag = False
        if len(same_context := self.same_context_candidates(candidates)) > 0:
            flag = True
            for (surface_form, context), problem_candidates in same_context.items():
                # we use only the first candidate
                candidate, annot_type = problem_candidates[0]
                problems = [
                    f"The tuple of surface form '{surface_form}' and context '{context}' is used for {len(problem_candidates)} candidates. Please differentiate to make the tuple unique. For following candidate it is used as {annot_type}."
                ]
                candidate = self.send_repair_problem_request(text, candidates, candidate, problems)

                for i, c in enumerate(candidates):
                    if c.id == candidate.id:
                        candidates[i] = candidate
                        break
        return flag

    def same_context_candidates(self, candidates: list[Candidate]) -> dict[
        tuple[str, str], list[tuple[Candidate, str]]]:
        """
        Finds candidates with the same context that are not distinguishable.

        :param candidates: list of candidates to check
        :return: dictionary mapping (surface form, context) to list of candidates having this context and description of context type
        """
        surface_forms = defaultdict(list)
        for i, c in enumerate(candidates):
            surface_forms[(c.surface_form, c.context)].append((c, "software name"))

        res = {}
        for k, lst in surface_forms.items():
            if len(lst) > 1:
                res[k] = lst
        return res

    def repair_ambiguous_contexts(self, text: str, candidates: list[Candidate]) -> bool:
        """
        Repairs candidates with ambiguous contexts by asking the model to differentiate them.

        :param text: original text from which the candidates were extracted
        :param candidates: list of candidates to repair/check
        :return: flag whether repair was done
        """
        flag = False
        if len(ambiguous_contexts := self.ambiguous_contexts_candidates(text, candidates)) > 0:
            flag = True
            for context, (candidates_with_annot_type, start_offsets) in ambiguous_contexts.items():
                for candidate, annot_type in candidates_with_annot_type:
                    problems = [
                        f"The context '{context}' appears {len(start_offsets)} times in the text, at offsets {start_offsets}. Please differentiate the context to make it unambiguous. The context is used for the candidate as {annot_type}."
                    ]
                    candidate = self.send_repair_problem_request(text, candidates, candidate, problems)
                    for i, c in enumerate(candidates):
                        if c.id == candidate.id:
                            candidates[i] = candidate
                            break
        return flag

    def ambiguous_contexts_candidates(self, text: str, candidates: list[Candidate]) -> dict[
        str, tuple[list[tuple[Candidate, str]], list[int]]]:
        """
        Finds candidates with ambiguous contexts. A context is considered ambiguous if it appears in more than one place in the text.

        :param text: original text from which the candidates were extracted
        :param candidates: list of candidates to check
        :return: dictionary mapping context to (list of candidates with description of context type, list of start offsets in the text)
        """

        res = {}

        def proc_context(ctx: str, ctype: str, can: Candidate):
            matches = text.count(ctx)
            if matches > 1:
                if ctx not in res:
                    all_start_offsets = [m.start() for m in re.finditer(re.escape(ctx), text)]
                    res[ctx] = ([(can, ctype)], all_start_offsets)
                else:
                    res[ctx][0].append((can, ctype))

        for c in candidates:
            proc_context(c.context, "software name", c)

        return res

    def verify_candidates(self, text: str, candidates: list[Candidate]) -> list[Candidate]:
        """
        Verifies software mentions candidates in the text.

        :param text: The input text to verify software mentions.
        :param candidates: A list of software mentions candidates to verify.
        :return: A list of verified software mentions.
        """

        res = []
        for c in candidates:
            v = self.verify_candidate(text, candidates, c)
            if v is not None:
                res.append(v)
        return res

    def verify_candidate(self, text: str, candidates: list[Candidate], target: Candidate) -> Optional[Candidate]:
        """
        Verifies a single software mention candidate in the text.

        :param text: The input text to verify software mentions.
        :param candidates: A list of software mentions candidates to verify.
        :param target: The software mention candidate to verify.
        :return: verified software mention or None if not verified.
        """

        search_query = self.obtain_search_query(text, target)
        search_results = self.search.search(search_query).results
        try:
            start_offset = self.get_start_offset(text, target.context, target.surface_form)
            seen_in_contexts = self.software_database.get_contexts_sorted(
                target.surface_form,
                self.get_context_window(text, start_offset, target.surface_form, self.context_window_for_database)
            )
        except ValueError:
            # not possible to obtain start offset, we skip similarity sorting
            seen_in_contexts = self.software_database.get_contexts(target.surface_form)

        messages = [
            {"role": "system", "content": self.verification_system_prompt.render(
                data={"text": text, "candidates": candidates, "target": target, "search_results": search_results,
                      "seen_in_contexts": seen_in_contexts})},
        ]
        if self.verification_few_shot is not None:
            for role, content in self.verification_few_shot:
                messages.append({"role": role, "content": content})

        messages.append(
            {"role": "user", "content": self.verification_prompt.render(
                data={"text": text, "candidates": candidates, "target": target, "search_results": search_results,
                      "seen_in_contexts": seen_in_contexts})},
        )
        request = self.request_factory(
            custom_id="verify_candidate",
            model=self.model,
            message=messages,
            options=self.requests_options,
            response_format=VerifyCandidateResponse,
            reasoning=self.use_reasoning,
            fake_structured=self.fake_structured_format
        )
        api_output = self.model_api.process_single_request(request)
        # get the tool calls from structured output, as it is not currently supported by ollama to use tool calling and structured output together
        reply = api_output.response.get_raw_content()
        reply = json_repair.loads(reply)
        try:
            reply = VerifyCandidateResponse.model_validate(reply)
        except ValidationError as e:
            # try to repair the output
            repair_request = self.request_factory(
                custom_id="verify_candidate_repair_output",
                model=self.model,
                message=messages + [
                    {"role": "assistant", "content": json.dumps(reply)},
                    {"role": "user", "content": self.verification_repair_output_prompt.render(
                        data={"text": text, "candidates": candidates, "target": target,
                              "search_results": search_results, "error": str(e)})}
                ],
                options=self.requests_options,
                response_format=VerifyCandidateResponse,
                fake_structured=self.fake_structured_format
            )
            reply = self.model_api.process_single_request(repair_request).response.get_raw_content()
            reply = json_repair.loads(reply)
            try:
                reply = VerifyCandidateResponse.model_validate(reply)
            except ValidationError:
                print(
                    f"Warning: Could not validate repaired output in verify_candidate for candidate {target.surface_form}. Error: {e}",
                    file=sys.stderr)
                return None

        if not reply.is_software:
            return None
        return reply.verified

    def obtain_search_query(self, text: str, candidate: Candidate) -> str:
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
        raw_response = self.model_api.process_single_request(request).response.get_raw_content()
        response = json_repair.loads(raw_response)
        response = SearchQueryResponse.model_validate(response)
        return response.query.strip()

    def eval_function_calls(self, function_calls: list[FunctionCall]) -> list[tuple[bool, str]]:
        """
        Evaluates function calls and returns their results.

        :param function_calls: A list of function calls to evaluate.
        :return: A list of function call results. (error flag, result)
            If error flag is True, the result is an error message.
        """
        res = []
        for fc in function_calls:
            if fc.name == "search_engine":
                if "query" not in fc.arguments:
                    res.append((True, "Missing argument 'query' in function call"))
                    continue
                result = self.search.search(query=fc.arguments["query"])
                res.append((False, result.model_dump_json()))
            else:
                res.append((True, f"Unknown function call: {fc.name}"))
        return res
