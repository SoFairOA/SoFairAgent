# SoFairAgent
This tool allows to extract software mentions and associated metadata (version, publisher, url, and language).

## Installation
Clone the repository and install the required dependencies:
```bash
pip install -r requirements.txt
```

### Models
To use local models install the Ollama according to the instructions at https://github.com/ollama/ollama. This will be used to run local LLM models.

There is also verifier model that always run on the local machine. It will be downloaded automatically. The used model is available at https://huggingface.co/SoFairOA/SoFairVerifier .

## Usage
Firstly insert the URL to your API (Ollama) to `config.yaml`. An example configuration is shown below:

```yaml
      cls: OllamaAPI  # name of class that is subclass of API
      config: # configuration for defined class
        api_key: ollama  # API key.
        base_url: http://example.com:11431 # Base URL for API.
```

You can run the script on whole dataset using

```bash
./run.py extract_dataset path_to_dataset --id_field id --text_field text --split train -c config.yaml
```

The `path_to_dataset` can be HuggingFace dataset identifier or path to local dataset in json/csv format. The local dataset could be simple file or folder with files named as splits (train.jsonl, test.jsonl, ...).

Define the `--id_field` and `--text_field` parameters to specify which fields in the dataset contain the unique identifier and the text to be processed, respectively. The `--split` parameter allows you to choose which split of the dataset to process (e.g., train, test, validation).

There is support to parallel processing. You can define `--world_size` and `--rank` parameters to split the work between multiple processes. It will divide the dataset into `world_size` parts and process only the part with index `rank`.

For example, you can try the `SoFairOA/sofair_dataset_splits`:

```bash
./run.py extract_dataset SoFairOA/sofair_dataset_splits --id_field id --text_field text --subset documents --split test -c config.yaml
```

The `--subset` parameter is optional and allows to specify a particular subset of the dataset if it contains multiple subsets.


There is also simple command line interface:


```bash
./run.py test_extract -c config.yaml
```


## How it works
* Firstly an LLM is used to extract set of candidates from the text. 
* In the second step a search engine is used to obtain auxiliary information about the candidates. 
* After that step the verification model is used to obtain confidence score for each candidate. 
* Finally, the given LLM is used to fill in missing metadata fields for each verified candidate.

