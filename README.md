# Dataset Generation

## Overview

This repository provides a dataset generation framework designed to generate datasets that can be used to:
1. Evaluate Retrieval-Augmented Generation (RAG) using LLM-as-a-Judge (DeepEval, RAGAS).
2. Conduct empirical evaluations of RAG.
3. Train models like COLBERT and Linear Adapters.

Our dataset generation class was created to address **poor query quality** from LlamaIndex's `generate_qa_embedding_pairs` and DeepEval's `Synthesiser`. By using our custom implementation, we ensure **consistent datasets** across all three tasks, leading to **fairer evaluations** and **better control** over query quality.

More details of implementation decisions can be found in [my documentation](./docs/documentation.md).

---

## **Installation**

To install the required dependencies, run:
```bash
pip install -r requirements.txt
```

## Code Structure
- dataset_generation.py: Main script for generating datasets.
- utils.py: Stores all the prompt templates for dataset generation.
- notebooks/: Contains example runs and Jupyter notebooks for interactive testing.
- docs/: Contains detailed documentation on dataset generation and evaluations.

## Usage Examples

### Prerequisites:

* Document store being used (in our case, Milvus) must contain chunks for query generation

### Generating Multi-context Dataset:

```python
# Configure connections
document_store = MilvusDocumentStore(...)
llm = AzureOpenAIGenerator()

# Connect to Milvus
milvus_wrapper = MilvusDocumentStoreWrapper(document_store=document_store)

# Call DatasetGenerator class
generator = DatasetGenerator(document_store_wrapper=milvus_wrapper, model=llm, seed=42)

# Obtain train_val_test_split
train_set, val_set, test_set, train_sources, val_sources, test_sources = generator.train_val_test_split(split_ratio=[0.6, 0.4, 0])

# If no train_test split required (ie. not using COLBERT or Linear Adapters), can just use all chunks
chunks, sources = generator.get_all_chunks()

# Generate multi-context dataset
val_dataset = generator.generate_dataset(
    number_of_questions=5,                    # Number of queries to generate
    chunks=val_set,                           # Adjust accordingly
    generate_answers=True,                    # Set to False if don't need to generate relevant answers (in our case, only LLM-as-a-judge requires relevant answers)
    get_multi_context=True,                   # Set to False if only generating single chunk-query pair
    evolve_queries=True,                      # Set to False if evolution not required
    evolve_steps=["generalizing_evolution"],  # Type of query evolution, 
    json_path='./val_multi_dataset.json',     # Output path for json document
    sources=val_sources,                      # To prevent data leakage, required for multi-context
    chunk_size_threshold=200,                 # Character level threshold, higher means larger chunks
    max_chunks_per_context= 5,                # Maximum number of chunks per context for multi-context
    min_chunks_per_context= 2,                # Minimum number of chunks per context for multi-context
    similarity_threshold = 0.5                # Cosine similarity threshold value for when grouping chunks into context, higher means stricter
)
```

### Generating Single-context Dataset:

```python
single_chunk_query_dataset = generator.generate_dataset(
    number_of_questions=5,                    # Number of queries to generate
    chunks=chunks,                            # Adjust accordingly
    generate_answers=True,                    # Set to False if don't need to generate relevant answers (in our case, only LLM-as-a-judge requires relevant answers)
    get_multi_context=False,                  # Set to False if only generating single chunk-query pair
    evolve_queries=True,                      # Set to False if evolution not required
    evolve_steps=["generalizing_evolution"],  # Type of query evolution, 
    json_path='./test_single_dataset.json',   # Output path for json document
    chunk_size_threshold=200,                 # Character level threshold, higher means larger chunks
)
```
Implementation can also be observed in our [notebook](./notebooks/dataset_generation.ipynb).
## Developer Notes

### Using a Different Vector Database
If using a different vector database from Milvus, inherit the `DocumentStoreWrapper` abstract base class and implement the methods inside. Connecting to the document store is necessary to retrieve chunks, and prevent data leakage if doing train-test split. Methods within `DatasetGenerator` class that require connection to document store include `train_val_test_split`, `get_all_chunks`, and `get_n_contexts`. This means that if the methods within the `DocumentStoreWrapper` are not properly implemented, you will not be able to obtain the chunks required for generation, and you will not be able to generate multi-context queries.

### Compatible Generators
Our DatasetGenerator currently supports AzureOpenAIGenerator. However, it can be modified to use any Haystack-compatible generator.
See Haystack’s documentation [here](https://docs.haystack.deepset.ai/docs/generators) for compatible models.

**Example**:
```python
from haystack.components.generators import HuggingFaceLocalGenerator

hf_generator = HuggingFaceLocalGenerator(model="google/flan-t5-large", task="text2text-generation")
generator = DatasetGenerator(document_store_wrapper=milvus_wrapper, model=hf_generator, seed=42)
```

### utils.py
Prompt templates are kept in `utils.py`, and should be edited based on your use case. In particular, the evolution templates are not refined, and can continue to be improved.