####################
# Required Modules #
####################

# Generic/Built-in
import json
import os
import random
import uuid
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Dict
from tqdm import tqdm

# Libs
from haystack.components.generators import AzureOpenAIGenerator
from llama_index.finetuning import EmbeddingQAFinetuneDataset
from milvus_haystack import MilvusDocumentStore
from pandas import DataFrame
from scipy.spatial.distance import cosine

# Custom
from .utils import (
    format_answer_query_template,
    format_chunk_query_template,
    format_context_query_template,
    format_evaluate_chunk_template,
    format_separating_multi_query_template,
    reasoning_evolution,
    generalizing_evolution,
    multi_context_evolution,
    concretizing_evolution,
    comparative_question_evolution,
    constrained_evolution,
    hypothetical_scenario_evolution,
    in_breadth_evolution,
)

#############################
# Dataset Class - myDataset #
#############################

class myDataset(EmbeddingQAFinetuneDataset):
    """Inherits from LlamaIndex's EmbeddingQAFinetuneDataset class. This class is used to store the dataset generated,
    with additional attributes to store the expected answers for the questions generated. Contains queries, corpus,
    relevant_docs and expected_answers attributes to store the questions, contexts, and expected answers respectively. 
    Access any of these attributes to get the data stored in the dataset. Methods like save_json and from_json are used
    to save and load the dataset as a JSON file.

    """
    expected_answers: Optional[Dict[str, str]] = None

#########################################
# Abstract Class - DocumentStoreWrapper #
#########################################

class DocumentStoreWrapper(ABC):
    """Abstract class for document store wrappers. Document store wrappers are used to interact with the document store.
    For example, the MilvusDocumentStoreWrapper class is a wrapper for the MilvusDocumentStore class, which is used to
    interact with the Milvus database. To integrate with other document stores, a new wrapper class should be created
    that inherits from this class.

    """

    @abstractmethod
    def get_all_sources(self) -> List[str]:
        """Get all document sources from the document store. Document sources refer to file paths, and are used to filter
        chunks by their original documents. This allows for the generation of questions from specific sources, and is 
        necessary to run train_val_test_split and get_all_chunks methods.

        Returns:
            List[str]: List of sources.
        """
        raise NotImplementedError
    
    @abstractmethod
    def get_chunks_from_sources(
            self, 
            sources: List[str]
        ) -> List[Tuple[str, str]]:
        """Get chunks from the document store based on the sources provided. This allows for the generation of questions
        from specific sources.

        Args:
            sources (List[str]): List of sources to get chunks from.

        Returns:
            List[Tuple[str, str]]: List of chunks. Each chunk is a tuple in the form of (id, chunk).
        """
        raise NotImplementedError
    
    @abstractmethod
    def get_chunk_embedding(self, chunk: Tuple[str, str]) -> List[float]:
        """Get the embedding of a chunk. The embedding is used to retrieve similar chunks.

        Args:
            chunk (Tuple[str, str]): The chunk to get the embedding of. Chunk comes in the form of (id, chunk).

        Returns:
            List[float]: The embedding of the chunk.
        """
        raise NotImplementedError
    
    @abstractmethod
    def retrieve_similar_chunks(
            self, 
            chunk_embedding: List[float], 
            top_k: int, 
            sources: Optional[List[str]]
        ) -> List[str]:
        """Retrieve top_k chunks based on chunk_embedding provided. This method is run in get_n_contexts method. Provide 
        sources if you have specific sources to retrieve from (for eg. to prevent data leakage).

        Args:
            chunk_embedding (List[float]): The embedding of the chunk to retrieve similar chunks for.
            top_k (int): The number of similar chunks to retrieve.
            sources (List[str]): List of sources to be passed to get_n_random_chunks method, prevents data leakage during retrieval.

        Returns:
            List[str]: List of n similar chunks.
        """
        raise NotImplementedError
    
###################################################
# Milvus Integration - MilvusDocumentStoreWrapper #
###################################################
    
class MilvusDocumentStoreWrapper(DocumentStoreWrapper):
    """Wrapper class for the MilvusDocumentStore class. This class is used for the DatasetGenerator class to interact 
    with the Milvus database. 
    """

    def __init__(self, document_store: MilvusDocumentStore) -> None:
        self.document_store = document_store
    
    def get_all_sources(self) -> List[str]:
        """Get all document sources from Milvus document store. Document sources refer to file paths, and are used to filter
        chunks by their original documents. This allows for the generation of questions from specific sources, and is 
        necessary to run train_val_test_split and get_all_chunks methods.

        Returns:
            List[str]: List of sources.
        """
        sources = [doc["source"] for doc in self.document_store.col.query(
            expr="id != ''", 
            output_fields=["source"]
            )
        ]
        return list(set(sources))
    
    def get_chunks_from_sources(
            self, 
            sources: List[str]
        ) -> List[Tuple[str, str]]:
        """Get chunks from the document store based on the sources provided. This allows for the generation of questions
        from specific sources.

        Args:
            sources (List[str]): List of sources to get chunks from.

        Returns:
            List[Tuple[str, str]]: List of chunks. Each chunk is a tuple in the form of (id, chunk).
        """
        chunks = [(doc["id"], doc["text"]) 
                  for doc in self.document_store.col.query(
                      expr=f"source in {sources}", 
                      output_fields=["text"]
                    )
                ]
        return chunks
    
    def get_chunk_embedding(self, chunk: str) -> List[float]:
        """Get the embedding of a chunk. The embedding is used to retrieve similar chunks.

        Args:
            chunk (Tuple[str, str]): The chunk to get the embedding of. Chunk comes in the form of (id, chunk).

        Returns:
            List[float]: The embedding of the chunk.
        """
        return self.document_store.col.query(
            expr=f"id == '{chunk[0]}'", 
            output_fields=["vector"]
            )[0]['vector']
    
    def retrieve_similar_chunks(
            self, 
            chunk_embedding: List[float], 
            top_k: int, 
            sources: List[str]
        ) -> List[str]:
        """Retrieve top_k chunks based on chunk_embedding provided. This method is run in get_n_contexts method. Provide 
        sources if you have specific sources to retrieve from (for eg. to prevent data leakage).

        Args:
            chunk_embedding (List[float]): The embedding of the chunk to retrieve similar chunks for.
            top_k (int): The number of similar chunks to retrieve.
            sources (List[str]): List of sources to be passed to get_n_random_chunks method, prevents data leakage during retrieval.

        Returns:
            List[str]: List of n similar chunks.
        """
        if sources:
            filters = {
                "field": "source",
                "operator": "in",
                "value": sources,
            }
        else:
            filters = None
        return self.document_store._embedding_retrieval(
            chunk_embedding, 
            filters=filters, 
            top_k=top_k
        )

########################################
# Dataset Generator - DatasetGenerator #
########################################

class DatasetGenerator:
    """Generates a dataset of questions from a collection of documents using a language model. Dataset generated will
    consist of questions, contexts (chunks in Milvus database that the questions are generated from), and the expected
    answers to the questions. The dataset can be saved as a JSON file for later use.
    """

    def __init__(
            self, 
            document_store_wrapper: DocumentStoreWrapper, 
            model: AzureOpenAIGenerator, 
            seed: int = 42
        ) -> None:
        """Initialises the DatasetGenerator class.

        Args:
            document_store (MilvusDocumentStore): The document store containing the collection of documents to generate 
                questions from.
            model (AzureOpenAIGenerator): The language model to be used for generating questions. Any language model that 
                is compatible with the Haystack library can be used. Haystack library can be found at 
                https://docs.haystack.deepset.ai/docs/generators.
            seed (int): The random seed to be used for reproducibility (default: 42).
        """
        self.document_store_wrapper = document_store_wrapper
        self.model = model
        self.seed = seed

    def train_val_test_split(
            self,
            split_ratio: List = [0.6, 0.2, 0.2], 
        ) -> tuple[List[str], List[str], List[str]]:
        """
        Splits a collection of documents into training and validation sets based on a given ratio.

        Args:
            collection (list): A collection of documents to be split.
            split_ratio (float): The ratio of documents to be included in the training set (default: 0.8).

        Returns:
            tuple: A tuple containing the training and validation sets of documents.
        """
        random.seed(self.seed)
        sources = self.document_store_wrapper.get_all_sources()
        random.shuffle(sources)
        train_split_index = int(len(sources) * split_ratio[0])
        val_split_index = int(len(sources) * (split_ratio[0] + split_ratio[1]))
        train_sources = sources[:train_split_index]
        val_sources = sources[train_split_index:val_split_index]
        test_sources = sources[val_split_index:]
        train_chunks = self.document_store_wrapper.get_chunks_from_sources(train_sources)
        val_chunks = self.document_store_wrapper.get_chunks_from_sources(val_sources)
        test_chunks = self.document_store_wrapper.get_chunks_from_sources(test_sources)
        return train_chunks, val_chunks, test_chunks, train_sources, val_sources, test_sources
    
    def get_all_chunks(self) -> List[Tuple[str, str]]:
        """Get all chunks from the document store.

        Returns:
            List[Tuple[str, str]]: List of chunks with format (id, chunk).
        """
        sources = self.document_store_wrapper.get_all_sources()
        chunks = self.document_store_wrapper.get_chunks_from_sources(sources)
        return chunks

    def generate_dataset(
            self,
            number_of_questions: int,
            chunks: List[Tuple[str, str]],
            generate_answers: bool,
            get_multi_context: bool = False,
            evolve_queries: bool = False,
            evolve_steps: List[str] = ["reasoning_evolution", "generalizing_evolution"],
            json_path: Optional[str] = '',
            sources: Optional[List[str]] = None,
            chunk_size_threshold: Optional[int] = 200,
            max_chunks_per_context: int = 5,
            min_chunks_per_context: int = 2,
            similarity_threshold: Optional[float] = 0.5,
    ):
        """Generate a dataset of questions from a list of chunks. The dataset will consist of questions, contexts (chunks
        in Milvus database that the questions are generated from), and the expected answers to the questions. The dataset
        can be saved as a JSON file for later use. To get chunks for generation, run either train_val_test_split or 
        get_all_chunks. If get_multi_context is set to True, questions will be generated from multiple contexts.

        Args:
            number_of_questions (int): The number of questions to generate.
            chunks (List[str]): List of chunks to generate questions from in format (id, chunk).
            generate_answers (bool): Whether to generate answers for the questions.
            get_multi_context (bool): Whether to generate questions from multiple contexts.
            evolve_queries (bool): Whether to evolve the questions.
            steps (List[str]): List of steps to evolve the questions (default: ["reasoning_evolution", "generalizing_evolution"]).
            json_path (str): The file path to save the dataset as a JSON file (default: '').
            sources (List[str]): List of sources to be passed to get_n_random_chunks method, prevents data leakage during retrieval.
            chunk_size_threshold (int): The threshold for the size of the chunks to be considered for generating questions.
            max_chunks_per_context (int): The maximum number of chunks to concatenate together to form a context.
            min_chunks_per_context (int): The minimum number of chunks to concatenate together to form a context.
            similarity_threshold (float): The similarity threshold to be used for filtering similar chunks. Value should be
                between 0 and 1.

        Returns:
            myDataset: A dataset of question-context pairs.
        """
        basename = 'data'
        if not os.path.exists(basename):
            os.mkdir(basename)
        json_path = os.path.join(basename, json_path)
        if get_multi_context:
            dataset = self.generate_multi_context_queries(
                n = number_of_questions,
                chunks = chunks,
                sources = sources, 
                json_path = json_path,
                chunk_size_threshold = chunk_size_threshold,
                max_chunks_per_context = max_chunks_per_context,
                min_chunks_per_context = min_chunks_per_context,
                similarity_threshold = similarity_threshold,
            )
        else:
            dataset = self.generate_n_single_chunk_queries(
                n = number_of_questions,
                chunks = chunks,
                json_path = json_path,
                chunk_size_threshold = chunk_size_threshold,
            )
        if evolve_queries:
            dataset = self.evolve_questions(dataset, json_path, evolve_steps)
        if generate_answers:
            dataset = self.answer_query(dataset, json_path)
        return dataset

    def evolve_questions(
        self,
        data: myDataset, 
        json_path: str,
        evolve_steps: List[str] = ["reasoning_evolution", "generalizing_evolution"]
    ) -> myDataset:
        """
            Takes in a JSON file (in dictionary format) and evolves the queries,
            returning a dictionary with the evolved queries, corpus, and relevant 
            document mappings.

            By default, this function produces two types of evolved queries (you can add more):

            - Generalizing: Broadens the scope of the query to capture more general aspects.
            - Reasoning: Adds complexity to the query to prompt deeper reasoning.

            Example input format:
            {
                "queries": {
                    "4c3195c0-1a63-4960-b5ba-33d8955deef4": "How did the transition from zine to website contribute to Boing Boing's popularity?",
                    "1a2b3c4d-5678-9101-1121-314151617181": "What role did co-editors play in the success of Boing Boing?"
                },
                "corpus": {
                    "34847e52-107d-453e-a0bd-00272ae5d5b9": "Bloggers commenting on the change at the time mentioned the shift in media consumption...",
                    "eb9fb4e7-2dc8-4d46-ad97-3f9d170deb69": "Boing Boing's co-editors curated content that attracted a diverse readership..."
                },
                "relevant_docs": {
                    "4c3195c0-1a63-4960-b5ba-33d8955deef4": [
                        "34847e52-107d-453e-a0bd-00272ae5d5b9",
                        "eb9fb4e7-2dc8-4d46-ad97-3f9d170deb69"
                    ],
                    "1a2b3c4d-5678-9101-1121-314151617181": [
                        "eb9fb4e7-2dc8-4d46-ad97-3f9d170deb69"
                    ]
                }
            }

            Parameters:
            - data (myDataset): Input data of myDataset object type.

            Returns:
            - myDataset object

            Example Usage:
            >>> with open("input.json", "r") as f:
            >>>     data = json.load(f)
            >>> output = evolve_questions(data)
            >>> with open("output.json", "w") as f:
            >>>     json.dump(output, f, indent=4)
        """

        EVOLUTION_MAPPINGS = {
            "reasoning_evolution": reasoning_evolution,
            "generalizing_evolution": generalizing_evolution,
            "in_breadth_evolution": in_breadth_evolution,
            "concretizing_evolution": concretizing_evolution,
            "multi_context_evolution": multi_context_evolution,
            "constrained_evolution": constrained_evolution,
            "comparative_question_evolution": comparative_question_evolution,
            "hypothetical_scenario_evolution": hypothetical_scenario_evolution
        }

        # Temporary dictionaries to store new queries and relevant_docs
        new_queries = {}
        new_relevant_docs = {}

        for doc_key, context_keys in tqdm(data.relevant_docs.items(), total=len(data.relevant_docs)):
            # Get the original query and its context
            original_query = data.queries[doc_key]
            context_concat = ' '.join([data.corpus[context_key] for context_key in context_keys])

            # Store the original query (no change to doc_key)
            new_queries[doc_key] = original_query
            new_relevant_docs[doc_key] = context_keys

            # Perform query evolutions
            for step in evolve_steps:
                if step not in EVOLUTION_MAPPINGS:
                    raise NotImplementedError(f"Step '{step}' is not implemented.")
                evolved_query = self.model.run(EVOLUTION_MAPPINGS[step](original_query, context_concat))['replies'][0]
                evolved_query_uuid = str(uuid.uuid4())

                # Add the evolved query with a new UUID
                new_queries[evolved_query_uuid] = evolved_query
                # Link the evolved query to the same context keys in relevant_docs
                new_relevant_docs[evolved_query_uuid] = context_keys

        # Update data with the new queries and relevant_docs
        data.queries.update(new_queries)
        data.relevant_docs.update(new_relevant_docs)

        # Export checkpoint data to json path
        if json_path:
            data.save_json(json_path)

        return data

    def generate_multi_context_queries(
            self,
            n: int,
            chunks: List[Tuple[str, str]],
            sources: Optional[List[str]],
            json_path: Optional[str] = None,
            chunk_size_threshold: Optional[int] = 200,
            max_chunks_per_context: int = 5,
            min_chunks_per_context: int = 2,
            similarity_threshold: Optional[float] = 0.5,
        ) -> myDataset:
        """Generate questions from a list of contexts.

        Args:
            n (int): Number of chunks to generate from.
            chunks (List[List[Tuple[str, str]]]): List of contexts, where each context is a list of tuples in the form of
                [(id, chunk), (id, chunk), ...].
            sources (List[str]): List of sources to be passed to get_n_random_chunks method, prevents data leakage during retrieval.
            json_path (str): The file path to save the dataset as a JSON file (default: None).
            max_chunks_per_context (int): The maximum number of chunks to concatenate together to form a context.
            min_chunks_per_context (int): The minimum number of chunks to concatenate together to form a context.
            similarity_threshold (float): The similarity threshold to be used for filtering similar chunks. Value should be
            between 0 and 1.

        Returns:
            myDataset: A dataset of question-context pairs.
        """
        contexts = self.get_n_contexts(
            n, 
            chunks, 
            sources, 
            chunk_size_threshold=chunk_size_threshold,
            max_chunks_per_context=max_chunks_per_context,
            min_chunks_per_context=min_chunks_per_context,
            similarity_threshold=similarity_threshold, 
        )
        corpus = {}
        for context in contexts:
            for chunk in context:
                if chunk[0] not in corpus:
                    corpus[chunk[0]] = chunk[1]
        queries = {}
        relevant_docs = {}
        for context in tqdm(contexts, desc="Generating Queries"):
            prompt = format_context_query_template(context, len(context))
            query = json.loads(self.model.run(prompt)['replies'][0])
            query_id = str(uuid.uuid4())
            queries[query_id] = query
            relevant_docs[query_id] = [chunk[0] for chunk in context]
        original_dataset = myDataset(
            queries=queries, 
            corpus=corpus, 
            relevant_docs=relevant_docs
        )
        cleaned_dataset = self.separate_query(original_dataset, json_path)

        # Export checkpoint data to json path
        if json_path:
            cleaned_dataset.save_json(json_path)

        return cleaned_dataset

    def generate_n_single_chunk_queries(
            self,
            n: int,
            chunks: List[Tuple[str, str]],
            json_path: Optional[str] = None,
            chunk_size_threshold: Optional[int] = 200,
        ) -> myDataset:
        """Generate n questions from a list of chunks.

        Args:
            n (int): Number of questions to generate.
            chunks (List[str]): List of chunks to generate questions from in the format (id, chunk).
            json_path (str): The file path to save the dataset as a JSON file (default: None).
            chunk_size_threshold (int): The threshold for the size of the chunks to be considered for generating questions.

        Returns:
            myDataset: A dataset of question-chunk pairs.
        """
        chunks = [chunk for chunk in chunks if len(chunk[1]) > chunk_size_threshold]
        random_chunks = self.get_n_random_chunks(chunks, n)
        corpus = {chunk[0]: chunk[1] for chunk in random_chunks}
        queries = {}
        relevant_docs = {}
        for chunk in tqdm(random_chunks, desc="Generating Queries"):
            prompt = format_chunk_query_template(chunk[1])
            query = self.model.run(prompt)['replies'][0]
            query_id = str(uuid.uuid4())
            queries[query_id] = query
            relevant_docs[query_id] = [chunk[0]]
        dataset = myDataset(
            queries=queries, 
            corpus=corpus, 
            relevant_docs=relevant_docs
        )

        # Export checkpoint data to json path
        if json_path:
            dataset.save_json(json_path)

        return dataset

    def get_n_contexts(
            self,
            n: int,
            chunks: List[Tuple[str, str]],
            sources: Optional[List[str]],
            max_chunks_per_context: int = 5,
            min_chunks_per_context: int = 2,
            chunk_size_threshold: Optional[int] = 200,
            similarity_threshold: Optional[float] = 0.5,
        ) -> List[List[Tuple[str, str]]]:
        """Get n contexts from the chunks. Contexts are defined as a list of chunks, so as to provide sufficient context
        to generate questions from.

        Args:
            n (int): Number of contexts to generate.
            chunks (List[Tuple[str, str]]): List of chunks to generate contexts from.
            sources (List[str]): List of sources to be passed to get_n_random_chunks method, prevents data leakage during retrieval.
            max_chunks_per_context (int): The maximum number of chunks to concatenate together to form a context.
            min_chunks_per_context (int): The minimum number of chunks to concatenate together to form a context.
            chunk_size_threshold (int): The threshold for the size of the chunks to be considered for generating questions.
            similarity_threshold (float): The similarity threshold to be used for filtering similar chunks. Value should be
                between 0 and 1.

        Returns:
            List[List[Tuple[str, str]]]: List of n contexts, where each context is a list of tuples in the form of 
            [(id, chunk), (id, chunk), ...].
        """
        random_chunks = self.get_n_random_chunks(chunks, 5*n)
        contexts = []
        for random_chunk in tqdm(random_chunks, desc="Building Contexts"):
            contexts.append([random_chunk])
            chunk_embedding = self.document_store_wrapper.get_chunk_embedding(random_chunk)
            similar_chunks = self.document_store_wrapper.retrieve_similar_chunks(
                chunk_embedding=chunk_embedding, 
                top_k=10, 
                sources=sources
            )
            for chunk in similar_chunks:
                i = len(contexts) - 1
                if len(chunk.content) > chunk_size_threshold and self.evaluate_chunk(chunk.content) == 1:
                    similarity = 1 - cosine(chunk_embedding, chunk.embedding)
                    if similarity > similarity_threshold and chunk.content != random_chunk[1]:
                        contexts[i].append((chunk.id, chunk.content))
                    elif chunk.content != random_chunk[1]:
                        continue
                    else:
                        break
                if len(contexts[i]) == max_chunks_per_context:
                    break
            if len(contexts[i]) <= min_chunks_per_context:
                contexts.pop()
            if len(contexts) == n:
                break
        return contexts

    def get_n_random_chunks(
            self,
            chunks: List[Tuple[str, str]], 
            n: int,
        ) -> List[Tuple[str, str]]:
        """Get n random chunks that contain sufficient context to generate questions from from a list of chunks.

        Args:
            chunks (List[Tuple[str, str]]): List of chunks to get random chunks from in format (id, chunk).
            n (int): Number of random chunks to get.
            seed (int): Random seed.

        Returns:
            List[Tuple[str, str]]: List of n random chunks.
        """
        if n > len(chunks):
            raise ValueError(f"{n} chunks requested is greater than the number of usable chunks provided, {len(chunks)}.")
        usable_chunks = []
        random.seed(self.seed)
        random.shuffle(chunks)
        with tqdm(total=n, desc="Generating Random Chunks") as pbar:
            while len(usable_chunks) < n and len(chunks) != 0:
                chunk = chunks.pop()
                if self.evaluate_chunk(chunk) == 1:
                    usable_chunks.append(chunk)
                    pbar.update(1)
            if len(usable_chunks) < n:
                print(f"Only {len(usable_chunks)} chunks were generated.")
        return usable_chunks    

    def evaluate_chunk(self, chunk) -> float:
        """
        Calls LLM to evaluate chunk based on self_containment and not_metadata. Returns 1 only if chunk is self-contained 
        and not metadata.

        Args:
            chunk (str): The chunk to be evaluated.

        Returns:
            float: A score of 1 if the chunk is self-contained and not metadata, 0 otherwise.
        """
        prompt = format_evaluate_chunk_template(chunk)
        try:
            res = json.loads(self.model.run(prompt)['replies'][0])
        except json.JSONDecodeError as e:
            print(f"JSON decoding failed: {e}")
            print("Falling back to default score of 0.")
            return 0
        if res.get('self_containment') == 0 or res.get('not_metadata') == 0:
            return 0
        return 1
        
    def separate_query(self, dataset: myDataset, json_path: str) -> myDataset:
        """
        Separate queries that include more than one question within them. Keeps queries that require multiple chunks to answer.

        Args:
            dataset (myDataset): The dataset containing the queries and relevant chunks.

        Returns:
            myDataset: The updated dataset with separated queries.

        """
        queries = dataset.queries
        corpus = dataset.corpus
        relevant_docs = dataset.relevant_docs
        for query_id, query in queries.copy().items():
            chunks = []
            doc_ids = []
            question_words = ["what", "how", "why", "when", "where", "who", "which"]
            for i in question_words:
                if "and " + i in query:
                    doc_ids = relevant_docs[query_id]
                    queries.pop(query_id)
                    relevant_docs.pop(query_id)
            chunks = [corpus[doc_id] for doc_id in doc_ids]
            if len(chunks) > 0:
                prompt = format_separating_multi_query_template(query, chunks)
                res = json.loads(self.model.run(prompt)['replies'][0])
                for new_query, new_chunks in res.items():
                    if len(new_chunks) > 1:
                        chunks_for_new_query = []
                        doc_ids_for_new_query = []
                        new_query_id = str(uuid.uuid4())
                        queries.update({new_query_id: new_query})
                        for i in new_chunks:
                            idx = i-1
                            chunks_for_new_query.append(chunks[idx])
                            doc_ids_for_new_query.append(doc_ids[idx])
                        relevant_docs.update({new_query_id: doc_ids_for_new_query})
        dataset = myDataset(
            queries=queries, 
            corpus=corpus, 
            relevant_docs=relevant_docs
        )

        # Export checkpoint data to json path
        if json_path:
            dataset.save_json(json_path)

        return dataset

    def answer_query(self, dataset: myDataset, json_path: str) -> myDataset:
        """
        Answer a query based on the information provided in the chunks.

        Args:
            dataset (myDataset): The dataset containing the queries and relevant chunks.

        Returns:
            myDataset: The updated dataset with the expected answers.
        """
        answers = {}
        for query_id, query in tqdm(dataset.queries.items(), desc="Answering Queries"):
            chunk_ids = dataset.relevant_docs[query_id]
            chunks = [dataset.corpus[chunk_id] for chunk_id in chunk_ids]
            prompt = format_answer_query_template(query, chunks)
            res = self.model.run(prompt)['replies'][0]
            answers[query_id] = res
        dataset.expected_answers = answers
        
        # Export checkpoint data to json path
        if json_path:
            dataset.save_json(json_path)

        return dataset
    
    def dataset_mapping(
            self, 
            dataset: myDataset, 
            deep_eval_format: bool = False
        ) -> DataFrame:
        """
        Map the dataset to a DataFrame for easy visualisation. Turn deep_eval_format to True to format the dataset in the
        format required for deep evaluation.

        Args:
            dataset (myDataset): The dataset to be mapped.
            deep_eval_format (bool): Whether to format the dataset for deep eval evaluation, whereby entire context are 
                just all the chunks in context concatenated together.

        Returns:
            DataFrame: A DataFrame containing the dataset.
        """
        data = []
        for query_id, query in dataset.queries.items():
            context_ids = dataset.relevant_docs[query_id]
            if deep_eval_format:
                context_str = json.dumps([dataset.corpus[context_id] 
                    for context_id in context_ids])
                data.append({
                    "input": query,
                    "expected_output": dataset.expected_answers[query_id],
                    "context": context_str,
                })
            else:
                context_dict = {
                    f"context_{i + 1}": dataset.corpus[context_id] 
                    for i, context_id in enumerate(context_ids)
                }
                if dataset.expected_answers:
                    data.append({
                        "query": query,
                        "expected_answer": dataset.expected_answers[query_id],
                        **context_dict,
                    })
                else:
                    data.append({
                        "query": query,
                        **context_dict,
                    })
        return DataFrame(data)