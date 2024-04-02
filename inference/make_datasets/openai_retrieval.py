import os
import json
import shutil
import traceback
import numpy as np
from filelock import FileLock
from tqdm import tqdm
from typing import Any, List, Dict
from argparse import ArgumentParser
from datasets import load_from_disk, load_dataset
from pathlib import Path

from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

try:
    from bm25_retrieval import ContextManager, clone_repo, get_missing_ids, get_root_dir, get_remaining_instances, DOCUMENT_ENCODING_FUNCTIONS
    from utils import string_to_bool, list_files
except:
    from .bm25_retrieval import ContextManager, clone_repo, get_missing_ids, get_root_dir, get_remaining_instances, DOCUMENT_ENCODING_FUNCTIONS
    from utils import string_to_bool, list_files

import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

from dotenv import find_dotenv, load_dotenv

dotenv_file = find_dotenv()
result = load_dotenv(find_dotenv(), override=True)


def build_openai_index(
    sentences: List[str],
    metadatas: Dict[str, str], 
    index_path: Path, 
    do_summary: bool=False,
):
    embedding_function = OpenAIEmbeddings(
        model="text-embedding-3-large",
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        max_retries=10
    )
    
    logger.info("Starting to embed")

    index = FAISS.from_texts(sentences, embedding_function, metadatas=metadatas)
    logger.info("Done embedding")

    index.save_local(str(index_path))
    
    index.embedding_function = embedding_function.embed_query

    logger.info("faiss saved to pickle {}".format(str(index_path)))

    return index


def build_documents(repo_dir, commit, document_encoding_func):
    """
    Builds a dictionary of documents from a given repository directory and commit.

    Args:
        repo_dir (str): The path to the repository directory.
        commit (str): The commit hash to use.
        document_encoding_func (function): A function that takes a filename and a relative path and returns the encoded document text.

    """
    sentences = list()
    metadatas = list()
    with ContextManager(repo_dir, commit):
        filenames = list_files(repo_dir, include_tests=False)
        for relative_path in filenames:
            filename = os.path.join(repo_dir, relative_path)
            text = document_encoding_func(filename, relative_path)
            sentences.append(text)
            metadatas.append({"relative_path": relative_path})
    return sentences, metadatas


def make_index(
    repo_dir,
    root_dir,
    query,
    commit,
    document_encoding_func,
    instance_id,
):
    """
    Builds an index for a given set of documents using Pyserini.

    Args:
        repo_dir (str): The path to the repository directory.
        root_dir (str): The path to the root directory.
        query (str): The query to use for retrieval.
        commit (str): The commit hash to use for retrieval.
        document_encoding_func (function): The function to use for encoding documents.
        instance_id (int): The ID of the current instance.

    Returns:
        index_path (Path): The path to the built index.
    """
    index_path = Path(root_dir, f"index__{str(instance_id)}", "index")
    if index_path.exists():
        return index_path
    sentences, metadatas = build_documents(repo_dir, commit, document_encoding_func)
    build_openai_index(sentences, metadatas, index_path)
    return index_path


def get_index_paths_worker(
    instance,
    root_dir_name,
    document_encoding_func,
    token,
):
    index_path = None
    repo = instance["repo"]
    commit = instance["base_commit"]
    instance_id = instance["instance_id"]
    try:
        repo_dir = clone_repo(repo, root_dir_name, token)
        query = instance["problem_statement"]
        index_path = make_index(
            repo_dir,
            root_dir_name,
            query,
            commit,
            document_encoding_func,
            instance_id,
        )
    except:
        logger.error(f"Failed to process {repo}/{commit} (instance {instance_id})")
        logger.error(traceback.format_exc())
    return instance_id, index_path


def get_index_paths(
    remaining_instances: List[Dict[str, Any]],
    root_dir_name: str,
    document_encoding_func: Any,
    token: str,
    output_file: str,
) -> Dict[str, str]:
    """
    Retrieves the index paths for the given instances using multiple processes.

    Args:
        remaining_instances: A list of instances for which to retrieve the index paths.
        root_dir_name: The root directory name.
        document_encoding_func: A function for encoding documents.
        token: The token to use for authentication.
        output_file: The output file.
        num_workers: The number of worker processes to use.

    Returns:
        A dictionary mapping instance IDs to index paths.
    """
    all_index_paths = dict()
    for instance in tqdm(remaining_instances, desc="Indexing"):
        instance_id, index_path = get_index_paths_worker(
            instance,
            root_dir_name,
            document_encoding_func,
            token,
        )
        if index_path is None:
            continue
        all_index_paths[instance_id] = index_path
    return all_index_paths


def load_index(index_path: Path):
    embedding_function = OpenAIEmbeddings(
        model="text-embedding-3-large",
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        max_retries=10
    )
    
    assert index_path.exists(), f"{index_path} does not exist"

    index = FAISS.load_local(str(index_path), embedding_function, allow_dangerous_deserialization=True)
    
    index.embedding_function = embedding_function.embed_query
    
    return index


def search(instance, index_path):
    """
    Searches for relevant documents in the given index for the given instance.

    Args:
        instance (dict): The instance to search for.
        index_path (str): The path to the index to search in.

    Returns:
        dict: A dictionary containing the instance ID and a list of hits, where each hit is a dictionary containing the
        document ID and its score.
    """
    try:
        instance_id = instance["instance_id"]
        cutoff = len(instance["problem_statement"])
        index = load_index(index_path)
        while True:
            try:
                query_embedding = index.embedding_function(instance["problem_statement"][:cutoff])
                nearest_neighbor_distances, nearest_neighbor_indices = index.index.search(
                        np.array([query_embedding], dtype=np.float32), 20
                    )
                # Return nearest neighbor
                nearest_neighbor_indices = nearest_neighbor_indices[0]
                nearest_neighbor_distances = nearest_neighbor_distances[0]

                hits = []
                for i, nearest_neighbor_index in enumerate(nearest_neighbor_indices):
                    if nearest_neighbor_index == -1:
                        continue

                    idx = index.index_to_docstore_id[nearest_neighbor_index]
                    doc = index.docstore.search(idx)
                    file_relative_path = doc.metadata['relative_path']
                    score = nearest_neighbor_distances[i]
                    if isinstance(score, np.float32):
                        score = score.item()

                    hits.append(
                        {'docid': file_relative_path, 'score': score}
                    )

            except Exception as e:
                if "maxClauseCount" in str(e):
                    cutoff = int(round(cutoff * 0.8))
                    continue
                else:
                    raise e
            break
        results = {"instance_id": instance_id, "hits": []}
        for hit in hits:
            results["hits"].append({"docid": hit['docid'], "score": hit['score']})
        return results
    except Exception as e:
        logger.error(f"Failed to process {instance_id}")
        logger.error(traceback.format_exc())
        return None
    

def search_indexes(remaining_instance, output_file, all_index_paths):
    """
    Searches the indexes for the given instances and writes the results to the output file.

    Args:
        remaining_instance (list): A list of instances to search for.
        output_file (str): The path to the output file to write the results to.
        all_index_paths (dict): A dictionary mapping instance IDs to the paths of their indexes.
    """
    for instance in tqdm(remaining_instance, desc="Retrieving"):
        instance_id = instance["instance_id"]
        if instance_id not in all_index_paths:
            continue
        index_path = all_index_paths[instance_id]
        results = search(instance, index_path)
        if results is None:
            continue
        with FileLock(output_file.as_posix() + ".lock"):
            with open(output_file, "a") as out_file:
                print(json.dumps(results), file=out_file, flush=True)


def main(
    dataset_name_or_path,
    document_encoding_style,
    output_dir,
    shard_id,
    num_shards,
    splits,
    leave_indexes,
    debug,
):
    document_encoding_func = DOCUMENT_ENCODING_FUNCTIONS[document_encoding_style]
    token = os.environ.get("GITHUB_TOKEN", "git")
    if Path(dataset_name_or_path).exists():
        dataset = load_from_disk(dataset_name_or_path)
        dataset_name = os.path.basename(dataset_name_or_path)
    else:
        dataset = load_dataset(dataset_name_or_path)
        dataset_name = dataset_name_or_path.replace("/", "__")
    if shard_id is not None:
        for split in splits:
            dataset[split] = dataset[split].shard(num_shards, shard_id)
    instances = list()
    if set(splits) - set(dataset.keys()) != set():
        raise ValueError(f"Unknown splits {set(splits) - set(dataset.keys())}")
    for split in splits:
        instances += list(dataset[split])
    if debug:
        instances = instances[:2]
    output_file = Path(
        output_dir, dataset_name, document_encoding_style + ".retrieval.jsonl"
    )
    remaining_instances = get_remaining_instances(instances, output_file)
    root_dir, root_dir_name = get_root_dir(
        dataset_name, output_dir, document_encoding_style
    )
    try:
        all_index_paths = get_index_paths(
            remaining_instances,
            root_dir_name,
            document_encoding_func,
            token,
            output_file,
        )
    except KeyboardInterrupt:
        logger.info(f"Cleaning up {root_dir}")
        del_dirs = list(root_dir.glob("repo__*"))
        if leave_indexes:
            index_dirs = list(root_dir.glob("index__*"))
            del_dirs += index_dirs
        for dirname in del_dirs:
            shutil.rmtree(dirname, ignore_errors=True)
    logger.info(f"Finished indexing {len(all_index_paths)} instances")
    search_indexes(remaining_instances, output_file, all_index_paths)
    missing_ids = get_missing_ids(instances, output_file)
    logger.warning(f"Missing indexes for {len(missing_ids)} instances.")
    logger.info(f"Saved retrieval results to {output_file}")
    del_dirs = list(root_dir.glob("repo__*"))
    logger.info(f"Cleaning up {root_dir}")
    if leave_indexes:
        index_dirs = list(root_dir.glob("index__*"))
        del_dirs += index_dirs
    for dirname in del_dirs:
        shutil.rmtree(dirname, ignore_errors=True)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--dataset_name_or_path",
        type=str,
        default="princeton-nlp/SWE-bench",
        help="Dataset to use for test set from HuggingFace Datasets or path to a save_to_disk directory.",
    )
    parser.add_argument(
        "--document_encoding_style",
        choices=DOCUMENT_ENCODING_FUNCTIONS.keys(),
        default="file_name_and_contents",
    )
    parser.add_argument("--output_dir", default="./retreival_results")
    parser.add_argument("--splits", nargs="+", default=["train", "test"])
    parser.add_argument("--shard_id", type=int)
    parser.add_argument("--num_shards", type=int, default=20)
    parser.add_argument("--leave_indexes", type=string_to_bool, default=True)
    parser.add_argument("--debug", type=string_to_bool, default=False)
    args = parser.parse_args()
    main(**vars(args))