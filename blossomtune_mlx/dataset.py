"""blossomtunellm-mlx: A Flower client app for federated learning with MLX."""

import re
import json
from pathlib import Path
from typing import Tuple, Optional, Dict

from datasets import Dataset, DatasetDict
from transformers import PreTrainedTokenizer
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from jinja2 import Template, StrictUndefined
from jinja2.exceptions import TemplateError, UndefinedError
from mlx_lm.tuner.datasets import CacheDataset, create_dataset


FDS = {}  # Cache FederatedDataset


def load_local_dataset(
    data_path: Path,
    tokenizer: PreTrainedTokenizer,
    config: Dict,
    split: str = "train",
):
    def load_subset(path):
        if not path.exists():
            return []
        with open(path, "r") as fid:
            data = [json.loads(line) for line in fid]
        return create_dataset(data, tokenizer, config)

    return load_subset(data_path / f"{split}.jsonl")


def render_template(template: str, context: dict) -> str:
    """Render jinja2 template string."""
    return Template(template, undefined=StrictUndefined).render(**context)


def extract_field_names_from_template(template_string: str) -> list[str]:
    """
    Extracts all unique field names (placeholders) from a template string.
    E.g., "{a} {b} {c}" -> ['a', 'b', 'c']
    """
    if not template_string:  # Handle empty template string
        return []
    pattern = r"\{\{\s*(\w+)\s*\}\}"
    return list(set(re.findall(pattern, template_string)))


def reformat_dynamic(
    example: dict, prompt_template: str, completion_template: str
) -> dict:
    """
    Reformat a single example based on dynamic field names specified in templates.
    """
    prompt_field_names = extract_field_names_from_template(prompt_template)
    prompt_kwargs = {
        field_name: example.get(field_name, "") for field_name in prompt_field_names
    }

    try:
        prompt_value = render_template(prompt_template, prompt_kwargs).strip()
    except (TemplateError, UndefinedError) as e:
        print(
            f"Warning: Prompt formatting error ({e}). Template: '{prompt_template}', Kwargs: {prompt_kwargs}"
        )
        prompt_value = ""

    completion_field_names = extract_field_names_from_template(completion_template)
    completion_kwargs = {
        field_name: example.get(field_name, "") for field_name in completion_field_names
    }

    try:
        completion_value = render_template(
            completion_template, completion_kwargs
        ).strip()
    except (TemplateError, UndefinedError) as e:
        print(
            f"Warning: Completion formatting error ({e}). Template: '{completion_template}', Kwargs: {completion_kwargs}"
        )
        completion_value = ""

    example["prompt"] = prompt_value
    example["completion"] = completion_value
    return example


def process_dataset_dynamic(dataset, prompt_template: str, completion_template: str):
    """
    Apply the dynamic reformat function to a Hugging Face Dataset or DatasetDict.
    """

    def map_fn(ex):
        return reformat_dynamic(ex, prompt_template, completion_template)

    if isinstance(dataset, DatasetDict):
        for split in dataset:
            dataset[split] = dataset[split].map(map_fn)
    elif isinstance(dataset, Dataset):
        dataset = dataset.map(map_fn)
    else:
        raise TypeError("Input must be a Hugging Face Dataset or DatasetDict.")
    return dataset


def load_data(
    partition_id: int,
    num_partitions: int,
    dataset_name: str,
    prompt_template: str,
    completion_template: str,
    data_path_base: str,
    tokenizer: PreTrainedTokenizer,
    train_split: str = "train",
    validation_split: Optional[str] = None,
    split_ratio: Optional[int] = None,
) -> Tuple[Dataset, Optional[Dataset], str]:
    """
    Load, process, and save dataset partitions for a client.

    This function handles three scenarios for creating train/validation sets:
    1. If `validation_split` is provided, it loads a separate partition for validation.
    2. If `split_ratio` is provided, it splits the train partition.
    3. Otherwise, it only creates a training set.
    """
    global FDS

    # TODO: allow customization
    dataset_config = {
        "mask_prompt": False,
        "prompt_feature": "prompt",
        "text_feature": "text",
        "completion_feature": "completion",
        "chat_feature": "messages",
    }

    # --- Setup Paths ---
    dataset_slug = re.sub(r"[^a-zA-Z0-9_-]", "", dataset_name.replace("/", "_"))
    save_dir = (
        Path(data_path_base) / "datasets" / dataset_slug / f"partition_{partition_id}"
    )
    valid_save_path = save_dir / "validation.jsonl"
    train_save_path = save_dir / "train.jsonl"
    save_dir.mkdir(parents=True, exist_ok=True)

    client_trainset = None
    client_validset = None
    if train_save_path.exists():
        client_trainset = load_local_dataset(
            save_dir, tokenizer, dataset_config, split="train"
        )
        print(f"Client {partition_id}: Loaded training data from {train_save_path}")
        if valid_save_path.exists():
            client_validset = load_local_dataset(
                save_dir, tokenizer, dataset_config, split="validation"
            )
            print(
                f"Client {partition_id}: Loaded validation data from {valid_save_path}"
            )
        return (
            CacheDataset(client_trainset),
            CacheDataset(client_validset),
            str(save_dir),
        )

    # --- Load and Process Train Data ---
    if FDS.get(train_split) is None:
        partitioner = IidPartitioner(num_partitions=num_partitions)
        FDS[train_split] = FederatedDataset(
            dataset=dataset_name,
            partitioners={train_split: partitioner},
        )
    client_trainset = FDS[train_split].load_partition(partition_id, train_split)
    client_trainset = process_dataset_dynamic(
        client_trainset, prompt_template, completion_template
    )

    # --- Handle Validation Data ---
    if validation_split:
        # Scenario 1: A specific validation split is provided
        print(
            f"Client {partition_id}: Loading separate validation split '{validation_split}'."
        )
        if FDS.get(validation_split) is None:
            partitioner = IidPartitioner(num_partitions=num_partitions)
            FDS[validation_split] = FederatedDataset(
                dataset=dataset_name,
                partitioners={validation_split: partitioner},
            )
        client_validset = FDS[validation_split].load_partition(
            partition_id, validation_split
        )
        client_validset = process_dataset_dynamic(
            client_validset, prompt_template, completion_template
        )

    elif split_ratio and 0 < split_ratio < 100:
        # Scenario 2: Split the training data into train/validation
        print(
            f"Client {partition_id}: Splitting train data with ratio {split_ratio}/{100 - split_ratio}."
        )
        split_dict = client_trainset.train_test_split(
            test_size=(100 - split_ratio) / 100.0, seed=42
        )
        client_trainset = split_dict["train"]
        client_validset = split_dict["test"]

    # --- Save Datasets to Disk ---
    # Save the final training set
    client_trainset.to_json(train_save_path, orient="records")
    client_trainset = load_local_dataset(
        save_dir, tokenizer, dataset_config, split="train"
    )
    print(f"Client {partition_id}: Saved training data to {train_save_path}")

    # Save the validation set if it exists
    if client_validset:
        client_validset.to_json(valid_save_path, orient="records")
        client_validset = load_local_dataset(
            save_dir, tokenizer, dataset_config, split="validation"
        )
        print(f"Client {partition_id}: Saved validation data to {valid_save_path}")

    # Return the train set, validation set, and the path to their directory
    return CacheDataset(client_trainset), CacheDataset(client_validset), str(save_dir)
