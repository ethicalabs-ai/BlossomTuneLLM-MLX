"""blossomtunellm-mlx: A Flower client app for federated learning with MLX."""

import os
from datetime import datetime
from typing import Dict, Tuple, Union, Iterable
from pathlib import Path
from slugify import slugify

import mlx.optimizers as optim
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from flwr.common.typing import NDArrays, Scalar
from omegaconf import DictConfig, OmegaConf
from mlx_lm import load
from mlx_lm.tuner import train
from mlx_lm.tuner.utils import linear_to_lora_layers
from mlx_lm.tuner.trainer import TrainingCallback
from transformers import AutoTokenizer

from blossomtune_mlx.config import get_run_config
from blossomtune_mlx.dataset import load_data

from blossomtune_mlx.models import (
    get_parameters,
    set_parameters,
    get_training_args,
    cosine_annealing,
)


# Suppress warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class LossCallback(TrainingCallback):
    """A simple callback to store the training loss."""

    def __init__(self):
        super().__init__()
        self.train_losses = []
        self.val_losses = []

    def on_train_loss_report(
        self,
        info: Dict[str, Union[float, int]],
    ) -> None:
        """Append the reported loss to the list."""
        loss = info.get("train_loss")
        if loss is not None:
            self.train_losses.append(loss)

    def on_val_loss_report(
        self,
        info: Dict[str, Union[float, int]],
    ) -> None:
        loss = info.get("val_loss")
        if loss is not None:
            self.val_losses.append(loss)

    def get_average_train_loss(self) -> float:
        """Calculate the average loss for the training run."""
        if not self.train_losses:
            return -1.0
        return sum(self.train_losses) / len(self.train_losses)

    def get_average_val_loss(self) -> float:
        """Calculate the average loss for the training run."""
        if not self.val_losses:
            return -1.0
        return sum(self.val_losses) / len(self.val_losses)


class MLXClient(NumPyClient):
    """
    A Flower client for federated fine-tuning using the mlx-lm library directly.
    """

    def __init__(
        self,
        partition_id: int,
        model_cfg: DictConfig,
        train_cfg: DictConfig,
        data_path: str,
        results_path: str,
        client_trainset: Iterable,
        client_valset: Iterable,
        num_examples: int,
        num_rounds: int,
    ):
        self.model_cfg = model_cfg
        self.train_cfg = train_cfg
        self.data_path = data_path  # Path to the directory with train.jsonl
        self.results_path = (
            results_path  # Path to the directory where to store model adapters.
        )
        self.client_trainset = client_trainset
        self.client_valset = client_valset
        self.num_examples = num_examples
        self.num_rounds = num_rounds

        # Load the model and tokenizer upon initialization
        print("Client: Loading model...")
        self.partition_id = partition_id
        self.model, self.tokenizer = load(self.model_cfg.name)
        self.model_slug = slugify(self.model_cfg.name)

        # Freeze the model and apply LoRA layers for fine-tuning
        print("Client: Applying LoRA layers...")
        self.model.freeze()

        # Convert OmegaConf DictConfig to a standard Python dict for the function
        lora_parameters_dict = OmegaConf.to_container(
            self.train_cfg.lora_parameters, resolve=True
        )
        use_dora = train_cfg.fine_tune_type == "dora"
        linear_to_lora_layers(
            self.model,
            self.train_cfg.lora_layers,
            lora_parameters_dict,
            use_dora=use_dora,
        )
        print(
            f"Client: Model loaded and configured for {'DoRA' if use_dora else 'LoRA'}."
        )

    def fit(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[NDArrays, int, Dict]:
        """
        Perform a local training round using the programmatic API of `mlx-lm`.
        """
        # Update the local model with the global parameters from the server
        set_parameters(self.model, parameters)

        # Calculate the learning rate for the current round
        new_lr = cosine_annealing(
            int(config["current_round"]),
            self.num_rounds,
            self.train_cfg.learning_rate_max,
            self.train_cfg.learning_rate_min,
        )
        current_time = datetime.now()
        current_time_str = current_time.strftime("%Y-%m-%d_%H-%M-%S")
        # Adapter output. Should we set this to a temporary file? Or saving it, optionally, for client adapter analysis?
        adapter_file = (
            Path(self.results_path)
            / self.model_slug
            / "clients"
            / current_time_str
            / f"partition_{self.partition_id}"
            / "adapters.safetensors"
        )
        os.makedirs(adapter_file.parent, exist_ok=True)

        # Define training arguments for the `mlx_lm.tuner.train` function
        # Note: `learning_rate` is NOT an argument here.
        training_args = get_training_args(self.train_cfg, str(adapter_file))

        # Load the dataset using the tuner's utility
        # TODO: add dataset config to pyproject.toml
        train_dataset = self.client_trainset
        val_dataset = self.client_valset
        print(f"Client: Starting training for {training_args.iters} iterations...")

        # Initialize our custom callback to capture the loss
        loss_callback = LossCallback()

        # Execute the training
        train(
            model=self.model,
            args=training_args,
            optimizer=optim.AdamW(learning_rate=new_lr),
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            training_callback=loss_callback,
        )

        print("Client: Training finished.")

        # Return the updated local parameters and the captured metrics
        new_parameters = get_parameters(self.model)
        avg_train_loss = loss_callback.get_average_train_loss()
        avg_val_loss = loss_callback.get_average_val_loss()
        metrics = {"train_loss": avg_train_loss, "val_loss": avg_val_loss}
        print(f"Client: Average training loss: {avg_train_loss:.4f}")
        print(f"Client: Average validation loss: {avg_val_loss:.4f}")
        os.unlink(adapter_file)
        return (
            new_parameters,
            len(train_dataset),
            metrics,
        )


def client_fn(context: Context) -> NumPyClient:
    """
    Factory function to create an MLXClient instance.
    """
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    num_rounds = context.run_config["num-server-rounds"]
    cfg = get_run_config(context)
    data_path_base = cfg.get("data_path", "./data")
    results_path_base = cfg.get("results_path", "./results")

    # Load the client's data partition. The `load_data` function now also
    # handles saving the data to a .jsonl file and returns the path.
    client_trainset, client_valset, saved_data_path = load_data(
        partition_id=partition_id,
        num_partitions=num_partitions,
        dataset_name=cfg.dataset.name,
        prompt_template=cfg.dataset.prompt_template,
        completion_template=cfg.dataset.completion_template,
        train_split="train",
        split_ratio=80,
        data_path_base=data_path_base,
        tokenizer=AutoTokenizer.from_pretrained(cfg.model.name),
    )

    # Instantiate the client with the path to the saved data
    return MLXClient(
        partition_id=partition_id,
        model_cfg=cfg.model,
        train_cfg=cfg.train,
        data_path=saved_data_path,
        results_path=results_path_base,
        client_trainset=client_trainset,
        client_valset=client_valset,
        num_examples=len(client_trainset),
        num_rounds=num_rounds,
    ).to_client()


# Flower ClientApp
app = ClientApp(client_fn=client_fn)
