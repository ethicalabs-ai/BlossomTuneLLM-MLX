"""blossomtunellm-mlx: A Flower client app for federated learning with MLX."""

import os
from pathlib import Path
from slugify import slugify
from typing import Dict, Tuple
from omegaconf import OmegaConf

from mlx_lm.tuner.utils import linear_to_lora_layers
from flwr.common import Context, Parameters, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig

from blossomtune_mlx.config import get_run_config
from blossomtune_mlx.strategy import FlowerTuneLlm
from blossomtune_mlx.models import get_parameters, set_parameters, get_peft_config
from mlx_lm.utils import load, save_config


def get_evaluate_fn(
    model_cfg,
    train_cfg,
    save_every_round: int,
    total_rounds: int,
    save_path: str,
):
    """
    Return an evaluation function for saving the global LoRA adapter.
    This function loads a base model, applies the aggregated adapter weights,
    and saves the resulting adapter file.
    """

    def evaluate(
        server_round: int, parameters: Parameters, config: Dict
    ) -> Tuple[float, Dict]:
        # Save the adapter weights if it's a save round or the final round
        if server_round > 0 and (
            server_round == total_rounds or server_round % save_every_round == 0
        ):
            print(f"Server: Saving global adapter for round {server_round}...")

            # 1. Load the base model architecture.
            # We only need the structure, so no need for full weights if memory is a concern.
            model, _ = load(model_cfg.name)

            # 2. Configure the model for LoRA to create the adapter layers.
            model.freeze()
            lora_parameters_dict = OmegaConf.to_container(
                train_cfg.lora_parameters, resolve=True
            )
            linear_to_lora_layers(
                model,
                train_cfg.lora_layers,
                lora_parameters_dict,
                use_dora=(train_cfg.fine_tune_type == "dora"),
            )

            # 3. Apply the aggregated parameters from the Flower server.
            set_parameters(model, parameters)

            # 4. Define the save path and save the adapter weights.
            round_save_path = Path(save_path) / f"adapter_{server_round}"
            round_save_path.mkdir(parents=True, exist_ok=True)
            adapter_file = round_save_path / "adapters.safetensors"

            # 5. Use the model's own method to save the trainable parameters (the adapter).
            model.save_weights(str(adapter_file))
            peft_config = get_peft_config(model_cfg, train_cfg, adapter_file)
            save_config(peft_config, (adapter_file.parent / "adapter_config.json"))
            print(f"Global adapter and config saved to {adapter_file.parent}")

        # No actual evaluation is performed, so return a dummy loss.
        return 0.0, {}

    return evaluate


def get_on_fit_config(save_path):
    """Return a function that will be used to construct the config that the
    client's fit() method will receive."""

    def fit_config_fn(server_round: int):
        fit_config = {}
        fit_config["current_round"] = server_round
        fit_config["save_path"] = save_path
        return fit_config

    return fit_config_fn


def fit_weighted_average(metrics):
    """Aggregate (federated) evaluation metrics."""
    # Multiply accuracy of each client by number of examples used
    losses = [num_examples * m["train_loss"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"train_loss": sum(losses) / sum(examples)}


def server_fn(context: Context) -> ServerAppComponents:
    """Construct components that set the ServerApp behaviour."""
    # Create a unique output directory for this run
    cfg = get_run_config(context)
    model_path_or_name = context.run_config["model.name"]
    adapter_path = context.run_config.get("model.adapter_path", None)
    model_slug = slugify(model_path_or_name)
    save_path = os.path.join(
        context.run_config["save_path"],
        model_slug,
        "server",
    )

    # 1. Load the base model architecture.
    # We only need the structure, so no need for full weights if memory is a concern.
    init_model, _ = load(model_path_or_name, adapter_path=adapter_path)

    # 2. Configure the model for LoRA to create the adapter layers.
    init_model.freeze()
    if not adapter_path:
        lora_parameters_dict = OmegaConf.to_container(
            cfg.train.lora_parameters, resolve=True
        )
        linear_to_lora_layers(
            init_model,
            cfg.train.lora_layers,
            lora_parameters_dict,
            use_dora=(cfg.train.fine_tune_type == "dora"),
        )
    init_model_parameters = get_parameters(init_model)
    init_model_parameters = ndarrays_to_parameters(init_model_parameters)

    os.makedirs(save_path, exist_ok=True)
    print(f"Server: Saving global models to {save_path}")

    # Read configuration
    num_rounds = context.run_config["num_server_rounds"]

    # Define the strategy
    strategy = FlowerTuneLlm(
        fraction_fit=cfg.strategy.fraction_fit,
        fraction_evaluate=cfg.strategy.fraction_evaluate,
        min_fit_clients=cfg.strategy.min_fit_clients,  # Wait for at least one client to start
        min_available_clients=cfg.strategy.min_available_clients,
        on_fit_config_fn=get_on_fit_config(save_path),
        fit_metrics_aggregation_fn=lambda m: fit_weighted_average(m),
        evaluate_fn=get_evaluate_fn(
            cfg.model,
            cfg.train,
            cfg.train.save_every_round,
            num_rounds,
            save_path,
        ),
        initial_parameters=init_model_parameters,  # Start with initial parameters
    )

    config = ServerConfig(num_rounds=num_rounds)
    return ServerAppComponents(strategy=strategy, config=config)


# Flower ServerApp
app = ServerApp(server_fn=server_fn)
