"""blossomtunellm-mlx: A Flower client app for federated learning with MLX."""

from logging import INFO
from typing import List, Tuple, Union

from flwr.common import (
    FitIns,
    FitRes,
    Parameters,
    log,
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg


class CommunicationTracker:
    """A helper class to track communication costs over FL rounds."""

    def __init__(self):
        self.curr_comm_cost = 0.0

    @staticmethod
    def _compute_bytes(parameters: Parameters) -> int:
        """Compute the size of parameters in bytes."""
        return sum(len(tensor) for tensor in parameters.tensors)

    def track(self, fit_list: List[Union[FitIns, FitRes]]):
        """Update and log the total communication cost."""
        size_bytes_list = [
            self._compute_bytes(fit_ele.parameters) for fit_ele in fit_list
        ]
        comm_cost_mb = sum(size_bytes_list) / (1024**2)

        self.curr_comm_cost += comm_cost_mb
        log(
            INFO,
            "Communication cost: %.2f MB this round / %.2f MB total",
            comm_cost_mb,
            self.curr_comm_cost,
        )


class FlowerTuneLlm(FedAvg):
    """
    A customized FedAvg strategy that tracks communication costs.

    This strategy behaves exactly like the standard FedAvg but adds logging
    to monitor the amount of data being sent between the server and clients
    during the `fit` phase of each round.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.comm_tracker = CommunicationTracker()

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training and track outgoing data."""
        client_instructions = super().configure_fit(
            server_round, parameters, client_manager
        )

        # Track the size of parameters sent to clients
        fit_ins_list = [fit_ins for _, fit_ins in client_instructions]
        self.comm_tracker.track(fit_ins_list)

        return client_instructions

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Parameters | None, dict]:
        """Aggregate fit results and track incoming data."""
        # Track the size of parameters received from clients
        fit_res_list = [fit_res for _, fit_res in results]
        self.comm_tracker.track(fit_res_list)

        # Perform the standard FedAvg aggregation
        return super().aggregate_fit(server_round, results, failures)
