"""
Abstract interface shared by discrete CTMC rollouts.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

import torch


class AbstractFlowProcess(ABC):
    """
    Minimal interface for discrete-time flow processes.
    """

    @abstractmethod
    def sample_prior(self) -> torch.Tensor:
        """
        Sample an initial latent state x_0.
        """

    @abstractmethod
    def rollout(self, x0: torch.Tensor) -> List[torch.Tensor]:
        """
        Roll out a full trajectory [x_0, ..., x_T] given x_0.
        """

    @abstractmethod
    def rollout_from(self, x_t: torch.Tensor, t: int) -> List[torch.Tensor]:
        """
        Roll out [x_t, ..., x_T] from a particular time step t.
        """

    @abstractmethod
    def rollout_from_batch(self, x_t: torch.Tensor, t: int) -> List[torch.Tensor]:
        """
        Roll out [x_t, ..., x_T] for a batch of starting states.

        Args:
            x_t: tensor of shape (batch, seq_len)
            t: current timestep
        Returns:
            list of tensors, each of shape (batch, seq_len)
        """
