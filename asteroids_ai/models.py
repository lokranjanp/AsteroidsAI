from __future__ import annotations

from typing import Dict

import torch
from torch import nn


TensorObservation = Dict[str, torch.Tensor]


def flatten_observation(observation: TensorObservation) -> torch.Tensor:
    """Flatten the shared structured observation without losing the entity mask."""
    return torch.cat(
        (
            observation["global"],
            observation["entities"].flatten(start_dim=1),
            observation["entity_mask"],
        ),
        dim=1,
    )


class _DuelingHead(nn.Module):
    def __init__(self, input_dim: int, action_count: int) -> None:
        super().__init__()
        self.value = nn.Sequential(nn.Linear(input_dim, 128), nn.ReLU(), nn.Linear(128, 1))
        self.advantage = nn.Sequential(nn.Linear(input_dim, 128), nn.ReLU(), nn.Linear(128, action_count))

    def forward(self, encoded: torch.Tensor) -> torch.Tensor:
        value = self.value(encoded)
        advantage = self.advantage(encoded)
        return value + advantage - advantage.mean(dim=1, keepdim=True)


class MLPQNetwork(nn.Module):
    def __init__(self, global_features: int, max_entities: int, entity_features: int, action_count: int) -> None:
        super().__init__()
        input_dim = global_features + max_entities * entity_features + max_entities
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        self.head = _DuelingHead(256, action_count)

    def forward(self, observation: TensorObservation) -> torch.Tensor:
        return self.head(self.encoder(flatten_observation(observation)))


class EntityTransformerQNetwork(nn.Module):
    """Permutation-aware encoder over entities within each observation.

    Batch samples remain on the batch axis. Padding is hidden with a key-padding
    mask, fixing the cross-sample attention behavior of the legacy network.
    """

    def __init__(
        self,
        global_features: int,
        entity_features: int,
        action_count: int,
        model_dim: int = 128,
        heads: int = 4,
        layers: int = 2,
    ) -> None:
        super().__init__()
        self.global_projection = nn.Sequential(nn.Linear(global_features, model_dim), nn.LayerNorm(model_dim))
        self.entity_projection = nn.Sequential(nn.Linear(entity_features, model_dim), nn.LayerNorm(model_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=heads,
            dim_feedforward=model_dim * 2,
            dropout=0.0,
            activation="relu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.output_norm = nn.LayerNorm(model_dim)
        self.head = _DuelingHead(model_dim, action_count)

    def forward(self, observation: TensorObservation) -> torch.Tensor:
        global_token = self.global_projection(observation["global"]).unsqueeze(1)
        entity_tokens = self.entity_projection(observation["entities"])
        tokens = torch.cat((global_token, entity_tokens), dim=1)
        entity_padding = observation["entity_mask"] <= 0.0
        global_padding = torch.zeros(
            (entity_padding.shape[0], 1), dtype=torch.bool, device=entity_padding.device
        )
        padding_mask = torch.cat((global_padding, entity_padding), dim=1)
        encoded = self.encoder(tokens, src_key_padding_mask=padding_mask)
        return self.head(self.output_norm(encoded[:, 0]))


def build_q_network(
    model_type: str,
    global_features: int,
    max_entities: int,
    entity_features: int,
    action_count: int,
) -> nn.Module:
    if model_type == "mlp":
        return MLPQNetwork(global_features, max_entities, entity_features, action_count)
    if model_type == "transformer":
        return EntityTransformerQNetwork(global_features, entity_features, action_count)
    raise ValueError(f"Unknown model type: {model_type}")

