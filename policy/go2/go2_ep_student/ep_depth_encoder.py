"""Minimal Extreme Parkour depth encoder for offline / sidecar deploy.

Architecture matches rsl_rl.modules.depth_backbone.RecurrentDepthBackbone +
DepthOnlyFCBackbone58x87. Supports batched envs with per-env GRU hidden reset.
"""

from __future__ import annotations

from typing import Optional, Sequence, Union

import torch
import torch.nn as nn


class DepthOnlyFCBackbone58x87(nn.Module):
    def __init__(self, scandots_output_dim: int = 32, output_activation: str = "tanh", num_frames: int = 1):
        super().__init__()
        self.num_frames = num_frames
        activation = nn.ELU()
        self.image_compression = nn.Sequential(
            nn.Conv2d(in_channels=self.num_frames, out_channels=32, kernel_size=5),
            nn.MaxPool2d(kernel_size=2, stride=2),
            activation,
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            activation,
            nn.Flatten(),
            nn.Linear(64 * 25 * 39, 128),
            activation,
            nn.Linear(128, scandots_output_dim),
        )
        self.output_activation = nn.Tanh() if output_activation == "tanh" else activation

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # images: [B, H, W] or [B, 1, H, W]
        if images.dim() == 3:
            images = images.unsqueeze(1)
        compressed = self.image_compression(images)
        return self.output_activation(compressed)


class RecurrentDepthBackbone(nn.Module):
    def __init__(self, n_proprio: int = 53) -> None:
        super().__init__()
        activation = nn.ELU()
        self.base_backbone = DepthOnlyFCBackbone58x87(32, output_activation="tanh")
        self.combination_mlp = nn.Sequential(
            nn.Linear(32 + n_proprio, 128),
            activation,
            nn.Linear(128, 32),
        )
        self.rnn = nn.GRU(input_size=32, hidden_size=512, batch_first=True)
        self.output_mlp = nn.Sequential(nn.Linear(512, 32 + 2), nn.Tanh())
        # GRU hidden: [num_layers, batch, hidden_size] — batch == num_envs
        self.hidden_states: Optional[torch.Tensor] = None

    def forward(self, depth_image: torch.Tensor, proprioception: torch.Tensor) -> torch.Tensor:
        depth_feat = self.base_backbone(depth_image)
        depth_latent = self.combination_mlp(torch.cat((depth_feat, proprioception), dim=-1))
        depth_latent, self.hidden_states = self.rnn(depth_latent[:, None, :], self.hidden_states)
        return self.output_mlp(depth_latent.squeeze(1))

    def reset_hidden(self) -> None:
        """Clear all env hiddens (e.g. cold start)."""
        self.hidden_states = None

    def reset_hidden_envs(self, env_ids: Union[torch.Tensor, Sequence[int]]) -> None:
        """Zero GRU hidden for selected envs only (episode reset)."""
        if self.hidden_states is None:
            return
        if isinstance(env_ids, torch.Tensor):
            if env_ids.numel() == 0:
                return
            ids = env_ids.long().to(self.hidden_states.device).view(-1)
        else:
            if len(env_ids) == 0:
                return
            ids = torch.as_tensor(env_ids, device=self.hidden_states.device, dtype=torch.long)
        # hidden_states: [1, num_envs, 512]
        self.hidden_states[:, ids, :] = 0.0
