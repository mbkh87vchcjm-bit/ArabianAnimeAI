from abc import abstractmethod
import torch
from typing import Optional
from models.base import BaseModel

class BaseVideoTransformer(BaseModel):
    """
    Abstract Interface for Video Generation Diffusion Transformer (DiT) component.
    Operates on Video VAE latent space, denoising latent video representations guided by conditioning inputs.
    """

    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Predicts noise residual or velocity vector given noisy latents, timestep, and conditioning context.
        """
        pass
