from abc import abstractmethod
import torch
from models.base import BaseModel

class BaseVideoVAE(BaseModel):
    """
    Abstract Interface for 3D Video Variational Autoencoder (VAE) component.
    Compresses RGB video frame tensors (B, C, T, H, W) into continuous latent representations and decodes them back.
    """

    @abstractmethod
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Compresses video tensor (B, C, T, H, W) to latent space."""
        pass

    @abstractmethod
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decodes latent tensor (B, C_latent, T_latent, H_latent, W_latent) back to video space."""
        pass

    @abstractmethod
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Full reconstruction pass."""
        pass
