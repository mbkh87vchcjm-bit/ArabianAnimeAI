from abc import ABC, abstractmethod
import torch
from typing import List, Optional

class BaseVideoAssembler(ABC):
    """
    Abstract Interface for Video Assembler component.
    Stitches multiple generated shot chunks, applies transition blending, and exports final video file format.
    """

    @abstractmethod
    def assemble_shots(self, shot_chunks: List[torch.Tensor], output_path: str, fps: int = 24) -> str:
        """
        Assembles sequence of shot tensors into a final exported video file.
        """
        pass
