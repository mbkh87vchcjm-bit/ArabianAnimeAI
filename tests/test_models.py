import pytest
import torch
import torch.nn as nn
from models.base import BaseModel
from models.vae.base import BaseVideoVAE
from models.transformer.base import BaseVideoTransformer

class DummyTestModel(BaseModel):
    def __init__(self, in_dim=10, out_dim=5):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)

def test_base_model_instantiation_and_num_params():
    model = DummyTestModel(in_dim=10, out_dim=5)
    assert model.get_num_params() == 55

    x = torch.randn(2, 10)
    out = model(x)
    assert out.shape == (2, 5)

def test_base_video_vae_abstract_contract():
    with pytest.raises(TypeError):
        # Cannot instantiate abstract class BaseVideoVAE without implementing encode, decode, forward
        BaseVideoVAE()

def test_base_video_transformer_abstract_contract():
    with pytest.raises(TypeError):
        # Cannot instantiate abstract class BaseVideoTransformer without implementing forward
        BaseVideoTransformer()
