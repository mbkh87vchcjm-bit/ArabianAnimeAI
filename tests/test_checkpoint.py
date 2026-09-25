import os
import shutil
import tempfile
import torch
from training.checkpoint import CheckpointManager, CheckpointMetadata

def test_checkpoint_save_and_load():
    temp_dir = tempfile.mkdtemp()
    try:
        manager = CheckpointManager(temp_dir)
        meta = CheckpointMetadata(
            step=500,
            epoch=5,
            loss=0.123,
            experiment_name="test_run",
            metrics={"fid": 12.5, "lpips": 0.08}
        )
        state_dict = {"weight": torch.tensor([1.0, 2.0, 3.0])}

        ckpt_path = manager.save_checkpoint("model_step_500.pt", state_dict, meta)
        assert os.path.exists(ckpt_path)
        assert os.path.exists(ckpt_path + ".json")

        checkpoints = manager.list_checkpoints()
        assert "model_step_500.pt" in checkpoints

        payload = manager.load_checkpoint("model_step_500.pt")
        assert torch.equal(payload["state_dict"]["weight"], state_dict["weight"])

        loaded_meta = manager.load_metadata("model_step_500.pt")
        assert loaded_meta.step == 500
        assert loaded_meta.epoch == 5
        assert loaded_meta.loss == 0.123
        assert loaded_meta.metrics["fid"] == 12.5
    finally:
        shutil.rmtree(temp_dir)
