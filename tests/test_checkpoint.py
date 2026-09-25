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

def test_checkpoint_resume_with_optimizer_and_scheduler():
    temp_dir = tempfile.mkdtemp()
    try:
        manager = CheckpointManager(temp_dir)
        meta = CheckpointMetadata(
            step=1000,
            epoch=10,
            loss=0.045,
            experiment_name="resume_run",
            metrics={"loss": 0.045}
        )

        dummy_model_state = {"layer.weight": torch.tensor([[0.5, -0.5], [1.0, 2.0]])}
        dummy_opt_state = {"state": {0: {"step": 1000}}, "param_groups": [{"lr": 0.0001}]}
        dummy_sched_state = {"last_epoch": 10, "_step_count": 1001}

        ckpt_path = manager.save_checkpoint(
            filename="ckpt_resume.pt",
            state_dict=dummy_model_state,
            metadata=meta,
            optimizer_state=dummy_opt_state,
            scheduler_state=dummy_sched_state
        )

        assert os.path.exists(ckpt_path)
        payload = manager.load_checkpoint("ckpt_resume.pt")

        # Verify tensor and metadata values are preserved precisely
        assert torch.equal(payload["state_dict"]["layer.weight"], dummy_model_state["layer.weight"])
        assert payload["optimizer_state"]["param_groups"][0]["lr"] == 0.0001
        assert payload["scheduler_state"]["last_epoch"] == 10

        loaded_meta = manager.load_metadata("ckpt_resume.pt")
        assert loaded_meta.step == 1000
        assert loaded_meta.epoch == 10
        assert loaded_meta.loss == 0.045
        assert loaded_meta.experiment_name == "resume_run"
    finally:
        shutil.rmtree(temp_dir)
