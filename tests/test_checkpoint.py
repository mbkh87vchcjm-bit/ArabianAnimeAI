import os
import shutil
import tempfile
import torch
import torch.nn as nn
import torch.optim as optim
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


def test_real_training_resume_validation():
    """
    Real training resume validation test:
    1. Instantiates actual PyTorch model, SGD optimizer, and StepLR scheduler.
    2. Runs training steps and records loss & state.
    3. Saves checkpoint payload (model state_dict, optimizer state_dict, scheduler state_dict, metadata).
    4. Instantiates fresh model, optimizer, and scheduler.
    5. Loads checkpoint and applies state_dicts via load_state_dict().
    6. Verifies exact restoration of model parameters, optimizer momentum/LR, scheduler step, loss, and metrics.
    7. Executes post-resume step to confirm training seamlessly continues.
    """
    temp_dir = tempfile.mkdtemp()
    try:
        # 1. Setup initial model, optimizer, scheduler
        torch.manual_seed(42)
        model1 = nn.Linear(4, 2)
        optimizer1 = optim.SGD(model1.parameters(), lr=0.1, momentum=0.9)
        scheduler1 = optim.lr_scheduler.StepLR(optimizer1, step_size=2, gamma=0.5)

        # 2. Perform 2 training steps
        x = torch.randn(3, 4)
        target = torch.randn(3, 2)

        for epoch in range(2):
            optimizer1.zero_grad()
            output1 = model1(x)
            loss1 = nn.functional.mse_loss(output1, target)
            loss1.backward()
            optimizer1.step()
            scheduler1.step()

        # Save state right after 2 steps
        initial_weight = model1.weight.clone().detach()
        initial_lr = scheduler1.get_last_lr()[0]

        manager = CheckpointManager(temp_dir)
        meta = CheckpointMetadata(
            step=2,
            epoch=2,
            loss=loss1.item(),
            experiment_name="real_resume_test",
            metrics={"train_loss": loss1.item()}
        )

        ckpt_path = manager.save_checkpoint(
            filename="checkpoint_epoch2.pt",
            state_dict=model1.state_dict(),
            metadata=meta,
            optimizer_state=optimizer1.state_dict(),
            scheduler_state=scheduler1.state_dict()
        )

        assert os.path.exists(ckpt_path)

        # 3. Instantiate fresh, un-trained model2, optimizer2, scheduler2
        torch.manual_seed(100) # different init weights
        model2 = nn.Linear(4, 2)
        optimizer2 = optim.SGD(model2.parameters(), lr=0.1, momentum=0.9)
        scheduler2 = optim.lr_scheduler.StepLR(optimizer2, step_size=2, gamma=0.5)

        # Confirm model2 is different prior to load
        assert not torch.equal(model2.weight, initial_weight)

        # 4. Load checkpoint and restore states
        payload = manager.load_checkpoint("checkpoint_epoch2.pt")
        model2.load_state_dict(payload["state_dict"])
        optimizer2.load_state_dict(payload["optimizer_state"])
        scheduler2.load_state_dict(payload["scheduler_state"])
        loaded_meta = manager.load_metadata("checkpoint_epoch2.pt")

        # 5. Verify restored state values
        assert torch.equal(model2.weight, initial_weight)
        assert scheduler2.get_last_lr()[0] == initial_lr
        assert loaded_meta.step == 2
        assert loaded_meta.epoch == 2
        assert loaded_meta.loss == loss1.item()

        # 6. Perform step 3 on both models to verify identical forward/backward trajectory post-resume
        optimizer1.zero_grad()
        out1 = model1(x)
        l1 = nn.functional.mse_loss(out1, target)
        l1.backward()
        optimizer1.step()

        optimizer2.zero_grad()
        out2 = model2(x)
        l2 = nn.functional.mse_loss(out2, target)
        l2.backward()
        optimizer2.step()

        assert torch.equal(model1.weight, model2.weight)
        assert torch.equal(out1, out2)

    finally:
        shutil.rmtree(temp_dir)
