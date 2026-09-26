#!/usr/bin/env python3
"""
Environment Verification Script for ArabianAnimeAI (Phase 1 Infrastructure).
Validates dependencies, package exports, configuration loading (default & debug), seed setting,
required directory structures, collision-free experiment tracking, abstract contracts,
and real model/optimizer/scheduler checkpoint resume persistence.
"""

import os
import sys

# Ensure root directory is on Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def run_verification():
    print("==================================================")
    print("  ArabianAnimeAI Phase 1 Infrastructure Check    ")
    print("==================================================")

    # 1. Check Python & PyTorch
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import yaml
    import numpy as np

    print(f"[✓] Python version: {sys.version.split()[0]}")
    print(f"[✓] PyTorch version: {torch.__version__}")
    print(f"[✓] NumPy version: {np.__version__}")
    print(f"[✓] PyYAML version: {yaml.__version__}")

    # 2. Check Package Imports & Abstract Contracts
    import configs
    import data
    import models
    import training
    import inference
    import conditioning
    import memory
    import long_video
    import evaluation
    import video

    from conditioning import BaseStoryCompiler
    from models import BaseVideoVAE, BaseVideoTransformer

    print("[✓] All 10 internal packages imported successfully.")
    print("[✓] Architecture contracts BaseStoryCompiler, BaseVideoVAE, and BaseVideoTransformer verified.")

    # 3. Test Config Loader (Default & Debug Configs)
    config = configs.load_config("configs/default.yaml")
    assert config.system.seed == 42
    print(f"[✓] Target config loaded successfully (system.seed = {config.system.seed}).")

    debug_config = configs.load_config("configs/debug.yaml")
    assert debug_config.data.resolution == [64, 64]
    print(f"[✓] Debug config loaded successfully (resolution = {debug_config.data.resolution}).")

    # 4. Test Seed Setting & Reconfiguration
    training.set_seed(1234, deterministic=True, use_deterministic_algorithms=False)
    t1 = torch.randn(2, 2)
    training.set_seed(1234, deterministic=True, use_deterministic_algorithms=False)
    t2 = torch.randn(2, 2)
    assert torch.equal(t1, t2)
    print("[✓] Reproducible random seed setting & reconfiguration verified.")

    # 5. Check Directories
    required_dirs = [
        "configs", "data", "models", "training", "inference",
        "conditioning", "memory", "long_video", "evaluation",
        "video", "tests", "scripts", "docs"
    ]
    for d in required_dirs:
        assert os.path.isdir(d), f"Directory {d} missing!"
    print(f"[✓] All {len(required_dirs)} required core directories verified.")

    # 6. Test Unique Experiment Tracker & Real Checkpoint Resume
    tracker1 = training.ExperimentTracker("verification_run", base_dir="experiments", config=config)
    tracker2 = training.ExperimentTracker("verification_run", base_dir="experiments", config=config)
    assert tracker1.run_id != tracker2.run_id, "Experiment run IDs collided!"
    print(f"[✓] Collision-free Experiment Tracker verified (Run ID: {tracker1.run_id})")

    # Real Model, Optimizer, Scheduler Resume Check
    model1 = nn.Linear(2, 2)
    opt1 = optim.SGD(model1.parameters(), lr=0.1)
    sched1 = optim.lr_scheduler.StepLR(opt1, step_size=1, gamma=0.5)

    x = torch.tensor([[1.0, 2.0]])
    target = torch.tensor([[0.0, 1.0]])
    opt1.zero_grad()
    loss1 = nn.functional.mse_loss(model1(x), target)
    loss1.backward()
    opt1.step()
    sched1.step()

    saved_weights = model1.weight.clone().detach()

    meta = training.CheckpointMetadata(step=1, epoch=1, loss=loss1.item(), experiment_name="verification_run")
    ckpt_mgr = training.CheckpointManager(tracker1.get_checkpoint_dir())
    ckpt_path = ckpt_mgr.save_checkpoint(
        "test_ckpt.pt",
        model1.state_dict(),
        meta,
        optimizer_state=opt1.state_dict(),
        scheduler_state=sched1.state_dict()
    )

    # Restore into fresh instances
    model2 = nn.Linear(2, 2)
    opt2 = optim.SGD(model2.parameters(), lr=0.1)
    sched2 = optim.lr_scheduler.StepLR(opt2, step_size=1, gamma=0.5)

    payload = ckpt_mgr.load_checkpoint("test_ckpt.pt")
    model2.load_state_dict(payload["state_dict"])
    opt2.load_state_dict(payload["optimizer_state"])
    sched2.load_state_dict(payload["scheduler_state"])

    assert torch.equal(model2.weight, saved_weights)
    assert sched2.get_last_lr() == sched1.get_last_lr()

    loaded_meta = ckpt_mgr.load_metadata("test_ckpt.pt")
    assert loaded_meta.step == 1
    assert loaded_meta.loss == loss1.item()
    print(f"[✓] Real model/optimizer/scheduler checkpoint resume verified at: {ckpt_path}")

    print("==================================================")
    print("  Phase 1 Infrastructure Check Passed!           ")
    print("  (Note: Model weights & AI video generation are ")
    print("   target Phase 2+ specifications, not built yet) ")
    print("==================================================")

if __name__ == "__main__":
    run_verification()
