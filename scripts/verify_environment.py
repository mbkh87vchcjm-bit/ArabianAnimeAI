#!/usr/bin/env python3
"""
Environment Verification Script for ArabianAnimeAI (Phase 1 Infrastructure).
Validates dependencies, package imports, configuration loading (default & debug), seed setting,
required directory structures, collision-free experiment tracking, and checkpoint/metadata persistence.
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
    import yaml
    import numpy as np

    print(f"[✓] Python version: {sys.version.split()[0]}")
    print(f"[✓] PyTorch version: {torch.__version__}")
    print(f"[✓] NumPy version: {np.__version__}")
    print(f"[✓] PyYAML version: {yaml.__version__}")

    # 2. Check Package Imports
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

    print("[✓] All 10 internal packages imported successfully.")

    # 3. Test Config Loader (Default & Debug Configs)
    config = configs.load_config("configs/default.yaml")
    assert config.system.seed == 42
    print(f"[✓] Target config loaded successfully (system.seed = {config.system.seed}).")

    debug_config = configs.load_config("configs/debug.yaml")
    assert debug_config.data.resolution == [64, 64]
    print(f"[✓] Debug config loaded successfully (resolution = {debug_config.data.resolution}).")

    # 4. Test Seed Setting
    training.set_seed(1234, deterministic=True)
    t1 = torch.randn(2, 2)
    training.set_seed(1234, deterministic=True)
    t2 = torch.randn(2, 2)
    assert torch.equal(t1, t2)
    print("[✓] Reproducible random seed setting verified.")

    # 5. Check Directories
    required_dirs = [
        "configs", "data", "models", "training", "inference",
        "conditioning", "memory", "long_video", "evaluation",
        "video", "tests", "scripts", "docs"
    ]
    for d in required_dirs:
        assert os.path.isdir(d), f"Directory {d} missing!"
    print(f"[✓] All {len(required_dirs)} required core directories verified.")

    # 6. Test Unique Experiment Tracker & Checkpoint Metadata Persistence
    tracker1 = training.ExperimentTracker("verification_run", base_dir="experiments", config=config)
    tracker2 = training.ExperimentTracker("verification_run", base_dir="experiments", config=config)
    assert tracker1.run_id != tracker2.run_id, "Experiment run IDs collided!"
    print(f"[✓] Collision-free Experiment Tracker verified (Run ID: {tracker1.run_id})")

    meta = training.CheckpointMetadata(step=0, epoch=1, loss=0.5, experiment_name="verification_run")
    ckpt_mgr = training.CheckpointManager(tracker1.get_checkpoint_dir())
    dummy_state = {"param": torch.tensor([1.0, 2.0])}
    dummy_sched_state = {"last_epoch": 1}
    ckpt_path = ckpt_mgr.save_checkpoint("test_ckpt.pt", dummy_state, meta, scheduler_state=dummy_sched_state)

    payload = ckpt_mgr.load_checkpoint("test_ckpt.pt")
    assert torch.equal(payload["state_dict"]["param"], dummy_state["param"])
    assert payload["scheduler_state"]["last_epoch"] == 1

    loaded_meta = ckpt_mgr.load_metadata("test_ckpt.pt")
    assert loaded_meta.step == 0
    assert loaded_meta.loss == 0.5
    print(f"[✓] Checkpoint state_dict & metadata saved and loaded successfully at: {ckpt_path}")

    print("==================================================")
    print("  Phase 1 Infrastructure Check Passed!           ")
    print("  (Note: Model weights & AI video generation are ")
    print("   target Phase 2+ specifications, not built yet)")
    print("==================================================")

if __name__ == "__main__":
    run_verification()
