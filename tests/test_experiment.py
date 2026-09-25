import os
import shutil
import tempfile
from configs.config_loader import load_config
from training.experiment import ExperimentTracker

def test_experiment_tracker_creation():
    temp_dir = tempfile.mkdtemp()
    try:
        config = load_config("configs/default.yaml")
        tracker = ExperimentTracker("test_experiment", base_dir=temp_dir, config=config)

        assert os.path.exists(tracker.get_experiment_dir())
        assert os.path.exists(tracker.get_checkpoint_dir())
        assert os.path.exists(os.path.join(tracker.get_experiment_dir(), "logs"))
        assert os.path.exists(os.path.join(tracker.get_experiment_dir(), "config_snapshot.yaml"))

        metadata_path = tracker.save_metadata({"status": "initialized", "user": "test_user"})
        assert os.path.exists(metadata_path)
    finally:
        shutil.rmtree(temp_dir)

def test_experiment_tracker_unique_run_ids():
    temp_dir = tempfile.mkdtemp()
    try:
        config = load_config("configs/default.yaml")
        tracker1 = ExperimentTracker("run", base_dir=temp_dir, config=config)
        tracker2 = ExperimentTracker("run", base_dir=temp_dir, config=config)

        assert tracker1.run_id != tracker2.run_id
        assert tracker1.get_experiment_dir() != tracker2.get_experiment_dir()
        assert os.path.exists(tracker1.get_experiment_dir())
        assert os.path.exists(tracker2.get_experiment_dir())
    finally:
        shutil.rmtree(temp_dir)
