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
