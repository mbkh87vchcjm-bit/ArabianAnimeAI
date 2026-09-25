import os

def test_mandatory_directories_exist():
    required_dirs = [
        "configs",
        "data",
        "models",
        "training",
        "inference",
        "conditioning",
        "memory",
        "long_video",
        "evaluation",
        "video",
        "tests",
        "scripts",
        "docs"
    ]
    for directory in required_dirs:
        assert os.path.exists(directory), f"Directory '{directory}' does not exist."
        assert os.path.isdir(directory), f"Path '{directory}' is not a directory."
