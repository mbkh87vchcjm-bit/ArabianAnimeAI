# ArabianAnimeAI

An open-source, scalable PyTorch research & development framework for text-to-anime video generation focused on visual character and location consistency across multi-shot scenes and long videos.

---

## Current Status: Phase 1 (Foundation & Architecture Contracts)

> **Important Note:** This repository is currently in **Phase 1 (Foundation & Architecture Contracts)**. The 9-component target architecture described below represents the long-term design roadmap. AI generation models (such as Video VAE, Diffusion Transformer, and Text Encoders) are **abstract contracts / interfaces** and are **not yet trained or generated** in this phase.

---

## Architecture Contracts & Planned Components

1. **Story / Prompt Compiler (`conditioning/story_compiler.py`)** *(Abstract Interface: `BaseStoryCompiler`)*:
   - Parses multi-shot scripts, scene storyboards, and character interaction descriptions into structured shot sequence specifications.

2. **Text Conditioning (`conditioning/text_conditioning.py`)** *(Abstract Interface: `BaseTextConditioner`)*:
   - Converts natural text prompts and scene descriptions into latent context vectors for cross-attention layers.

3. **Character Memory (`memory/character_memory.py`)** *(Abstract Interface: `BaseCharacterMemory`)*:
   - Stores visual feature embeddings, face identity vectors, and reference character representations across shots.

4. **World / Location Memory (`memory/world_memory.py`)** *(Abstract Interface: `BaseWorldMemory`)*:
   - Stores scene background features, architectural keys, and environment lighting palettes across camera cuts.

5. **Video VAE (`models/vae/base.py`)** *(Abstract Interface: `BaseVideoVAE`)*:
   - Inherits from `BaseModel`. Abstract 3D Spatiotemporal Variational Autoencoder contract (`encode`, `decode`, `forward`) to compress RGB video tensors into compact latent spaces.

6. **Video Generation Transformer (`models/transformer/base.py`)** *(Abstract Interface: `BaseVideoTransformer`)*:
   - Inherits from `BaseModel`. Abstract 3D/Spatiotemporal Diffusion Transformer (DiT) contract (`forward`) for denoising latents guided by conditioning vectors.

7. **Long Video Engine (`long_video/engine.py`)** *(Abstract Interface: `BaseLongVideoEngine`)*:
   - Manages extended frame generation beyond single-chunk training windows using sliding windows and overlapping temporal chunks.

8. **Consistency Checker (`evaluation/consistency.py`)** *(Abstract Interface: `BaseConsistencyChecker`)*:
   - Automated evaluation contract measuring character identity feature distance, temporal coherence, frame flicker, and text-video alignment scores.

9. **Video Assembler (`video/assembler.py`)** *(Abstract Interface: `BaseVideoAssembler`)*:
   - Combines output video shot chunks, applies shot transitions, and exports final encoded video files (`MP4`, `WebM`).

---

## Directory Structure

```
ArabianAnimeAI/
├── configs/
│   ├── __init__.py
│   ├── config_loader.py       # Config loader with dictionary attribute wrapper
│   ├── debug.yaml             # Lightweight development & debugging configuration
│   └── default.yaml           # Target research YAML configuration file
├── data/
│   ├── __init__.py
│   └── dataset.py             # Base dataset abstract contract (BaseDataset)
├── models/
│   ├── __init__.py
│   ├── base.py                # Abstract base model class (BaseModel)
│   ├── vae/
│   │   ├── __init__.py
│   │   └── base.py            # Abstract Video VAE contract (BaseVideoVAE)
│   └── transformer/
│       ├── __init__.py
│       └── base.py            # Abstract Video Transformer contract (BaseVideoTransformer)
├── training/
│   ├── __init__.py
│   ├── seed.py                # Reproducible random seed manager
│   ├── experiment.py          # Experiment run directory tracker & snapshot manager
│   └── checkpoint.py          # Checkpoint & structured metadata persistence
├── inference/
│   ├── __init__.py
│   └── pipeline.py            # End-to-end inference pipeline abstract contract
├── conditioning/
│   ├── __init__.py
│   ├── story_compiler.py      # Abstract Story Compiler contract (BaseStoryCompiler)
│   └── text_conditioning.py   # Abstract Text conditioning contract (BaseTextConditioner)
├── memory/
│   ├── __init__.py
│   ├── character_memory.py    # Character identity memory abstraction
│   └── world_memory.py        # Location/environment memory abstraction
├── long_video/
│   ├── __init__.py
│   └── engine.py              # Long video chunk generation engine interface
├── evaluation/
│   ├── __init__.py
│   └── consistency.py         # Visual & temporal consistency evaluator interface
├── video/
│   ├── __init__.py
│   └── assembler.py           # Video stitching and assembly interface
├── tests/
│   ├── test_checkpoint.py     # Real training resume validation & checkpoint tests
│   ├── test_config.py         # Unit tests for config loading and overriding
│   ├── test_dataset.py        # Unit tests for dataset base interface
│   ├── test_directories.py    # Unit tests for directory layout existence
│   ├── test_experiment.py     # Unit tests for experiment tracking and run IDs
│   ├── test_imports.py        # Unit tests for clean package imports
│   ├── test_models.py         # Unit tests for model interfaces and parameter counts
│   └── test_seed.py           # Unit tests for seed determinism & reconfiguration
├── scripts/
│   └── verify_environment.py  # Environment verification utility script
├── docs/
│   └── architecture.md        # Comprehensive architecture specification
├── .gitignore
├── LICENSE
├── pyproject.toml
├── README.md
└── requirements.txt
```

---

## Quick Start & Verification (Phase 1 Infrastructure)

### 1. Requirements & Setup

Ensure Python 3.10+ and PyTorch are installed in a clean Python environment:

```bash
pip install -r requirements.txt
```

To install the project in editable development mode:

```bash
pip install -e .
```

### 2. Verify System Setup

Run the environment verification script to confirm package imports, directory existence, seed setting, configuration loading, and real checkpoint resume logging:

```bash
python3 scripts/verify_environment.py
```

### 3. Run Test Suite

Run the comprehensive pytest suite:

```bash
pytest -q
```

---

## Experiment & Checkpoint Management

All experiments are logged inside `experiments/<experiment_name>_<timestamp>_<microseconds>_<uuid>/`:

- `config_snapshot.yaml`: Copy of the exact config used for the run.
- `metadata.json`: Run metadata including start timestamp and parameters.
- `checkpoints/`: Model checkpoint files (`.pt`) alongside metadata JSON sidecars (`.pt.json`) preserving model state_dict, optimizer state_dict, scheduler state_dict, step count, epoch, loss metrics, and model versions.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
