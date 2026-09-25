# ArabianAnimeAI

An open-source, scalable PyTorch research & development framework for text-to-anime video generation focused on visual character and location consistency across multi-shot scenes and long videos.

---

## Current Status: Phase 1 (Foundation & Research Infrastructure)

> **Important Note:** This repository is currently in **Phase 1 (Foundation & Research Infrastructure)**. The 9-component target architecture described below represents the long-term design roadmap. AI generation models (such as Video VAE, Diffusion Transformer, and Text Encoders) are **interfaces/planned components** and are **not yet trained or generated** in this phase.

---

## Project Overview & Mission

**ArabianAnimeAI** is built from scratch using PyTorch to solve one of the hardest challenges in AI video generation: **maintaining visual consistency** (character identities, costumes, artistic style, and location backgrounds) across multi-shot animated episodes and long video clips.

### Key Technical Principles:
1. **Zero External Paid API Dependency:** Fully self-contained open-source model architecture built natively on PyTorch.
2. **Zero Pre-trained Video Weights Assumption:** Built with modular component abstractions designed for custom training from scratch.
3. **Professional Original Anime Aesthetic:** Designed for high-quality, professional TV anime aesthetics (cel-shading, clean line-art, expressive lighting) without copying or infringing on copyrighted characters or specific existing shows.
4. **Experiment Preservation & Reproducibility:** Integrated experiment tracking, timestamped output logging, and structured checkpoint metadata management to guarantee that no successful run is lost.

---

## Target System Architecture (Planned Roadmap)

The ArabianAnimeAI framework is designed around 9 decoupled, modular sub-systems:

```
                              ┌──────────────────────────────────┐
                              │     1. Story / Prompt Compiler   │
                              └────────────────┬─────────────────┘
                                               │
                                               ▼
                              ┌──────────────────────────────────┐
                              │      2. Text Conditioning        │
                              └────────────────┬─────────────────┘
                                               │
        ┌──────────────────────────────────────┼──────────────────────────────────────┐
        │                                      │                                      │
        ▼                                      ▼                                      ▼
┌──────────────┐                       ┌──────────────┐                       ┌──────────────┐
│  3. Character│                       │  4. World    │                       │ 5. Video     │
│     Memory   │                       │    Memory    │                       │    VAE       │
└───────┬──────┘                       └───────┬──────┘                       └───────┬──────┘
        │                                      │                                      │
        └──────────────────────────────────────┼──────────────────────────────────────┘
                                               │
                                               ▼
                              ┌──────────────────────────────────┐
                              │  6. Video Generation Transformer │
                              └────────────────┬─────────────────┘
                                               │
                                               ▼
                              ┌──────────────────────────────────┐
                              │     7. Long Video Engine         │
                              └────────────────┬─────────────────┘
                                               │
                                               ▼
                              ┌──────────────────────────────────┐
                              │    8. Consistency Checker        │
                              └────────────────┬─────────────────┘
                                               │
                                               ▼
                              ┌──────────────────────────────────┐
                              │     9. Video Assembler           │
                              └──────────────────────────────────┘
```

### Component Breakdown & Design Specs

1. **Story / Prompt Compiler (`conditioning/`)** *(Target Interface)*:
   - Planned component to compile multi-shot scripts, scene storyboards, and character interaction descriptions into structured shot sequence graphs.

2. **Text Conditioning (`conditioning/text_conditioning.py`)** *(Target Interface)*:
   - Planned interface to convert natural text prompts and scene descriptions into latent context vectors for cross-attention layers.

3. **Character Memory (`memory/character_memory.py`)** *(Target Interface)*:
   - Planned interface to store visual feature embeddings, face identity vectors, and reference character representations across shots.

4. **World / Location Memory (`memory/world_memory.py`)** *(Target Interface)*:
   - Planned interface to store scene background features, architectural keys, and environment lighting palettes across camera cuts.

5. **Video VAE (`models/base.py`)** *(Target Interface)*:
   - Planned 3D Spatiotemporal Variational Autoencoder interface to compress RGB video tensors into compact latent spaces.

6. **Video Generation Transformer (`models/base.py`)** *(Target Interface)*:
   - Planned 3D/Spatiotemporal Diffusion Transformer (DiT) base interface for denoising latents guided by conditioning vectors.

7. **Long Video Engine (`long_video/engine.py`)** *(Target Interface)*:
   - Planned interface to manage extended frame generation beyond single-chunk training windows using sliding windows and overlapping temporal chunks.

8. **Consistency Checker (`evaluation/consistency.py`)** *(Target Interface)*:
   - Planned evaluation interface measuring character identity feature distance, temporal coherence, frame flicker, and text-video alignment scores.

9. **Video Assembler (`video/assembler.py`)** *(Target Interface)*:
   - Planned interface to combine output video shot chunks, apply shot transitions, and export final encoded video files (`MP4`, `WebM`).

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
│   └── dataset.py             # Base dataset interface contract
├── models/
│   ├── __init__.py
│   └── base.py                # Abstract base model class
├── training/
│   ├── __init__.py
│   ├── seed.py                # Reproducible random seed manager
│   ├── experiment.py          # Experiment run directory tracker & snapshot manager
│   └── checkpoint.py          # Checkpoint & structured metadata persistence
├── inference/
│   ├── __init__.py
│   └── pipeline.py            # End-to-end inference pipeline contract
├── conditioning/
│   ├── __init__.py
│   └── text_conditioning.py   # Text conditioning interface
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
│   ├── test_checkpoint.py     # Unit tests for checkpoint persistence & metadata
│   ├── test_config.py         # Unit tests for config loading and overriding
│   ├── test_directories.py    # Unit tests for directory layout existence
│   ├── test_experiment.py     # Unit tests for experiment tracking and run IDs
│   ├── test_imports.py        # Unit tests for clean package imports
│   └── test_seed.py           # Unit tests for seed determinism
├── scripts/
│   └── verify_environment.py  # Verification utility script
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

Run the environment verification script to confirm package imports, directory existence, seed setting, configuration loading, and checkpoint metadata logging:

```bash
python3 scripts/verify_environment.py
```

### 3. Run Test Suite

Run the comprehensive pytest suite:

```bash
pytest
```

---

## Experiment & Checkpoint Management

All experiments are logged inside `experiments/<experiment_name>_<timestamp>_<microseconds>/`:

- `config_snapshot.yaml`: Copy of the exact config used for the run.
- `metadata.json`: Run metadata including start timestamp and parameters.
- `checkpoints/`: Model checkpoint files (`.pt`) alongside metadata JSON sidecars (`.pt.json`) preserving step count, loss metrics, and model versions.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
