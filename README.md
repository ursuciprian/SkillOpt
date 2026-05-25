# SkillOpt: Executive Strategy for Self-Evolving Agent Skills

*Train agent skills like you train neural networks — with epochs, learning rates, and validation gates — but without touching model weights.*

[![Project Page](https://img.shields.io/badge/Project%20Page-SkillOpt-8dbb3c?style=for-the-badge&labelColor=4c4c4c)](https://microsoft.github.io/SkillOpt/) [![Paper](https://img.shields.io/badge/Paper-arXiv-b31b1b?style=for-the-badge&labelColor=4c4c4c)](https://arxiv.org/abs/2605.23904) [![Project Video](https://img.shields.io/badge/Project%20Video-Watch%20Demo-ff0000?style=for-the-badge&labelColor=4c4c4c)](https://youtu.be/JUBMDTCiM0M) [![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&labelColor=4c4c4c)](https://www.python.org/) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge&labelColor=4c4c4c)](LICENSE)

## 🎬 SkillOpt Demo Video

https://github.com/user-attachments/assets/eb12d3bc-371c-467f-904d-91b61f339ed7

<p align="center">
  <a href="https://youtu.be/JUBMDTCiM0M"><b>▶ Watch the full demo on YouTube</b></a>
</p>

---

## Install

**Requirements:** Python 3.10+

```bash
git clone https://github.com/microsoft/SkillOpt.git
cd SkillOpt
pip install -e .

# For ALFWorld benchmark (optional):
pip install -e ".[alfworld]"
alfworld-download
```

### Configure API Credentials

```bash
cp .env.example .env
# Edit .env with your API credentials, then:
source .env
```

**Azure OpenAI** (recommended):
```bash
export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com/"
# Option 1: API key auth
export AZURE_OPENAI_API_KEY="your-key"
# Option 2: Azure CLI auth (no API key needed)
export AZURE_OPENAI_AUTH_MODE="azure_cli"
```

> **Note:** `AZURE_OPENAI_ENDPOINT` is always required. Without it, all LLM calls will fail.

**OpenAI** directly:
```bash
export OPENAI_API_KEY="sk-..."
```

**Anthropic Claude**:
```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

**Qwen (local vLLM)**:
```bash
export QWEN_CHAT_BASE_URL="http://localhost:8000/v1"
export QWEN_CHAT_MODEL="Qwen/Qwen3.5-4B"
```

---

## Data Preparation

SkillOpt expects data in a **split directory** with `train/`, `val/`, `test/` subdirectories, each containing a JSON file (e.g., `items.json`).

```
data/my_split/
├── train/items.json
├── val/items.json
└── test/items.json
```

Each JSON file is an array of task items. The required fields depend on the benchmark. For example, SearchQA items look like:

```json
[
  {
    "id": "unique_item_id",
    "question": "Who wrote the novel ...",
    "context": "[DOC] relevant passage text ...",
    "answers": ["expected answer"]
  }
]
```

See `skillopt/envs/<benchmark>/dataloader.py` for the exact format each benchmark expects.

> **Note:** Benchmark datasets are not included in this repository. Prepare your own data following the format above.

### Supported Benchmarks

| Benchmark | Type | Config |
|---|---|---|
| SearchQA | QA | `configs/searchqa/default.yaml` |
| ALFWorld | Embodied agent | `configs/alfworld/default.yaml` |
| DocVQA | Document QA | `configs/docvqa/default.yaml` |
| LiveMathematicianBench | Math | `configs/livemathematicianbench/default.yaml` |
| SpreadsheetBench | Code generation | `configs/spreadsheetbench/default.yaml` |
| OfficeQA | Tool-augmented QA | `configs/officeqa/default.yaml` |

---

## Quick Start

### Training

```bash
# Minimal example — train on SearchQA:
python scripts/train.py \
    --config configs/searchqa/default.yaml \
    --split_dir /path/to/your/searchqa_split \
    --azure_openai_endpoint https://your-resource.openai.azure.com/ \
    --optimizer_model gpt-5.5 \
    --target_model gpt-5.5

# Train on LiveMathematicianBench:
python scripts/train.py \
    --config configs/livemathematicianbench/default.yaml \
    --split_dir /path/to/your/livemath_split \
    --azure_openai_endpoint https://your-resource.openai.azure.com/ \
    --optimizer_model gpt-5.5 \
    --target_model gpt-5.5

# Train on ALFWorld:
python scripts/train.py \
    --config configs/alfworld/default.yaml \
    --split_dir /path/to/your/alfworld_split \
    --azure_openai_endpoint https://your-resource.openai.azure.com/ \
    --optimizer_model gpt-5.5 \
    --target_model gpt-5.5
```

Key CLI arguments:

| Argument | Description | Example |
|---|---|---|
| `--config` | Benchmark config YAML | `configs/searchqa/default.yaml` |
| `--split_dir` | Path to data split directory | `/path/to/split` |
| `--azure_openai_endpoint` | Azure OpenAI endpoint URL | `https://your-resource.openai.azure.com/` |
| `--optimizer_model` | Optimizer model deployment name | `gpt-5.5` |
| `--target_model` | Target model deployment name | `gpt-5.5` |
| `--num_epochs` | Number of training epochs | `4` |
| `--batch_size` | Batch size per step | `40` |
| `--workers` | Parallel rollout workers | `8` |
| `--out_root` | Output directory | `outputs/my_run` |

### Eval Only

Evaluate a trained skill on specific data splits without training:

```bash
# Evaluate on test set only:
python scripts/eval_only.py \
  --config configs/searchqa/default.yaml \
  --skill outputs/my_run/best_skill.md \
  --split valid_unseen \
  --split_dir /path/to/searchqa_split \
  --azure_openai_endpoint https://your-resource.openai.azure.com/

# Evaluate on all splits (train + val + test):
python scripts/eval_only.py \
  --config configs/searchqa/default.yaml \
  --skill outputs/my_run/best_skill.md \
  --split all \
  --split_dir /path/to/searchqa_split \
  --azure_openai_endpoint https://your-resource.openai.azure.com/
```

| Split | Description |
|---|---|
| `valid_unseen` | Test set |
| `valid_seen` | Validation set |
| `train` | Training set |
| `all` | All splits combined (default) |

### Output Structure

Each run writes to a structured output directory:

```
outputs/<run_name>/
├── config.json              # Flattened runtime config
├── history.json             # Per-step training history
├── runtime_state.json       # Resume checkpoint
├── best_skill.md            # Best validated skill document
├── skills/skill_vXXXX.md   # Skill snapshot per step
├── steps/step_XXXX/        # Per-step artifacts (patches, evals)
├── slow_update/epoch_XX/   # Slow update logs
└── meta_skill/epoch_XX/    # Meta skill logs
```

Re-running the same command auto-resumes from the last completed step.

---

## WebUI

Launch the monitoring dashboard (optional):

```bash
pip install -e ".[webui]"
python -m skillopt_webui.app
```

| Flag | Default | Description |
|---|---|---|
| `--port` | 7860 | Server port |
| `--host` | `0.0.0.0` | Bind address |
| `--share` | off | Create a public Gradio share link |

```bash
# With public share link (useful for remote servers)
python -m skillopt_webui.app --share
```

---

## Citation

```bibtex
@article{skillopt2026,
  title={SKILLOPT: Executive Strategy for Self-Evolving Agent Skills},
  author={SkillOpt Team},
  year={2026}
}
```

