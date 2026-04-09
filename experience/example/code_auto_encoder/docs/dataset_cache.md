# Dataset Cache Documentation

## Overview

`prepare_dataset.py` supports dataset caching for:
1. **Fixed experiment input**: Ensure different models use exactly the same test data
2. **Accelerate repeated experiments**: Avoid regenerating random samples every run
3. **Incremental expansion**: Generate more samples based on existing cache

## Quick Start

### Three LLM Method Tests

The system supports three `--llm-method` options. Below are complete example commands for each method:

#### 1. raw_llm_api (Direct API Call)

The simplest approach, directly calling the LLM API:

```bash
python experience/example/code_auto_encoder/test_baseline.py \
    --llm-method raw_llm_api \
    --num-iterations 1 \
    --total-batch-size 1 \
    --dataset-cache-dir ./my_experiment \
    --seed 42
```

#### 2. coding_agent (Coding Agent Mode)

Uses the coding agent framework:

```bash
python experience/example/code_auto_encoder/test_baseline.py \
    --llm-method coding_agent \
    --num-iterations 1 \
    --total-batch-size 1 \
    --dataset-cache-dir ./my_experiment \
    --seed 42
```

#### 3. tmux_cc (Tmux Interactive Mode)

Runs ducc via tmux with visual observation support:

```bash
python experience/example/code_auto_encoder/test_baseline.py \
    --llm-method tmux_cc \
    --interactive \
    --tmux-session manual_ducc \
    --num-iterations 1 \
    --total-batch-size 1 \
    --dataset-cache-dir ./my_experiment \
    --seed 42
```

Observe execution in another terminal:
```bash
tmux attach -t manual_ducc
```

### Method Comparison

| Method | Features | Use Cases |
|--------|----------|-----------|
| `raw_llm_api` | Direct API call, fastest | Quick testing, batch evaluation |
| `coding_agent` | With agent framework | Requires file operation capabilities |
| `tmux_cc` | Visual interaction | Debugging, observing agent behavior |

### Python API Usage

```python
from experience.example.code_auto_encoder.prepare_dataset import parepare_dataset

# Generate and cache
paths, contents, gt, info = parepare_dataset(
    total_batch_size=16,
    dataset_dir="./codebase",
    tmpdir="/tmp/tensors",
    dataset_cache_dir="./my_experiment",  # Cache directory
    seed=42,                               # Random seed
)

# Subsequent load (auto-detect cache)
paths, contents, gt, info = parepare_dataset(
    total_batch_size=16,
    dataset_dir="./codebase",
    tmpdir="/tmp/tensors2",
    dataset_cache_dir="./my_experiment",
    seed=42,
)
# Output: Loaded 16 samples from cache: ./my_experiment/seed_42
```

## Parameter Description

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `total_batch_size` | int | Required | Number of samples needed |
| `dataset_dir` | str | Required | Source code directory (containing .py files) |
| `tmpdir` | str | Required | Symbolic tensor storage directory |
| `dataset_cache_dir` | str | None | Cache directory, None means no caching |
| `seed` | int | None | Random seed, None means not fixed |

## Cache Directory Structure

```
my_experiment/                      # dataset_cache_dir
├── seed_42/                        # Data for seed=42
│   ├── metadata.json               # Metadata
│   ├── source_files/               # Original source files (stored once)
│   │   ├── index.json              # File path list
│   │   └── contents/               # File contents
│   │       ├── 0.txt
│   │       ├── 1.txt
│   │       └── ...
│   └── samples/                    # Sample mask indices
│       ├── 0.json
│       ├── 1.json
│       └── ...
├── seed_123/                       # Data for seed=123
│   └── ...
└── seed_none/                      # Data for seed=None
    └── ...
```

## File Formats

### metadata.json

Records basic cache information:

```json
{
  "version": 2,
  "dataset_dir": "/absolute/path/to/codebase",
  "seed": 42,
  "num_files": 45,
  "cached_samples": 16
}
```

| Field | Description |
|-------|-------------|
| `version` | Cache format version (currently 2) |
| `dataset_dir` | Absolute path of source code directory |
| `seed` | Random seed |
| `num_files` | Number of source files |
| `cached_samples` | Number of cached samples |

### source_files/index.json

Source file path list (relative paths):

```json
[
  "symbolic_tensor/tensor_util/assign_tensor.py",
  "symbolic_tensor/tensor_util/load_tensor.py",
  "symbolic_tensor/function/coding_agent.py",
  ...
]
```

### source_files/contents/{i}.txt

Complete content of the i-th source file.

### samples/{i}.json

Mask index information for the i-th sample:

```json
{
  "file_idx": 40,
  "mask_start": 5,
  "mask_end": 10,
  "ground_truth": "def _get_storage_path(tensor, coords):\n    ..."
}
```

| Field | Description |
|-------|-------------|
| `file_idx` | Index of the masked file in index.json |
| `mask_start` | Mask start line number (0-indexed) |
| `mask_end` | Mask end line number (exclusive) |
| `ground_truth` | Actual masked code content |

## Use Cases

### Case 1: Model Comparison Experiments

Ensure different models use exactly the same input:

```bash
# Model A
python test_baseline.py --llm-method raw_llm_api \
    --dataset-cache-dir ./exp_v1 --seed 42 --total-batch-size 20

# Model B (using same data)
python test_baseline.py --llm-method tmux_cc \
    --dataset-cache-dir ./exp_v1 --seed 42 --total-batch-size 20

# Model C (using same data)
python test_baseline.py --llm-method coding_agent \
    --dataset-cache-dir ./exp_v1 --seed 42 --total-batch-size 20
```

### Case 2: Incremental Sample Expansion

Start with small dataset for validation, then expand:

```bash
# First use 5 samples for quick validation
python test_baseline.py --dataset-cache-dir ./exp_v1 --seed 42 --total-batch-size 5

# After validation, expand to 50 samples (reuses first 5)
python test_baseline.py --dataset-cache-dir ./exp_v1 --seed 42 --total-batch-size 50
# Output: Cache has 5 samples, need 50, will generate more
#         Loaded 5 cached samples from: ./exp_v1/seed_42
#         Saved 45 new samples to cache: ./exp_v1/seed_42
```

### Case 3: Multiple Control Group Experiments

Use different seeds to generate multiple independent datasets:

```bash
# Experiment group 1
python test_baseline.py --dataset-cache-dir ./exp --seed 42 --total-batch-size 20

# Experiment group 2 (different data)
python test_baseline.py --dataset-cache-dir ./exp --seed 123 --total-batch-size 20

# Experiment group 3 (different data)
python test_baseline.py --dataset-cache-dir ./exp --seed 456 --total-batch-size 20
```

Cache structure:
```
exp/
├── seed_42/    # Experiment group 1 data
├── seed_123/   # Experiment group 2 data
└── seed_456/   # Experiment group 3 data
```

## Cache Behavior

| Scenario | Behavior |
|----------|----------|
| Cache does not exist | Generate data → Save to cache |
| Cache exists with enough samples | Load directly from cache |
| Cache exists but insufficient samples | Load existing + Generate new + Update cache |
| Different seed | Stored in different subdirectories, no interference |
| dataset_dir mismatch | Ignore cache, regenerate |

## Storage Optimization

New format (v2) significantly saves space compared to old format:

| Comparison | Old Format | New Format (v2) |
|------------|------------|-----------------|
| Storage method | Store all file contents per sample | Store source files once + sample indices only |
| 16 samples × 45 files | ~720 file copies | 45 files + 16 indices |
| Space complexity | O(samples × files) | O(files + samples) |

## Notes

1. **Importance of seed**: Same seed guarantees same random sequence, different seeds generate completely different data

2. **dataset_dir matching**: Cache validates absolute path of dataset_dir, regeneration needed if source directory changes

3. **Incremental consistency**: During incremental generation, random numbers for cached samples are skipped to ensure new samples are consistent with fresh generation

4. **Without cache**: If `dataset_cache_dir` is not provided, data is regenerated every run (using seed ensures reproducibility)

## Parameter Combination Comparison

| Command Parameters | Data Consistency | Caching | Use Cases |
|-------------------|------------------|---------|-----------|
| `--seed 42` | ✅ Same every time | ❌ No cache | Debugging, quick validation |
| `--dataset-cache-dir ./exp` | ❌ Different every time | ✅ Cached | ⚠️ Not recommended |
| `--seed 42 --dataset-cache-dir ./exp` | ✅ Same every time | ✅ Cached | **Production experiments (recommended)** |

**Notes**:
- **seed only**: Reproducible, but regenerates data every time (wastes computation time)
- **cache-dir only**: Caches to `seed_none/` directory, but data is not reproducible
- **Both**: ✅ Best practice, both reproducible and cache-accelerated
