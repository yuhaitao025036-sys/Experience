"""Auto-encoder cloze experiment following claude.viba.

Pipeline:
  PrepareDataSet -> BaselineCodingAgentModel -> get_edit_distance_ratio

PrepareDataSet:
  - Fetch all .py files from experience/
  - Build (1, num_files) symbolic tensors for paths and contents
  - Per batch: copy, mask a random range in one file, record ground truth
  - Stack into (batch_size, num_files) tensors

BaselineCodingAgentModel:
  - Merge path+content per file via st_attention (all files attend to last column)
  - coding_agent on merged (batch,) tensor
"""

import os
import copy
import random
import tempfile
import torch
from typing import List, Tuple

from experience.symbolic_tensor.tensor_util.make_tensor import make_tensor
from experience.symbolic_tensor.function.coding_agent import coding_agent
from experience.symbolic_tensor.function.get_edit_distance_ratio import get_edit_distance_ratio_impl


kMaskedHint = "<AUTOENCODER-CLOZE-MASK-PLACEHOLDER>"

EXPERIENCE_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "experience")


def _read_storage(tensor: torch.Tensor, flat_index: int) -> str:
    digits = list(str(flat_index))
    path = os.path.join(
        tensor.st_relative_to, tensor.st_tensor_uid,
        "storage", os.path.join(*digits), "data",
    )
    with open(path) as f:
        return f.read()


def _write_storage(tensor: torch.Tensor, flat_index: int, content: str):
    digits = list(str(flat_index))
    path = os.path.join(
        tensor.st_relative_to, tensor.st_tensor_uid,
        "storage", os.path.join(*digits), "data",
    )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


# ── PrepareDataSet ──

def _fetch_all_files(root_dir: str) -> List[Tuple[str, str]]:
    """Fetch all .py files as (relative_path, content)."""
    files = []
    for dirpath, _dirs, filenames in os.walk(root_dir):
        for fname in sorted(filenames):
            if not fname.endswith(".py"):
                continue
            fpath = os.path.join(dirpath, fname)
            rel = os.path.relpath(fpath, root_dir)
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    content = f.read()
            except (UnicodeDecodeError, OSError):
                continue
            if content.strip():
                files.append((rel, content))
    return sorted(files)


def _find_maskable_range(content: str) -> Tuple[int, int]:
    """Find the range of lines eligible for masking: skip comments and __main__ block."""
    lines = content.splitlines(keepends=True)
    # Find __main__ block start
    main_start = len(lines)
    for i, line in enumerate(lines):
        if line.strip().startswith("if __name__"):
            main_start = i
            break
    # Find first non-comment, non-blank, non-import line
    code_start = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and not stripped.startswith("import ") and not stripped.startswith("from "):
            code_start = i
            break
    return code_start, main_start


def _get_random_mask_range(content: str, min_size: int = 20) -> Tuple[int, int, str]:
    """Pick random line range to mask. Skips comments and __main__ block. Range >= min_size."""
    lines = content.splitlines(keepends=True)
    code_start, main_start = _find_maskable_range(content)
    maskable_len = main_start - code_start
    if maskable_len < min_size:
        # File too small — mask whatever we can
        mask_len = max(1, maskable_len)
        start = code_start
    else:
        mask_len = random.randint(min_size, min(maskable_len, max(min_size, maskable_len // 2)))
        start = random.randint(code_start, main_start - mask_len)
    end = start + mask_len
    return start, end, "".join(lines[start:end])


def _apply_mask(content: str, start: int, end: int) -> str:
    lines = content.splitlines(keepends=True)
    return "".join(lines[:start] + [kMaskedHint + "\n"] + lines[end:])


def prepare_dataset(
    total_batch_size: int,
    dataset_dir: str,
    tmpdir: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[str]]:
    """PrepareDataSet from claude.viba.

    Returns:
        masked_file_path_tensor:    Symbolic[Tensor(batch, num_files)]
        masked_file_content_tensor: Symbolic[Tensor(batch, num_files)]
        ground_truth_tensor:        Symbolic[Tensor(batch,)]
        file_info: list of "file_path:start-end" for logging
    """
    files = _fetch_all_files(dataset_dir)
    # Filter files with enough maskable code (>= 20 lines outside __main__)
    files = [(fp, fc) for fp, fc in files
             if _find_maskable_range(fc)[1] - _find_maskable_range(fc)[0] >= 20]
    num_files = len(files)
    assert num_files > 0, f"No sufficiently large .py files in {dataset_dir}"

    # Base tensors: (1, num_files) — shared template
    file_paths = [fp for fp, _ in files]
    file_contents = [fc for _, fc in files]

    all_masked_paths = []
    all_masked_contents = []
    all_ground_truths = []
    file_info = []

    for _ in range(total_batch_size):
        file_idx = random.randint(0, num_files - 1)
        fp, fc = files[file_idx]
        start, end, ground_truth = _get_random_mask_range(fc)
        masked_content = _apply_mask(fc, start, end)

        batch_paths = list(file_paths)
        batch_contents = list(file_contents)
        batch_contents[file_idx] = masked_content

        all_masked_paths.append(batch_paths)
        all_masked_contents.append(batch_contents)
        all_ground_truths.append(ground_truth)
        file_info.append(f"{fp}:{start+1}-{end}")

    masked_path_tensor = make_tensor(all_masked_paths, tmpdir)
    masked_content_tensor = make_tensor(all_masked_contents, tmpdir)
    gt_tensor = make_tensor(all_ground_truths, tmpdir)

    return masked_path_tensor, masked_content_tensor, gt_tensor, file_info


# ── BaselineCodingAgentModel ──

def _build_prefix_to_last_mask(batch_size: int, num_files: int) -> torch.Tensor:
    """Build attention mask: all files attend to last column (concat context → last).

    Shape: (batch, num_files, num_files), dtype bool.
    mask[b, i, j] = True means position i attends to position j.
    Last column (j=num_files-1) attends to ALL positions (gathers full context).
    Other columns attend only to themselves (identity).
    """
    mask = torch.eye(num_files, dtype=torch.bool).unsqueeze(0).expand(batch_size, -1, -1).clone()
    # Last row attends to all columns
    mask[:, num_files - 1, :] = True
    return mask


def baseline_coding_agent_model(
    masked_path_tensor: torch.Tensor,
    masked_content_tensor: torch.Tensor,
    llm_method: str = "raw_llm_api",
) -> torch.Tensor:
    """BaselineCodingAgentModel from claude.viba (simplified for token limits).

    Instead of st_attention over all files (exceeds context),
    merge path+content per file, then provide only the masked file's
    merged content plus a file listing as context.
    """
    batch_size, num_files = masked_path_tensor.shape
    tmpdir = masked_path_tensor.st_relative_to

    # Build (batch,) input: for each batch, find the masked file and
    # provide it with a file listing as context
    input_data = []
    for b in range(batch_size):
        # Find the masked file (the one containing kMaskedHint)
        file_listing = []
        masked_file_content = None
        masked_file_path = None
        for f in range(num_files):
            flat_idx = b * num_files + f
            path_text = _read_storage(masked_path_tensor, flat_idx)
            content_text = _read_storage(masked_content_tensor, flat_idx)
            file_listing.append(path_text)
            if kMaskedHint in content_text:
                masked_file_content = content_text
                masked_file_path = path_text

        context = f"# Repository files ({num_files} total):\n"
        context += "\n".join(f"  {fp}" for fp in file_listing)
        context += f"\n\n# Target file: {masked_file_path}\n"
        context += f"# Lines masked with {kMaskedHint}\n\n"
        context += masked_file_content
        input_data.append(context)

    input_tensor = make_tensor(input_data, tmpdir)

    task_prompt = (
        "You are an auto-encoder for source code.\n"
        f"The input contains a Python file with a masked region marked by {kMaskedHint}.\n"
        "Your task: reconstruct ONLY the masked lines.\n"
        "Output ONLY the missing source code lines. No explanations."
    )

    output = coding_agent(
        input_tensor,
        task_prompt=task_prompt,
        llm_method=llm_method,
    )
    return output


# ── Main ──

def run_experiment(total_batch_size: int = 16, llm_method: str = "raw_llm_api"):
    root_dir = os.path.realpath(EXPERIENCE_ROOT)
    print(f"Dataset: {root_dir}")

    tmpdir = tempfile.mkdtemp()
    masked_path_tensor, masked_content_tensor, gt_tensor, file_info = prepare_dataset(
        total_batch_size, root_dir, tmpdir,
    )
    print(f"Batch={total_batch_size}, files={masked_path_tensor.shape[1]}")
    for i, info in enumerate(file_info):
        gt_preview = _read_storage(gt_tensor, i)[:60].replace("\n", "\\n")
        print(f"  [{i}] {info} -> {gt_preview}...")

    print(f"\nRunning baseline (llm_method={llm_method})...")
    output = baseline_coding_agent_model(
        masked_path_tensor, masked_content_tensor, llm_method=llm_method,
    )

    loss = get_edit_distance_ratio_impl(output, gt_tensor)
    print(f"\n{'='*50}")
    print(f"Results (edit_distance_ratio, lower=better):")
    print(f"{'='*50}")
    for i in range(total_batch_size):
        actual = _read_storage(output, i)
        gt = _read_storage(gt_tensor, i)
        r = loss[i].item()
        print(f"  [{i}] {file_info[i]}  loss={r:.4f}")
        print(f"       gt:     {gt[:80].replace(chr(10), '\\n')}")
        print(f"       actual: {actual[:80].replace(chr(10), '\\n')}")

    mean_loss = loss.float().mean().item()
    print(f"\nMean loss: {mean_loss:.4f}")
    return mean_loss


if __name__ == "__main__":
    import subprocess

    result = subprocess.run(
        ["bash", "-c", "source ~/.anthropic.sh && env"],
        capture_output=True, text=True,
    )
    for line in result.stdout.splitlines():
        if "=" in line:
            key, _, val = line.partition("=")
            os.environ[key] = val
    os.environ.pop("CLAUDECODE", None)

    run_experiment(total_batch_size=16, llm_method="raw_llm_api")
