import torch
import torch.nn as nn
from typing import Callable, Dict, Optional, Type

from experience.symbolic_tensor.function.slice_attention import slice_attention
from experience.symbolic_tensor.function.merge import merge
from experience.fs_util.text_merger import TextMerger


class Attention(nn.Module):
    """Symbolic attention module: slice_attention + merge last dim.

    Forward pipeline:
      input (batch, seq_len) + attention_mask (batch, seq_len, seq_len)
        -> slice_attention -> (batch, seq_len, seq_len)
        -> merge(axis=-1)  -> (batch, seq_len)

    Each output token is the merged representation of all tokens it attends to.

    Args:
        return_view: If True, use symlinks (views) instead of copies in slice_attention.
        grad_input_prompt: Custom prompt callable for slice_attention backward. None uses default.
        task_prompt: High-level task description.
        llm_method: LLM backend to use.
        llm_env: Environment variable dict for LLM client.
        text_merger: Class with pack/unpack for merge. Defaults to TextMerger.
    """

    def __init__(
        self,
        return_view: bool = False,
        grad_input_prompt: Optional[Callable[..., str]] = None,
        task_prompt: str = "",
        llm_method: str = "raw_llm_api",
        llm_env: Optional[Dict[str, str]] = None,
        text_merger: Optional[Type] = None,
    ):
        super().__init__()
        self.return_view = return_view
        self.grad_input_prompt = grad_input_prompt
        self.task_prompt = task_prompt
        self.llm_method = llm_method
        self.llm_env = llm_env
        self.text_merger = text_merger

    def forward(
        self,
        input: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        # slice_attention: (batch, seq_len) -> (batch, seq_len, seq_len)
        sliced = slice_attention(
            input, attention_mask,
            self.return_view, self.grad_input_prompt,
            self.task_prompt, self.llm_method, self.llm_env,
        )
        # merge last dim: (batch, seq_len, seq_len) -> (batch, seq_len)
        return merge(sliced, -1, self.text_merger)


if __name__ == "__main__":
    import os
    import tempfile
    from experience.symbolic_tensor.tensor_util.make_tensor import make_tensor
    from experience.fs_util.text_merger import kFrameMarker

    print("Running Attention module tests...\n")

    def run_test(name: str, condition: bool, expected=None, actual=None):
        if condition:
            print(f"  \u2713 {name}")
        else:
            print(f"  \u2717 {name}")
            if expected is not None and actual is not None:
                print(f"    expected: {expected}")
                print(f"    actual:   {actual}")

    def read_storage(tensor, flat_index):
        digits = list(str(flat_index))
        path = os.path.join(
            tensor.st_relative_to,
            tensor.st_tensor_uid,
            "storage",
            os.path.join(*digits),
            "data",
        )
        path = os.path.realpath(path)
        if not os.path.isfile(path):
            return None
        with open(path) as f:
            return f.read()

    # Test 1: Construction with defaults
    print("Test 1: Construction with defaults")
    module = Attention()
    run_test("is nn.Module", isinstance(module, nn.Module))
    run_test("return_view default False", module.return_view is False)
    run_test("grad_input_prompt default None", module.grad_input_prompt is None)
    run_test("task_prompt default ''", module.task_prompt == "")
    run_test("llm_method default 'raw_llm_api'", module.llm_method == "raw_llm_api")
    run_test("llm_env default None", module.llm_env is None)
    run_test("text_merger default None", module.text_merger is None)

    # Test 2: Construction with custom params
    print("Test 2: Construction with custom params")
    module = Attention(
        return_view=True,
        task_prompt="Test task",
    )
    run_test("return_view True", module.return_view is True)
    run_test("task_prompt stored", module.task_prompt == "Test task")

    # Test 3: Forward — causal 1x3, output shape is (batch, seq_len)
    print("Test 3: Causal 1x3 forward")
    with tempfile.TemporaryDirectory() as tmpdir:
        module = Attention()
        inp = make_tensor([["a", "b", "c"]], tmpdir)
        mask = torch.tril(torch.ones(1, 3, 3, dtype=torch.bool))

        output = module(inp, mask)
        run_test("Output shape (1, 3)", list(output.shape) == [1, 3])
        run_test("Output has st_relative_to", hasattr(output, "st_relative_to"))
        run_test("Output has st_tensor_uid", hasattr(output, "st_tensor_uid"))
        # output[0, 0] = merge of [a] (only attends to self)
        content_0 = read_storage(output, 0)
        run_test("output[0,0] not None", content_0 is not None)
        frames_0 = TextMerger.unpack(content_0)
        run_test("output[0,0] 1 frame", len(frames_0) == 1)
        run_test("output[0,0] frame = 'a'", frames_0[0][2] == "a")
        # output[0, 1] = merge of [a, b]
        content_1 = read_storage(output, 1)
        frames_1 = TextMerger.unpack(content_1)
        run_test("output[0,1] 2 frames", len(frames_1) == 2)
        run_test("output[0,1] frame[0] = 'a'", frames_1[0][2] == "a")
        run_test("output[0,1] frame[1] = 'b'", frames_1[1][2] == "b")
        # output[0, 2] = merge of [a, b, c]
        content_2 = read_storage(output, 2)
        frames_2 = TextMerger.unpack(content_2)
        run_test("output[0,2] 3 frames", len(frames_2) == 3)
        run_test("output[0,2] frame[0] = 'a'", frames_2[0][2] == "a")
        run_test("output[0,2] frame[2] = 'c'", frames_2[2][2] == "c")

    # Test 4: Forward — partial mask
    print("Test 4: Partial mask 1x3")
    with tempfile.TemporaryDirectory() as tmpdir:
        module = Attention()
        inp = make_tensor([["x", "y", "z"]], tmpdir)
        mask = torch.zeros(1, 3, 3, dtype=torch.bool)
        mask[0, 0, 0] = True
        mask[0, 2, 0] = True
        mask[0, 2, 2] = True

        output = module(inp, mask)
        run_test("Output shape (1, 3)", list(output.shape) == [1, 3])
        # output[0, 0] = merge of [x]
        frames_0 = TextMerger.unpack(read_storage(output, 0))
        run_test("output[0,0] 1 frame = 'x'", len(frames_0) == 1 and frames_0[0][2] == "x")
        # output[0, 1] = no attention → empty/zero
        run_test("output[0,1] coeff zero", output.data[0, 1].item() == 0.0)
        # output[0, 2] = merge of [x, z]
        frames_2 = TextMerger.unpack(read_storage(output, 2))
        run_test("output[0,2] 2 frames", len(frames_2) == 2)
        run_test("output[0,2] frame[0] = 'x'", frames_2[0][2] == "x")
        run_test("output[0,2] frame[1] = 'z'", frames_2[1][2] == "z")

    # Test 5: Multi-batch 2x2
    print("Test 5: Multi-batch 2x2")
    with tempfile.TemporaryDirectory() as tmpdir:
        module = Attention()
        inp = make_tensor([["p", "q"], ["r", "s"]], tmpdir)
        mask = torch.zeros(2, 2, 2, dtype=torch.bool)
        mask[0, 1, 0] = True
        mask[0, 1, 1] = True
        mask[1, 0, 0] = True

        output = module(inp, mask)
        run_test("Output shape (2, 2)", list(output.shape) == [2, 2])
        # batch 0, token 1: attends to [p, q]
        frames_b0t1 = TextMerger.unpack(read_storage(output, 1))
        run_test("b0[1] 2 frames", len(frames_b0t1) == 2)
        run_test("b0[1] frame[0] = 'p'", frames_b0t1[0][2] == "p")
        run_test("b0[1] frame[1] = 'q'", frames_b0t1[1][2] == "q")
        # batch 1, token 0: attends to [r]
        frames_b1t0 = TextMerger.unpack(read_storage(output, 2))
        run_test("b1[0] 1 frame = 'r'", len(frames_b1t0) == 1 and frames_b1t0[0][2] == "r")

    # Test 6: Empty mask
    print("Test 6: Empty mask")
    with tempfile.TemporaryDirectory() as tmpdir:
        module = Attention()
        inp = make_tensor([["a", "b"]], tmpdir)
        mask = torch.zeros(1, 2, 2, dtype=torch.bool)

        output = module(inp, mask)
        run_test("Output shape (1, 2)", list(output.shape) == [1, 2])
        run_test("all zero", (output == 0).all().item())

    # Test 7: Custom text_merger
    print("Test 7: Custom text_merger")

    class SimpleMerger:
        @staticmethod
        def pack(frames):
            return " | ".join(f"{idx}:{text}" for idx, _, text in frames)

        @staticmethod
        def unpack(merged):
            parts = merged.split(" | ")
            return [(int(p.split(":", 1)[0]), 1.0, p.split(":", 1)[1]) for p in parts]

    with tempfile.TemporaryDirectory() as tmpdir:
        module = Attention(text_merger=SimpleMerger)
        inp = make_tensor([["a", "b"]], tmpdir)
        mask = torch.ones(1, 2, 2, dtype=torch.bool)

        output = module(inp, mask)
        run_test("Output shape (1, 2)", list(output.shape) == [1, 2])
        content_1 = read_storage(output, 1)
        run_test("custom format", " | " in content_1)

    # Test 8: return_view=True
    print("Test 8: return_view=True")
    with tempfile.TemporaryDirectory() as tmpdir:
        module = Attention(return_view=True)
        inp = make_tensor([["alpha", "beta"]], tmpdir)
        mask = torch.tensor([[[True, False], [True, True]]])

        output = module(inp, mask)
        run_test("Output shape (1, 2)", list(output.shape) == [1, 2])
        frames_1 = TextMerger.unpack(read_storage(output, 1))
        run_test("output[0,1] 2 frames", len(frames_1) == 2)
        run_test("frame[0] = 'alpha'", frames_1[0][2] == "alpha")
        run_test("frame[1] = 'beta'", frames_1[1][2] == "beta")

    print("\nAll tests completed.")
