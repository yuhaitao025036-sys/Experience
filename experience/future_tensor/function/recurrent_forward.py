"""
recurrent_forward :=
    $output FutureTensor[($prefix_dims ...)]
    <- $input FutureTensor[($prefix_dims ..., $recurrent_dim int, 3)]
    # inline
    <- $output.ft_async_get recurrent_forward_async_get

recurrent_forward_async_get :=
    Awaitable[(str, float)]
    <- $cooridates list[int]
    <- $prompt str
    # captured args
    <- $input FutureTensor[($prefix_dims ..., $recurrent_dim int, 3)]
    # inline
    <- $cur_prompt ($prompt)
    <- (Loop[$i int] <- range <- $recurrent_dim)
    <- (($cur_output str, $cur_output_confidence float)
        <- $input.ft_async_get
        <- {[*cooridates, i, 0]}
        <- $cur_prompt)
    <- (($cur_output_status ("Ok"|"Error"), $cur_output_status_confidence float)
        <- $input.ft_async_get
        <- {[*cooridates, i, 1]}
        <- $cur_prompt)
    <- { return (cur_output, 1.0) if cur_output_status is "Ok" }
    <- (($error_details str, _)
        <- $input.ft_async_get
        <- {[*cooridates, i, 2]}
        <- $cur_prompt)
    <- {
        # for next iteration
        cur_prompt = (
            f"{cur_prompt}\\n"
            f"====[Iteration {i}]====\\n"
            f"Output:\\n\\n{cur_output}\\n\\n"
            f"Status:{cur_output_status}\\n\\n"
            f"Error Details:\\n\\n{error_details}\\n\\n"
        )
    }
    <- {
        if all trials failed:
            return ($cur_output, cur_output_confidence * cur_output_status_confidence)
            where cur_output_confidence * cur_output_status_confidence is the max
     }
"""

from typing import List, Tuple

from experience.future_tensor.future_tensor import FutureTensor


def recurrent_forward(input: FutureTensor) -> FutureTensor:
    """Recurrent forward: retry loop encoded in tensor dimensions.

    Takes input of shape (*prefix_dims, recurrent_dim, 3) where the last two
    dims encode a generate-validate-diagnose loop. Returns output of shape
    (*prefix_dims).

    For each prefix coordinate, iterates i in range(recurrent_dim):
      - [*coords, i, 0]: generator  -> (output, confidence)
      - [*coords, i, 1]: validator  -> (status "Ok"|"Error", confidence)
      - [*coords, i, 2]: diagnoser  -> (error_details, _)
    Returns on first "Ok", or best result (by confidence product) if all fail.

    Args:
        input: FutureTensor of shape (*prefix_dims, recurrent_dim, 3).

    Returns:
        FutureTensor of shape (*prefix_dims).
    """
    input_shape = input.shape
    assert len(input_shape) >= 2, (
        f"Input must have at least 2 dims, got shape {input_shape}"
    )
    assert input_shape[-1] == 3, (
        f"Last dim must be 3 (generator, validator, diagnoser), got {input_shape[-1]}"
    )

    recurrent_dim = input_shape[-2]
    prefix_shape = input_shape[:-2]

    async def recurrent_forward_async_get(
        coordinates: List[int], prompt: str
    ) -> Tuple[str, float]:
        cur_prompt = prompt

        best_output = ""
        best_score = -1.0

        for i in range(recurrent_dim):
            # Generator: [*coordinates, i, 0]
            cur_output, cur_output_confidence = await input.ft_async_get(
                [*coordinates, i, 0], cur_prompt
            )

            # Validator: [*coordinates, i, 1]
            cur_output_status, cur_output_status_confidence = await input.ft_async_get(
                [*coordinates, i, 1], cur_prompt
            )

            # Early return on Ok
            if cur_output_status == "Ok":
                return (cur_output, 1.0)

            # Track best by confidence product
            score = cur_output_confidence * cur_output_status_confidence
            if score > best_score:
                best_score = score
                best_output = cur_output

            # Diagnoser: [*coordinates, i, 2]
            error_details, _ = await input.ft_async_get(
                [*coordinates, i, 2], cur_prompt
            )

            # Build prompt for next iteration
            cur_prompt = (
                f"{cur_prompt}\n"
                f"====[Iteration {i}]====\n"
                f"Output:\n\n{cur_output}\n\n"
                f"Status:{cur_output_status}\n\n"
                f"Error Details:\n\n{error_details}\n\n"
            )

        # All trials failed — return best result
        return (best_output, max(0.0, min(1.0, best_score)))

    output = FutureTensor(prefix_shape, input.st_relative_to, recurrent_forward_async_get)
    return output


if __name__ == "__main__":
    import asyncio
    import os
    import tempfile

    from experience.symbolic_tensor.tensor_util.make_tensor import make_tensor as st_make_tensor
    from experience.symbolic_tensor.tensor_util.assign_tensor import assign_tensor

    print("Running 100 tests for recurrent_forward...\n")

    passed = 0
    failed = 0

    def run_test(name: str, condition: bool, expected=None, actual=None):
        global passed, failed
        if condition:
            passed += 1
        else:
            failed += 1
            print(f"  ✗ {name}")
            if expected is not None:
                print(f"    expected: {expected}, actual: {actual}")

    def _storage_path(ft, flat_index):
        digits = list(str(flat_index))
        return os.path.join(
            ft.st_relative_to, ft.st_tensor_uid,
            "storage", os.path.join(*digits), "data",
        )

    def read_ft_element(ft, flat_index):
        path = _storage_path(ft, flat_index)
        if not os.path.isfile(path):
            return None
        with open(path) as f:
            return f.read()

    # ── Helper: build an input FutureTensor from a response table ─────

    def make_input_ft(shape, response_table, tmpdir):
        """Create a FutureTensor with a scripted ft_async_get.

        response_table: dict mapping tuple(coordinates) -> (str, float)
        """
        async def scripted_get(coords, prompt):
            key = tuple(coords)
            if key in response_table:
                val = response_table[key]
                if callable(val):
                    return val(prompt)
                return val
            return ("???", 0.0)

        return FutureTensor(shape, tmpdir, scripted_get)

    # === Group 1: Single element, Ok on first try (tests 1-10) ===

    with tempfile.TemporaryDirectory() as tmpdir:
        # shape: (1, 3) -> prefix (), recurrent_dim=1
        # [0, 0] = generator, [0, 1] = validator, [0, 2] = diagnoser
        table = {
            (0, 0): ("hello world", 0.9),
            (0, 1): ("Ok", 0.95),
            (0, 2): ("no error", 0.0),
        }
        inp = make_input_ft([1, 3], table, tmpdir)
        out = recurrent_forward(inp)

        run_test("1: output shape []", out.shape == [])
        run_test("2: not forwarded", out.ft_forwarded is False)

        prompt_t = st_make_tensor("test_prompt", tmpdir)
        out.ft_forward(prompt_t)

        run_test("3: forwarded", out.ft_forwarded is True)
        run_test("4: content", read_ft_element(out, 0) == "hello world")
        run_test("5: confidence 1.0 (Ok)",
                 abs(out.tensor.data.flatten()[0].item() - 1.0) < 0.01)

    with tempfile.TemporaryDirectory() as tmpdir:
        # shape: (3, 3) -> prefix (), recurrent_dim=3, Ok on first
        table = {
            (0, 0): ("result_0", 0.8),
            (0, 1): ("Ok", 0.9),
        }
        inp = make_input_ft([3, 3], table, tmpdir)
        out = recurrent_forward(inp)
        run_test("6: shape [] with recurrent_dim=3", out.shape == [])
        prompt_t = st_make_tensor("p", tmpdir)
        out.ft_forward(prompt_t)
        run_test("7: content from i=0", read_ft_element(out, 0) == "result_0")
        run_test("8: confidence 1.0", abs(out.tensor.data.flatten()[0].item() - 1.0) < 0.01)

    with tempfile.TemporaryDirectory() as tmpdir:
        # prefix [2], recurrent_dim=1 -> shape [2, 1, 3]
        table = {
            (0, 0, 0): ("out_a", 0.9),
            (0, 0, 1): ("Ok", 0.8),
            (1, 0, 0): ("out_b", 0.7),
            (1, 0, 1): ("Ok", 0.6),
        }
        inp = make_input_ft([2, 1, 3], table, tmpdir)
        out = recurrent_forward(inp)
        run_test("9: prefix [2] shape", out.shape == [2])
        prompt_t = st_make_tensor(["pa", "pb"], tmpdir)
        out.ft_forward(prompt_t)
        run_test("10: elem 0 content", read_ft_element(out, 0) == "out_a")

    # === Group 2: Ok on second try (tests 11-25) ===

    with tempfile.TemporaryDirectory() as tmpdir:
        # recurrent_dim=2, fail first, ok second
        call_log = []

        async def logged_get(coords, prompt):
            call_log.append((tuple(coords), prompt))
            key = tuple(coords)
            if key == (0, 0):
                return ("bad_output", 0.3)
            if key == (0, 1):
                return ("Error", 0.2)
            if key == (0, 2):
                return ("missing semicolon", 0.0)
            if key == (1, 0):
                return ("good_output", 0.95)
            if key == (1, 1):
                return ("Ok", 0.9)
            return ("???", 0.0)

        inp = FutureTensor([2, 3], tmpdir, logged_get)
        out = recurrent_forward(inp)
        prompt_t = st_make_tensor("initial", tmpdir)
        out.ft_forward(prompt_t)

        run_test("11: content is good_output", read_ft_element(out, 0) == "good_output")
        run_test("12: confidence 1.0 (Ok on i=1)",
                 abs(out.tensor.data.flatten()[0].item() - 1.0) < 0.01)

        # Check call sequence
        run_test("13: first call is generator i=0", call_log[0][0] == (0, 0))
        run_test("14: second call is validator i=0", call_log[1][0] == (0, 1))
        run_test("15: third call is diagnoser i=0", call_log[2][0] == (0, 2))
        run_test("16: fourth call is generator i=1", call_log[3][0] == (1, 0))
        run_test("17: fifth call is validator i=1", call_log[4][0] == (1, 1))
        # Should NOT call diagnoser for i=1 since Ok
        run_test("18: total 5 calls (no diagnoser for Ok)",
                 len(call_log) == 5, 5, len(call_log))

        # Check prompt accumulation for i=1
        i1_prompt = call_log[3][1]
        run_test("19: i=1 prompt has 'Iteration 0'", "Iteration 0" in i1_prompt)
        run_test("20: i=1 prompt has bad_output", "bad_output" in i1_prompt)
        run_test("21: i=1 prompt has Error", "Error" in i1_prompt)
        run_test("22: i=1 prompt has missing semicolon", "missing semicolon" in i1_prompt)
        run_test("23: i=1 prompt starts with initial", i1_prompt.startswith("initial"))

    with tempfile.TemporaryDirectory() as tmpdir:
        # Ok on third try (recurrent_dim=3)
        async def ok_on_third(coords, prompt):
            key = tuple(coords)
            if key == (0, 0): return ("out0", 0.5)
            if key == (0, 1): return ("Error", 0.3)
            if key == (0, 2): return ("err0", 0.0)
            if key == (1, 0): return ("out1", 0.6)
            if key == (1, 1): return ("Error", 0.4)
            if key == (1, 2): return ("err1", 0.0)
            if key == (2, 0): return ("out2", 0.9)
            if key == (2, 1): return ("Ok", 0.8)
            return ("???", 0.0)

        inp = FutureTensor([3, 3], tmpdir, ok_on_third)
        out = recurrent_forward(inp)
        prompt_t = st_make_tensor("start", tmpdir)
        out.ft_forward(prompt_t)
        run_test("24: content from i=2", read_ft_element(out, 0) == "out2")
        run_test("25: confidence 1.0", abs(out.tensor.data.flatten()[0].item() - 1.0) < 0.01)

    # === Group 3: All fail — best result returned (tests 26-40) ===

    with tempfile.TemporaryDirectory() as tmpdir:
        # recurrent_dim=3, all Error
        async def all_fail(coords, prompt):
            key = tuple(coords)
            i = key[0]
            slot = key[1]
            if slot == 0:
                confs = [0.3, 0.7, 0.5]
                return (f"output_{i}", confs[i])
            if slot == 1:
                confs = [0.4, 0.8, 0.6]
                return ("Error", confs[i])
            if slot == 2:
                return (f"error_{i}", 0.0)
            return ("???", 0.0)

        inp = FutureTensor([3, 3], tmpdir, all_fail)
        out = recurrent_forward(inp)
        prompt_t = st_make_tensor("p", tmpdir)
        out.ft_forward(prompt_t)

        # Best score: i=1 -> 0.7 * 0.8 = 0.56, i=2 -> 0.5*0.6=0.30, i=0 -> 0.3*0.4=0.12
        run_test("26: best content is output_1", read_ft_element(out, 0) == "output_1")
        run_test("27: confidence = 0.56",
                 abs(out.tensor.data.flatten()[0].item() - 0.56) < 0.02)

    with tempfile.TemporaryDirectory() as tmpdir:
        # All fail, best is last iteration
        async def best_last(coords, prompt):
            key = tuple(coords)
            i, slot = key
            if slot == 0: return (f"out_{i}", 0.1 * (i + 1))
            if slot == 1: return ("Error", 0.1 * (i + 1))
            if slot == 2: return ("err", 0.0)
            return ("???", 0.0)

        inp = FutureTensor([3, 3], tmpdir, best_last)
        out = recurrent_forward(inp)
        prompt_t = st_make_tensor("p", tmpdir)
        out.ft_forward(prompt_t)
        # i=0: 0.1*0.1=0.01, i=1: 0.2*0.2=0.04, i=2: 0.3*0.3=0.09
        run_test("28: best is out_2", read_ft_element(out, 0) == "out_2")
        run_test("29: confidence = 0.09",
                 abs(out.tensor.data.flatten()[0].item() - 0.09) < 0.02)

    with tempfile.TemporaryDirectory() as tmpdir:
        # All fail, best is first iteration
        async def best_first(coords, prompt):
            key = tuple(coords)
            i, slot = key
            if slot == 0:
                confs = [0.9, 0.2, 0.1]
                return (f"out_{i}", confs[i])
            if slot == 1:
                confs = [0.9, 0.2, 0.1]
                return ("Error", confs[i])
            if slot == 2: return ("err", 0.0)
            return ("???", 0.0)

        inp = FutureTensor([3, 3], tmpdir, best_first)
        out = recurrent_forward(inp)
        prompt_t = st_make_tensor("p", tmpdir)
        out.ft_forward(prompt_t)
        # i=0: 0.81, i=1: 0.04, i=2: 0.01
        run_test("30: best is out_0", read_ft_element(out, 0) == "out_0")
        run_test("31: confidence = 0.81",
                 abs(out.tensor.data.flatten()[0].item() - 0.81) < 0.02)

    with tempfile.TemporaryDirectory() as tmpdir:
        # recurrent_dim=1, single fail
        async def single_fail(coords, prompt):
            key = tuple(coords)
            if key == (0, 0): return ("only_out", 0.6)
            if key == (0, 1): return ("Error", 0.5)
            if key == (0, 2): return ("detail", 0.0)
            return ("???", 0.0)

        inp = FutureTensor([1, 3], tmpdir, single_fail)
        out = recurrent_forward(inp)
        prompt_t = st_make_tensor("p", tmpdir)
        out.ft_forward(prompt_t)
        run_test("32: content only_out", read_ft_element(out, 0) == "only_out")
        run_test("33: confidence 0.3", abs(out.tensor.data.flatten()[0].item() - 0.3) < 0.02)

    with tempfile.TemporaryDirectory() as tmpdir:
        # recurrent_dim=2, tied scores — first one wins
        async def tied(coords, prompt):
            key = tuple(coords)
            i, slot = key
            if slot == 0: return (f"out_{i}", 0.5)
            if slot == 1: return ("Error", 0.5)
            if slot == 2: return ("err", 0.0)
            return ("???", 0.0)

        inp = FutureTensor([2, 3], tmpdir, tied)
        out = recurrent_forward(inp)
        prompt_t = st_make_tensor("p", tmpdir)
        out.ft_forward(prompt_t)
        # Both have score 0.25; first (i=0) stays since > not >=
        run_test("34: tied -> first wins", read_ft_element(out, 0) == "out_0")
        run_test("35: confidence 0.25", abs(out.tensor.data.flatten()[0].item() - 0.25) < 0.02)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Zero confidence
        async def zero_conf(coords, prompt):
            key = tuple(coords)
            if key[1] == 0: return ("out", 0.0)
            if key[1] == 1: return ("Error", 0.0)
            if key[1] == 2: return ("err", 0.0)
            return ("???", 0.0)

        inp = FutureTensor([2, 3], tmpdir, zero_conf)
        out = recurrent_forward(inp)
        prompt_t = st_make_tensor("p", tmpdir)
        out.ft_forward(prompt_t)
        run_test("36: zero confidence clamped to 0",
                 abs(out.tensor.data.flatten()[0].item()) < 0.01)

    with tempfile.TemporaryDirectory() as tmpdir:
        # High confidence product clamped to 1.0
        async def high_conf(coords, prompt):
            key = tuple(coords)
            if key[1] == 0: return ("out", 1.0)
            if key[1] == 1: return ("Error", 1.0)
            if key[1] == 2: return ("err", 0.0)
            return ("???", 0.0)

        inp = FutureTensor([1, 3], tmpdir, high_conf)
        out = recurrent_forward(inp)
        prompt_t = st_make_tensor("p", tmpdir)
        out.ft_forward(prompt_t)
        run_test("37: product 1.0 clamped", abs(out.tensor.data.flatten()[0].item() - 1.0) < 0.01)

        # Verify shape assertions
        try:
            bad = FutureTensor([5], tmpdir, high_conf)
            recurrent_forward(bad)
            run_test("38: 1D should fail", False)
        except AssertionError:
            run_test("38: 1D assertion", True)

        try:
            bad = FutureTensor([2, 4], tmpdir, high_conf)
            recurrent_forward(bad)
            run_test("39: last dim != 3 should fail", False)
        except AssertionError:
            run_test("39: last dim assertion", True)

        run_test("40: shape [1,3] -> scalar", out.shape == [])

    # === Group 4: Prompt accumulation (tests 41-55) ===

    with tempfile.TemporaryDirectory() as tmpdir:
        prompts_seen = {}

        async def capture_prompts(coords, prompt):
            key = tuple(coords)
            prompts_seen.setdefault(key, []).append(prompt)
            i, slot = key
            if slot == 0: return (f"gen_{i}", 0.5)
            if slot == 1: return ("Error", 0.5)
            if slot == 2: return (f"diag_{i}", 0.0)
            return ("???", 0.0)

        inp = FutureTensor([3, 3], tmpdir, capture_prompts)
        out = recurrent_forward(inp)
        prompt_t = st_make_tensor("INITIAL", tmpdir)
        out.ft_forward(prompt_t)

        # i=0 generator sees original prompt
        run_test("41: i=0 gen sees INITIAL", prompts_seen[(0, 0)][0] == "INITIAL")
        run_test("42: i=0 val sees INITIAL", prompts_seen[(0, 1)][0] == "INITIAL")
        run_test("43: i=0 diag sees INITIAL", prompts_seen[(0, 2)][0] == "INITIAL")

        # i=1 generator sees accumulated prompt
        p1 = prompts_seen[(1, 0)][0]
        run_test("44: i=1 starts with INITIAL", p1.startswith("INITIAL"))
        run_test("45: i=1 has Iteration 0", "====[Iteration 0]====" in p1)
        run_test("46: i=1 has gen_0", "gen_0" in p1)
        run_test("47: i=1 has Error", "Error" in p1)
        run_test("48: i=1 has diag_0", "diag_0" in p1)

        # i=2 generator sees both iterations
        p2 = prompts_seen[(2, 0)][0]
        run_test("49: i=2 has Iteration 0", "====[Iteration 0]====" in p2)
        run_test("50: i=2 has Iteration 1", "====[Iteration 1]====" in p2)
        run_test("51: i=2 has gen_0", "gen_0" in p2)
        run_test("52: i=2 has gen_1", "gen_1" in p2)
        run_test("53: i=2 has diag_0", "diag_0" in p2)
        run_test("54: i=2 has diag_1", "diag_1" in p2)
        run_test("55: i=2 has Status:Error twice", p2.count("Status:Error") == 2)

    # === Group 5: Multi-element prefix (tests 56-75) ===

    with tempfile.TemporaryDirectory() as tmpdir:
        # prefix [3], recurrent_dim=2 -> shape [3, 2, 3]
        async def multi_elem(coords, prompt):
            prefix, i, slot = coords
            if slot == 0:
                if i == 0: return (f"p{prefix}_bad", 0.3)
                else: return (f"p{prefix}_good", 0.9)
            if slot == 1:
                if i == 0: return ("Error", 0.2)
                else: return ("Ok", 0.8)
            if slot == 2:
                return (f"err_p{prefix}_i{i}", 0.0)
            return ("???", 0.0)

        inp = FutureTensor([3, 2, 3], tmpdir, multi_elem)
        out = recurrent_forward(inp)
        run_test("56: prefix [3] shape", out.shape == [3])

        prompt_t = st_make_tensor(["pa", "pb", "pc"], tmpdir)
        out.ft_forward(prompt_t)

        run_test("57: elem 0 content", read_ft_element(out, 0) == "p0_good")
        run_test("58: elem 1 content", read_ft_element(out, 1) == "p1_good")
        run_test("59: elem 2 content", read_ft_element(out, 2) == "p2_good")
        for i in range(3):
            run_test(f"60+{i}: elem {i} conf 1.0",
                     abs(out.tensor.data.flatten()[i].item() - 1.0) < 0.01)

    with tempfile.TemporaryDirectory() as tmpdir:
        # prefix [2, 2], recurrent_dim=1 -> shape [2, 2, 1, 3]
        async def prefix_2d(coords, prompt):
            r, c, i, slot = coords
            if slot == 0: return (f"r{r}c{c}", 0.8)
            if slot == 1: return ("Ok", 0.9)
            return ("???", 0.0)

        inp = FutureTensor([2, 2, 1, 3], tmpdir, prefix_2d)
        out = recurrent_forward(inp)
        run_test("63: prefix [2,2] shape", out.shape == [2, 2])

        prompt_t = st_make_tensor([["p00", "p01"], ["p10", "p11"]], tmpdir)
        out.ft_forward(prompt_t)

        run_test("64: [0,0]", read_ft_element(out, 0) == "r0c0")
        run_test("65: [0,1]", read_ft_element(out, 1) == "r0c1")
        run_test("66: [1,0]", read_ft_element(out, 2) == "r1c0")
        run_test("67: [1,1]", read_ft_element(out, 3) == "r1c1")

    with tempfile.TemporaryDirectory() as tmpdir:
        # prefix [4], recurrent_dim=3, mixed Ok/fail per element
        async def mixed_per_elem(coords, prompt):
            prefix, i, slot = coords
            ok_at = [0, 2, 1, -1]  # elem 3 never Ok
            if slot == 0: return (f"e{prefix}_i{i}", 0.5 + i * 0.1)
            if slot == 1:
                if i == ok_at[prefix]:
                    return ("Ok", 0.9)
                return ("Error", 0.4)
            if slot == 2: return ("err", 0.0)
            return ("???", 0.0)

        inp = FutureTensor([4, 3, 3], tmpdir, mixed_per_elem)
        out = recurrent_forward(inp)
        run_test("68: shape [4]", out.shape == [4])
        prompt_t = st_make_tensor(["p"] * 4, tmpdir)
        out.ft_forward(prompt_t)

        run_test("69: elem 0 Ok at i=0", read_ft_element(out, 0) == "e0_i0")
        run_test("70: elem 0 conf 1.0", abs(out.tensor.data.flatten()[0].item() - 1.0) < 0.01)
        run_test("71: elem 1 Ok at i=2", read_ft_element(out, 1) == "e1_i2")
        run_test("72: elem 2 Ok at i=1", read_ft_element(out, 2) == "e2_i1")
        # elem 3: all fail, best = i=2 (0.7*0.4=0.28 > 0.6*0.4=0.24 > 0.5*0.4=0.20)
        run_test("73: elem 3 all fail best i=2", read_ft_element(out, 3) == "e3_i2")
        run_test("74: elem 3 conf ~0.28",
                 abs(out.tensor.data.flatten()[3].item() - 0.28) < 0.02)
        run_test("75: elem 1 conf 1.0", abs(out.tensor.data.flatten()[1].item() - 1.0) < 0.01)

    # === Group 6: recurrent_dim=1 edge case (tests 76-82) ===

    with tempfile.TemporaryDirectory() as tmpdir:
        async def single_ok(coords, prompt):
            if coords[-1] == 0: return ("result", 0.9)
            if coords[-1] == 1: return ("Ok", 0.85)
            return ("err", 0.0)

        inp = FutureTensor([1, 3], tmpdir, single_ok)
        out = recurrent_forward(inp)
        run_test("76: recurrent_dim=1 Ok shape", out.shape == [])
        prompt_t = st_make_tensor("p", tmpdir)
        out.ft_forward(prompt_t)
        run_test("77: content", read_ft_element(out, 0) == "result")
        run_test("78: conf 1.0", abs(out.tensor.data.flatten()[0].item() - 1.0) < 0.01)

    with tempfile.TemporaryDirectory() as tmpdir:
        async def single_err(coords, prompt):
            if coords[-1] == 0: return ("result", 0.7)
            if coords[-1] == 1: return ("Error", 0.6)
            return ("detail", 0.0)

        inp = FutureTensor([1, 3], tmpdir, single_err)
        out = recurrent_forward(inp)
        prompt_t = st_make_tensor("p", tmpdir)
        out.ft_forward(prompt_t)
        run_test("79: single fail content", read_ft_element(out, 0) == "result")
        run_test("80: single fail conf 0.42",
                 abs(out.tensor.data.flatten()[0].item() - 0.42) < 0.02)

    with tempfile.TemporaryDirectory() as tmpdir:
        # recurrent_dim=5
        async def dim5(coords, prompt):
            i, slot = coords
            if slot == 0: return (f"out_{i}", 0.1 * (i + 1))
            if slot == 1:
                return ("Ok" if i == 4 else "Error", 0.1 * (i + 1))
            if slot == 2: return ("err", 0.0)
            return ("???", 0.0)

        inp = FutureTensor([5, 3], tmpdir, dim5)
        out = recurrent_forward(inp)
        run_test("81: recurrent_dim=5 shape", out.shape == [])
        prompt_t = st_make_tensor("p", tmpdir)
        out.ft_forward(prompt_t)
        run_test("82: Ok at i=4", read_ft_element(out, 0) == "out_4")

    # === Group 7: Prompt-dependent responses (tests 83-90) ===

    with tempfile.TemporaryDirectory() as tmpdir:
        # Validator response depends on prompt content
        async def prompt_dependent(coords, prompt):
            i, slot = coords
            if slot == 0: return (f"gen_{i}", 0.8)
            if slot == 1:
                # Only Ok if prompt contains the magic word
                if "FIXED" in prompt:
                    return ("Ok", 0.9)
                return ("Error", 0.5)
            if slot == 2:
                return ("hint: add FIXED to prompt", 0.0)
            return ("???", 0.0)

        inp = FutureTensor([2, 3], tmpdir, prompt_dependent)
        out = recurrent_forward(inp)

        # Without FIXED in initial prompt, i=0 fails
        # i=0 error details say "hint: add FIXED to prompt"
        # But prompt accumulation just includes that as error detail,
        # i=1 prompt does contain "FIXED" via error details
        prompt_t = st_make_tensor("initial", tmpdir)
        out.ft_forward(prompt_t)
        run_test("83: Ok on i=1 (prompt contains FIXED from error)",
                 read_ft_element(out, 0) == "gen_1")
        run_test("84: conf 1.0", abs(out.tensor.data.flatten()[0].item() - 1.0) < 0.01)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Already has FIXED in initial prompt
        async def prompt_dependent2(coords, prompt):
            i, slot = coords
            if slot == 0: return (f"gen_{i}", 0.8)
            if slot == 1:
                if "FIXED" in prompt: return ("Ok", 0.9)
                return ("Error", 0.5)
            if slot == 2: return ("err", 0.0)
            return ("???", 0.0)

        inp = FutureTensor([2, 3], tmpdir, prompt_dependent2)
        out = recurrent_forward(inp)
        prompt_t = st_make_tensor("FIXED already", tmpdir)
        out.ft_forward(prompt_t)
        run_test("85: Ok on i=0 with FIXED in initial", read_ft_element(out, 0) == "gen_0")
        run_test("86: conf 1.0", abs(out.tensor.data.flatten()[0].item() - 1.0) < 0.01)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Generator output improves with more context
        async def improving(coords, prompt):
            i, slot = coords
            if slot == 0:
                quality = min(0.9, 0.3 + 0.3 * prompt.count("Iteration"))
                return (f"v{i}_q{quality:.1f}", quality)
            if slot == 1: return ("Error", 0.5)
            if slot == 2: return ("try again", 0.0)
            return ("???", 0.0)

        inp = FutureTensor([3, 3], tmpdir, improving)
        out = recurrent_forward(inp)
        prompt_t = st_make_tensor("start", tmpdir)
        out.ft_forward(prompt_t)
        # i=0: 0.3, i=1: 0.6, i=2: 0.9 (best)
        run_test("87: best output from i=2", read_ft_element(out, 0) == "v2_q0.9")
        run_test("88: best score 0.9*0.5=0.45",
                 abs(out.tensor.data.flatten()[0].item() - 0.45) < 0.02)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Each element gets its own prompt
        async def per_prompt(coords, prompt):
            prefix, i, slot = coords
            if slot == 0: return (f"echo:{prompt[:3]}", 0.8)
            if slot == 1: return ("Ok", 0.9)
            return ("???", 0.0)

        inp = FutureTensor([3, 1, 3], tmpdir, per_prompt)
        out = recurrent_forward(inp)
        prompt_t = st_make_tensor(["AAA", "BBB", "CCC"], tmpdir)
        out.ft_forward(prompt_t)
        run_test("89: elem 0 uses its prompt", read_ft_element(out, 0) == "echo:AAA")
        run_test("90: elem 2 uses its prompt", read_ft_element(out, 2) == "echo:CCC")

    # === Group 8: Composition with slice/unsqueeze (tests 91-100) ===

    with tempfile.TemporaryDirectory() as tmpdir:
        from experience.future_tensor.function.slice_forward import slice_forward

        # prefix [4], recurrent_dim=2 -> shape [4, 2, 3]
        async def comp_get(coords, prompt):
            prefix, i, slot = coords
            if slot == 0: return (f"p{prefix}_i{i}", 0.8)
            if slot == 1: return ("Ok", 0.9)
            return ("???", 0.0)

        inp = FutureTensor([4, 2, 3], tmpdir, comp_get)
        out = recurrent_forward(inp)
        run_test("91: shape [4]", out.shape == [4])

        # Slice the output
        sliced = slice_forward(out, [slice(1, 3)])
        run_test("92: sliced shape [2]", sliced.shape == [2])

        prompt_t = st_make_tensor(["p0", "p1"], tmpdir)
        sliced.ft_forward(prompt_t)
        # sliced[0] -> out[1], sliced[1] -> out[2]
        run_test("93: sliced[0] = p1_i0", read_ft_element(sliced, 0) == "p1_i0")
        run_test("94: sliced[1] = p2_i0", read_ft_element(sliced, 1) == "p2_i0")

    with tempfile.TemporaryDirectory() as tmpdir:
        from experience.future_tensor.function.unsqueeze_forward import unsqueeze_forward

        async def unsq_get(coords, prompt):
            i, slot = coords
            if slot == 0: return (f"item_{i}", 0.85)
            if slot == 1: return ("Ok", 0.9)
            return ("???", 0.0)

        inp = FutureTensor([2, 3], tmpdir, unsq_get)
        out = recurrent_forward(inp)
        run_test("95: before unsqueeze shape []", out.shape == [])

        unsq = unsqueeze_forward(out, 0)
        run_test("96: unsqueezed shape [1]", unsq.shape == [1])

        prompt_t = st_make_tensor(["p"], tmpdir)
        unsq.ft_forward(prompt_t)
        run_test("97: unsqueezed content", read_ft_element(unsq, 0) == "item_0")

    with tempfile.TemporaryDirectory() as tmpdir:
        # 2D prefix + recurrent, slice prefix then forward
        from experience.future_tensor.function.slice_forward import slice_forward

        async def big_get(coords, prompt):
            r, c, i, slot = coords
            if slot == 0: return (f"r{r}c{c}_i{i}", 0.8)
            if slot == 1:
                return ("Ok" if i == 1 else "Error", 0.7)
            if slot == 2: return ("err", 0.0)
            return ("???", 0.0)

        inp = FutureTensor([3, 3, 2, 3], tmpdir, big_get)
        out = recurrent_forward(inp)
        run_test("98: 2D prefix shape [3,3]", out.shape == [3, 3])

        sliced = slice_forward(out, [1, slice(None)])
        run_test("99: sliced row shape [3]", sliced.shape == [3])

        prompt_t = st_make_tensor(["p0", "p1", "p2"], tmpdir)
        sliced.ft_forward(prompt_t)
        run_test("100: sliced[0] = r1c0_i1", read_ft_element(sliced, 0) == "r1c0_i1")

    print(f"\n  Passed: {passed}, Failed: {failed}, Total: {passed + failed}")
    print("All recurrent_forward tests completed.")
