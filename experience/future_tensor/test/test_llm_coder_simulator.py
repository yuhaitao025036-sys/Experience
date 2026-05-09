"""
test_llm_coder_simulator: LLM-powered coder simulator using ft_expert + ft_recurrent.

Architecture:
  ft_recurrent(
    ft_coder_validator(workspace_ft, experience)
  )

  ft_coder_validator: harness future op that per iteration:
    1. ft_expert decides action (text:cmd or ctrl:key)
    2. ft_tmux_send_text / ft_tmux_send_ctrl executes it
    3. ft_tmux_capture_pane observes the terminal
    4. check_terminator_last_line decides termination

All inputs to tmux ops are LLM-generated (via ft_expert).
Only Chinese prompts allowed for ft_forward (anti-hack).
No CLI commands exposed in prompts — LLM decides everything.
"""

import os
import sys
import subprocess
import tempfile

import sympy
import libtmux

from experience.future_tensor.future_tensor import FutureTensor
from experience.future_tensor.status import Status
from experience.future_tensor.function.tmux_session import tmux_session_prefix
from experience.future_tensor.function.ft_make_forwarded import ft_make_forwarded
from experience.future_tensor.function.ft_tmux_create_session import ft_tmux_create_session
from experience.future_tensor.function.ft_tmux_send_text import ft_tmux_send_text
from experience.future_tensor.function.ft_tmux_send_ctrl import ft_tmux_send_ctrl
from experience.future_tensor.function.ft_tmux_capture_pane import ft_tmux_capture_pane
from experience.future_tensor.function.ft_sleep import ft_sleep
from experience.future_tensor.function.ft_sequential import ft_sequential
from experience.future_tensor.function.ft_recurrent import ft_recurrent
from experience.future_tensor.function.ft_expert import ft_expert
from experience.symbolic_tensor.tensor_util.make_tensor import make_tensor as st_make_tensor


# ─── Source LLM env ───

result = subprocess.run(
    ["bash", "-c", "source ~/.anthropic.sh && env"],
    capture_output=True, text=True,
)
for line in result.stdout.splitlines():
    if "=" in line:
        key, _, val = line.partition("=")
        os.environ[key] = val
os.environ.pop("CLAUDECODE", None)


# ─── Termination check ───

def check_terminator_last_line(captured_text: str) -> bool:
    """Check if terminal shows command completion: fresh prompt on last line."""
    lines = [l for l in captured_text.split("\n") if l.strip()]
    if len(lines) < 3:
        return False
    last_line = lines[-1]
    return "λ" in last_line or "$ " in last_line or last_line.rstrip().endswith("$")


# ─── Experience ───

def make_experience(tmpdir: str):
    """Experience for human-level terminal decisions — format examples only."""
    return st_make_tensor([
        ["格式\n示例\ntext\n输入文本\n命令",
         "输入文本格式示例",
         "text:ls"],
        ["格式\n示例\nctrl\n控制键\nenter",
         "发送控制键格式示例",
         "ctrl:Enter"],
    ], tmpdir)


# ─── Parse ───

def _parse_decision(raw: str) -> str:
    """Extract first text:/ctrl: line from LLM output."""
    if not raw:
        return ""
    for line in raw.strip().split("\n"):
        line = line.strip()
        if line.startswith("text:") or line.startswith("ctrl:"):
            return line
    for line in raw.strip().split("\n"):
        line = line.strip()
        if line:
            return line
    return ""


# ─── ft_coder_validator: harness future op ───

def ft_coder_validator(
    workspace_ft: FutureTensor,
    experience,
    tmpdir: str,
    max_iters: int = 30,
) -> FutureTensor:
    """Harness future op: LLM-powered terminal validator.

    Each iteration (driven by ft_recurrent):
      1. Reads task from prompt (passed through ft_forward → ft_recurrent)
      2. Calls ft_expert to decide action (text:content or ctrl:key)
      3. Sends to tmux via ft_tmux_send_text or ft_tmux_send_ctrl
      4. Captures pane, appends to trajectory
      5. Returns confidence when terminal shows fresh prompt (task done)

    Args:
        workspace_ft: Pre-forwarded FutureTensor with tmux instance ID.
        experience: Experience tensor for ft_expert.
        tmpdir: Working directory.
        max_iters: Maximum iterations (recurrent dim).

    Returns:
        A FutureTensor[max_iters] suitable as input to ft_recurrent.
    """
    relative_to = workspace_ft.ft_static_tensor.st_relative_to

    # Shared state across iterations
    step = [0]
    pane_trajectory = []

    # Reusable capture op
    capture_op = ft_tmux_capture_pane(workspace_ft)

    def _format_trajectory():
        if not pane_trajectory:
            return "（终端刚启动，尚无观察记录）"
        recent = pane_trajectory[-5:]
        parts = []
        start_idx = len(pane_trajectory) - len(recent)
        for j, pane in enumerate(recent):
            parts.append(f"--- 第{start_idx + j}步后的终端 ---\n{pane}")
        return "\n".join(parts)

    async def _validator_async_get(coords, prompt):
        import asyncio

        i = step[0]
        traj = _format_trajectory()

        # 1. Decision context — prompt carries the task from ft_forward
        decision_context = (
            f"任务目标：{prompt}\n"
            f"操作轨迹（终端观察序列）：\n{traj}\n"
            f"这是第{i}步。你是一个人类用户在操作终端。\n"
            f"规则：\n"
            f"1. 观察终端最后一行（提示符后面的内容）——那是当前已输入但未执行的命令\n"
            f"2. 如果当前命令行已经有内容，说明你之前已经输入过了，不要重复输入\n"
            f"3. 如果命令行的内容就是你想执行的命令，直接按回车：ctrl:Enter\n"
            f"4. 如果命令行为空，输入你想执行的命令：text:你的命令\n"
            f"回答格式（只输出一行）：text:内容 或 ctrl:键名\n"
        )

        # 2. ft_expert: materialized input → LLM decision
        input_ft = ft_make_forwarded(tmpdir, [1], [decision_context])
        output_ft, _, _ = ft_expert(
            input_ft, experience,
            task_prompt="你是一个人类终端用户。根据任务目标和终端观察轨迹，决定下一步操作。只输出一行：text:内容 或 ctrl:键名。不要复制示例中的命令，根据任务目标生成正确的操作。",
            topk=2,
        )
        raw_result, _ = await output_ft.ft_async_get([0], "决策")
        decision_result = _parse_decision(raw_result)

        # 3. Parse
        action_type = "text"
        payload = ""
        if decision_result:
            if ":" in decision_result:
                parts = decision_result.split(":", 1)
                kind = parts[0].strip().lower()
                payload = parts[1].strip()
                if kind == "ctrl":
                    action_type = "ctrl"
                else:
                    action_type = "text"
            else:
                if decision_result.lower() in ("enter", "tab", "escape", "backspace"):
                    action_type = "ctrl"
                    payload = decision_result.capitalize()
                else:
                    payload = decision_result

        if not payload:
            payload = "Enter" if action_type == "ctrl" else " "

        # 4. Execute
        payload_forwarded = ft_make_forwarded(tmpdir, [1], [payload])
        if action_type == "text":
            op = ft_tmux_send_text(payload_forwarded, workspace_ft)
        else:
            op = ft_tmux_send_ctrl(payload_forwarded, workspace_ft)

        await op.ft_async_get([0], "执行")
        await asyncio.sleep(0.3)

        step[0] += 1

        # 5. Capture pane → trajectory
        captured_text, _ = await capture_op.ft_async_get([0], "观察")
        pane_trajectory.append(captured_text)

        print(f"  [step {i}] {action_type}:{repr(payload)}")

        # 6. Termination check
        if i >= 1 and check_terminator_last_line(captured_text):
            return (captured_text, Status.confidence(1.0))
        else:
            return ("", Status.self_confidence_but_failed(0.9))

    validator = FutureTensor(relative_to, _validator_async_get, [sympy.Integer(max_iters)])
    validator.ft_capacity_shape = [max_iters]
    return validator


# ─── Helpers ───

def read_ft_element(ft, flat_index):
    digits = list(str(flat_index))
    path = os.path.join(
        ft.ft_static_tensor.st_relative_to, ft.ft_static_tensor.st_tensor_uid,
        "storage", os.path.join(*digits), "data",
    )
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        return f.read()


# ─── Test ───

print("Running test_llm_coder_simulator...\n")

server = libtmux.Server()
INSTANCE_ID = "llm_coder_test"
SESSION_NAME = f"{tmux_session_prefix}{INSTANCE_ID}"

with tempfile.TemporaryDirectory() as tmpdir:
    workspace_ft = ft_make_forwarded(tmpdir, [1], [INSTANCE_ID])

    # Setup: create session (or clear if exists) + delay
    setup = ft_sequential(ft_tmux_create_session(workspace_ft), ft_sleep(workspace_ft, 0.5))
    setup.ft_forward(st_make_tensor(["启动终端会话"], tmpdir))

    # Build pipeline: ft_recurrent(ft_coder_validator(...))
    experience = make_experience(tmpdir)
    validator = ft_coder_validator(workspace_ft, experience, tmpdir)
    output, _ = ft_recurrent(validator)

    # ONE ft_forward — the prompt IS the task
    output.ft_forward(st_make_tensor("在终端中输出问候语hello world", tmpdir))

    # Verify
    content = read_ft_element(output, 0)
    ok = content is not None and "hello" in content.lower()
    print(f"\n  {'✓' if ok else '✗'} pane contains 'hello'")
    print(f"\n  Pane:\n{content}\n")

if not ok:
    sys.exit(1)
print("test_llm_coder_simulator passed.")
