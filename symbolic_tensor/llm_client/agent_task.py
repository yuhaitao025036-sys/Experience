from dataclasses import dataclass


@dataclass
class AgentTask:
    workspace_dir: str
    output_relative_dir: str
    prompt: str
