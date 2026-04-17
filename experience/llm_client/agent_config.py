"""Agent configuration abstraction layer.

This module provides a unified configuration interface for all LLM methods:
- raw_llm_api: Direct OpenAI-compatible API calls
- coding_agent: Claude-based coding agent in programmatic mode
- tmux_cc: Claude-based coding agent in interactive/batch mode

All configurations hide implementation details from the query layer.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass
class AgentConfig(ABC):
    """Abstract configuration for all LLM methods.
    
    This interface hides implementation details from the query layer.
    Different LLM methods can provide their own config.
    """
    
    @abstractmethod
    def get_method_name(self) -> str:
        """Get the LLM method name (raw_llm_api, coding_agent, tmux_cc)."""
        pass


@dataclass
class RawLlmConfig(AgentConfig):
    """Configuration for raw_llm_api method (OpenAI-compatible API).
    
    This encapsulates all OpenAI API-specific details.
    """
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    model: Optional[str] = None
    username: Optional[str] = None
    custom_headers: Optional[Dict[str, str]] = None
    
    def get_method_name(self) -> str:
        return "raw_llm_api"
    
    def to_env(self) -> Dict[str, str]:
        """Convert to environment variables for raw_llm_query."""
        env = {}
        if self.base_url:
            env["LLM_BASE_URL"] = self.base_url
        if self.api_key:
            env["LLM_API_KEY"] = self.api_key
        if self.model:
            env["LLM_MODEL"] = self.model
        if self.username:
            env["LLM_USERNAME"] = self.username
        return env


@dataclass
class ClaudeAgentConfig(AgentConfig):
    """Configuration for coding_agent method (agent in programmatic mode).
    
    This encapsulates all agent-specific details for coding_agent.
    """
    cli_path: Optional[str] = None
    settings_path: Optional[str] = None
    setting_sources: Optional[str] = None
    model: Optional[str] = None
    cwd: Optional[str] = None
    allowed_tools: Optional[list[str]] = None
    permission_mode: str = "acceptEdits"
    
    def get_method_name(self) -> str:
        return "coding_agent"
    
    def to_options(self) -> Dict[str, Any]:
        """Convert to ClaudeAgentOptions dict."""
        options = {
            "permission_mode": self.permission_mode,
        }
        
        if self.cli_path is not None:
            options["cli_path"] = self.cli_path
        if self.settings_path is not None:
            options["settings"] = self.settings_path
        if self.setting_sources is not None:
            options["setting_sources"] = self.setting_sources
        if self.model is not None:
            options["model"] = self.model
        if self.cwd is not None:
            options["cwd"] = self.cwd
        if self.allowed_tools is not None:
            options["allowed_tools"] = self.allowed_tools
            
        return options


@dataclass
class TmuxCcConfig(AgentConfig):
    """Configuration for tmux_cc method (agent in tmux interactive/batch mode).
    
    This encapsulates all agent-specific details for tmux_cc handler.
    """
    cli_path: Optional[str] = None
    settings_path: Optional[str] = None
    model: Optional[str] = None
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    workspace_root: Optional[str] = None
    
    def get_method_name(self) -> str:
        return "tmux_cc"
    
    def get_cli_path(self) -> Optional[str]:
        """Get agent CLI path."""
        return self.cli_path
    
    def get_settings_path(self) -> Optional[str]:
        """Get agent settings.json path."""
        return self.settings_path
