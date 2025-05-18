"""Agent imports for AMAS."""

from .base import BaseAgent, register_agent, get_agent_registry
from typing import Any

# Import specific agent implementations to register them
from .planner import PlannerAgent
from .coder import CoderAgent
# Removed import of DeciderAgent as the agent is being removed

def get_registered_agents() -> dict[str, Any]:
    """Returns the agent registry dictionary."""
    return get_agent_registry()