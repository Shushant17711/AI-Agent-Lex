import logging
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

class AgentRegistry:
    _agents: Dict[str, Dict[str, Any]] # Class-level annotation
    """
    Manages the registration and discovery of agents and their capabilities.
    """
    _instance = None

    def __new__(cls) -> 'AgentRegistry':
        if cls._instance is None:
            cls._instance = super(AgentRegistry, cls).__new__(cls)
            # Initialize the dictionary here, annotation is now at class level
            cls._instance._agents = {}
            logger.info("AgentRegistry initialized.")
        return cls._instance    
    
    def register_agent(self, agent_id: str, capabilities: List[str], agent_instance: Any):
        """
        Registers an agent with the registry.

        Args:
            agent_id: A unique identifier for the agent.
            capabilities: A list of strings describing the agent's capabilities.
            agent_instance: The actual instance of the agent.

        Example:
            >>> registry = AgentRegistry()
            >>> class MyAgent: pass
            >>> my_agent = MyAgent()
            >>> registry.register_agent("planner_01", ["planning"], my_agent)

        Raises:
            ValueError: If agent_id, capabilities, or agent_instance are invalid.
        """
        if not agent_id or not isinstance(agent_id, str):
            raise ValueError("Agent ID must be a non-empty string")
        if not capabilities or not isinstance(capabilities, list):
            raise ValueError("Capabilities must be a non-empty list")
        if agent_instance is None:
            raise ValueError("Agent instance cannot be None")

        if agent_id in self._agents:
            logger.warning(f"Agent with ID '{agent_id}' is already registered. Overwriting.")
        self._agents[agent_id] = {
            "capabilities": capabilities,
            "instance": agent_instance
        }
        logger.info(f"Agent '{agent_id}' registered with capabilities: {capabilities}")

    def agent_exists(self, agent_id: str) -> bool:
        """Check if an agent with the given ID exists in the registry."""
        return agent_id in self._agents


    def unregister_agent(self, agent_id: str):
        """
        Removes an agent from the registry.

        Args:
            agent_id: The ID of the agent to remove.
        """
        if agent_id in self._agents:
            del self._agents[agent_id]
            logger.info(f"Agent '{agent_id}' unregistered.")
        else:
            logger.warning(f"Attempted to unregister non-existent agent '{agent_id}'.")

    def get_agent_instance(self, agent_id: str) -> Optional[Any]:
        """
        Retrieves the instance of a registered agent.

        Args:
            agent_id: The ID of the agent.

        Returns:
            The agent instance, or None if not found.
        """
        agent_info = self._agents.get(agent_id)
        return agent_info.get("instance") if agent_info else None

    def list_agents(self) -> Dict[str, List[str]]:
        """
        Lists all registered agents and their capabilities.

        Returns:
            A dictionary where keys are agent IDs and values are lists of capabilities.
        """
        return {agent_id: info["capabilities"] for agent_id, info in self._agents.items()}

    def find_agents_by_capability(self, required_capability: str) -> List[str]:
        """
        Finds agents that possess a specific capability.

        Args:
            required_capability: The capability to search for.

        Returns:
            A list of agent IDs that have the required capability.
        """
        matching_agents = [
            agent_id for agent_id, info in self._agents.items()
            if required_capability in info["capabilities"]
        ]
        logger.debug(f"Agents found for capability '{required_capability}': {matching_agents}")
        return matching_agents

    def get_all_capabilities(self) -> List[str]:
        """
        Gets a unique list of all capabilities declared by registered agents.

        Returns:
            A list of unique capability strings.
        """
        all_caps = set()
        for info in self._agents.values():
            all_caps.update(info["capabilities"])
        return sorted(list(all_caps))

# Singleton instance
agent_registry = AgentRegistry()

if __name__ == '__main__':
    # Example Usage (for testing)
    logging.basicConfig(level=logging.INFO)

    class MockAgent:
        def __init__(self, name):
            self.name = name
        def __repr__(self):
            return f"MockAgent({self.name})"

    agent1 = MockAgent("PlannerAgent")
    agent2 = MockAgent("CoderAgent")
    agent3 = MockAgent("FileSystemAgent")

    registry = AgentRegistry() # Get the singleton instance

    registry.register_agent("planner_01", ["planning", "task_decomposition"], agent1)
    registry.register_agent("coder_01", ["coding", "python", "debugging"], agent2)
    registry.register_agent("coder_02", ["coding", "javascript"], MockAgent("JsCoder"))
    registry.register_agent("fs_agent", ["file_system_access", "read", "write"], agent3)

    print("\nRegistered Agents:")
    print(registry.list_agents())

    print("\nFinding agents with 'coding' capability:")
    coding_agents = registry.find_agents_by_capability("coding")
    print(coding_agents)
    for agent_id in coding_agents:
        print(f" - Instance: {registry.get_agent_instance(agent_id)}")


    print("\nFinding agents with 'planning' capability:")
    planning_agents = registry.find_agents_by_capability("planning")
    print(planning_agents)

    print("\nFinding agents with 'read' capability:")
    read_agents = registry.find_agents_by_capability("read")
    print(read_agents)

    print("\nAll declared capabilities:")
    print(registry.get_all_capabilities())

    registry.unregister_agent("coder_02")
    print("\nRegistered Agents after unregistering coder_02:")
    print(registry.list_agents())