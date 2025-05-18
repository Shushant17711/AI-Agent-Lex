# Project: Autonomous Multi-Agent System (AMAS)

**Version:** 0.2.0
**Date:** 2025-04-07

## 1. Introduction & Goal

This project aims to develop an Autonomous Multi-Agent System (AMAS) capable of tackling complex tasks, particularly focused on software development workflows (e.g., planning, coding, debugging, testing). The system will leverage Large Language Models (LLMs) to power specialized agents that collaborate to achieve user-defined objectives, inspired by systems like Cursor AI and Claude Coder.

The initial goal is to create a proof-of-concept system that can take a high-level requirement (e.g., "Create a simple Flask API endpoint") and generate the corresponding code along with basic tests, demonstrating core agent collaboration and tool usage principles.

## 2. Core Concepts

* **Multi-Agent System (MAS):** A system composed of multiple autonomous agents interacting within an environment to solve problems beyond the capabilities of any single agent.
* **Specialized Agents:** Each agent possesses specific skills (planning, coding, debugging) powered by tailored LLM prompts, potentially different underlying AI models, and specific tools.
* **Orchestration:** A central component (Orchestrator Agent) responsible for managing the overall workflow, decomposing tasks, assigning them to appropriate agents, managing permissions, interpreting agent requests (including tool use), and synthesizing results.
* **Shared Context:** A mechanism (e.g., database, shared file system, state manager) allowing agents to access relevant information like the overall goal, current code state, project files, and outputs from other agents.
* **Tool Usage & Permissions:** Agents can request to use predefined tools (e.g., file I/O, terminal execution) to interact with the environment. Tool usage is mediated by the Orchestrator and requires explicit user permission for sensitive actions.
* **Iterative Refinement:** The system operates iteratively, potentially involving cycles of planning, coding, testing, and debugging until the requirements are met or a satisfactory state is achieved.

## 3. System Architecture (Initial Proposal)

* **Model:** Centralized Orchestrator Model.
* **Components:**
    * **User Interface (CLI/API):** Entry point for user requests and permission approvals.
    * **Orchestrator Agent:** The central coordinator. Manages state, delegates tasks, interprets agent outputs (including structured tool requests), manages permissions, interacts with other agents.
    * **Specialized Agents:** (Planner, Coder, Debugger, Tester, etc.) - Each potentially running on different optimized AI models.
    * **Communication Bus:** Simple API-based communication or direct function calls initially. (Future: Message Queues).
    * **Shared Context Store:** File system or basic database (SQLite, JSON files) for project state. (Future: Persistent DB).
    * **LLM Service Interface:** Module abstracting interactions with various LLM APIs.
    * **Tool Execution Engine:** A secure component, managed by the Orchestrator, that executes tool commands requested by agents after permission approval.

```
+-----------------+      +---------------------+      +-------------------+
| User Interface  |----->| Orchestrator Agent  |<---->| LLM Service I/F   |
| (CLI / API +    |<-----|(Handles Permissions, |      |(Supports multiple |
| Permission UI)  |      | Tool Interpretation)|      | LLM APIs)         |
+-----------------+      +----------+----------+      +-------------------+
                                    |
                      +-------------+-------------+
                      |   Communication Bus       |
                      +--+----------+----------+--+
                         |          |          |
           +-------------+--+     +-+----------+---+     +-------------------+
           | Planner Agent  |     | Coder Agent    | ... | Debugger / Tester |
           | (Model A)      |     | (Model B)      |     | (Model C / Tools) |
           +----------------+     +----------------+     +-------------------+
                         |          |          |
                         +----------+----------+----------+----------+
                                    |                     |          |
                                    v                     | (Tool    |
                             +-----------------+          | Request) v
                             | Shared Context  |          |      +---------------------+
                             | (Files / DB)    |          +----->| Tool Execution Eng. |
                             +-----------------+                 +---------------------+

```

## 4. Agent Design Principles

* **Detailed System Prompts:** Each agent's behavior is heavily defined by its system prompt. These prompts must be meticulously crafted to include:
    * **Persona & Role:** Define the agent's identity (e.g., "You are a senior Python developer specializing in API design").
    * **Capabilities & Limitations:** Clearly state what the agent can and cannot do.
    * **Instructions & Constraints:** Provide specific guidelines on how to perform its task (e.g., "Write code compliant with PEP 8").
    * **Output Format:** Specify the expected output structure (e.g., JSON, Markdown, code blocks). Crucially, this includes the format for requesting tool usage.
    * **Tool Usage Protocol:** Define *how* an agent signals its intent to use a tool. This typically involves outputting a specific, structured format (e.g., a JSON object like `{"tool": "file_write", "params": {"filename": "app.py", "content": "..."}}`) that the Orchestrator can parse. The prompt should guide the agent on *when* it's appropriate to request tool use.
* **Modularity:** Agents should be designed as independent modules with clear interfaces (inputs/outputs).
* **Statelessness (where possible):** Aim for agents to be largely stateless, relying on the Orchestrator and Shared Context for state management, simplifying scaling and resilience.

## 5. Agent Roles & Responsibilities (Initial Set)

* **Orchestrator:**
    * Receives/parses user requests.
    * Manages overall task state and shared context updates.
    * Delegates planning/tasks to specialized agents.
    * **Parses agent outputs, specifically identifying structured requests for tool usage.**
    * **Manages the permission workflow for tool usage, prompting the user for approval.**
    * **Instructs the Tool Execution Engine upon approval.**
    * Integrates results and handles error recovery.
* **Planner:** (Input: Goal; Output: Plan; LLM Focus: Decomposition)
* **Coder:** (Input: Task, Context; Output: Code / Tool Request; LLM Focus: Code Gen)
* **Debugger:** (Input: Code, Error; Output: Analysis / Fix / Tool Request; LLM Focus: Analysis, Fixing)
* **Tester:** (Input: Code, Req; Output: Tests, Results / Tool Request; LLM Focus: Test Gen, Assertion)
    * *Note: Agents like Coder, Debugger, Tester might request tools like `file_write`, `read_file`, `run_terminal`.*

## 6. Workflow Example (with Tool Use)

1.  User: "Refactor `utils.py` to improve readability and add type hints."
2.  Orchestrator -> Planner -> Plan: `["1. Read content of `utils.py`.", "2. Analyze and refactor code.", "3. Write updated content back to `utils.py`."]`
3.  Orchestrator assigns Task 1 to an appropriate agent (e.g., Coder or a dedicated File Agent).
4.  Agent outputs: `{"tool": "read_file", "params": {"filename": "utils.py"}}`
5.  Orchestrator sees tool request. Since `read_file` might be pre-approved or deemed low-risk, it executes it via the Tool Execution Engine.
6.  Tool Engine returns file content. Orchestrator adds content to context, passes Task 2 + content to Coder/Refactorer.
7.  Coder/Refactorer analyzes, generates refactored code, and outputs it.
8.  Orchestrator passes Task 3 + refactored code to Coder/File Agent.
9.  Agent outputs: `{"tool": "write_file", "params": {"filename": "utils.py", "content": "<refactored_code>"}}`
10. Orchestrator sees `write_file` request. **Prompts User:** "Agent requests permission to overwrite `utils.py`. Allow?"
11. User approves.
12. Orchestrator instructs Tool Execution Engine to write the file.
13. Orchestrator confirms completion to the user.

## 7. Technology Stack (Proposed)

* **Language:** Python 3.10+
* **LLM APIs:** Configurable interface for multiple providers (Google Gemini, OpenAI, Anthropic, local models via Ollama/LM Studio). **Allow specifying different models per agent role** (e.g., Gemini Pro for Planner, GPT-4 for Coder, a fine-tuned model for Debugger).
* **Agent Framework:** Evaluate LangChain, AutoGen, CrewAI for managing agents, prompts, and tool integration.
* **API (if needed):** FastAPI for potential web UI or external interaction.
* **Context Store:** File system / SQLite initially. (Future: PostgreSQL/MongoDB).
* **Tool Execution:** Secure subprocess management, potentially sandboxing (e.g., Docker containers) for terminal commands.
* **Dependencies:** `requests`, `python-dotenv`, LLM SDKs, framework libraries.

## 8. Potential Challenges

* **Complex Prompt Engineering:** Crafting prompts that reliably guide agents, especially regarding tool use and output formatting.
* **Tool Security:** Ensuring tools (especially terminal access) are properly sandboxed and permissions are strictly enforced.
* **State Synchronization:** Keeping the shared context consistent across asynchronous agent operations.
* **Error Handling:** Robustly handling failures in LLM calls, tool execution, or agent logic.
* **Cost Optimization:** Managing costs associated with potentially using multiple (sometimes expensive) LLMs.
* **Multi-Model Management:** Handling different API interfaces and capabilities if using diverse LLMs.

## 9. Future Enhancements

* Add more specialized agents (Refactorer, Researcher, Security Analyst, Documentation Writer).
* Implement robust message queues (RabbitMQ, Kafka) for asynchronous agent communication.
* Develop a persistent, scalable database (PostgreSQL, MongoDB) for context, history, and project states.
* Integrate tightly with version control (Git) for branching, committing, and tracking changes made by agents.
* Expand the secure Tool Execution Engine:
    * Refine the permission model (e.g., granular permissions, session-based approvals).
    * Add more tools (web browsing, API interaction, database queries).
    * Improve sandboxing for terminal commands.
* Develop a graphical user interface (Web UI) for interaction, visualization of plans, and managing permissions.
* Explore agent fine-tuning on specific tasks/domains.
* Implement mechanisms for inter-agent dialogue and negotiation.

## 10. Initial Directory Structure

```
amas/
├── agents/                 # Core logic for each agent type
│   ├── __init__.py
│   ├── base_agent.py       # Abstract base class for agents
│   ├── planner_agent.py
│   ├── coder_agent.py
│   ├── debugger_agent.py
│   ├── tester_agent.py
│   └── orchestrator.py     # The main orchestrator logic
├── core/                   # Core components
│   ├── __init__.py
│   ├── llm_service_gemini/        # LLM interface (multi-provider/model support)
│   │   └── __init__.py
│   ├── context_manager.py  # Shared state/context handling
│   ├── communication.py    # Agent communication logic (if needed)
│   ├── permissions.py      # Permission management logic
│   └── tools/              # Tool definitions and execution engine
│       ├── __init__.py
│       ├── base_tool.py
│       ├── file_system_tool.py
│       └── terminal_tool.py
│       └── execution_engine.py
├── prompts/                # Prompt templates (structured per agent)
│   ├── __init__.py
│   ├── planner.yaml        # Example using YAML/JSON for prompts
│   └── coder.yaml
│   └── ...
├── utils/                  # Utility functions
│   └── __init__.py
├── main.py                 # Main entry point (e.g., CLI)
├── config.yaml             # Configuration (API keys, model choices per agent)
├── requirements.txt        # Project dependencies
├── project.md              # This file
└── .env.example            # Example environment variables
└── .gitignore
```

