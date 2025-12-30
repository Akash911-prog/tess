from libs.response_format import (
    CommandContext, AgentContext, SystemContext, SessionContext, State
)
import time

class ContextManager:
    def __init__(self):
        self.session = SessionContext()
        self.commands = CommandContext()
        self.agent = AgentContext()
        self.system = SystemContext()

    def set_state(self, state: State):
        """
        Sets the current operational state of the system.

        Args:
            state (State): New operational state

        Example:
            >>> context_manager.set_state(State.LISTENING)
        """
        self.session.state = state

    def log_command(self, intent: str, params: dict, executed: bool):
        """
        Logs a command execution to the command context and updates session metadata.

        This method records the command in the command history and updates the
        session's last intent and action time.

        Args:
            intent (str): The command intent (e.g., "open_app", "send_message")
            params (dict): Parameters passed with the command
            executed (bool): Whether the command was successfully executed

        Example:
            >>> context_manager.log_command(
            ...     intent="open_browser",
            ...     params={"url": "example.com"},
            ...     executed=True
            ... )
        """
        self.commands.log(intent, params, executed)
        self.session.last_intent = intent
        self.session.last_action_time = time.time()

    def log_agent_invocation(self, reason: str, tools_used: list[str]):
        """
        Logs an agent invocation to the agent context and updates session metadata.

        This method records when and why the AI agent was invoked, including
        which tools were used, and updates the session's last action time.

        Args:
            reason (str): Why the agent was invoked (e.g., "chat", "search", "fallback")
            tools_used (list[str]): List of tools/APIs the agent utilized

        Example:
            >>> context_manager.log_agent_invocation(
            ...     reason="search",
            ...     tools_used=["web_search", "summarization"]
            ... )
        """
        self.agent.log(reason, tools_used)
        self.session.last_action_time = time.time()

    def log_intent_and_entity(self, intent: str, entity: str | None = None):
        """
        Updates the session with the most recent intent and entity.

        This is typically called after parsing user input to track what
        the user intended and what entities were extracted.

        Args:
            intent (str): The detected user intent
            entity (str | None, optional): Extracted entity from user input. Defaults to None.

        Example:
            >>> context_manager.log_intent_and_entity(
            ...     intent="play_music",
            ...     entity="Beatles"
            ... )
        """
        self.session.last_intent = intent
        self.session.last_entity = entity
        self.session.last_action_time = time.time()

    def update_system_state(
        self,
        active_apps: list[str] | None = None,
        fullscreen_app: str | None = None,
        unsaved_work: bool | None = None,
        critical_apps_open: bool | None = None
    ):
        """
        Updates the system context with current application and system state.

        This method allows selective updates to system state. Only provided
        arguments will update the corresponding attributes.

        Args:
            active_apps (list[str] | None, optional): List of currently running applications
            fullscreen_app (str | None, optional): Name of the app in fullscreen mode
            unsaved_work (bool | None, optional): Flag indicating if there's unsaved work
            critical_apps_open (bool | None, optional): Flag for critical applications running

        Example:
            >>> context_manager.update_system_state(
            ...     active_apps=["browser", "terminal"],
            ...     unsaved_work=True
            ... )
        """
        if active_apps is not None:
            self.system.active_apps = active_apps
        if fullscreen_app is not None:
            self.system.fullscreen_app = fullscreen_app
        if unsaved_work is not None:
            self.system.unsaved_work = unsaved_work
        if critical_apps_open is not None:
            self.system.critical_apps_open = critical_apps_open

    def log_complete_interaction(
        self,
        intent: str,
        params: dict,
        executed: bool,
        entity: str | None = None,
        agent_reason: str | None = None,
        tools_used: list[str] | None = None
    ):
        """
        Logs a complete user interaction including command, intent, and optional agent invocation.

        This is a convenience method that logs multiple context records in a single call,
        useful for tracking complete interactions that involve both command execution
        and agent invocations.

        Args:
            intent (str): The command/user intent
            params (dict): Parameters passed with the command
            executed (bool): Whether the command was successfully executed
            entity (str | None, optional): Extracted entity from user input. Defaults to None.
            agent_reason (str | None, optional): Reason for agent invocation. Defaults to None.
            tools_used (list[str] | None, optional): Tools used by agent. Defaults to None.

        Example:
            >>> context_manager.log_complete_interaction(
            ...     intent="search_web",
            ...     params={"query": "weather today"},
            ...     executed=True,
            ...     entity="weather",
            ...     agent_reason="search",
            ...     tools_used=["web_search", "location_service"]
            ... )
        """
        # Log command
        self.log_command(intent, params, executed)
        
        # Update entity if provided
        if entity is not None:
            self.session.last_entity = entity
        
        # Log agent invocation if applicable
        if agent_reason is not None and tools_used is not None:
            self.agent.log(agent_reason, tools_used)

    def get_recent_history(self, n: int = 5) -> dict:
        """
        Retrieves recent history from all context types.

        Args:
            n (int, optional): Number of recent records to retrieve. Defaults to 5.

        Returns:
            dict: Dictionary containing recent commands and agent invocations

        Example:
            >>> history = context_manager.get_recent_history(3)
            >>> print(history['commands'])
            >>> print(history['agent_invocations'])
        """
        return {
            'commands': self.commands.recent(n),
            'agent_invocations': self.agent.recent(n)
        }

    def reset_session(self):
        """
        Resets the session context while preserving command and agent history.

        This is useful for starting a new session without losing historical data
        that might be useful for analytics or debugging.

        Example:
            >>> context_manager.reset_session()
        """
        self.session = SessionContext()

    def full_reset(self):
        """
        Performs a complete reset of all contexts.

        This creates fresh instances of all context objects, clearing all history
        and state information.

        Example:
            >>> context_manager.full_reset()
        """
        self.session = SessionContext()
        self.commands = CommandContext()
        self.agent = AgentContext()
        self.system = SystemContext()

