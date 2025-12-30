"""
Context Management System for Conversational AI

This module provides a comprehensive context tracking system for managing state,
commands, agent interactions, and system information in a voice assistant or
conversational AI application.
"""

from dataclasses import dataclass, field
import time
import uuid
from enum import Enum


class State(Enum):
    """
    Represents the current operational state of the system.
    
    Attributes:
        IDLE: System is inactive, waiting for input
        LISTENING: Actively receiving user input
        PROCESSING: Processing user request
        SPEAKING: Delivering response to user
    """
    IDLE = 0
    LISTENING = 1
    PROCESSING = 2
    SPEAKING = 3


@dataclass
class SessionContext:
    """
    Tracks information about the current user session.
    
    This class maintains session-level state including unique identification,
    timing information, and the most recent user interactions.
    
    Attributes:
        session_id (str): Unique identifier for the session, auto-generated using UUID4
        started_at (float): Unix timestamp when the session began
        last_intent (str | None): Most recently detected user intent
        last_entity (str | None): Most recently extracted entity from user input
        last_action_time (float | None): Unix timestamp of the last action
        state (State): Current operational state (defaults to IDLE)
    
    Example:
        >>> session = SessionContext()
        >>> session.state = State.LISTENING
        >>> session.last_intent = "open_app"
    """
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    started_at: float = field(default_factory=time.time)

    last_intent: str | None = None
    last_entity: str | None = None
    last_action_time: float | None = None

    state: State = State.IDLE  # idle | executing | awaiting_confirmation


@dataclass
class CommandRecord:
    """
    Represents a single command execution record.
    
    This class captures all relevant information about a command that was
    issued and potentially executed by the system.
    
    Attributes:
        intent (str): The command intent (e.g., "open_app", "send_message")
        params (dict): Parameters passed with the command
        executed (bool): Whether the command was successfully executed
        timestamp (float): Unix timestamp when the command was recorded
    
    Example:
        >>> record = CommandRecord(
        ...     intent="open_browser",
        ...     params={"url": "example.com"},
        ...     executed=True
        ... )
    """
    intent: str
    params: dict
    executed: bool
    timestamp: float = field(default_factory=time.time)


@dataclass
class CommandContext:
    """
    Maintains a chronological history of all commands.
    
    This class provides methods to log, retrieve, and query command execution
    history throughout the session.
    
    Attributes:
        history (list[CommandRecord]): Complete list of command records
    
    Example:
        >>> cmd_context = CommandContext()
        >>> cmd_context.log("open_app", {"app": "browser"}, True)
        >>> last_cmd = cmd_context.last()
        >>> recent_cmds = cmd_context.recent(3)
    """
    history: list[CommandRecord] = field(default_factory=list)

    def log(self, intent: str, params: dict, executed: bool):
        """
        Adds a new command record to the history.
        
        Args:
            intent (str): The command intent
            params (dict): Command parameters
            executed (bool): Whether the command was successfully executed
        
        Example:
            >>> context.log("play_music", {"song": "Yesterday"}, True)
        """
        self.history.append(CommandRecord(intent, params, executed))

    def last(self) -> CommandRecord | None:
        """
        Returns the most recent command record.
        
        Returns:
            CommandRecord | None: The last command record, or None if history is empty
        
        Example:
            >>> last_command = context.last()
            >>> if last_command:
            ...     print(last_command.intent)
        """
        return self.history[-1] if self.history else None

    def recent(self, n=5):
        """
        Returns the last n command records.
        
        Args:
            n (int, optional): Number of recent commands to return. Defaults to 5.
        
        Returns:
            list[CommandRecord]: List of the n most recent command records
        
        Example:
            >>> recent_commands = context.recent(3)
            >>> for cmd in recent_commands:
            ...     print(cmd.intent)
        """
        return self.history[-n:]


@dataclass
class AgentInvocation:
    """
    Records a single agent invocation event.
    
    This class captures information about when and why the AI agent was invoked,
    along with which tools it utilized.
    
    Attributes:
        reason (str): Why the agent was invoked (e.g., "chat", "search", "fallback")
        tools_used (list[str]): List of tools/APIs the agent utilized
        timestamp (float): Unix timestamp of the invocation
    
    Example:
        >>> invocation = AgentInvocation(
        ...     reason="search",
        ...     tools_used=["web_search", "summarization"]
        ... )
    """
    reason: str                 # "chat" | "search" | "fallback"
    tools_used: list[str]
    timestamp: float = field(default_factory=time.time)


@dataclass
class AgentContext:
    """
    Tracks the history of AI agent invocations.
    
    This class maintains a record of all times the AI agent was invoked,
    including the reasons and tools used.
    
    Attributes:
        history (list[AgentInvocation]): Complete list of agent invocation records
    
    Example:
        >>> agent_context = AgentContext()
        >>> agent_context.log("search", ["web_search"])
        >>> recent = agent_context.recent(5)
    """
    history: list[AgentInvocation] = field(default_factory=list)

    def log(self, reason: str, tools_used: list[str]):
        """
        Records a new agent invocation.
        
        Args:
            reason (str): Why the agent was invoked
            tools_used (list[str]): List of tools the agent utilized
        
        Example:
            >>> context.log("fallback", ["general_knowledge", "reasoning"])
        """
        self.history.append(AgentInvocation(reason, tools_used))

    def recent(self, n=5):
        """
        Returns the last n agent invocations.
        
        Args:
            n (int, optional): Number of recent invocations to return. Defaults to 5.
        
        Returns:
            list[AgentInvocation]: List of the n most recent agent invocations
        
        Example:
            >>> recent_invocations = context.recent(3)
            >>> for inv in recent_invocations:
            ...     print(f"{inv.reason}: {inv.tools_used}")
        """
        return self.history[-n:]


@dataclass
class SystemContext:
    """
    Monitors the current system and application state.
    
    This class tracks information about running applications and system state
    that may be relevant for decision-making or safety checks.
    
    Attributes:
        active_apps (list[str]): List of currently running applications
        fullscreen_app (str | None): Name of the app in fullscreen mode, if any
        unsaved_work (bool): Flag indicating if there's unsaved work
        critical_apps_open (bool): Flag indicating if critical applications are running
    
    Example:
        >>> sys_context = SystemContext()
        >>> sys_context.active_apps = ["browser", "terminal", "editor"]
        >>> sys_context.unsaved_work = True
        >>> sys_context.critical_apps_open = True
    """
    active_apps: list[str] = field(default_factory=list)
    fullscreen_app: str | None = None
    unsaved_work: bool = False
    critical_apps_open: bool = False