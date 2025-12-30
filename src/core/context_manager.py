from libs.response_format import (
    CommandContext, AgentContext, SystemContext, SessionContext
)

class ContextManager:
    def __init__(self):
        self.session = SessionContext()
        self.commands = CommandContext()
        self.agent = AgentContext()
        self.system = SystemContext()

