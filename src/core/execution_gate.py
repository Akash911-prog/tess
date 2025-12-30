from enum import Enum

from configs.execution_gate import (
    EXECUTABLE_INTENTS,
    NO_PARAM_INTENTS,
    REQUIRED_PARAMS
)

class ExecutionType(Enum):
    LLM = 'llm'
    PLUGINS = 'plugins'
    MISSING_PARAMS = 'missing params'



class ExecutionGate:
    def __init__(self):
        self.executable_intents = EXECUTABLE_INTENTS
        self.no_param_intents = NO_PARAM_INTENTS
        self.required_params = REQUIRED_PARAMS
        

    def is_executable(self, result):

        if result.intent not in self.executable_intents:
            return (False, "Intent not executable", ExecutionType.LLM)

        for required_param in self.required_params[result.intent]:
            if required_param not in result.params:
                return (False, f"Missing required parameter: {required_param}", ExecutionType.MISSING_PARAMS)
        

        return (True, "Intent executable", ExecutionType.PLUGINS)