"""
Custom exceptions for the LangChain Platform
"""

class ComponentException(Exception):
    """Base exception for component-related errors"""
    def __init__(self, message: str = "", **kwargs):
        super().__init__(message)
        self.message = message
        self.metadata = kwargs

class ExecutionException(ComponentException):
    """Exception raised during component execution"""
    def __init__(self, message: str = "", execution_time: float = 0.0, component_id: str = "", **kwargs):
        super().__init__(message, **kwargs)
        self.execution_time = execution_time
        self.component_id = component_id

class ValidationException(ComponentException):
    """Exception raised during input validation"""
    pass

class RegistrationException(ComponentException):
    """Exception raised during component registration"""
    pass

class FlowException(Exception):
    """Base exception for flow-related errors"""
    def __init__(self, message: str = "", **kwargs):
        super().__init__(message)
        self.message = message
        self.metadata = kwargs

class FlowValidationException(FlowException):
    """Exception raised during flow validation"""
    pass

class FlowExecutionException(FlowException):
    """Exception raised during flow execution"""
    pass