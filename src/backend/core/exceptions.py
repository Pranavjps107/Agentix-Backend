"""
Custom exceptions for the LangChain Platform
"""

class ComponentException(Exception):
    """Base exception for component-related errors"""
    pass

class ExecutionException(ComponentException):
    """Exception raised during component execution"""
    pass

class ValidationException(ComponentException):
    """Exception raised during input validation"""
    pass

class RegistrationException(ComponentException):
    """Exception raised during component registration"""
    pass

class FlowException(Exception):
    """Base exception for flow-related errors"""
    pass

class FlowValidationException(FlowException):
    """Exception raised during flow validation"""
    pass

class FlowExecutionException(FlowException):
    """Exception raised during flow execution"""
    pass