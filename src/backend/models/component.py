"""
Component Data Models
"""
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
from uuid import uuid4
from datetime import datetime

class ComponentExecutionRequest(BaseModel):
    """Request to execute a component"""
    component_id: str = Field(default_factory=lambda: str(uuid4()))
    inputs: Dict[str, Any]
    timeout: Optional[float] = Field(default=300.0, description="Execution timeout in seconds")
    cache_result: bool = Field(default=True, description="Whether to cache the result")
    
    class Config:
        json_schema_extra = {
            "example": {
                "component_id": "llm-component-123",
                "inputs": {
                    "prompt": "Hello, world!",
                    "temperature": 0.7,
                    "max_tokens": 100
                },
                "timeout": 300.0,
                "cache_result": True
            }
        }

class ComponentResponse(BaseModel):
    """Response from component execution"""
    success: bool
    outputs: Dict[str, Any]
    execution_time: float
    component_id: str
    component_name: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    cached: bool = False
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "outputs": {
                    "response": "Hello! How can I help you today?",
                    "usage": {
                        "prompt_tokens": 3,
                        "completion_tokens": 8,
                        "total_tokens": 11
                    }
                },
                "execution_time": 1.23,
                "component_id": "llm-component-123",
                "component_name": "OpenAI LLM",
                "timestamp": "2024-01-01T12:00:00Z",
                "cached": False
            }
        }

class ComponentStats(BaseModel):
    """Component execution statistics"""
    component_name: str
    total_executions: int = 0
    successful_executions: int = 0
    failed_executions: int = 0
    average_execution_time: float = 0.0
    last_execution: Optional[datetime] = None
    cache_hit_rate: float = 0.0

class ComponentInputSchema(BaseModel):
    """Schema for component input validation"""
    name: str
    type: str
    required: bool = True
    default: Any = None
    description: str = ""
    options: Optional[List[Any]] = None
    validation_rules: Optional[Dict[str, Any]] = None

class ComponentOutputSchema(BaseModel):
    """Schema for component output definition"""
    name: str
    type: str
    description: str = ""
    example: Any = None