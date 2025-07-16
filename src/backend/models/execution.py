"""
Execution Data Models
"""
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
from datetime import datetime
from enum import Enum
from uuid import uuid4

class ExecutionStatus(str, Enum):
    """Execution status enumeration"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"

class ExecutionResult(BaseModel):
    """Result of component or flow execution"""
    execution_id: str = Field(default_factory=lambda: str(uuid4()))
    status: ExecutionStatus
    outputs: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None
    execution_time: float = 0.0
    start_time: datetime = Field(default_factory=datetime.utcnow)
    end_time: Optional[datetime] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        schema_extra = {
            "example": {
                "execution_id": "exec-123",
                "status": "completed",
                "outputs": {"result": "Success"},
                "execution_time": 2.5,
                "start_time": "2024-01-01T12:00:00Z",
                "end_time": "2024-01-01T12:00:02Z",
                "metadata": {"cached": False}
            }
        }

class TaskInfo(BaseModel):
    """Information about an async task"""
    task_id: str
    status: ExecutionStatus
    progress: float = 0.0  # 0.0 to 100.0
    start_time: datetime = Field(default_factory=datetime.utcnow)
    end_time: Optional[datetime] = None
    result: Optional[ExecutionResult] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        schema_extra = {
            "example": {
                "task_id": "task-456",
                "status": "running",
                "progress": 45.0,
                "start_time": "2024-01-01T12:00:00Z",
                "metadata": {"flow_id": "flow-1"}
            }
        }

class ExecutionStep(BaseModel):
    """Individual step in execution"""
    step_id: str = Field(default_factory=lambda: str(uuid4()))
    component_id: str
    component_name: str
    status: ExecutionStatus
    inputs: Dict[str, Any] = Field(default_factory=dict)
    outputs: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None
    execution_time: float = 0.0
    start_time: datetime = Field(default_factory=datetime.utcnow)
    end_time: Optional[datetime] = None
    
    class Config:
        schema_extra = {
            "example": {
                "step_id": "step-789",
                "component_id": "node-1",
                "component_name": "OpenAI LLM",
                "status": "completed",
                "inputs": {"prompt": "Hello"},
                "outputs": {"response": "Hi there!"},
                "execution_time": 1.2,
                "start_time": "2024-01-01T12:00:00Z",
                "end_time": "2024-01-01T12:00:01Z"
            }
        }

class ExecutionLog(BaseModel):
    """Complete execution log"""
    execution_id: str
    flow_id: Optional[str] = None
    component_id: Optional[str] = None
    steps: List[ExecutionStep] = Field(default_factory=list)
    total_execution_time: float = 0.0
    status: ExecutionStatus
    start_time: datetime = Field(default_factory=datetime.utcnow)
    end_time: Optional[datetime] = None
    user_id: Optional[str] = None
    
    class Config:
        schema_extra = {
            "example": {
                "execution_id": "exec-123",
                "flow_id": "flow-1",
                "steps": [],
                "total_execution_time": 5.7,
                "status": "completed",
                "start_time": "2024-01-01T12:00:00Z",
"end_time": "2024-01-01T12:00:05Z",
               "user_id": "user-456"
           }
       }

class PerformanceMetrics(BaseModel):
   """Performance metrics for executions"""
   component_type: str
   average_execution_time: float
   min_execution_time: float
   max_execution_time: float
   total_executions: int
   success_rate: float
   error_rate: float
   last_updated: datetime = Field(default_factory=datetime.utcnow)
   
   class Config:
       schema_extra = {
           "example": {
               "component_type": "OpenAI LLM",
               "average_execution_time": 2.5,
               "min_execution_time": 0.8,
               "max_execution_time": 8.2,
               "total_executions": 150,
               "success_rate": 0.96,
               "error_rate": 0.04,
               "last_updated": "2024-01-01T12:00:00Z"
           }
       }