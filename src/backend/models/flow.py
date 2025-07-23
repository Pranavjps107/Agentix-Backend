"""
Flow Data Models
"""
from pydantic import BaseModel, Field, validator
from typing import Dict, Any, List, Optional
from uuid import uuid4
from datetime import datetime
from enum import Enum

class FlowStatus(str, Enum):
    """Flow execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class FlowNode(BaseModel):
    """Flow node definition"""
    id: str = Field(default_factory=lambda: str(uuid4()))
    component_type: str
    position: Dict[str, float] = Field(default_factory=lambda: {"x": 0, "y": 0})
    data: Optional[Dict[str, Any]] = None
    label: Optional[str] = None
    disabled: bool = False
    
    @validator('position')
    def validate_position(cls, v):
        if 'x' not in v or 'y' not in v:
            raise ValueError('Position must contain x and y coordinates')
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "id": "node-1",
                "component_type": "OpenAI LLM",
                "position": {"x": 100, "y": 100},
                "data": {
                    "model": "gpt-3.5-turbo",
                    "temperature": 0.7
                },
                "label": "Main LLM",
                "disabled": False
            }
        }

class FlowEdge(BaseModel):
    """Flow edge definition"""
    id: str = Field(default_factory=lambda: str(uuid4()))
    source: str
    target: str
    source_handle: Optional[str] = None
    target_handle: Optional[str] = None
    label: Optional[str] = None
    animated: bool = False
    
    class Config:
        schema_extra = {
            "example": {
                "id": "edge-1",
                "source": "node-1",
                "target": "node-2",
                "source_handle": "response",
                "target_handle": "prompt",
                "label": "Text Flow",
                "animated": False
            }
        }

class FlowDefinition(BaseModel):
    """Complete flow definition"""
    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    description: Optional[str] = None
    nodes: List[FlowNode]
    edges: List[FlowEdge]
    version: str = "1.0.0"
    metadata: Optional[Dict[str, Any]] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    @validator('nodes')
    def validate_nodes(cls, v):
        if not v:
            raise ValueError('Flow must contain at least one node')
        
        # Check for duplicate node IDs
        node_ids = [node.id for node in v]
        if len(node_ids) != len(set(node_ids)):
            raise ValueError('Duplicate node IDs found')
        
        return v
    
    @validator('edges')
    def validate_edges(cls, v, values):
        if 'nodes' not in values:
            return v
        
        node_ids = {node.id for node in values['nodes']}
        
        for edge in v:
            if edge.source not in node_ids:
                raise ValueError(f'Edge source "{edge.source}" not found in nodes')
            if edge.target not in node_ids:
                raise ValueError(f'Edge target "{edge.target}" not found in nodes')
        
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "id": "flow-1",
                "name": "Simple Chat Flow",
                "description": "A basic chatbot flow",
                "nodes": [
                    {
                        "id": "input-1",
                        "component_type": "Text Input",
                        "position": {"x": 100, "y": 100}
                    },
                    {
                        "id": "llm-1", 
                        "component_type": "OpenAI LLM",
                        "position": {"x": 300, "y": 100},
                        "data": {"model": "gpt-3.5-turbo"}
                    }
                ],
                "edges": [
                    {
                        "id": "edge-1",
                        "source": "input-1",
                        "target": "llm-1"
                    }
                ],
                "version": "1.0.0"
            }
        }

class FlowExecutionRequest(BaseModel):
    """Request to execute a flow"""
    flow_definition: FlowDefinition
    inputs: Optional[Dict[str, Any]] = None
    async_execution: bool = False
    timeout: Optional[float] = Field(default=600.0, description="Flow execution timeout in seconds")
    save_intermediate_results: bool = True
    
    class Config:
        schema_extra = {
            "example": {
                "flow_definition": {
                    "id": "flow-1",
                    "name": "Test Flow",
                    "nodes": [],
                    "edges": []
                },
                "inputs": {"user_input": "Hello"},
                "async_execution": False,
                "timeout": 600.0,
                "save_intermediate_results": True
            }
        }

class FlowExecutionResponse(BaseModel):
    """Response from flow execution"""
    success: bool
    outputs: Optional[Dict[str, Any]] = None
    execution_time: Optional[float] = None
    flow_id: str
    task_id: Optional[str] = None  # For async execution
    async_execution: bool = False
    error: Optional[str] = None
    component_outputs: Optional[Dict[str, Any]] = None
    execution_order: Optional[List[str]] = None
    status: FlowStatus = FlowStatus.COMPLETED
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "outputs": {"final_response": "Hello! How can I help?"},
                "execution_time": 5.23,
                "flow_id": "flow-1",
                "async_execution": False,
                "component_outputs": {
                    "node-1": {"response": "Hello! How can I help?"}
                },
                "execution_order": ["node-1"],
                "status": "completed",
                "timestamp": "2024-01-01T12:00:00Z"
            }
        }

class FlowTemplate(BaseModel):
    """Predefined flow template"""
    id: str
    name: str
    description: str
    category: str
    flow_definition: FlowDefinition
    tags: List[str] = []
    difficulty: str = "beginner"  # beginner, intermediate, advanced
    estimated_time: Optional[str] = None
    
    class Config:
        schema_extra = {
            "example": {
                "id": "template-chatbot",
                "name": "Basic Chatbot",
                "description": "A simple chatbot template",
                "category": "conversational",
                "flow_definition": {},
                "tags": ["chatbot", "llm", "conversation"],
                "difficulty": "beginner",
                "estimated_time": "5 minutes"
            }
        }