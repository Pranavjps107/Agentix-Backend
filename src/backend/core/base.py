# src/backend/core/base.py (Fixed version)
"""
Base component system for LangChain Platform
"""
from __future__ import annotations
import asyncio
import inspect
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type, Union, get_type_hints
from uuid import uuid4
from pydantic import BaseModel, Field
from langchain_core.runnables import Runnable
from langchain_core.callbacks import BaseCallbackHandler
from .exceptions import ValidationException, ExecutionException

class ComponentInput(BaseModel):
    """Defines input configuration for components"""
    name: str
    display_name: str
    field_type: str
    required: bool = True
    default: Any = None
    description: str = ""
    options: Optional[List[str]] = None
    input_types: Optional[List[str]] = None
    password: bool = False
    multiline: bool = False
    
    class Config:
        arbitrary_types_allowed = True

class ComponentOutput(BaseModel):
    """Defines output configuration for components"""
    name: str
    display_name: str
    field_type: str
    description: str = ""
    method: str
    
    class Config:
        arbitrary_types_allowed = True

class ComponentMetadata(BaseModel):
    """Component metadata"""
    display_name: str
    description: str
    icon: str = "🔧"
    category: str
    tags: List[str] = []
    version: str = "1.0.0"
    author: str = "LangChain Platform"
    
    class Config:
        arbitrary_types_allowed = True

class BaseLangChainComponent(ABC):
    """Base class for all LangChain components"""
    
    def __init__(self):
        self.id = str(uuid4())
        self.inputs: List[ComponentInput] = []
        self.outputs: List[ComponentOutput] = []
        self.metadata: ComponentMetadata = ComponentMetadata(
            display_name="Base Component",
            description="Base component",
            category="base"
        )
        self._execution_count = 0
        self._last_execution_time = None
        self._setup_component()
    
    @abstractmethod
    def _setup_component(self):
        """Setup component inputs and outputs"""
        pass
    
    @abstractmethod
    async def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute component logic"""
        pass
    
    def get_schema(self) -> Dict[str, Any]:
        """Get component schema for frontend"""
        return {
            "id": self.id,
            "metadata": self.metadata.dict(),
            "inputs": [inp.dict() for inp in self.inputs],
            "outputs": [out.dict() for out in self.outputs],
            "execution_stats": {
                "execution_count": self._execution_count,
                "last_execution_time": self._last_execution_time
            }
        }
    
    def validate_inputs(self, inputs: Dict[str, Any]) -> bool:
        """Validate input data"""
        for inp in self.inputs:
            if inp.required and inp.name not in inputs:
                raise ValidationException(f"Required input '{inp.name}' missing for component {self.metadata.display_name}")
            
            if inp.name in inputs:
                value = inputs[inp.name]
                # Type validation
                if inp.field_type == "int" and not isinstance(value, int):
                    try:
                        inputs[inp.name] = int(value)
                    except (ValueError, TypeError):
                        raise ValidationException(f"Input '{inp.name}' must be an integer")
                
                elif inp.field_type == "float" and not isinstance(value, (int, float)):
                    try:
                        inputs[inp.name] = float(value)
                    except (ValueError, TypeError):
                        raise ValidationException(f"Input '{inp.name}' must be a number")
                
                elif inp.field_type == "bool" and not isinstance(value, bool):
                    if isinstance(value, str):
                        inputs[inp.name] = value.lower() in ('true', '1', 'yes', 'on')
                    else:
                        inputs[inp.name] = bool(value)
                
                # Options validation
                if inp.options and value not in inp.options:
                    raise ValidationException(f"Input '{inp.name}' must be one of {inp.options}")
        
        return True
    
    async def _execute_with_error_handling(self, **kwargs) -> Dict[str, Any]:
        """Execute with error handling and stats tracking"""
        import time
        start_time = time.time()
        
        try:
            self.validate_inputs(kwargs)
            result = await self.execute(**kwargs)
            
            self._execution_count += 1
            self._last_execution_time = time.time()
            
            execution_time = time.time() - start_time
            
            if isinstance(result, dict):
                result["_execution_metadata"] = {
                    "execution_time": execution_time,
                    "component_id": self.id,
                    "component_type": self.metadata.display_name,
                    "success": True
                }
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            raise ExecutionException(
                f"Component {self.metadata.display_name} execution failed: {str(e)}",
                execution_time=execution_time,
                component_id=self.id
            )
    
    def get_input_by_name(self, name: str) -> Optional[ComponentInput]:
        """Get input configuration by name"""
        for inp in self.inputs:
            if inp.name == name:
                return inp
        return None
    
    def get_output_by_name(self, name: str) -> Optional[ComponentOutput]:
        """Get output configuration by name"""
        for out in self.outputs:
            if out.name == name:
                return out
        return None

# Register component decorator - this must be imported from registry
def register_component(component_class: Type[BaseLangChainComponent]):
    """Decorator to auto-register components"""
    from .registry import ComponentRegistry
    ComponentRegistry.register(component_class)
    return component_class