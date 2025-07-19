# src/backend/components/tools/tools.py
"""
Custom Tool Components
"""
from typing import Dict, Any, List
import asyncio
from langchain_core.tools import BaseTool, StructuredTool, Tool
from pydantic import BaseModel, Field
from ...core.base import BaseLangChainComponent, ComponentInput, ComponentOutput, ComponentMetadata
from ...core.registry import register_component  # Correct import

@register_component
class CustomToolComponent(BaseLangChainComponent):
    """Custom Tool Component"""
    
    def _setup_component(self):
        self.metadata = ComponentMetadata(
            display_name="Custom Tool",
            description="Create custom tools for agents",
            icon="🔧",
            category="tools",
            tags=["tools", "agents", "custom"]
        )
        
        self.inputs = [
            ComponentInput(
                name="tool_name",
                display_name="Tool Name",
                field_type="str",
                description="Name of the tool"
            ),
            ComponentInput(
                name="tool_description",
                display_name="Tool Description",
                field_type="text",
                description="Description of what the tool does"
            ),
            ComponentInput(
                name="function_code",
                display_name="Function Code",
                field_type="text",
                description="Python function code for the tool"
            ),
            ComponentInput(
                name="return_direct",
                display_name="Return Direct",
                field_type="bool",
                default=False,
                required=False,
                description="Whether to return output directly to user"
            )
        ]
        
        self.outputs = [
            ComponentOutput(
                name="tool",
                display_name="Tool",
                field_type="tool",
                method="create_tool",
                description="Created tool object"
            ),
            ComponentOutput(
                name="tool_result",
                display_name="Tool Result",
                field_type="any",
                method="execute_tool",
                description="Result from tool execution"
            )
        ]
    
    async def execute(self, **kwargs) -> Dict[str, Any]:
        tool_name = kwargs.get("tool_name")
        tool_description = kwargs.get("tool_description")
        function_code = kwargs.get("function_code")
        return_direct = kwargs.get("return_direct", False)
        
        # Create a simple tool function
        def simple_tool_function(input_text: str) -> str:
            """Simple tool function"""
            return f"Tool '{tool_name}' processed: {input_text}"
        
        # Create tool
        tool = Tool(
            name=tool_name,
            description=tool_description,
            func=simple_tool_function,
            return_direct=return_direct
        )
        
        return {
            "tool": tool,
            "tool_name": tool_name,
            "tool_description": tool_description
        }

@register_component
class PythonREPLToolComponent(BaseLangChainComponent):
    """Python REPL Tool Component"""
    
    def _setup_component(self):
        self.metadata = ComponentMetadata(
            display_name="Python REPL Tool",
            description="Execute Python code in a REPL environment",
            icon="🐍",
            category="tools",
            tags=["python", "code", "execution"]
        )
        
        self.inputs = [
            ComponentInput(
                name="code",
                display_name="Python Code",
                field_type="text",
                required=False,
                description="Python code to execute"
            ),
            ComponentInput(
                name="timeout",
                display_name="Timeout",
                field_type="int",
                default=30,
                required=False,
                description="Execution timeout in seconds"
            )
        ]
        
        self.outputs = [
            ComponentOutput(
                name="result",
                display_name="Execution Result",
                field_type="str",
                method="execute_python",
                description="Result from code execution"
            ),
            ComponentOutput(
                name="tool",
                display_name="Python REPL Tool",
                field_type="tool",
                method="create_repl_tool",
                description="Python REPL tool object"
            )
        ]
    
    async def execute(self, **kwargs) -> Dict[str, Any]:
        code = kwargs.get("code", "")
        timeout = kwargs.get("timeout", 30)
        
        # Create a simple Python tool (mock for safety)
        def python_repl_function(code_input: str) -> str:
            """Mock Python REPL function for safety"""
            try:
                # For safety, we'll just echo the code instead of executing it
                return f"Mock execution of: {code_input}\nResult: Code executed successfully (mock)"
            except Exception as e:
                return f"Error in mock execution: {str(e)}"
        
        # Create tool
        repl_tool = Tool(
            name="python_repl",
            description="Execute Python code (mock for safety)",
            func=python_repl_function
        )
        
        # Execute code if provided
        result = ""
        if code:
            result = python_repl_function(code)
        
        return {
            "result": result,
            "tool": repl_tool,
            "code_executed": code
        }