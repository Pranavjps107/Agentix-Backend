# src/backend/components/tools/__init__.py
"""
Tool Components
"""

# Import components that exist
try:
    from .tools import CustomToolComponent, PythonREPLToolComponent
except ImportError:
    # Create placeholder classes if the file doesn't exist
    from ...core.base import BaseLangChainComponent, ComponentMetadata
    from ...core.registry import register_component  # Correct import
    
    @register_component
    class CustomToolComponent(BaseLangChainComponent):
        def _setup_component(self):
            self.metadata = ComponentMetadata(
                display_name="Custom Tool",
                description="Custom tool component (placeholder)",
                icon="🔧",
                category="tools",
                tags=["tools"]
            )
            self.inputs = []
            self.outputs = []
        
        async def execute(self, **kwargs):
            return {"message": "Custom tool component not fully implemented"}
    
    @register_component
    class PythonREPLToolComponent(BaseLangChainComponent):
        def _setup_component(self):
            self.metadata = ComponentMetadata(
                display_name="Python REPL Tool",
                description="Python REPL tool (placeholder)",
                icon="🐍",
                category="tools",
                tags=["python"]
            )
            self.inputs = []
            self.outputs = []
        
        async def execute(self, **kwargs):
            return {"message": "Python REPL component not fully implemented"}

try:
    from .web_search import WebSearchToolComponent
except ImportError:
    # Create placeholder if web_search doesn't exist
    @register_component
    class WebSearchToolComponent(BaseLangChainComponent):
        def _setup_component(self):
            self.metadata = ComponentMetadata(
                display_name="Web Search Tool",
                description="Web search tool (placeholder)",
                icon="🔍",
                category="tools",
                tags=["search"]
            )
            self.inputs = []
            self.outputs = []
        
        async def execute(self, **kwargs):
            return {"message": "Web search component not fully implemented"}

__all__ = [
    "CustomToolComponent",
    "PythonREPLToolComponent", 
    "WebSearchToolComponent"
]