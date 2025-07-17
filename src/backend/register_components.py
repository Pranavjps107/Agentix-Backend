"""
Component Registration Module
This module ensures all components are registered when imported
"""
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

def register_all_components():
    """Register all components explicitly"""
    logger.info("🔧 Starting explicit component registration...")
    
    from .core.registry import ComponentRegistry
    from .core.base import BaseLangChainComponent, ComponentInput, ComponentOutput, ComponentMetadata, register_component
    
    # Create basic components inline
    
    @register_component
    class TextInputComponent(BaseLangChainComponent):
        def _setup_component(self):
            self.metadata = ComponentMetadata(
                display_name="Text Input",
                description="Text input component",
                icon="📝",
                category="inputs"
            )
            self.inputs = [ComponentInput(name="text", display_name="Text", field_type="str")]
            self.outputs = [ComponentOutput(name="text", display_name="Text", field_type="str", method="get_text")]
        
        async def execute(self, **kwargs):
            return {"text": kwargs.get("text", "")}
    
    @register_component  
    class ChatModelComponent(BaseLangChainComponent):
        def _setup_component(self):
            self.metadata = ComponentMetadata(
                display_name="ChatModel", 
                description="Chat model component",
                icon="💬",
                category="chat_models"
            )
            self.inputs = [ComponentInput(name="messages", display_name="Messages", field_type="list")]
            self.outputs = [ComponentOutput(name="response", display_name="Response", field_type="str", method="chat")]
        
        async def execute(self, **kwargs):
            return {"response": "Mock chat response"}
    
    @register_component
    class WebSearchToolComponent(BaseLangChainComponent):
        def _setup_component(self):
            self.metadata = ComponentMetadata(
                display_name="Web Search Tool",
                description="Web search tool",
                icon="🔍", 
                category="tools"
            )
            self.inputs = [ComponentInput(name="query", display_name="Query", field_type="str")]
            self.outputs = [ComponentOutput(name="results", display_name="Results", field_type="list", method="search")]
        
        async def execute(self, **kwargs):
            return {"results": ["Mock search result"]}
    
    @register_component
    class OpenAIFunctionsAgentComponent(BaseLangChainComponent):
        def _setup_component(self):
            self.metadata = ComponentMetadata(
                display_name="OpenAI Functions Agent",
                description="OpenAI functions agent",
                icon="🤖",
                category="agents"
            )
            self.inputs = [ComponentInput(name="input", display_name="Input", field_type="str")]
            self.outputs = [ComponentOutput(name="output", display_name="Output", field_type="str", method="run")]
        
        async def execute(self, **kwargs):
            return {"output": "Mock agent response"}
    
    @register_component
    class AgentExecutorComponent(BaseLangChainComponent):
        def _setup_component(self):
            self.metadata = ComponentMetadata(
                display_name="Agent Executor",
                description="Agent executor",
                icon="⚡",
                category="agents"
            )
            self.inputs = [ComponentInput(name="input", display_name="Input", field_type="str")]
            self.outputs = [ComponentOutput(name="output", display_name="Output", field_type="str", method="execute")]
        
        async def execute(self, **kwargs):
            return {"output": "Mock execution result"}
    
    count = len(ComponentRegistry._components)
    logger.info(f"✅ Registered {count} components explicitly")
    return count

# Auto-register when module is imported
try:
    register_all_components()
except Exception as e:
    logger.error(f"❌ Failed to auto-register components: {e}")