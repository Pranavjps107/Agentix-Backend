"""
Simple Components for Basic Workflows
"""
from typing import Dict, Any
from ..core.base import BaseLangChainComponent, ComponentInput, ComponentOutput, ComponentMetadata
from ..core.registry import register_component  # Fix: Import from registry, not base
from ..core.registry import register_component  # Import from registry, not base
@register_component
class SimpleInputComponent(BaseLangChainComponent):
    """Simple Input Component for basic text input"""
    
    def _setup_component(self):
        self.metadata = ComponentMetadata(
            display_name="Simple Input",
            description="Basic text input component",
            icon="📝",
            category="inputs",
            tags=["input", "text", "basic"]
        )
        
        self.inputs = [
            ComponentInput(
                name="text",
                display_name="Text",
                field_type="str",
                default="",
                description="Input text"
            )
        ]
        
        self.outputs = [
            ComponentOutput(
                name="output",
                display_name="Output Text",
                field_type="str",
                method="get_output",
                description="The input text passed through"
            )
        ]
    
    async def execute(self, **kwargs) -> Dict[str, Any]:
        text = kwargs.get("text", "")
        
        return {
            "output": text,
            "length": len(text),
            "word_count": len(text.split()) if text else 0
        }

@register_component 
class SimpleLLMComponent(BaseLangChainComponent):
    """Simple LLM Component for basic text generation"""
    
    def _setup_component(self):
        self.metadata = ComponentMetadata(
            display_name="Simple LLM",
            description="Basic LLM component for text generation",
            icon="🤖",
            category="language_models",
            tags=["llm", "simple", "generation"]
        )
        
        self.inputs = [
            ComponentInput(
                name="prompt",
                display_name="Prompt",
                field_type="str",
                description="Input prompt for the LLM"
            ),
            ComponentInput(
                name="model",
                display_name="Model",
                field_type="str",
                default="fake",
                options=["fake", "openai", "anthropic"],
                required=False,
                description="Model type to use"
            ),
            ComponentInput(
                name="temperature",
                display_name="Temperature",
                field_type="float",
                default=0.7,
                required=False,
                description="Sampling temperature"
            )
        ]
        
        self.outputs = [
            ComponentOutput(
                name="response",
                display_name="Response",
                field_type="str",
                method="generate_response",
                description="Generated response"
            ),
            ComponentOutput(
                name="prompt_length",
                display_name="Prompt Length",
                field_type="int",
                method="get_prompt_length",
                description="Length of input prompt"
            )
        ]
    
    async def execute(self, **kwargs) -> Dict[str, Any]:
        prompt = kwargs.get("prompt", "")
        model = kwargs.get("model", "fake")
        temperature = kwargs.get("temperature", 0.7)
        
        if not prompt.strip():
            return {
                "response": "Error: Empty prompt provided",
                "prompt_length": 0,
                "model_used": model,
                "success": False
            }
        
        # For demo purposes, create a simple response
        if model == "fake":
            response = f"Simple LLM Response to: '{prompt[:50]}...' (Temperature: {temperature})"
        else:
            # Here you could integrate with actual LLM providers
            response = f"Mock {model.upper()} response to: {prompt}"
        
        return {
            "response": response,
            "prompt_length": len(prompt),
            "model_used": model,
            "temperature": temperature,
            "success": True
        }