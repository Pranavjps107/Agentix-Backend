"""
Text Input Component
"""
from typing import Dict, Any
from ...core.base import BaseLangChainComponent, ComponentInput, ComponentOutput, ComponentMetadata
from ...core.registry import register_component

@register_component
class TextInputComponent(BaseLangChainComponent):
    """Text Input Component for user input"""
    
    def _setup_component(self):
        self.metadata = ComponentMetadata(
            display_name="Text Input",
            description="Text input component for user interaction",
            icon="📝",
            category="inputs",
            tags=["input", "text", "user"]
        )
        
        self.inputs = [
            ComponentInput(
                name="text",
                display_name="Text",
                field_type="str",
                default="",
                description="Input text"
            ),
            ComponentInput(
                name="placeholder",
                display_name="Placeholder",
                field_type="str",
                default="Enter your question...",
                required=False,
                description="Placeholder text"
            ),
            ComponentInput(
                name="required",
                display_name="Required",
                field_type="bool",
                default=True,
                required=False,
                description="Whether input is required"
            )
        ]
        
        self.outputs = [
            ComponentOutput(
                name="text",
                display_name="Output Text",
                field_type="str",
                method="get_output",
                description="The input text"
            )
        ]
    
    async def execute(self, **kwargs) -> Dict[str, Any]:
        text = kwargs.get("text", "")
        placeholder = kwargs.get("placeholder", "Enter your question...")
        required = kwargs.get("required", True)
        
        # Use data from node configuration if available
        if "data" in kwargs:
            data = kwargs["data"]
            text = data.get("text", text)
            placeholder = data.get("placeholder", placeholder)
            required = data.get("required", required)
        
        return {
            "text": text,
            "length": len(text),
            "word_count": len(text.split()) if text else 0,
            "placeholder": placeholder,
            "required": required
        }