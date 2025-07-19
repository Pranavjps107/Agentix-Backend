"""
Text Input Component
"""
from typing import Dict, Any
from ...core.base import BaseLangChainComponent, ComponentInput, ComponentOutput, ComponentMetadata
from ...core.registry import register_component

@register_component
class TextInputComponent(BaseLangChainComponent):
    """Text Input Component for workflow start"""
    
    def _setup_component(self):
        self.metadata = ComponentMetadata(
            display_name="Text Input",
            description="Start your workflow with user input",
            icon="📝",
            category="inputs",
            tags=["input", "text", "start"],
            version="1.0.0"
        )
        
        self.inputs = [
            ComponentInput(
                name="placeholder",
                display_name="Placeholder",
                field_type="str",
                default="Enter text...",
                required=False,
                description="Placeholder text for input field"
            ),
            ComponentInput(
                name="text",
                display_name="Input Text",
                field_type="str",
                multiline=True,
                description="The input text from user"
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
                display_name="Text Output",
                field_type="str",
                method="get_text",
                description="The input text"
            ),
            ComponentOutput(
                name="length",
                display_name="Text Length",
                field_type="int",
                method="get_length",
                description="Length of the input text"
            )
        ]
    
    async def execute(self, **kwargs) -> Dict[str, Any]:
        text = kwargs.get("text", "")
        placeholder = kwargs.get("placeholder", "Enter text...")
        required = kwargs.get("required", True)
        
        if required and not text.strip():
            raise ValueError("Text input is required but was empty")
        
        return {
            "text": text,
            "length": len(text),
            "placeholder": placeholder,
            "word_count": len(text.split()) if text else 0
        }