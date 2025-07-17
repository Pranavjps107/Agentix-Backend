"""
Text Input Component
"""
from typing import Dict, Any
from ...core.base import BaseLangChainComponent, ComponentInput, ComponentOutput, ComponentMetadata, register_component

@register_component
class TextInputComponent(BaseLangChainComponent):
    """Text Input Component for user input"""
    
    def _setup_component(self):
        self.metadata = ComponentMetadata(
            display_name="Text Input",
            description="Capture text input from user",
            icon="📝",
            category="inputs",
            tags=["input", "text", "user"],
            version="1.0.0"
        )
        
        self.inputs = [
            ComponentInput(
                name="placeholder",
                display_name="Placeholder",
                field_type="str",
                default="Enter text...",
                required=False,
                description="Placeholder text for input"
            ),
            ComponentInput(
                name="user_input",
                display_name="User Input",
                field_type="str",
                required=False,
                description="The actual user input"
            )
        ]
        
        self.outputs = [
            ComponentOutput(
                name="text",
                display_name="Text Output",
                field_type="str",
                method="get_text",
                description="The user input text"
            )
        ]
    
    async def execute(self, **kwargs) -> Dict[str, Any]:
        user_input = kwargs.get("user_input", "")
        placeholder = kwargs.get("placeholder", "Enter text...")
        
        return {
            "text": user_input,
            "placeholder": placeholder,
            "length": len(user_input)
        }