"""
Test Component to verify registration works
"""
from typing import Dict, Any
from ..core.base import BaseLangChainComponent, ComponentInput, ComponentOutput, ComponentMetadata
from ..core.registry import register_component

@register_component
class TestComponent(BaseLangChainComponent):
    """Simple test component"""
    
    def _setup_component(self):
        self.metadata = ComponentMetadata(
            display_name="Test Component",
            description="Simple test component for debugging",
            icon="🧪",
            category="test",
            tags=["test", "debug"],
        )
        
        self.inputs = [
            ComponentInput(
                name="input_text",
                display_name="Input Text",
                field_type="str",
                description="Test input"
            )
        ]
        
        self.outputs = [
            ComponentOutput(
                name="output_text",
                display_name="Output Text",
                field_type="str",
                method="process_text",
                description="Test output"
            )
        ]
    
    async def execute(self, **kwargs) -> Dict[str, Any]:
        input_text = kwargs.get("input_text", "")
        return {
            "output_text": f"Processed: {input_text}",
            "success": True
        }