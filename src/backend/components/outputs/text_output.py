# src/backend/components/outputs/text_output.py
"""
Text Output Component
"""
from typing import Dict, Any
from ...core.base import BaseLangChainComponent, ComponentInput, ComponentOutput, ComponentMetadata
from ...core.registry import register_component  # Correct import location

@register_component
class TextOutputComponent(BaseLangChainComponent):
    """Text Output Component for displaying results"""
    
    def _setup_component(self):
        self.metadata = ComponentMetadata(
            display_name="Text Output",
            description="Display text output to users",
            icon="📄",
            category="outputs",
            tags=["output", "text", "display"],
            version="1.0.0"
        )
        
        self.inputs = [
            ComponentInput(
                name="text",
                display_name="Text Content",
                field_type="str",
                description="Text content to display"
            ),
            ComponentInput(
                name="format",
                display_name="Output Format",
                field_type="str",
                options=["plain", "markdown", "html"],
                default="plain",
                required=False,
                description="Format for displaying the text"
            )
        ]
        
        self.outputs = [
            ComponentOutput(
                name="formatted_text",
                display_name="Formatted Output",
                field_type="str",
                method="format_output",
                description="Formatted text output"
            ),
            ComponentOutput(
                name="metadata",
                display_name="Output Metadata",
                field_type="dict",
                method="get_output_metadata",
                description="Metadata about the output"
            )
        ]
    
    async def execute(self, **kwargs) -> Dict[str, Any]:
        text = kwargs.get("text", "")
        format_type = kwargs.get("format", "plain")
        
        # Format the text based on type
        if format_type == "markdown":
            formatted_text = f"```\n{text}\n```"
        elif format_type == "html":
            formatted_text = f"<div class='output'>{text}</div>"
        else:
            formatted_text = text
        
        metadata = {
            "format": format_type,
            "length": len(text),
            "word_count": len(text.split()) if text else 0,
            "line_count": len(text.split('\n')) if text else 0
        }
        
        return {
            "formatted_text": formatted_text,
            "metadata": metadata
        }