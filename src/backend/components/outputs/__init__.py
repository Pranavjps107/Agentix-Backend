# src/backend/components/outputs/__init__.py
"""
Output Components
"""

try:
    from .text_output import TextOutputComponent
except ImportError as e:
    # Fallback component
    import logging
    logging.warning(f"Failed to import TextOutputComponent: {e}")
    
    from ...core.base import BaseLangChainComponent, ComponentMetadata
    from ...core.registry import register_component  # Correct import
    
    @register_component
    class TextOutputComponent(BaseLangChainComponent):
        def _setup_component(self):
            self.metadata = ComponentMetadata(
                display_name="Text Output",
                description="Text output component (placeholder)",
                icon="📄",
                category="outputs",
                tags=["output"]
            )
            self.inputs = []
            self.outputs = []
        
        async def execute(self, **kwargs):
            return {"formatted_text": "Demo output", "metadata": {}}

__all__ = [
    "TextOutputComponent"
]